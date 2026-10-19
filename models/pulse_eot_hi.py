"""Pulse-EoT-hi v0: a causal, CPU-cheap Hindi/Hinglish end-of-turn hazard model.

At every 100 ms hop during user silence it fuses

* silence duration from a streaming Silero VAD (the hazard's time axis),
* prosody of the speech before the pause (final f0 vs speaker median, f0 / energy slopes,
  final lengthening, voicing, syllable-rate proxy), see core/prosody.py,
* a lexical head over the trailing ASR tokens that had *arrived* by t, plus their
  script-normalised classes (HOLD connective/filler, VERB, TAG closer, DIGIT), see
  models/lexical.py and core/text.py,
* ASR state (utterance closed by a final, time since the last partial, token count),

in a small gradient-boosted tree ensemble that outputs p_eot(t) = P(turn over | still silent
at t, features). Because training examples are drawn at every silence step at which the span
is still silent (the risk set), the classifier is a discrete-time hazard model over silence
duration. It is scored at every hop, including while the VAD still reports speech (the VAD lags the
true acoustic offset by 100 to 300 ms, and gating on it made v0 unable to act in the first
300 ms of a pause). The head ships as ONNX (models/artifacts/).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.asr import Partial, visible_text
from core.prosody import FEATURE_NAMES as PROSODY, ProsodyTracker
from core.text import CLASSES, is_tag_question, normalise, trailing_classes
from core.vad import SileroStream
from models.lexical import LexicalHead

ART = Path(__file__).resolve().parent / "artifacts"
MISSING = -999.0  # tree-friendly sentinel (also keeps the ONNX graph free of NaN handling)

LEX_CLASSES = [c for c in CLASSES if c != "NONE"]
FEATURES = (
    ["sil_s", "speech_8s", "vad_p_last", "vad_p_mean_160ms", "e_last_rel", *PROSODY]
    + ["has_text", "lex_logit", "n_tokens", "asr_closed", "since_partial_s", "tag_q"]
    + [f"c1_{c}" for c in LEX_CLASSES]
    + [f"c2_{c}" for c in LEX_CLASSES]
)
AUDIO_FEATURES = ["sil_s", "speech_8s", "vad_p_last", "vad_p_mean_160ms", "e_last_rel", *PROSODY]

VARIANTS = {
    "full": {"features": FEATURES, "drop_classes": []},
    "audio_only": {"features": AUDIO_FEATURES, "drop_classes": []},
    "no_lexhead": {"features": [f for f in FEATURES if f != "lex_logit"], "drop_classes": []},
    **{f"no_{c}": {"features": FEATURES, "drop_classes": [c]} for c in ("HOLD", "VERB", "TAG", "DIGIT")},
    # v0.1 "stacked": + frozen Smart Turn v3.2 probability, fused on clips Smart Turn never trained on
    "stacked": {"features": [*FEATURES, "st_prob"], "drop_classes": [], "smart_turn": True},
    "stacked_audio_only": {"features": [*AUDIO_FEATURES, "st_prob"], "drop_classes": [], "smart_turn": True},
    "smart_turn_only": {"features": ["sil_s", "st_prob"], "drop_classes": [], "smart_turn": True},
}


@lru_cache(maxsize=None)
def _shared_smart_turn():
    """One Smart Turn session per process, shared by every concurrent call (ORT run is thread-safe)."""
    from core.encoders import SmartTurnV32

    return SmartTurnV32()


@lru_cache(maxsize=None)
def _shared_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class FeatureState:
    """Streaming state shared by training-time featurisation and inference."""

    def __init__(self, lexical: LexicalHead | None, drop_classes: frozenset[str] = frozenset(), smart_turn: bool = False) -> None:
        self.lexical = lexical
        self.drop = drop_classes
        self.encoder = None
        if smart_turn:
            from adapters.baselines import AudioRing

            self.encoder, self.ring = _shared_smart_turn(), AudioRing()
        self.start_turn(16000)

    def start_turn(self, sr: int) -> None:
        self.sr = sr
        self.vad = SileroStream(sr=sr)
        self.pros = ProsodyTracker(sr=sr)
        self.partials: list[Partial] = []
        self._lex_cache: tuple[str, float] = ("", float("nan"))
        self._st: float = float("nan")  # Smart Turn prob for the current pause
        if self.encoder is not None:
            self.ring.reset(sr)

    def push_audio(self, frame: np.ndarray) -> None:
        self.vad.push(frame)
        self.pros.push(frame)
        if self.encoder is not None:
            self.ring.push(frame)
            if self.vad.probs and self.vad.probs[-1][1] >= 0.5:
                self._st = float("nan")  # speech again: the next pause gets a fresh call

    def push_partial(self, p: Partial) -> None:
        self.partials.append(p)

    def in_speech(self) -> bool:
        return self.vad.in_speech

    def features(self, t: float) -> dict[str, float]:
        sil = self.vad.trailing_silence(t)
        f: dict[str, float] = {"sil_s": sil}
        probs = self.vad.probs[-250:]  # last 8 s of 32 ms windows
        f["speech_8s"] = float(sum(1 for _, p in probs if p >= 0.5) * self.vad.win / self.sr)
        # raw VAD evidence: the hysteresis VAD lags the true offset by 100 to 300 ms, so the
        # head also sees the latest window probabilities and the energy drop directly
        f["vad_p_last"] = float(probs[-1][1]) if probs else float("nan")
        f["vad_p_mean_160ms"] = float(np.mean([p for _, p in probs[-5:]])) if probs else float("nan")
        e = np.array(self.pros.energy)
        f["e_last_rel"] = float(e[-5:].mean() - np.percentile(e, 90)) if len(e) >= 10 else float("nan")
        f.update(self.pros.features(sil))
        if self.encoder is not None:
            quiet = len(probs) >= 6 and all(p < 0.5 for _, p in probs[-6:])  # ~200 ms of raw non-speech
            if not np.isfinite(self._st) and quiet:
                self._st = self.encoder.prob(self.ring.buf)
            f["st_prob"] = self._st
        text, closed = visible_text(self.partials, t)
        toks = normalise(text)
        f["has_text"] = float(bool(toks))
        if text != self._lex_cache[0]:
            self._lex_cache = (text, self.lexical.logit(text) if (self.lexical and toks) else float("nan"))
        f["lex_logit"] = self._lex_cache[1]
        f["n_tokens"] = float(len(toks))
        f["asr_closed"] = float(closed)
        arrived = [p.t_arrival for p in self.partials if p.t_arrival <= t]
        f["since_partial_s"] = t - arrived[-1] if arrived else float("nan")
        f["tag_q"] = float(is_tag_question(toks)) if "TAG" not in self.drop else 0.0
        cls = trailing_classes(toks, 2)
        for i, c in enumerate(cls):
            c = "OTHER" if c in self.drop else c
            for k in LEX_CLASSES:
                f[f"c{i + 1}_{k}"] = float(c == k)
        return f


def vectorise(feats: dict[str, float], names: list[str]) -> np.ndarray:
    v = np.array([feats.get(n, np.nan) for n in names], dtype=np.float32)
    v[~np.isfinite(v)] = MISSING
    return v


class PulseEoTHi:
    """StreamingDetector implementation (adapters/base.py) backed by the ONNX fusion head."""

    score_point = None

    def __init__(self, variant: str, head_path: Path, lexical: LexicalHead | None, meta: dict) -> None:
        self.sess = _shared_session(str(head_path))
        self.input = self.sess.get_inputs()[0].name
        self.names = meta["features"]
        self.variant = variant
        self.name = f"pulse-eot-hi-v0:{variant}"
        self.uses_text = any(n in self.names for n in ("lex_logit", "has_text"))
        self.state = FeatureState(lexical, frozenset(meta.get("drop_classes", [])), bool(meta.get("smart_turn")))
        self.frozen_knobs = meta.get("dev_frozen_knobs_lat5")  # chosen on dev, never on a test set

    @classmethod
    def load(cls, variant: str = "full", tag: str = "") -> "PulseEoTHi":
        stem = f"pulse_eot_hi_{variant}{tag}"
        meta = json.loads((ART / f"{stem}.json").read_text())
        lex = ART / meta["lexical"] if meta.get("lexical") else None
        return cls(variant, ART / f"{stem}.onnx", LexicalHead.load(lex) if lex else None, meta)

    def start_turn(self, sr: int, agent_text: str | None) -> None:
        self.state.start_turn(sr)

    def push_audio(self, frame: np.ndarray) -> None:
        self.state.push_audio(frame)

    def push_partial(self, partial: Partial) -> None:
        self.state.push_partial(partial)

    def silence_s(self, t: float) -> float:
        return self.state.vad.trailing_silence(t)

    def score(self, t: float, need: bool) -> float:
        x = vectorise(self.state.features(t), self.names)[None]
        return float(self.sess.run(None, {self.input: x})[0][0, 0])
