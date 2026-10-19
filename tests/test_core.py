import numpy as np
import pytest

from core.asr import Partial, visible_text
from core.audio import frames, mulaw_roundtrip
from core.text import ends_with_question, is_tag_question, normalise, token_class, trailing_classes
from eval.data import Span, Turn
from eval.deployed import events, score_turn
from eval.harness import bench_rows, grid_times, replay


def test_lexical_classes_survive_script():
    for roman, deva in [("matlab", "मतलब"), ("toh", "तो"), ("hai", "है"), ("ji", "जी")]:
        assert token_class(roman) == token_class(deva)
    assert token_class("कि") == "HOLD"
    assert token_class("करूंगा") == "VERB" and token_class("dunga") == "VERB"
    assert token_class("4572") == "DIGIT" and token_class("चार") == "DIGIT"
    assert trailing_classes(normalise("main kal branch mein jaake jama kar dunga")) == ["VERB", "OTHER"]
    assert is_tag_question(normalise("theek hai")) and is_tag_question(normalise("aa jaoge na"))
    assert ends_with_question("Aap kab tak pay kar payenge?")


def test_visible_text_respects_arrival():
    ps = [Partial(1.0, "haan", False), Partial(1.4, "haan dekhiye", False), Partial(2.0, "haan dekhiye woh", True), Partial(3.0, "matlab", False)]
    assert visible_text(ps, 0.9) == ("", False)
    assert visible_text(ps, 1.5) == ("haan dekhiye", False)
    assert visible_text(ps, 2.5) == ("haan dekhiye woh", True)
    assert visible_text(ps, 3.0)[0] == "haan dekhiye woh matlab"


def test_frames_are_20ms_pcm16():
    x = np.linspace(-1, 1, 16000 * 1 + 7, dtype=np.float32)
    fs = list(frames(x, 16000))
    assert all(len(f) == 320 for f in fs) and len(fs) == 51
    assert np.allclose(fs[0] * 32768, np.round(fs[0] * 32768))


def test_mulaw_removes_high_band():
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    assert np.sqrt(np.mean(mulaw_roundtrip(x, sr) ** 2)) < 0.02


def test_grid_matches_eot_bench():
    assert grid_times(1.0, 1.25) == [1.0, 1.1, 1.2, 1.25]
    assert grid_times(3.8, 4.0) == [3.8, 3.9, 4.0]


class Spy:
    """Records what it was shown; scores the time of the last partial it saw."""

    name, score_point, uses_text = "spy", None, True

    def start_turn(self, sr, agent_text):
        self.n, self.seen = 0, []

    def push_audio(self, f):
        self.n += len(f)

    def push_partial(self, p):
        self.seen.append(p.t_arrival)

    def silence_s(self, t):
        return 0.0

    def score(self, t, need):
        # causality: never more audio than t, never a partial from the future
        assert self.n <= int(round(t * 16000)) and all(a <= t + 1e-9 for a in self.seen)
        return float(len(self.seen))


def test_replay_is_causal_and_rows_follow_spans():
    turn = Turn("x", np.zeros(16000 * 2, np.float32), 16000, [Span(0.5, 0.8, "hold", 0), Span(1.5, 2.0, "eot", 1)])
    tr = replay(Spy(), turn, [Partial(0.55, "a", False), Partial(1.62, "b", True)])
    rows = bench_rows(turn, tr, None)
    assert [r["timestamp"] for r in rows if r["span_index"] == 0] == [0.5, 0.6, 0.7, 0.8]
    by_t = {r["timestamp"]: r["p_eot"] for r in rows}
    assert by_t[0.5] == 0 and by_t[0.6] == 1 and by_t[1.6] == 1 and by_t[1.7] == 2


def test_deployed_policy():
    t = np.round(np.arange(1, 31) * 0.1, 6)
    sil = np.where(t < 1.0, 0.0, t - 1.0)
    p = np.where(sil > 0.25, 0.9, 0.1)
    ev = events(t, p, sil, threshold=0.5, action_delay=0.2, timeout=2.0)
    assert ev == [pytest.approx(1.3)]
    turn = Turn("x", np.zeros(48000, np.float32), 16000, [Span(1.0, 3.0, "eot", 0)])
    assert score_turn(turn, ev, 3.0)["latency"] == pytest.approx(0.3)
