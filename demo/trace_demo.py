"""Offline demo: the live p_eot trace of one turn vs a 600 ms silence timer (demo shot 1 to 3).

    python -m demo.trace_demo --turn hi__4344          # an eot-bench hi turn (with Vosk partials)
    python -m demo.trace_demo --wav my_call.wav        # any 8/16 kHz mono wav (Vosk run live)

Writes demo/<name>.png: waveform, labelled pauses (if known), p_eot per 100 ms hop, the
model's eot events at its dev-frozen knobs and the 600 ms timer's events. A browser-mic / phone
demo needs the live Pulse socket (PENDING key): sidecar/eot_sidecar.py is that path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.asr import transcribe_stream  # noqa: E402
from core.audio import resample  # noqa: E402
from eval.data import Turn, load_eot_bench_hi  # noqa: E402
from eval.deployed import events  # noqa: E402
from eval.harness import replay  # noqa: E402
from models.pulse_eot_hi import ART, PulseEoTHi  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn")
    ap.add_argument("--wav")
    a = ap.parse_args()
    if a.turn:
        turn = next(t for t in load_eot_bench_hi() if t.id == a.turn)
    else:
        import soundfile as sf

        x, sr = sf.read(a.wav, dtype="float32")
        turn = Turn(Path(a.wav).stem, resample(x.mean(axis=1) if x.ndim > 1 else x, sr, 16000), 16000, [])
    parts = transcribe_stream(turn.audio, turn.sr)
    det = PulseEoTHi.load("stacked")
    tr = replay(det, turn, parts, pad_s=1.5, lazy=False)
    knobs = json.loads((ART / "pulse_eot_hi_stacked.json").read_text())["dev_frozen_knobs_lat5"]
    ev_model = events(tr.t, tr.p, tr.sil, knobs["threshold"], knobs["action_delay"], knobs["timeout"])
    ev_timer = events(tr.t, tr.p, tr.sil, -1.0, 0.6, np.inf)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    tx = np.arange(len(turn.audio)) / turn.sr
    a1.plot(tx, turn.audio, lw=0.4, color="0.4")
    for s in turn.spans:
        a1.axvspan(s.start, s.end, color="tab:green" if s.label == "eot" else "tab:orange", alpha=0.3)
    for p in parts:
        if p.is_final:
            a1.text(p.t_arrival, 0.9 * np.max(np.abs(turn.audio)), p.text[-25:], fontsize=6, rotation=20)
    a1.set_title(f"{turn.id}: orange = hold, green = true end of turn")
    a2.plot(tr.t, tr.p, label="p_eot (Pulse-EoT-hi v0.1)")
    a2.axhline(knobs["threshold"], ls=":", color="k", lw=0.8)
    for e in ev_model:
        a2.axvline(e, color="tab:blue", lw=2)
    for e in ev_timer:
        a2.axvline(e, color="tab:red", ls="--", lw=1)
    a2.set_xlabel("time (s); blue = model eot event, red dashed = 600 ms timer event")
    a2.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / f"{turn.id}.png"
    fig.savefig(path, dpi=120)
    print(json.dumps({"png": str(path), "model_events": ev_model, "timer600_events": ev_timer, "spans": [(s.start, s.end, s.label) for s in turn.spans]}))


if __name__ == "__main__":
    main()
