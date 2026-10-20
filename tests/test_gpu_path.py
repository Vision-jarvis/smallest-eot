"""GPU path on CPU: Smart Turn torch rebuild matches the ONNX; pause labelling rules."""

import numpy as np
import pytest

from bharat_turn.indicvoices_episodes import Cfg, pauses, recording_key, speech_runs

torch = pytest.importorskip("torch")


def test_smart_turn_torch_matches_published_onnx():
    from models.gpu.smart_turn_torch import ONNX_FP32, load_from_onnx, parity

    if not ONNX_FP32.exists():
        pytest.skip("Smart Turn fp32 ONNX not downloaded")
    assert parity(load_from_onnx(), n=2) < 1e-4


def test_speech_runs_merge_and_pauses():
    cfg = Cfg(merge_gap_s=0.1, min_speech_s=0.1)
    hop = 0.032
    p = np.zeros(200)
    p[10:40] = 0.9   # 0.32 .. 1.28 s
    p[42:60] = 0.9   # 64 ms gap: merged
    p[80:120] = 0.9  # 0.64 s pause before this run: a hold
    runs = speech_runs(p, hop, cfg)
    assert len(runs) == 2
    ps = pauses(runs, dur=200 * hop, is_last_segment=True, cfg=cfg)
    assert [x["label"] for x in ps] == ["hold", "eot"]
    assert ps[0]["end"] - ps[0]["start"] == pytest.approx(20 * hop)
    assert pauses(runs, 200 * hop, is_last_segment=False, cfg=cfg)[-1]["label"] == "hold"  # no EoT mid-recording


def test_recording_key():
    cfg = Cfg()
    assert recording_key("a/b/rec_0042_7.wav", cfg) == ("rec_0042", 7)
    assert recording_key("nochunk.wav", cfg) is None
