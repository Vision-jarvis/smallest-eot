import json

import numpy as np
import pytest

from models.export_onnx import export
from sidecar.eot_sidecar import SidecarCore
from sidecar.pulse_protocol import eot_event, pulse_url


def test_hgb_onnx_parity(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 6)).astype(np.float32)
    X[:, 3] = -999.0 * (rng.random(2000) < 0.3)  # sentinel for missing, as at runtime
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + (X[:, 3] < -1) > 0.7).astype(int)
    m = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X, y)
    assert export(m, 6, tmp_path / "h.onnx", X) < 1e-4


def test_pulse_url_and_event_shapes():
    u = pulse_url(sample_rate=8000, endpointing=False, eou_timeout_ms=1000)
    assert "model=pulse" in u and "language=hi" in u and "vad_events=true" in u and "eou_timeout_ms=1000" in u
    with pytest.raises(ValueError):
        pulse_url(eou_timeout_ms=50)
    ev = eot_event("s", 0.912345, 340.2, 12.4801)
    assert ev == {"type": "eot", "eot_finalized": True, "p_eot": 0.9123, "silence_ms": 340, "timestamp": 12.48, "session_id": "s"}
    json.dumps(ev)


class FakeDet:
    """Silence starts at 1.0 s; p_eot jumps after 0.3 s of silence."""

    def start_turn(self, sr, agent):
        pass

    def push_audio(self, f):
        pass

    def push_partial(self, p):
        self.last = p

    def silence_s(self, t):
        return max(0.0, t - 1.0)

    def score(self, t, need):
        return 0.9 if self.silence_s(t) >= 0.3 else 0.1


def test_sidecar_emits_one_eot_per_pause():
    core = SidecarCore(FakeDet(), {"threshold": 0.5, "action_delay": 0.2, "timeout": 2.0})
    core.on_pulse({"type": "transcription", "transcript": "kar dunga", "is_final": False})
    out = []
    for _ in range(150):  # 3 s of 20 ms frames
        out += core.on_audio(np.zeros(320, np.float32))
    assert len(out) == 1 and out[0]["type"] == "eot" and out[0]["timestamp"] == pytest.approx(1.3)
    assert core.detector.last.text == "kar dunga"
