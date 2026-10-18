"""Our sweep must reproduce LiveKit's committed Hindi numbers exactly (validates the yardstick)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from eval import metrics as my

BASE = Path(__file__).resolve().parents[1] / "third_party/eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/hi"

# Published Hindi rows (eot-bench comparison report): FCR@300, FCR@600, Lat@5%
PUBLISHED = {
    "smart_turn_audio_adapter__cb4f57229a": (0.426, 0.065, 0.639),
    "vap_adapter__d8cef83e57": (0.495, 0.074, 0.655),
    "live_kit_turn_detector_adapter__turn_detector_v1": (0.105, 0.020, 0.454),
    "live_kit_turn_detector_mini_adapter__7cd280604d": (0.274, 0.054, 0.624),
}


@pytest.mark.skipif(not BASE.exists(), reason="clone livekit/eot-bench into third_party/ first")
@pytest.mark.parametrize("run", sorted(PUBLISHED))
def test_reproduces_published_hindi_rows(run):
    sp = json.loads((BASE / run / "manifest.json").read_text()).get("score_point")
    pred = pd.read_parquet(BASE / run / "predictions.parquet")
    _, s = my.summarise(pred, sp)
    fcr300, fcr600, lat5 = PUBLISHED[run]
    assert s["FCR@300"] == pytest.approx(fcr300, abs=6e-4)
    assert s["FCR@600"] == pytest.approx(fcr600, abs=6e-4)
    assert s["Lat@5%"] == pytest.approx(lat5, abs=6e-4)


@pytest.mark.skipif(not BASE.exists(), reason="clone livekit/eot-bench into third_party/ first")
def test_vad_baseline_matches_published():
    pred = pd.read_parquet(BASE / "smart_turn_audio_adapter__cb4f57229a/predictions.parquet")
    vb = my.vad_baseline(pred)
    assert vb[vb.mean_latency <= 0.3 + 1e-9].cutoff_rate.min() == pytest.approx(0.495, abs=6e-4)
    assert vb[vb.mean_latency <= 0.6 + 1e-9].cutoff_rate.min() == pytest.approx(0.074, abs=6e-4)
    assert vb[vb.cutoff_rate <= 0.05 + 1e-9].mean_latency.min() == pytest.approx(0.7, abs=1e-6)


def test_policy_on_toy_spans():
    # one hold of 0.5 s where p crosses 0.8 at 0.3 s, one eot where it crosses at 0.2 s
    rows = []
    for t, p in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.9), (0.4, 0.9), (0.5, 0.9)]:
        rows.append(dict(id="a", language="hi", span_index=0, timestamp=1 + t, silence_dur=t, p_eot=p, label="hold"))
    for t, p in [(0.0, 0.1), (0.1, 0.5), (0.2, 0.95), (0.3, 0.95)]:
        rows.append(dict(id="a", language="hi", span_index=1, timestamp=3 + t, silence_dur=t, p_eot=p, label="eot"))
    pred = pd.DataFrame(rows)
    sw = my.sweep(pred, None, thresholds=[0.8, 0.92], action_delays=[0.2], timeouts=[1.0])
    a = sw[sw.threshold == 0.8].iloc[0]
    assert a.cutoff_rate == 1.0 and a.mean_latency == pytest.approx(0.2)
    b = sw[sw.threshold == 0.92].iloc[0]
    assert b.cutoff_rate == 0.0 and b.mean_latency == pytest.approx(0.2)
