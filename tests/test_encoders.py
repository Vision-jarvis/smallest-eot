import numpy as np
import pytest

from core.encoders import MEL_PATH, log_mel, right_align

transformers = pytest.importorskip("transformers")


@pytest.mark.skipif(not MEL_PATH.exists(), reason="download models first (data/README.md)")
@pytest.mark.parametrize("normalise", [True, False])
def test_log_mel_matches_whisper_feature_extractor(normalise):
    x = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    fe = transformers.WhisperFeatureExtractor(chunk_length=8)
    ref = fe(right_align(x), sampling_rate=16000, return_tensors="np", padding="max_length", max_length=128000, truncation=True, do_normalize=normalise).input_features[0]
    assert np.abs(ref - log_mel(x, normalise)).max() < 1e-4
