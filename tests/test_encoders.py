import numpy as np
import pytest

from core.encoders import log_mel, mel_filters, right_align

transformers = pytest.importorskip("transformers")


def test_mel_filters_match_transformers():
    from transformers.audio_utils import mel_filter_bank

    ref = mel_filter_bank(num_frequency_bins=201, num_mel_filters=80, min_frequency=0.0, max_frequency=8000.0,
                          sampling_rate=16000, norm="slaney", mel_scale="slaney")
    assert np.abs(ref.T - mel_filters()).max() < 1e-6


@pytest.mark.parametrize("normalise", [True, False])
def test_log_mel_matches_whisper_feature_extractor(normalise):
    x = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    fe = transformers.WhisperFeatureExtractor(chunk_length=8)
    ref = fe(right_align(x), sampling_rate=16000, return_tensors="np", padding="max_length", max_length=128000, truncation=True, do_normalize=normalise).input_features[0]
    assert np.abs(ref - log_mel(x, normalise)).max() < 1e-4
