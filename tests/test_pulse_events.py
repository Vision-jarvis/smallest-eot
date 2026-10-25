"""Pulse log -> end-of-turn events (shapes as logged from the live API, docs/pulse_api_notes.md)."""

from adapters.pulse_endpointing import final_times, partials_from_log


def _tr(t, text, final, last=False):
    return {"t_arrival": t, "msg": {"type": "transcription", "transcript": text, "is_final": final, "is_last": last}}


LOG = [
    {"t_arrival": 0.9, "msg": {"type": "speech_started", "timestamp": 0.5}},
    _tr(1.5, "a", False),
    _tr(3.0, "a b c", True),  # word-count final in the middle of speech
    _tr(3.6, "d", False),
    {"t_arrival": 4.6, "msg": {"type": "speech_ended", "timestamp": 4.0}},
    _tr(4.7, "d e", True),  # silence final (eou window expired)
    _tr(8.0, "", True, last=True),  # close_stream reply
]


def test_all_finals_excludes_is_last():
    assert final_times(LOG, "all") == [3.0, 4.7]


def test_silence_finals_need_speech_ended():
    assert final_times(LOG, "silence") == [4.7]


def test_partials_skip_empty():
    assert [p[1] for p in partials_from_log(LOG)] == ["a", "a b c", "d", "d e"]
