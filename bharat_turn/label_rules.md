# Bharat-Turn-8k label rules

Implemented in `bharat_turn/labels.py`, tested in `tests/test_labels.py`. These follow Section 5 of the problem statement.

1. **Voice activity.** Per-speaker speech comes from separate channels if IndicVoices conversations are stereo. If they are mono, it comes from diarisation: Pulse batch `diarize=true&word_timestamps=true`, or pyannote.audio. A 30-minute dev subset is hand-corrected to measure diarisation error.
2. **IPUs.** An inter-pausal unit is a run of one speaker's words separated by less than 200 ms of silence. The 200 ms threshold matches eot-bench's minimum hold span.
3. **Silence between two IPUs of speaker A** (0.2 to 5.0 s; longer holds are dropped, as in eot-bench `metrics.py`):
   - **shift (EoT):** B takes the floor in the gap. B's non-backchannel IPU starts in the gap, or starts up to 1 s before A's next IPU (≤ 1 s overlap allowed).
   - **backchannel:** B produces under 1 s of speech made only of {haan, ha, hmm, achha, ji, haanji, ok, theek} and A continues. The silence still counts as a **hold** for A.
   - **hold:** otherwise, i.e. A resumes before B speaks.
4. **Barge-in:** B speech longer than 1 s that starts inside an A IPU, after which A stops within 1.5 s.
5. **Slices:** a hold is *filler/connective-preceded* if the last token before it is in `core.text.HOLD` (matlab, woh, ki, toh, aur, lekin, so, actually...). It is *digit/entity* if that token is a numeral. Channel slices (μ-law 8 kHz, MUSAN 10/5 dB) are applied at evaluation time by `eval/conditions.py`.
6. **Splits:** train/dev/test are speaker-disjoint and conversation-disjoint (`build_dataset.split_of`). The test split is never used for training, threshold tuning or early stopping.
