"""Bharat-Turn-8k label construction from per-speaker voice activity + word timings.

Implements the rules in bharat_turn/label_rules.md (problem statement Section 5):

1. IPUs: per-speaker speech runs separated by >= 200 ms of silence.
2. A within-speaker silence (0.2 to 5.0 s, eot-bench's hold filter) between two IPUs of
   speaker A is
     * `hold`        if A resumes before the other speaker B speaks, or B only backchannels;
     * `backchannel` (recorded on the B side) if B produces < 1 s of speech made only of
                     backchannel tokens and A continues;
     * `shift` (EoT) if B takes the floor (B's IPU starts in the gap, or starts before A's
                     next IPU with <= 1 s overlap at the start).
3. Barge-in: B speech > 1 s that starts while A is speaking and A stops within 1.5 s.

Input is model-agnostic: `segments` per speaker as (start, end, text) IPUs, which come from
Pulse batch diarisation + word timestamps or pyannote + an ASR, see build_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.text import is_backchannel_only, normalise

MIN_GAP = 0.2
MAX_HOLD = 5.0
BC_MAX_S = 1.0
SHIFT_MAX_OVERLAP = 1.0
BARGE_MIN_S = 1.0
BARGE_STOP_S = 1.5


@dataclass(frozen=True)
class IPU:
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Event:
    kind: str  # hold | shift | backchannel | barge_in
    speaker: str  # whose silence (hold/shift) or who produced the overlap (backchannel/barge_in)
    start: float
    end: float
    prev_text: str = ""  # text of the IPU before the silence (filler / digit slices)


def merge_ipus(words: list[tuple[float, float, str]], speaker: str, min_gap: float = MIN_GAP) -> list[IPU]:
    """Word timings of one speaker -> IPUs (runs separated by >= min_gap silence)."""
    out: list[list] = []
    for s, e, w in sorted(words):
        if out and s - out[-1][1] < min_gap:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2] += " " + w
        else:
            out.append([s, e, w])
    return [IPU(speaker, a, b, t.strip()) for a, b, t in out]


def is_bc(ipu: IPU) -> bool:
    return ipu.dur < BC_MAX_S and is_backchannel_only(normalise(ipu.text))


def label_conversation(ipus: list[IPU]) -> list[Event]:
    ev: list[Event] = []
    speakers = sorted({u.speaker for u in ipus})
    by = {s: sorted((u for u in ipus if u.speaker == s), key=lambda u: u.start) for s in speakers}
    for a in speakers:
        others = [u for u in ipus if u.speaker != a]
        mine = by[a]
        for cur, nxt in zip(mine, mine[1:]):
            gap = (cur.end, nxt.start)
            if gap[1] - gap[0] < MIN_GAP:
                continue
            in_gap = [o for o in others if o.start < gap[1] and o.end > gap[0]]
            taker = [o for o in in_gap if not is_bc(o) and o.start >= gap[0] - SHIFT_MAX_OVERLAP]
            if taker:
                ev.append(Event("shift", a, gap[0], min(gap[1], taker[0].start) if taker[0].start > gap[0] else gap[0], cur.text))
                continue
            if gap[1] - gap[0] > MAX_HOLD:
                continue
            for o in in_gap:
                if is_bc(o):
                    ev.append(Event("backchannel", o.speaker, o.start, o.end, cur.text))
            ev.append(Event("hold", a, gap[0], gap[1], cur.text))
        if mine:  # last IPU of A followed by B: a shift if B speaks next
            last = mine[-1]
            nxt_o = [o for o in others if o.start >= last.end - SHIFT_MAX_OVERLAP and not is_bc(o)]
            if nxt_o:
                first = min(nxt_o, key=lambda o: o.start)
                ev.append(Event("shift", a, last.end, max(last.end, first.start), last.text))
        # barge-ins: B speech > 1 s starting inside an A IPU, A stops within 1.5 s
        for u in mine:
            for o in others:
                if u.start < o.start < u.end and o.dur > BARGE_MIN_S and u.end - o.start <= BARGE_STOP_S:
                    ev.append(Event("barge_in", o.speaker, o.start, o.end, u.text))
    return sorted(ev, key=lambda e: (e.start, e.kind))
