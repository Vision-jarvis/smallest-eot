"""eot-bench `hi` split into a 200-turn dev-test half and a 200-turn SEALED test half.

Why: all 400 turns were scored for v0, v0.1 and nine ablations, and the v0 -> v0.1 design
change was made after seeing v0's score (see REPORT section "What was looked at"). From
2026-10-20 the 400 turns are split by a salted SHA-256 of the turn id:

* `devtest` (200 turns): iterate here. Every number produced after 2026-10-20 that informs a
  decision is computed on this half only.
* `sealed` (200 turns): scored exactly once, for the final frozen-knob v1 run. Any run that
  touches it must pass `--final` and is appended to results/SEALED_LEDGER.md.

The sealed half was part of the 400 that informed one design iteration before the split
existed; it is "sealed from now on", not "never seen". The split is rank-based (the 200
smallest hashes are sealed), so it is exactly 200/200 and does not depend on file order.
LiveKit's committed predictions can be rescored on either half, so comparability survives.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALT = "pulse-eot-hi/eot-bench-hi-split/2026-10-20"
N_SEALED = 200
SPLIT_CSV = ROOT / "eval/eot_bench_hi_split.csv"
LEDGER = ROOT / "results/SEALED_LEDGER.md"
HALVES = ("devtest", "sealed", "all")


def _h(turn_id: str) -> str:
    return hashlib.sha256(f"{SALT}:{turn_id}".encode()).hexdigest()


def assign(ids: list[str], n_sealed: int = N_SEALED) -> dict[str, str]:
    """turn id -> 'sealed' | 'devtest'. Deterministic and order-independent."""
    ranked = sorted(set(ids), key=_h)
    sealed = set(ranked[:n_sealed])
    return {i: ("sealed" if i in sealed else "devtest") for i in ids}


def load_assignment() -> dict[str, str]:
    """The committed split (eval/eot_bench_hi_split.csv), so it never silently changes."""
    import pandas as pd

    df = pd.read_csv(SPLIT_CSV)
    return dict(zip(df["id"].astype(str), df["half"]))


def ids_of(half: str) -> set[str]:
    a = load_assignment()
    if half == "all":
        return set(a)
    return {i for i, h in a.items() if h == half}


def guard(half: str, final: bool, what: str) -> None:
    """Refuse to touch the sealed half without --final; log every run that does."""
    if half == "devtest":
        return
    if not final:
        raise SystemExit(
            f"Refusing to score the '{half}' half: it contains the sealed 200 turns. Iterate on "
            "--half devtest. For the one final frozen-knob run pass --final (it is logged in "
            "results/SEALED_LEDGER.md)."
        )
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = "unknown"
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Sealed-half ledger\n\nEvery run that scored the sealed eot-bench hi half. It should have one row per final model.\n\n| UTC time | git HEAD | half | what |\n|---|---|---|---|\n")
        f.write(f"| {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} | {rev} | {half} | {what} |\n")


def write_split_csv() -> None:
    """One-off: python -m eval.split  (writes eval/eot_bench_hi_split.csv from the data file)."""
    import pandas as pd

    from eval.data import EOT_BENCH_HI

    ids = pd.read_parquet(EOT_BENCH_HI, columns=["id"])["id"].astype(str).tolist()
    a = assign(ids)
    pd.DataFrame({"id": ids, "half": [a[i] for i in ids]}).to_csv(SPLIT_CSV, index=False)
    print(f"wrote {SPLIT_CSV}: " + ", ".join(f"{h}={sum(v == h for v in a.values())}" for h in ("devtest", "sealed")))


if __name__ == "__main__":
    write_split_csv()
