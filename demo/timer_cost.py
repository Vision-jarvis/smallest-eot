"""The Day-1 picture: what each silence-timer setting costs on Hindi (eot-bench hi, real VAD).

    python -m demo.timer_cost      # reads results/eot_bench_hi/clean/deployed.csv, writes results/timer_cost_hi.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "results/eot_bench_hi/clean/deployed.csv")
    t = df[df.system.str.startswith("silence timer")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t.mean_latency * 1000, t.fcr * 100, marker="o", color="#1f6feb")
    for r in t.itertuples():
        ms = int(r.action_delay * 1000)
        if ms in (400, 600, 800, 1000):
            ax.annotate(f"{ms} ms timer\n{100 * r.fcr:.1f}% cut-offs", (r.mean_latency * 1000, r.fcr * 100), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("mean wait after the caller really finished (ms)")
    ax.set_ylabel("pauses where the agent cuts in (%)")
    ax.set_title("Silence timers on 400 real Hindi turns (eot-bench hi, Silero VAD)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "results/timer_cost_hi.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
