"""Checks on results/models_events.csv: (1) why paired timing differences are exactly 0, with the mean difference as an
alternative summary; (2) whether seed 0 is typical for HGB; (3) which participants drive the false alarms.

  python comparison/check_oddities.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from events import PARTICIPANTS  # noqa: E402
from stats import paired_timing  # noqa: E402
from stats_tools import hodges_lehmann, holm, signed_rank_test  # noqa: E402

events, parts = pd.read_csv(ROOT / "results/models_events.csv"), pd.read_csv(ROOT / "results/models_participants.csv")
models = [m for m in dict.fromkeys(events.detector) if m != "HGB"]
print("timing vs HGB, per-participant MEAN paired difference (model - HGB), runs averaged; share of pairs with the same alert time")
p, rows = {}, {}
for m in models:
    d = paired_timing(events, "HGB", m)
    d["d"] = -d.d
    x = d.groupby(["participant", "run"]).d.mean().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
    p[m] = signed_rank_test(x, "two-sided")["p"]
    rows[m] = (hodges_lehmann(x, 0.95, "two-sided"), (d.d.abs() < 1e-9).mean(), d.d.abs().median())
adjusted = holm(p)
for m, (c, same, _) in rows.items():
    print(f"  {m:12s} {c['estimate']:+.3f} s [{c['lo']:+.3f}, {c['hi']:+.3f}]  p = {p[m]:.4f}  Holm {adjusted[m]:.4f}  same time {100 * same:.0f} %")
print("\ncaught (of 514) and false alarms per run")
g = parts.groupby(["detector", "run"])[["hits", "false"]].sum().unstack("run")
print(g.to_string())
print("\nfalse alarms per participant, mean over runs")
f = parts.groupby(["detector", "participant"])["false"].mean().unstack("participant").reindex(columns=PARTICIPANTS)
print(f.round(1).to_string())
