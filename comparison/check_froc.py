"""Best caught share (%) at or below each false-alarm rate, mean over runs, from results/froc_counts.csv.

  python comparison/check_froc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "froc_counts.csv")
rates = [0, 0.43, 0.85, 1.7, 2.6, 3.4, 4.3, 5.0, 10, 20, 40]
rows = {}
for m, g in d.groupby("model", sort=False):
    rows[m] = [100 * np.mean([r[r.fa_per_hour <= x].caught.max() if (r.fa_per_hour <= x).any() else 0 for _, r in g.groupby("run")])
               for x in rates]
print(pd.DataFrame(rows, index=[f"<={x:g}/h" for x in rates]).T.round(1).to_string())
g = d[(d.model == "B3")].sort_values("theta")
print("\nB3 curve (theta, caught %, FA/h):")
print("  ".join(f"{t:.2f}:{100 * c:.0f}/{f:.1f}" for t, c, f in zip(g.theta, g.caught, g.fa_per_hour) if 0.4 < t < 1.3)[:1500])
g = d[(d.model == "logreg")].sort_values("theta")
print("\nlogreg curve:")
print("  ".join(f"{t:.4f}:{100 * c:.0f}/{f:.1f}" for t, c, f in zip(g.theta, g.caught, g.fa_per_hour) if c > 0.5)[:1500])
