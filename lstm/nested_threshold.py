"""Primary LSTM result: single models, threshold chosen by strict nesting (the study protocol, §6).

For every run (seed) k and held-out participant i:
  1. inner outputs: runs/nested_final/outer_<i>/test_<j>/seed<k>   (10 participants j, models never saw i)
  2. sweep theta over 0.30 ... 0.95 (lock-out 5 s, tolerance 0.5 s), pooled over the 10 j:
     keep thetas that catch >= --min-recall of the stand-ups; take the fewest false alarms; ties -> lowest theta
     (if none reaches the floor: the theta catching the most; flagged)
  3. apply that theta to i's outer model output: runs/loo_final/test_<i>/seed<k>
Per run: pooled hits / false alarms / onset latency over the 11 participants. Across the 6 runs: median [min-max].

Writes <out>/nested_threshold_per_participant.csv and nested_threshold_summary.csv.

  python lstm/nested_threshold.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_trigger import SAMPLE_RATE, score

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.round(np.arange(0.30, 0.951, 0.05), 2)


def load(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(next(folder.glob("*_predictions.csv")))
    covered = frame["video_covered"].astype(bool).to_numpy()
    return frame["expected"].to_numpy()[covered], frame["lstm"].to_numpy()[covered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nested", type=Path, default=HERE / "runs" / "nested_final")
    parser.add_argument("--outer", type=Path, default=HERE / "runs" / "loo_final")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--refractory-s", type=float, default=5.0)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    args = parser.parse_args()

    participants = sorted(p.name.removeprefix("test_") for p in args.outer.glob("test_*") if p.is_dir())
    seeds = sorted(p.name for p in (args.outer / f"test_{participants[0]}").glob("seed*"))
    rows, incomplete = [], []
    for seed in seeds:
        for i in participants:
            inner_dirs = [args.nested / f"outer_{i}" / f"test_{j}" / seed for j in participants if j != i]
            missing = [d for d in inner_dirs if not (d / "metrics.json").exists()]
            if missing:
                incomplete.append((seed, i, len(missing)))
                continue
            inner = [load(d) for d in inner_dirs]
            sweep = []
            for theta in THRESHOLDS:
                results = [score(s, e, theta, 0, args.refractory_s, args.tolerance_s) for e, s in inner]
                hits = sum(r["hits"] for r in results)
                moves = sum(r["movements"] for r in results)
                sweep.append((theta, hits / moves, sum(r["false"] for r in results)))
            feasible = [t for t in sweep if t[1] >= args.min_recall]
            if feasible:
                theta, inner_recall, inner_false = min(feasible, key=lambda t: (t[2], t[0]))
                fallback = False
            else:
                theta, inner_recall, inner_false = max(sweep, key=lambda t: (t[1], -t[0]))
                fallback = True
            expected, signal = load(args.outer / f"test_{i}" / seed)
            r = score(signal, expected, theta, 0, args.refractory_s, args.tolerance_s)
            rows.append({"seed": seed, "participant": i, "theta": theta, "inner_recall": round(inner_recall, 4),
                         "inner_false": inner_false, "fallback_below_floor": fallback,
                         "movements": r["movements"], "hits": r["hits"], "false": r["false"],
                         "minutes": len(expected) / SAMPLE_RATE / 60, "latency_median_s": r["latency_median_s"]})

    if incomplete:
        print(f"Skipped {len(incomplete)} (seed, participant) pairs with unfinished inner runs, e.g. {incomplete[:3]}")
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(args.nested / "nested_threshold_per_participant.csv", index=False)

    per_run = table.groupby("seed").agg(participants=("participant", "count"), hits=("hits", "sum"),
                                        movements=("movements", "sum"), false=("false", "sum"),
                                        minutes=("minutes", "sum"), latency=("latency_median_s", "median"),
                                        theta_min=("theta", "min"), theta_max=("theta", "max"),
                                        fallbacks=("fallback_below_floor", "sum"))
    per_run["recall"] = per_run["hits"] / per_run["movements"]
    per_run["false_per_hour"] = per_run["false"] / per_run["minutes"] * 60
    per_run.to_csv(args.nested / "nested_threshold_summary.csv")
    print(f"Single models, theta chosen by strict nesting (fewest false alarms with >= {args.min_recall:.0%} caught "
          f"on the 10 inner participants)\n")
    print(per_run[["participants", "hits", "movements", "recall", "false", "false_per_hour", "latency",
                   "theta_min", "theta_max", "fallbacks"]].to_string(float_format=lambda v: f"{v:.3f}"))
    complete = per_run[per_run["participants"] == len(participants)]
    if len(complete):
        def spread(column, fmt):
            v = complete[column]
            return f"{fmt.format(v.median())} [{fmt.format(v.min())}-{fmt.format(v.max())}]"
        print(f"\nOver {len(complete)} complete runs, median [min-max]: caught {spread('recall', '{:.1%}')}, "
              f"false alarms {spread('false_per_hour', '{:.1f}')} per hour, onset latency {spread('latency', '{:+.2f}')} s")


if __name__ == "__main__":
    main()
