"""Can a different trigger threshold (or a minimum time above it) remove false triggers, and at what cost?

Uses the saved leave-one-out outputs (3-seed average per held-out participant), so nothing is retrained.

1. Pooled sweep over threshold x min-above: hits, misses, false triggers, median latency.
2. Peak LSTM value of every false trigger burst at the current rule (how high would the threshold need to be?).
3. Cross-validated choice: for each held-out participant the rule is picked on the OTHER 10 participants
   only (max pooled event F1, ties -> earlier/lower rule) and then applied to that participant.
   This is the honest number; picking the best rule on all 11 and reporting it is optimistic.

  python lstm/threshold_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_trigger import SAMPLE_RATE, score, triggers, movement_spans
from kfold_summary import load_fold

HERE = Path(__file__).resolve().parent


def f1(hits: int, movements: int, false: int) -> float:
    return 2 * hits / (movements + hits + false) if movements else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE / "runs" / "loo")
    parser.add_argument("--refractory-s", type=float, default=5.0)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    args = parser.parse_args()

    thresholds = np.round(np.arange(0.30, 0.951, 0.05), 2)
    min_above = [0.0, 0.05, 0.1, 0.2, 0.3]
    folds = {}
    for folder in sorted(p for p in args.root.glob("test_*") if p.is_dir()):
        expected, signals, _ = load_fold(folder)
        folds[folder.name.removeprefix("test_")] = (expected, np.mean(signals, axis=0))
    minutes = sum(len(e) for e, _ in folds.values()) / SAMPLE_RATE / 60

    # results[name][(thr, above)] = score dict
    results = {name: {(t, a): score(s, e, t, a, args.refractory_s, args.tolerance_s)
                      for t in thresholds for a in min_above} for name, (e, s) in folds.items()}

    def pooled(names, rule):
        rows = [results[n][rule] for n in names]
        hits = sum(r["hits"] for r in rows)
        moves = sum(r["movements"] for r in rows)
        false = sum(r["false"] for r in rows)
        lat = np.median(np.concatenate([[r["latency_median_s"]] for r in rows]))
        return hits, moves, false, lat

    print(f"Pooled over {len(folds)} participants, {minutes:.0f} min, lock-out {args.refractory_s:g} s "
          f"(latency = median over participants of their median, from the start mark)\n")
    print(f"{'thr':>5} " + "  ".join(f"{'above ' + format(a, 'g') + ' s':>22}" for a in min_above))
    print(f"{'':>5} " + "  ".join(f"{'hit/miss/false lat':>22}" for _ in min_above))
    for t in thresholds:
        cells = []
        for a in min_above:
            h, m, fa, lat = pooled(folds, (t, a))
            cells.append(f"{h:>3}/{m - h:<2}/{fa:<3} {lat:+.2f}s".rjust(22))
        print(f"{t:5.2f} " + "  ".join(cells))

    # 2. Peak of each false trigger burst at the current rule (0.5, 0 s).
    print("\nFalse triggers at threshold 0.5: peak of the averaged LSTM output in the 2 s after the trigger")
    tol = round(args.tolerance_s * SAMPLE_RATE)
    for name, (expected, signal) in folds.items():
        spans = movement_spans(expected)
        for i in triggers(signal, 0.5, 0.0, args.refractory_s):
            if not any(s - tol <= i < e + tol for s, e in spans):
                peak = signal[i:i + 2 * SAMPLE_RATE].max()
                above = (signal[i:i + 2 * SAMPLE_RATE] > 0.5).sum() / SAMPLE_RATE
                print(f"  {name:14s} t={i / SAMPLE_RATE:7.1f}s  peak {peak:.2f}  time above 0.5: {above:.2f} s")

    # 3. Cross-validated rule choice.
    print("\nRule chosen on the other 10 participants (max pooled F1), applied to the held-out one:")
    rules = [(t, a) for a in min_above for t in thresholds]          # order = preference on ties: earlier first
    totals = np.zeros(3, dtype=int)
    for name in folds:
        others = [n for n in folds if n != name]
        best = max(rules, key=lambda r: (round(f1(*(lambda h, m, fa, _: (h, m, fa))(*pooled(others, r))), 6),
                                         -r[1], -r[0]))
        r = results[name][best]
        totals += (r["hits"], r["movements"], r["false"])
        print(f"  {name:14s} rule thr {best[0]:.2f}, above {best[1]:g} s  ->  {r['hits']}/{r['movements']} hits, "
              f"{r['false']} false, latency {r['latency_median_s']:+.2f} s")
    base = pooled(folds, (0.5, 0.0))
    print(f"\n  cross-validated total: {totals[0]}/{totals[1]} hits, {totals[2]} false "
          f"({totals[2] / minutes * 10:.2f} per 10 min)")
    print(f"  fixed 0.5 / 0 s:       {base[0]}/{base[1]} hits, {base[2]} false ({base[2] / minutes * 10:.2f} per 10 min)")


if __name__ == "__main__":
    main()
