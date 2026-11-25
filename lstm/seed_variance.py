"""How much do the leave-one-out results move between training runs?

For a runs folder with 6 seeds per fold: pooled hits / false triggers / latency for
  * every single seed
  * the two disjoint 3-seed averages (seeds 0-2, seeds 3-5)
  * the 6-seed average
each with the fixed rule (0.5, 0 s) and with the threshold chosen on the other participants
(same cross-validated procedure as threshold_analysis.py, threshold only, min-above 0).

  python lstm/seed_variance.py --root lstm/runs/loo_final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_trigger import SAMPLE_RATE, score
from kfold_summary import load_fold

HERE = Path(__file__).resolve().parent
THRESHOLDS = np.round(np.arange(0.30, 0.951, 0.05), 2)


def pooled(rows: list[dict]) -> tuple[int, int, int, float]:
    return (sum(r["hits"] for r in rows), sum(r["movements"] for r in rows), sum(r["false"] for r in rows),
            float(np.median([r["latency_median_s"] for r in rows])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE / "runs" / "loo_final")
    parser.add_argument("--refractory-s", type=float, default=5.0)
    args = parser.parse_args()

    folds = {}
    for folder in sorted(p for p in args.root.glob("test_*") if p.is_dir()):
        expected, signals, _ = load_fold(folder)
        folds[folder.name.removeprefix("test_")] = (expected, np.array(signals))
    seeds = len(next(iter(folds.values()))[1])
    minutes = sum(len(e) for e, _ in folds.values()) / SAMPLE_RATE / 60

    def evaluate(label: str, pick: list[int]) -> tuple:
        sweep = {name: {t: score(s[pick].mean(axis=0), e, t, 0, args.refractory_s, 0.5) for t in THRESHOLDS}
                 for name, (e, s) in folds.items()}
        fixed = pooled([sweep[n][0.5] for n in folds])
        chosen_rows, chosen = [], []
        for name in folds:
            others = [n for n in folds if n != name]

            def f1(t):
                h, m, fa, _ = pooled([sweep[o][t] for o in others])
                return 2 * h / (m + h + fa)
            best = max(THRESHOLDS, key=lambda t: (round(f1(t), 6), -t))
            chosen.append(best)
            chosen_rows.append(sweep[name][best])
        cv = pooled(chosen_rows)
        print(f"{label:22s} fixed 0.5: {fixed[0]}/{fixed[1]} hits, {fixed[2]:2d} false ({fixed[2] / minutes * 10:.2f}/10 min), "
              f"latency {fixed[3]:+.2f} s   |   CV threshold ({min(chosen):.2f}-{max(chosen):.2f}): {cv[0]}/{cv[1]} hits, "
              f"{cv[2]:2d} false ({cv[2] / minutes * 10:.2f}/10 min), latency {cv[3]:+.2f} s")
        return fixed, cv

    print(f"{len(folds)} participants, {seeds} seeds, {minutes:.0f} min\n")
    singles = [evaluate(f"seed {k}", [k]) for k in range(seeds)]
    print()
    evaluate("average seeds 0-2", [0, 1, 2])
    if seeds >= 6:
        evaluate("average seeds 3-5", [3, 4, 5])
        evaluate("average seeds 0-5", list(range(6)))
    print("\nSingle seeds, spread (min-max): fixed 0.5 hits "
          f"{min(s[0][0] for s in singles)}-{max(s[0][0] for s in singles)}, false "
          f"{min(s[0][2] for s in singles)}-{max(s[0][2] for s in singles)}; CV hits "
          f"{min(s[1][0] for s in singles)}-{max(s[1][0] for s in singles)}, false "
          f"{min(s[1][2] for s in singles)}-{max(s[1][2] for s in singles)}")


if __name__ == "__main__":
    main()
