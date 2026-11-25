"""Summarise the leave-one-recording-out runs in runs/kfold (see run_kfold.sh).

For each held-out recording: trigger scores of every seed and of the 3-seed average,
curve metrics from metrics.json, a threshold sweep chart, and kfold_summary.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_trigger import SAMPLE_RATE, score

HERE = Path(__file__).resolve().parent
# Validated categorical slots 1-5 (light surface); low-contrast slots are direct-labelled.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, GRID, INK, INK_2, MUTED, AXIS = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def load_fold(folder: Path):
    seeds = sorted(folder.glob("seed*"))
    frames, metrics = [], []
    for seed in seeds:
        csv = next(seed.glob("*_predictions.csv"))
        frames.append(pd.read_csv(csv))
        m = json.loads((seed / "metrics.json").read_text())
        metrics.append(m.get("test", m))
    covered = frames[0]["video_covered"].astype(bool).to_numpy()
    expected = frames[0]["expected"].to_numpy()[covered]
    signals = [frame["lstm"].to_numpy()[covered] for frame in frames]
    return expected, signals, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "runs" / "kfold")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refractory-s", type=float, default=5.0)
    args = parser.parse_args()

    thresholds = np.round(np.arange(0.2, 0.951, 0.05), 2)
    rows, sweeps = [], {}
    for folder in sorted(p for p in args.root.glob("test_*") if p.is_dir()):
        name = folder.name.removeprefix("test_")
        expected, signals, metrics = load_fold(folder)
        minutes = len(expected) / SAMPLE_RATE / 60
        ensemble = np.mean(signals, axis=0)
        per_seed = [score(s, expected, args.threshold, 0, args.refractory_s, 0.5) for s in signals]
        ens = score(ensemble, expected, args.threshold, 0, args.refractory_s, 0.5)
        sweeps[name] = [score(ensemble, expected, t, 0, args.refractory_s, 0.5) for t in thresholds]
        n = ens["movements"]
        recall, false_rate = ens["hits"] / n, ens["false"] / minutes * 10
        precision = ens["hits"] / max(1, ens["hits"] + ens["false"])
        rows.append({
            "test": name, "movements": n, "minutes": round(minutes, 1),
            "ens_hits": ens["hits"], "ens_false": ens["false"],
            "ens_recall": recall, "ens_false_per_10min": false_rate,
            "ens_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "ens_latency_s": ens["latency_median_s"],
            "seed_hits": "/".join(str(r["hits"]) for r in per_seed),
            "seed_false": "/".join(str(r["false"]) for r in per_seed),
            "correlation": np.mean([m["correlation"] for m in metrics]),
            "timed_within_0.5s": np.mean([m["detected_within_tolerance"] / m["movements"] for m in metrics]),
        })
    table = pd.DataFrame(rows).sort_values("ens_f1", ascending=False)
    table.to_csv(args.root / "kfold_summary.csv", index=False)

    print(f"Trigger at threshold {args.threshold}, refractory {args.refractory_s:g} s; ens = average of seeds\n")
    print(f"{'held-out test':14s} {'moves':>5} {'ens caught':>10} {'ens false':>9} {'false/10min':>11} {'F1':>5} "
          f"{'latency':>8} {'seed caught':>12} {'seed false':>10} {'corr':>5} {'timed':>6}")
    for _, r in table.iterrows():
        print(f"{r.test:14s} {r.movements:>5} {r.ens_hits:>4}/{r.movements:<5} {r.ens_false:>9} "
              f"{r.ens_false_per_10min:>11.1f} {r.ens_f1:>5.2f} {r.ens_latency_s:>+7.2f}s {r.seed_hits:>12} "
              f"{r.seed_false:>10} {r.correlation:>5.2f} {r['timed_within_0.5s']:>6.0%}")

    # Threshold sweep: two separate charts side by side (no shared/dual axis).
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    order = list(table["test"])
    panels = [("Movements caught (%)", lambda r: 100 * r["hits"] / r["movements"]),
              ("False triggers per 10 min", None)]
    for axis, (title, _) in zip(axes, panels):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
        axis.tick_params(colors=MUTED, labelsize=9)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(title, loc="left", color=INK, fontsize=11)
        axis.set_xlabel("Trigger threshold", color=INK_2)
        axis.axvline(args.threshold, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    minutes_of = dict(zip(table["test"], table["minutes"]))
    caught_of = {n: [100 * r["hits"] / r["movements"] for r in sweeps[n]] for n in sweeps}
    false_of = {n: [r["false"] / minutes_of[n] * 10 for r in sweeps[n]] for n in sweeps}
    if len(sweeps) <= len(COLOURS):
        # Few folds: one colour per fold, direct labels at the line ends pushed apart so they don't collide.
        ends = sorted(((caught_of[n][-1], n) for n in sweeps), reverse=True)
        label_y, previous = {}, None
        for value, name in ends:
            y = value if previous is None else min(value, previous - 3.2)
            label_y[name], previous = y, y
        for slot, name in enumerate(sorted(sweeps)):
            for axis, values in ((axes[0], caught_of[name]), (axes[1], false_of[name])):
                axis.plot(thresholds, values, color=COLOURS[slot], linewidth=2, marker="o", markersize=4,
                          markeredgecolor=SURFACE, markeredgewidth=1, label=name)
            axes[0].annotate(name, (thresholds[-1] + 0.015, label_y[name]), va="center", fontsize=9, color=INK_2,
                             annotation_clip=False)
        axes[1].legend(title="Held-out test recording", frameon=False, fontsize=9, title_fontsize=9,
                       labelcolor=INK_2, loc="upper right")
    else:
        # Many folds: each fold as a thin neutral line, the mean over folds as the one coloured line.
        for name in sweeps:
            for axis, values in ((axes[0], caught_of[name]), (axes[1], false_of[name])):
                axis.plot(thresholds, values, color=AXIS, linewidth=1, alpha=0.9)
        for axis, of in ((axes[0], caught_of), (axes[1], false_of)):
            mean = np.mean([of[n] for n in sweeps], axis=0)
            axis.plot(thresholds, mean, color=COLOURS[0], linewidth=2.4, marker="o", markersize=4,
                      markeredgecolor=SURFACE, markeredgewidth=1)
            axis.annotate(f"mean of {len(sweeps)} folds", (thresholds[-1] + 0.015, mean[-1]), va="center",
                          fontsize=9, color=COLOURS[0], annotation_clip=False)
        axes[0].annotate("one line per held-out recording", (thresholds[0], 61.5), fontsize=9, color=MUTED)
    lowest = min(min(v) for v in caught_of.values())
    axes[0].set_ylim(min(60, 5 * np.floor(lowest / 5) - 2), 102)
    axes[1].set_ylim(bottom=0)
    axes[0].set_xlim(thresholds[0] - 0.02, thresholds[-1] + 0.17)
    axes[1].set_xlim(thresholds[0] - 0.02, thresholds[-1] + (0.17 if len(sweeps) > len(COLOURS) else 0.02))
    seeds = len(list(next(p for p in args.root.glob("test_*") if p.is_dir()).glob("seed*")))
    fig.suptitle(f"Leave-one-recording-out: train on the other {len(sweeps) - 1}, test on one  ·  "
                 f"average of {seeds} seed{'s' if seeds != 1 else ''}  ·  refractory {args.refractory_s:g} s",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    out = args.root / "kfold_threshold_sweep.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"\nWrote {args.root / 'kfold_summary.csv'} and {out}")


if __name__ == "__main__":
    main()
