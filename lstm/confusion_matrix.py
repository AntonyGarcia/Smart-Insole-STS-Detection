"""Event-level confusion matrix with 95 % confidence intervals for the leave-one-participant-out runs.

Events (per held-out participant, on the samples the video covers):
  positive   a labelled stand-up: span start ... end + hold, widened by the tolerance on both sides
  negative   a labelled period WITHOUT a stand-up: each stretch between those widened spans
             (also before the first and after the last), ignoring stretches shorter than --min-negative-s
  TP / FN    stand-up with / without a trigger inside its span
  FP / TN    no-stand-up period with / without any trigger inside it

Triggers are exactly those of kfold_summary.py: the seed-averaged output crosses --threshold,
then no new trigger for --refractory-s.

Confidence intervals: participant-level (cluster) bootstrap, i.e. resample the participants with
replacement and recompute every rate from the pooled counts (events within one person are not
independent). Wilson score intervals on the pooled counts are written to the CSV as well.

Outputs (in --root): confusion_matrix.png, confusion_matrix.pdf, confusion_matrix.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_trigger import SAMPLE_RATE, movement_spans, triggers
from kfold_summary import load_fold

HERE = Path(__file__).resolve().parent

# Sequential blue ramp (light -> dark) and chart ink.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
SURFACE, INK, INK_2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    on = np.r_[0, mask.astype(int), 0]
    edges = np.flatnonzero(np.diff(on))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def count_events(signal, expected, threshold, refractory_s, tolerance_s, min_negative_s):
    n = len(expected)
    tol = round(tolerance_s * SAMPLE_RATE)
    fired = triggers(signal, threshold, 0.0, refractory_s)
    spans = movement_spans(expected)

    tp = sum(bool(np.any((fired >= s - tol) & (fired < e + tol))) for s, e in spans)
    fn = len(spans) - tp

    positive_zone = np.zeros(n, dtype=bool)
    for s, e in spans:
        positive_zone[max(0, s - tol):min(n, e + tol)] = True
    fp = tn = dropped = 0
    false_triggers = int((~positive_zone[fired]).sum()) if fired.size else 0
    for a, b in runs(~positive_zone):
        has_trigger = bool(np.any((fired >= a) & (fired < b)))
        if b - a < min_negative_s * SAMPLE_RATE:
            dropped += has_trigger
            continue
        fp += has_trigger
        tn += not has_trigger
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "false_triggers": false_triggers, "false_triggers_in_dropped": dropped}


def rates(tp, fn, fp, tn) -> dict:
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "sensitivity (recall)": sens,
        "miss rate (FNR)": 1 - sens,
        "specificity": spec,
        "false positive rate": 1 - spec,
        "precision (PPV)": prec,
        "NPV": npv,
        "accuracy": (tp + tn) / (tp + fn + fp + tn),
        "F1": 2 * prec * sens / (prec + sens),
        "balanced accuracy": (sens + spec) / 2,
    }


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (centre - half, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE / "runs" / "loo")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refractory-s", type=float, default=5.0)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    parser.add_argument("--min-negative-s", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    per = []
    for folder in sorted(p for p in args.root.glob("test_*") if p.is_dir()):
        expected, signals, _ = load_fold(folder)
        counts = count_events(np.mean(signals, axis=0), expected, args.threshold, args.refractory_s,
                              args.tolerance_s, args.min_negative_s)
        per.append({"participant": folder.name.removeprefix("test_"), **counts})
    table = pd.DataFrame(per)
    cells = ["tp", "fn", "fp", "tn"]
    total = table[cells].sum()
    tp, fn, fp, tn = (int(total[c]) for c in cells)
    point = rates(tp, fn, fp, tn)

    # Participant-level bootstrap.
    rng = np.random.default_rng(args.seed)
    matrix = table[cells].to_numpy()
    draws = []
    for _ in range(args.bootstrap):
        pick = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        draws.append(rates(*pick))
    draws = pd.DataFrame(draws)
    lo, hi = draws.quantile(0.025), draws.quantile(0.975)

    pooled = {"sensitivity (recall)": (tp, tp + fn), "miss rate (FNR)": (fn, tp + fn),
              "specificity": (tn, tn + fp), "false positive rate": (fp, tn + fp),
              "precision (PPV)": (tp, tp + fp), "NPV": (tn, tn + fn), "accuracy": (tp + tn, tp + fn + fp + tn)}
    rows = []
    for name, value in point.items():
        w = wilson(*pooled[name]) if name in pooled else (np.nan, np.nan)
        rows.append({"metric": name, "value": value, "boot_lo": lo[name], "boot_hi": hi[name],
                     "wilson_lo": w[0], "wilson_hi": w[1]})
    summary = pd.DataFrame(rows)

    out_csv = args.root / "confusion_matrix.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# event-level, threshold {args.threshold}, refractory {args.refractory_s} s, tolerance "
                     f"{args.tolerance_s} s, bootstrap {args.bootstrap} over {len(table)} participants\n")
        summary.to_csv(handle, index=False)
        handle.write("\n")
        table.to_csv(handle, index=False)

    print(table.to_string(index=False))
    print(f"\nTP {tp}  FN {fn}  FP {fp}  TN {tn}   ({len(table)} participants)")
    if table["false_triggers_in_dropped"].sum():
        print(f"note: {table['false_triggers_in_dropped'].sum()} false trigger(s) fell in no-stand-up stretches "
              f"shorter than {args.min_negative_s} s and are not counted")
    print(f"\n{'metric':22s} {'value':>7}   {'95% CI (bootstrap)':>20}   {'95% CI (Wilson)':>18}")
    for r in rows:
        wil = "" if np.isnan(r["wilson_lo"]) else f"[{r['wilson_lo']:.1%}, {r['wilson_hi']:.1%}]"
        print(f"{r['metric']:22s} {r['value']:7.1%}   [{r['boot_lo']:6.1%}, {r['boot_hi']:6.1%}]   {wil:>18}")

    plot(tp, fn, fp, tn, point, lo, hi, len(table), args)
    print(f"\nWrote {out_csv}, {args.root / 'confusion_matrix.png'}, {args.root / 'confusion_matrix.pdf'}")


def plot(tp, fn, fp, tn, point, lo, hi, participants, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    # Rows = labelled truth, columns = detector output; colour = share of the row.
    grid = [
        [("TP", tp, "sensitivity (recall)"), ("FN", fn, "miss rate (FNR)")],
        [("FP", fp, "false positive rate"), ("TN", tn, "specificity")],
    ]
    figure, axis = plt.subplots(figsize=(6.4, 5.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    gap = 0.03
    for r, row in enumerate(grid):
        for c, (tag, count, metric) in enumerate(row):
            share = point[metric]
            colour = cmap(share)
            luminance = 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
            ink, ink_2 = ("#ffffff", "#e8eefa") if luminance < 0.45 else (INK, INK_2)
            axis.add_patch(FancyBboxPatch((c + gap, 1 - r + gap), 1 - 2 * gap, 1 - 2 * gap,
                                          boxstyle="round,pad=0,rounding_size=0.03", facecolor=colour,
                                          edgecolor=SURFACE, linewidth=2))
            cx, cy = c + 0.5, 1 - r + 0.5
            axis.text(cx, cy + 0.16, tag, ha="center", va="center", fontsize=10, color=ink_2)
            axis.text(cx, cy + 0.01, f"{count}", ha="center", va="center", fontsize=24, color=ink,
                      fontweight="semibold")
            axis.text(cx, cy - 0.17, f"{share:.1%} of row", ha="center", va="center", fontsize=10, color=ink)
            axis.text(cx, cy - 0.29, f"95% CI {lo[metric]:.1%}–{hi[metric]:.1%}", ha="center", va="center",
                      fontsize=9, color=ink_2)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_aspect("equal")
    axis.set_xticks([0.5, 1.5], ["Trigger", "No trigger"])
    axis.set_yticks([1.5, 0.5], [f"Stand-up\n(n = {tp + fn})", f"No stand-up\n(n = {fp + tn})"])
    axis.xaxis.set_ticks_position("top")
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Detector output", color=INK_2, labelpad=8)
    axis.set_ylabel("Labelled from video", color=INK_2, labelpad=8)
    axis.tick_params(length=0, colors=INK, labelsize=10)
    for side in axis.spines.values():
        side.set_visible(False)

    lines = [
        f"Sensitivity {point['sensitivity (recall)']:.1%} ({lo['sensitivity (recall)']:.1%}–{hi['sensitivity (recall)']:.1%})   "
        f"Specificity {point['specificity']:.1%} ({lo['specificity']:.1%}–{hi['specificity']:.1%})",
        f"Precision {point['precision (PPV)']:.1%} ({lo['precision (PPV)']:.1%}–{hi['precision (PPV)']:.1%})   "
        f"F1 {point['F1']:.3f} ({lo['F1']:.3f}–{hi['F1']:.3f})",
        f"Leave-one-participant-out, {participants} participants · 95% CIs: participant bootstrap · "
        f"threshold {args.threshold:g}, lock-out {args.refractory_s:g} s",
    ]
    figure.text(0.5, 0.075, lines[0], ha="center", fontsize=9.5, color=INK)
    figure.text(0.5, 0.045, lines[1], ha="center", fontsize=9.5, color=INK)
    figure.text(0.5, 0.012, lines[2], ha="center", fontsize=8, color=MUTED)
    figure.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.15)
    for suffix in ("png", "pdf"):
        figure.savefig(args.root / f"confusion_matrix.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    main()
