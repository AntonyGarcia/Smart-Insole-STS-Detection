"""Event-level, threshold-free comparison (the study protocol, §9b): FROC = caught share vs false alarms per hour.

For each model and run, 100 thresholds (quantiles of the model's own scores, pooled over participants) are scored with
the event rule of analysis/events.py. Summary: mean caught share over 0-5 false alarms/h, pooled over participants and
averaged over runs. Difference vs HGB: exact test swapping the two models within participants (2^11), BCa bootstrap CI.

Writes results/froc_counts.csv, froc_summary.csv and results/froc.md.

  python comparison/froc.py -j 12
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "comparison"))
from events import PARTICIPANTS, lstm_alert_times, score, window_alert_times  # noqa: E402
from score_detectors import MINUTES, RUNS as LSTM_RUNS, STAND_UPS, b3_alert_times, lstm_output  # noqa: E402
from score_models import DL, ML, RES, RUNS, outputs, runs_of  # noqa: E402
from stats_tools import bca, holm  # noqa: E402

MAX_FA, POINTS, REFERENCE = 5.0, 100, "HGB"
X = np.linspace(0, MAX_FA, 501)
HOURS = np.array([MINUTES[p] for p in PARTICIPANTS]) / 60
EVENTS = np.array([len(STAND_UPS[p]) for p in PARTICIPANTS])


def load(model: str, run: str):
    """participant -> (scores, function theta -> alert times)"""
    if model == "B3":
        s = {p: np.load(ROOT / "baselines" / "runs" / "b3" / f"{p}.npy").astype(float) for p in PARTICIPANTS}
        return {p: (v, lambda t, v=v: b3_alert_times(v, t)) for p, v in s.items()}
    if model == "LSTM_ramp":
        s = {p: lstm_output(LSTM_RUNS / "loo_final" / f"test_{p}" / run).astype(float) for p in PARTICIPANTS}
        return {p: (v, lambda t, v=v: lstm_alert_times(v, t)) for p, v in s.items()}
    out = outputs(model, run)
    return {p: (o["probability"].astype(float),
                lambda t, o=o: window_alert_times(o["probability"].astype(float), o["window_end_s"], t)) for p, o in out.items()}


def counts(args: tuple[str, str]) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    model, run = args
    data = load(model, run)
    pooled = np.concatenate([v for v, _ in data.values()])
    # dense where the score is high: only the top share of the scores can be an alert
    thresholds = np.quantile(pooled, 1 - np.geomspace(5e-4, 0.6, POINTS))
    hits, false = np.zeros((POINTS, len(PARTICIPANTS))), np.zeros((POINTS, len(PARTICIPANTS)))
    for j, p in enumerate(PARTICIPANTS):
        for i, t in enumerate(thresholds):
            s = score(data[p][1](t), STAND_UPS[p])
            hits[i, j], false[i, j] = s.hits, s.false
    return model, run, thresholds, hits, false


def area(hits: np.ndarray, false: np.ndarray, weights: np.ndarray) -> float:
    """Mean caught share over 0-5 FA/h. hits, false: (runs, points, participants); weights: (participants,)."""
    caught = hits @ weights / (EVENTS @ weights)
    rate = false @ weights / (HOURS @ weights)
    values = []
    for c, r in zip(caught, rate):
        order = np.argsort(r, kind="stable")
        best = np.r_[0.0, np.maximum.accumulate(c[order])]
        values.append(best[np.searchsorted(r[order], X, side="right")].mean())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=12)
    args = parser.parse_args()
    models = [m for m in ML + DL if runs_of(m)] + ["B3", "LSTM_ramp"]
    tasks = [(m, r) for m in models for r in (runs_of(m) if m in ML + DL else ["seed0"] if m == "B3" else RUNS)]
    collected: dict[str, list] = {m: [] for m in models}
    rows = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for model, run, thresholds, hits, false in pool.map(counts, tasks):
            collected[model].append((hits, false))
            rows += [{"model": model, "run": run, "theta": t, "caught": h.sum() / EVENTS.sum(), "fa_per_hour": f.sum() / HOURS.sum()}
                     for t, h, f in zip(thresholds, hits, false)]
    pd.DataFrame(rows).to_csv(RES / "froc_counts.csv", index=False)

    def stack(m: str, runs: int) -> tuple[np.ndarray, np.ndarray]:           # single-run models are repeated
        c = collected[m] * (runs if len(collected[m]) == 1 else 1)
        return np.array([x[0] for x in c]), np.array([x[1] for x in c])

    n, one = len(PARTICIPANTS), np.ones(len(PARTICIPANTS))
    runs = len(collected[REFERENCE])
    hr, fr = stack(REFERENCE, runs)
    swaps = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    table = []
    for m in models:
        h, f = stack(m, runs)
        count = lambda idx: np.bincount(idx, minlength=n).astype(float)  # noqa: E731
        est, lo, hi = bca(lambda idx: area(h, f, count(idx)), n, resamples=2000)
        row = {"model": m, "area": est, "area_lo": lo, "area_hi": hi}
        if m != REFERENCE:
            observed = area(h, f, one) - area(hr, fr, one)
            extreme = 0
            for s in swaps[: 2 ** (n - 1)]:                    # the statistic is antisymmetric: half of the swaps suffice
                a = area(np.where(s, hr, h), np.where(s, fr, f), one) - area(np.where(s, h, hr), np.where(s, f, fr), one)
                extreme += abs(a) >= abs(observed) - 1e-12
            d, dlo, dhi = bca(lambda idx: area(h, f, count(idx)) - area(hr, fr, count(idx)), n, resamples=2000)
            row.update({"difference": d, "lo": dlo, "hi": dhi, "p": extreme / 2 ** (n - 1)})
        table.append(row)
        print(f"{m} done", flush=True)
    adjusted = holm({r["model"]: r["p"] for r in table if "p" in r})
    lines = ["# FROC: mean caught share over 0–5 false alarms/h (the study protocol, §9b, exploratory; reference HGB)\n",
             "Thresholds are swept on the test data: this compares score quality, not operating points.\n",
             "| Model | Area, % [95 % CI] | Difference vs HGB, points [95 % CI] | permutation p (Holm) |", "|---|---|---|---|"]
    for r in sorted(table, key=lambda r: -r["area"]):
        r["holm_p"] = adjusted.get(r["model"], np.nan)
        diff = (f"{100 * r['difference']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {r['p']:.4f} ({r['holm_p']:.4f})"
                if "p" in r else "– | –")
        lines.append(f"| {r['model']} | {100 * r['area']:.1f} [{100 * r['area_lo']:.1f}, {100 * r['area_hi']:.1f}] | {diff} |")
    pd.DataFrame(table).to_csv(RES / "froc_summary.csv", index=False)
    (RES / "froc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
