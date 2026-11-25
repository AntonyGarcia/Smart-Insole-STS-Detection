"""Tuned, strictly nested search (the study protocol, §9c): choose setting, threshold and winner for every left-out participant.

For outer participant i only outputs of models trained without i (and without the scored participant q) are used.
Writes comparison/tuning/counts/<model>_c<k>.npz (cache), comparison/tuning/selected.json, results/tuning_settings.csv.

  python comparison/tune_select.py -j 20
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "comparison"))
from events import PARTICIPANTS, choose_threshold, score, window_alert_times  # noqa: E402
from score_detectors import B3_THRESHOLDS, MINUTES, STAND_UPS, b3_alert_times  # noqa: E402
from tune_models import DL, ML, MODELS, OUT, PAIRS, SETTINGS, inner_path, settings  # noqa: E402

THETA = 1 / (1 + np.exp(-np.linspace(-5, 14.05, 128)))
X = np.linspace(0, 5.0, 501)
EVENTS = np.array([len(STAND_UPS[p]) for p in PARTICIPANTS], dtype=float)
HOURS = np.array([MINUTES[p] for p in PARTICIPANTS]) / 60
N = len(PARTICIPANTS)
FLOOR = 0.95


def count(args: tuple[str, int]):
    """hits[p, q, t], false[p, q, t]: participant p scored with the model trained without p and q. None if incomplete."""
    model, k = args
    cache = OUT / "counts" / f"{model}_c{k}.npz"
    if cache.exists():
        d = np.load(cache)
        return model, k, d["hits"], d["false"]
    if not all(inner_path(model, k, pair).exists() for pair in PAIRS):
        return model, k, None, None
    hits, false = np.zeros((N, N, len(THETA))), np.zeros((N, N, len(THETA)))
    for pair in PAIRS:
        d = np.load(inner_path(model, k, pair))
        for p, q in (pair, pair[::-1]):
            a, b = PARTICIPANTS.index(p), PARTICIPANTS.index(q)
            for t, theta in enumerate(THETA):
                s = score(window_alert_times(d[f"prob_{p}"], d[f"end_{p}"], theta), STAND_UPS[p])
                hits[a, b, t], false[a, b, t] = s.hits, s.false
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, hits=hits, false=false)
    return model, k, hits, false


def froc_area(caught: np.ndarray, rate: np.ndarray) -> float:
    order = np.argsort(rate, kind="stable")
    best = np.r_[0.0, np.maximum.accumulate(caught[order])]
    return float(best[np.searchsorted(rate[order], X, side="right")].mean())


def f1(hits, events, false):
    return 2 * hits / (events + hits + false)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=20)
    args = parser.parse_args()
    counts = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for model, k, hits, false in pool.map(count, [(m, k) for m in MODELS for k in range(SETTINGS)]):
            if hits is None:
                print(f"INCOMPLETE, setting excluded: {model} c{k}", flush=True)
            else:
                counts[model, k] = (hits, false)

    selected: dict = {m: {} for m in MODELS}
    rows = []
    for i, p in enumerate(PARTICIPANTS):
        others = np.array([q for q in range(N) if q != i])
        events, hours = EVENTS[others].sum(), HOURS[others].sum()
        for m in MODELS:
            best = None
            for k in range(SETTINGS):
                if (m, k) not in counts:
                    continue
                h, f = (c[others, i].sum(axis=0) for c in counts[m, k])
                a = froc_area(h / events, f / hours)
                rows.append({"model": m, "outer": p, "setting": k, "area": a})
                if best is None or a > best[0] + 1e-12:
                    best = (a, k, h, f)
            a, k, h, f = best
            theta, fallback = choose_threshold([(THETA[t], h[t], events, f[t]) for t in range(len(THETA))])
            t = int(np.flatnonzero(THETA == theta)[0])
            selected[m][p] = {"setting": k, "theta": float(theta), "fallback": bool(fallback), "area": a, "caught": h[t] / events,
                              "fa_per_hour": f[t] / hours, "f1": f1(h[t], events, f[t])}

    # B1 and B3: not trained on these data; B3's threshold from the other participants
    b1 = pd.read_csv(ROOT / "baselines" / "runs" / "b1" / "b1_alerts.csv")
    s1 = [score(b1.loc[b1.participant == p, "alert_s"].to_numpy(), STAND_UPS[p]) for p in PARTICIPANTS]
    b3 = {p: np.load(ROOT / "baselines" / "runs" / "b3" / f"{p}.npy").astype(float) for p in PARTICIPANTS}
    dense = np.linspace(0.2, 1.5, 131)

    def b3_counts(thresholds):
        s = [[score(b3_alert_times(b3[p], t), STAND_UPS[p]) for t in thresholds] for p in PARTICIPANTS]
        return np.array([[x.hits for x in r] for r in s], float), np.array([[x.false for x in r] for r in s], float)
    coarse, fine = b3_counts(B3_THRESHOLDS), b3_counts(dense)
    selected["B1"], selected["B3"], winners = {}, {}, {}
    for i, p in enumerate(PARTICIPANTS):
        others = np.array([q for q in range(N) if q != i])
        events, hours = EVENTS[others].sum(), HOURS[others].sum()
        h, f = sum(s1[q].hits for q in others), sum(s1[q].false for q in others)
        selected["B1"][p] = {"caught": h / events, "fa_per_hour": f / hours, "f1": f1(h, events, f), "area": -1.0}
        h, f = (c[others].sum(axis=0) for c in coarse)
        theta, fallback = choose_threshold([(t, h[j], events, f[j]) for j, t in enumerate(B3_THRESHOLDS)])
        t = int(np.flatnonzero(B3_THRESHOLDS == theta)[0])
        hd, fd = (c[others].sum(axis=0) for c in fine)
        selected["B3"][p] = {"theta": float(theta), "fallback": bool(fallback), "caught": h[t] / events, "fa_per_hour": f[t] / hours,
                             "f1": f1(h[t], events, f[t]), "area": froc_area(hd / events, fd / hours)}

        def rule_a(names):
            ok = [m for m in names if selected[m][p]["caught"] >= FLOOR - 1e-12]
            if not ok:
                return max(names, key=lambda m: selected[m][p]["caught"])
            return min(ok, key=lambda m: (round(selected[m][p]["fa_per_hour"], 9), -selected[m][p]["area"]))
        every = MODELS + ["B1", "B3"]
        winners[p] = {"A": rule_a(every), "A_ML": rule_a(ML), "A_DL": rule_a(DL),
                      "F1": max(every, key=lambda m: selected[m][p]["f1"]), "area": max(MODELS + ["B3"], key=lambda m: selected[m][p]["area"])}
    selected["_winners"] = winners
    (OUT / "selected.json").write_text(json.dumps(selected, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tuning_settings.csv", index=False)
    for m in MODELS:
        print(f"{m}: settings " + " ".join(f"{p}:c{selected[m][p]['setting']}" for p in PARTICIPANTS))
    print(pd.DataFrame(winners).T.to_string())
    print("distinct settings:", {m: len(settings(m)) for m in MODELS})


if __name__ == "__main__":
    main()
