"""RQ7 (exploratory, 2026-11-24): are all 48 channels needed? Nested NSGA-II channel selection with the reference HGB.

Design follows the author's ADNI protein-panel study (comparison/nsga2.py): two objectives, maximise the cross-validated
FROC area and minimise the number of channels; one representative panel = the smallest panel within 0.01 of the best;
selection frequency and the Nogueira stability index summarise which channels are chosen.

Outer loop  leave-one-participant-out (11 folds). The held-out participant is never used for the search, the
            threshold or the training of the evaluated models.
Genes       the 48 channels (6 IMU + 18 pressure per foot); a channel brings all 12 of its window features.
Fitness     (revised 2026-11-25; the first run used the mean window AUC, kept in comparison/ga_auc and results/ga_auc)
            HGB with the base setting of the reference model (max_iter 200, random_state 0) on the selected channels,
            5 participant-grouped inner folds of the 10 training participants (2 per fold) -> out-of-fold output for
            all 10. Alerts at the 128 logit-spaced thresholds of the tuned search, events matched, counts pooled over
            the 10 participants -> FROC area over 0-5 false alarms/h, as used to select tuned settings
            (tune_select.py). Event-level, threshold-free, weighted to the low false-alarm region the detector is
            used in, and every training participant is scored once.
NSGA-II     population 50, 40 generations, uniform crossover 0.8, mutation 1/48, panels of 1-48 channels,
            3 seeds per fold sharing one fitness cache.
Panel       per fold: the pooled final fronts of the 3 seeds; the smallest panel within 0.01 FROC area of the best.
Evaluation  the panel and all 48 channels, each: HGB trained on the 10 training participants; threshold from the
            same pooled out-of-fold counts with the paper's rule (fewest false alarms at >= 95 % caught, lowest
            threshold on ties; strictly nested); alerts and events scored on the held-out participant.
Test        exact Wilcoxon signed-rank on per-participant F1 (panel - all), Hodges-Lehmann estimate and 95 % CI.

  python comparison/ga_channels.py run -j 11      search + evaluation, one process per fold (restartable)
  python comparison/ga_channels.py analyse        tables: results/ga_channels.md, ga_channels_*.csv
  python comparison/ga_channels.py smoke          tiny GA on one fold, to /tmp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("analysis", "baselines", "comparison"):
    sys.path.insert(0, str(ROOT / folder))
from b2_hgb_window import cached  # noqa: E402
from events import PARTICIPANTS, choose_threshold, score, window_alert_times  # noqa: E402
from nsga2 import GASettings, _select_representative, nondominated_front, run_nsga2  # noqa: E402
from score_detectors import MINUTES, STAND_UPS  # noqa: E402

OUT = ROOT / "comparison" / "ga"
RES = ROOT / "results"
THETA = 1 / (1 + np.exp(-np.linspace(-5, 14.05, 128)))   # thresholds of the tuned search (tune_select.py)
X = np.linspace(0, 5.0, 501)                              # false alarms/h at which the FROC curve is read


def froc_area(caught: np.ndarray, rate: np.ndarray) -> float:
    """Mean over X of the best share caught at <= x false alarms/h (0 if none); same as tune_select.froc_area."""
    order = np.argsort(rate, kind="stable")
    best = np.r_[0.0, np.maximum.accumulate(caught[order])]
    return float(best[np.searchsorted(rate[order], X, side="right")].mean())
N_CHANNELS, INNER_FOLDS, GA_SEEDS, TOLERANCE = 48, 5, 3, 0.01
SETTINGS = GASettings(population_size=50, generations=40, crossover_probability=0.8, min_panel_size=1,
                      max_panel_size=N_CHANNELS, trace_every=5, near_optimal_tolerance=TOLERANCE)
CHANNEL_NAMES = [f"{side} {name}" for side in ("L", "R")
                 for name in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] + [f"p{i}" for i in range(18)]]
IMU = [c for c, n in enumerate(CHANNEL_NAMES) if "acc" in n or "gyro" in n]


def columns(channels) -> np.ndarray:
    """Feature columns of the given channels (layout of b2_hgb_window: 4 blocks of 48, then 2 blocks of 48 x 4)."""
    cols = []
    for c in channels:
        cols += [c, 48 + c, 96 + c, 144 + c] + [192 + 4 * c + q for q in range(4)] + [384 + 4 * c + q for q in range(4)]
    return np.array(sorted(cols))


def hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, random_state=0)


class Fold:
    """Data of one outer fold: 10 training participants (5 inner folds of 2) and the held-out participant."""

    def __init__(self, test: str, data: dict) -> None:
        self.test, self.data = test, data
        self.train = [p for p in PARTICIPANTS if p != test]
        order = np.random.default_rng([20261124, PARTICIPANTS.index(test)]).permutation(self.train)
        self.inner = [list(order[i::INNER_FOLDS]) for i in range(INNER_FOLDS)]
        self.events = sum(len(STAND_UPS[p]) for p in self.train)
        self.hours = sum(MINUTES[p] for p in self.train) / 60
        self._counts: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def stack(self, people, cols):
        return np.vstack([self.data[p][0][:, cols] for p in people]), np.concatenate([self.data[p][1] for p in people])

    def counts(self, channels) -> tuple[np.ndarray, np.ndarray]:
        """Hits and false alarms at each THETA, pooled over the out-of-fold outputs of the 10 training participants."""
        mask = np.zeros(N_CHANNELS, bool)
        mask[np.asarray(channels, int)] = True
        key = np.packbits(mask).tobytes()
        if key not in self._counts:
            oof = self.out_of_fold(np.flatnonzero(mask))
            hits, false = np.zeros(len(THETA)), np.zeros(len(THETA))
            for p in self.train:
                for t, theta in enumerate(THETA):
                    s = score(window_alert_times(oof[p], self.data[p][2], theta), STAND_UPS[p])
                    hits[t] += s.hits
                    false[t] += s.false
            self._counts[key] = (hits, false)
        return self._counts[key]

    def fitness(self, channels: np.ndarray) -> float:
        hits, false = self.counts(channels)
        return froc_area(hits / self.events, false / self.hours)

    def out_of_fold(self, channels) -> dict[str, np.ndarray]:
        cols, out = columns(channels), {}
        for valid in self.inner:
            x, y = self.stack([p for p in self.train if p not in valid], cols)
            model = hgb().fit(x, y)
            for p in valid:
                out[p] = model.predict_proba(self.data[p][0][:, cols])[:, 1]
        return out

    def evaluate(self, channels) -> dict:
        """Nested threshold from the inner out-of-fold outputs, then train on all 10 and score the held-out one."""
        hits, false = self.counts(channels)
        theta, fallback = choose_threshold([(float(THETA[t]), hits[t], self.events, false[t]) for t in range(len(THETA))])
        cols = columns(channels)
        x, y = self.stack(self.train, cols)
        prob = hgb().fit(x, y).predict_proba(self.data[self.test][0][:, cols])[:, 1]
        s = score(window_alert_times(prob, self.data[self.test][2], theta), STAND_UPS[self.test])
        caught = ~np.isnan(s.tau)
        return {"theta": theta, "fallback": bool(fallback), "events": s.events, "hits": s.hits, "false": s.false,
                "duplicates": s.duplicates, "minutes": MINUTES[self.test],
                "onset": float(np.median(s.onset_latency[caught])) if caught.any() else None,
                "lead": float(np.median(s.lead_time[caught])) if caught.any() else None,
                "f1": 2 * s.hits / (s.events + s.hits + s.false)}


def run_fold(args) -> str:
    test, settings, seeds, out_dir = args
    path = Path(out_dir) / f"fold_{test}.json"
    if path.exists():
        return f"{test}: exists"
    started = time.time()
    fold = Fold(test, {p: cached(p) for p in PARTICIPANTS})
    cache: dict[bytes, float] = {}
    runs, panels, metrics, sizes = [], [], [], []
    for seed_index in range(seeds):
        r = run_nsga2(fold.fitness, N_CHANNELS, settings, seed=int(1000 * PARTICIPANTS.index(test) + seed_index), cache=cache)
        runs.append({"seed_index": seed_index, "panel": r.selected_panel.tolist(), "fitness": r.selected_metric,
                     "front_panels": [p.tolist() for p in r.front_panels], "front_fitness": r.front_metric.tolist(),
                     "front_size": r.front_size.tolist(), "trace": r.trace, "evaluations": r.n_evaluations,
                     "cache_hits": r.n_cache_hits})
        panels += r.front_panels
        metrics += r.front_metric.tolist()
        sizes += r.front_size.tolist()
    metrics, sizes = np.array(metrics), np.array(sizes)
    keep = nondominated_front(np.column_stack([-metrics, sizes]))                 # pooled front of the seeds
    pooled = [panels[i] for i in np.flatnonzero(keep)]
    panel, panel_auc, _ = _select_representative(pooled, metrics[keep], sizes[keep], TOLERANCE)
    every = np.arange(N_CHANNELS)
    result = {"test": test, "inner_folds": fold.inner, "runs": runs, "panel": panel.tolist(), "fitness_metric": "FROC area 0-5 FA/h",
              "panel_fitness": panel_auc, "all_fitness": fold.fitness(every),
              "panel_outer": fold.evaluate(panel), "all_outer": fold.evaluate(every),
              "fitness_cache": len(cache), "minutes": (time.time() - started) / 60}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(path)
    return (f"{test}: panel of {len(panel)} channels, inner FROC area {panel_auc:.3f} (all 48: {result['all_fitness']:.3f}), "
            f"{result['minutes']:.0f} min")


def nogueira(panels: list[list[int]], d: int = N_CHANNELS) -> float:
    z = np.zeros((len(panels), d))
    for i, p in enumerate(panels):
        z[i, p] = 1
    m, k = len(panels), z.sum(axis=1).mean()
    s2 = m / (m - 1) * z.mean(axis=0) * (1 - z.mean(axis=0))
    return float(1 - s2.mean() / (k / d * (1 - k / d)))


def analyse() -> None:
    import pandas as pd
    from stats_tools import bca, hodges_lehmann, signed_rank_test
    folds = [json.loads((OUT / f"fold_{p}.json").read_text(encoding="utf-8")) for p in PARTICIPANTS]
    lines = ["# RQ7: nested NSGA-II channel selection with HGB, fitness = inner FROC area (exploratory)\n"]
    rows = []
    for f in folds:
        for arm in ("panel", "all"):
            o = f[f"{arm}_outer"]
            rows.append({"participant": f["test"], "arm": arm, "channels": len(f["panel"]) if arm == "panel" else N_CHANNELS, **o})
    per = pd.DataFrame(rows)
    per.to_csv(RES / "ga_channels_participants.csv", index=False)
    n = len(PARTICIPANTS)
    lines += ["| Channels | Caught % | False alarms/h | F1 | Onset s | Lead s |", "|---|---|---|---|---|---|"]
    summary = []
    for arm in ("panel", "all"):
        a = per[per.arm == arm].set_index("participant").reindex(PARTICIPANTS)
        g = {c: a[c].to_numpy(float) for c in ("events", "hits", "false", "minutes", "onset", "lead")}
        fns = {"caught": lambda i: 100 * g["hits"][i].sum() / g["events"][i].sum(),
               "fa_per_hour": lambda i: 60 * g["false"][i].sum() / g["minutes"][i].sum(),
               "f1": lambda i: 2 * g["hits"][i].sum() / (g["events"][i].sum() + g["hits"][i].sum() + g["false"][i].sum()),
               "onset": lambda i: np.nanmedian(g["onset"][i]), "lead": lambda i: np.nanmedian(g["lead"][i])}
        row = {"arm": arm}
        for k, fn in fns.items():
            row[k], row[k + "_lo"], row[k + "_hi"] = bca(fn, n)
        summary.append(row)
        label = "GA panel (per fold)" if arm == "panel" else "All 48"
        ci = lambda k, fmt: f"{row[k]:{fmt}} [{row[k + '_lo']:{fmt}}, {row[k + '_hi']:{fmt}}]"  # noqa: E731
        lines.append(f"| {label} | {ci('caught', '.1f')} | {ci('fa_per_hour', '.2f')} | {ci('f1', '.3f')} | "
                     f"{ci('onset', '.2f')} | {ci('lead', '.2f')} |")
    pd.DataFrame(summary).to_csv(RES / "ga_channels_summary.csv", index=False)

    wide = per.pivot(index="participant", columns="arm", values="f1").reindex(PARTICIPANTS)
    d = (wide["panel"] - wide["all"]).to_numpy()
    t, hl = signed_rank_test(d, "two-sided"), hodges_lehmann(d, 0.95, "two-sided")
    lines += ["", f"F1, panel − all 48 (exact signed-rank, n = {n}): {hl['estimate']:+.4f} [{hl['lo']:+.4f}, {hl['hi']:+.4f}], "
              f"p = {t['p']:.4f}"]
    sizes = [len(f["panel"]) for f in folds]
    lines.append(f"Panel size over folds: median {np.median(sizes):.0f}, range {min(sizes)}-{max(sizes)}")

    fold_panels = [f["panel"] for f in folds]
    seed_panels = [r["panel"] for f in folds for r in f["runs"]]
    freq = pd.DataFrame({"channel": CHANNEL_NAMES,
                         "sensor": ["IMU" if c in IMU else "pressure" for c in range(N_CHANNELS)],
                         "fold_panels": [np.mean([c in p for p in fold_panels]) for c in range(N_CHANNELS)],
                         "seed_panels": [np.mean([c in p for p in seed_panels]) for c in range(N_CHANNELS)]})
    freq.to_csv(RES / "ga_channels_frequency.csv", index=False)
    lines += [f"Nogueira stability: fold panels {nogueira(fold_panels):.2f}, seed panels {nogueira(seed_panels):.2f}",
              f"Panels with any IMU channel: {np.mean([any(c in IMU for c in p) for p in fold_panels]):.0%} of folds; "
              f"panels with channels of both feet: {np.mean([min(p) < 24 <= max(p) for p in fold_panels]):.0%}",
              "", "## Selection frequency (top 20 by fold panels)", "",
              "| Channel | Sensor | Fold panels | Seed panels |", "|---|---|---|---|"]
    lines += [f"| {r.channel} | {r.sensor} | {r.fold_panels:.2f} | {r.seed_panels:.2f} |"
              for r in freq.sort_values("fold_panels", ascending=False).head(20).itertuples()]
    fronts = [{"participant": f["test"], "seed": r["seed_index"], "channels": s, "froc_area": a}
              for f in folds for r in f["runs"] for s, a in zip(r["front_size"], r["front_fitness"])]
    pd.DataFrame(fronts).to_csv(RES / "ga_channels_fronts.csv", index=False)
    (RES / "ga_channels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "analyse", "smoke"])
    parser.add_argument("-j", "--jobs", type=int, default=11)
    args = parser.parse_args()
    if args.stage == "analyse":
        return analyse()
    if args.stage == "smoke":
        tiny = GASettings(population_size=8, generations=2, crossover_probability=0.8, min_panel_size=1,
                          max_panel_size=N_CHANNELS, trace_every=1, near_optimal_tolerance=TOLERANCE)
        print(run_fold((PARTICIPANTS[0], tiny, 1, str(Path(tempfile.gettempdir(), "sts_ga_smoke")))), flush=True)
        print(Path(tempfile.gettempdir(), "sts_ga_smoke", f"fold_{PARTICIPANTS[0]}.json").read_text()[:1500])
        return
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    tasks = [(p, SETTINGS, GA_SEEDS, str(OUT)) for p in PARTICIPANTS]
    with get_context("spawn").Pool(args.jobs) as pool:
        for message in pool.imap_unordered(run_fold, tasks):
            print(time.strftime("%F %T"), message, flush=True)
    print("ALL FOLDS DONE", flush=True)


if __name__ == "__main__":
    main()
