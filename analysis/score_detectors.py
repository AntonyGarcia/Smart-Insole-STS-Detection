"""Score every detector of the study protocol on data_final with one event rule (analysis/events.py).

Detector               output                                       threshold (fewest false alarms with >= 95 % caught)
LSTM          primary  lstm/runs/loo_final/test_i/seed_k            strict nesting: lstm/runs/nested_final/outer_i/test_j/seed_k
LSTM_approx            same outputs                                 other participants' outer outputs (same run)    [H4, H5]
LSTM_<cfg>             lstm/runs/ablation_<cfg>/test_i/seed_k       other participants' outer outputs (same run)    [H4]
B1                     baselines/runs/b1/b1_alerts.csv              none (fixed, deployed model)
B2                     baselines/runs/b2/test_i/seed_k.npz          other participants' outer outputs (same run)    [H5]
B3                     baselines/runs/b3/<ID>.npy                   other participants' outer outputs               [descriptive]
LSTM_ens3a/3b/6_*      average of runs {0,1,2} / {345,4,5} / all 6  fixed 0.5 ("_fixed") or other participants ("_sel")  [secondary]

LSTM_fullnorm_approx   lstm/runs/loo_fullnorm (pressure scale over the whole recording)   other participants  [sensitivity, §6]
LSTM_inside_approx     lstm/runs/loo_inside (primary target = B2's "inside" indicator)    other participants  [exploratory]
LSTM_W<W>s_approx      lstm/runs/window_<W>s, W = 1, 3, 4 s                              other participants  [exploratory W]
*_ext                  B2, LSTM_approx, LSTM_inside, LSTM_W4s with θ up to 0.99           other participants  [exploratory E3]

Writes results/events.csv (one row per detector x run x participant x stand-up: tau or empty if missed),
results/alerts.csv (every alert with its kind: hit / false / duplicate) and
results/participants.csv (one row per detector x run x participant: theta, counts, minutes).

  python analysis/score_detectors.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from events import (PARTICIPANTS, RATE, THRESHOLDS, LOCKOUT_S, choose_threshold, labels, lstm_alert_times,
                    minutes, rising_crossings, score, window_alert_times)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "lstm" / "runs"
OUT = ROOT / "results"
SEEDS = ["seed0", "seed1", "seed2", "seed345", "seed4", "seed5"]
CONFIGS = ["pressure", "imu", "left", "right"]
WINDOWS = [1, 3, 4]                     # exploratory window lengths in s (2 s = LSTM_approx)
THRESHOLDS_EXT = np.r_[THRESHOLDS, [0.96, 0.97, 0.98, 0.99]]   # exploratory E3: grid extended above 0.95
B3_THRESHOLDS = np.round(np.arange(0.20, 0.951, 0.05), 2)

STAND_UPS = {p: labels(p) for p in PARTICIPANTS}
MINUTES = {p: minutes(p) for p in PARTICIPANTS}
_cache: dict[Path, np.ndarray] = {}


def lstm_output(run_dir: Path) -> np.ndarray | None:
    if run_dir not in _cache:
        files = list(run_dir.glob("*_predictions.csv")) if (run_dir / "metrics.json").exists() else []
        _cache[run_dir] = pd.read_csv(files[0], usecols=["lstm"])["lstm"].to_numpy() if files else None
    return _cache[run_dir]


def b3_alert_times(signal: np.ndarray, theta: float) -> np.ndarray:
    above = signal > theta
    crossings = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    seated = [i for i in crossings if i >= RATE and not above[i - RATE:i].any()]   # >= 1 s at or below before
    fired = rising_crossings(np.isin(np.arange(len(signal)), seated).astype(float), 0.5, LOCKOUT_S,
                             np.arange(len(signal)) / RATE)
    return (fired + 1) / RATE


def sweep(alerts_at, participant: str, thresholds) -> dict[float, tuple[int, int, int]]:
    """theta -> (hits, events, false) for one output."""
    result = {}
    for theta in thresholds:
        s = score(alerts_at(theta), STAND_UPS[participant])
        result[theta] = (s.hits, s.events, s.false)
    return result


def pooled(sweeps: list[dict], thresholds) -> list[tuple[float, int, int, int]]:
    return [(t, sum(s[t][0] for s in sweeps), sum(s[t][1] for s in sweeps), sum(s[t][2] for s in sweeps))
            for t in thresholds]


class Writer:
    def __init__(self) -> None:
        self.events, self.participants, self.alerts = [], [], []

    def add(self, detector: str, run: str, participant: str, alerts: np.ndarray, theta, fallback, selection: str):
        s = score(alerts, STAND_UPS[participant])
        self.alerts += [{"detector": detector, "run": run, "participant": participant, "alert_s": t, "kind": k}
                        for t, k in s.alerts]
        self.participants.append({"detector": detector, "run": run, "participant": participant, "theta": theta,
                                  "fallback_below_floor": fallback, "selection": selection, "events": s.events,
                                  "hits": s.hits, "false": s.false, "duplicates": s.duplicates,
                                  "minutes": MINUTES[participant]})
        for k, ((start, end), tau) in enumerate(zip(STAND_UPS[participant], s.tau), start=1):
            self.events.append({"detector": detector, "run": run, "participant": participant, "stand_up": k,
                                "start_s": start, "end_s": end, "tau": tau})


def other_participants_rule(writer: Writer, detector: str, run: str, outputs: dict[str, object], alerts_fn, thresholds):
    """Threshold for i chosen on the outputs of the other participants (same detector, same run)."""
    if any(v is None for v in outputs.values()):
        print(f"  {detector} {run}: incomplete, skipped")
        return
    sweeps = {p: sweep(lambda t, o=outputs[p]: alerts_fn(o, t), p, thresholds) for p in PARTICIPANTS}
    for p in PARTICIPANTS:
        theta, fallback = choose_threshold(pooled([sweeps[q] for q in PARTICIPANTS if q != p], thresholds))
        writer.add(detector, run, p, alerts_fn(outputs[p], theta), theta, fallback, "other participants")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    w = Writer()
    lstm_alerts = lstm_alert_times

    # B1: fixed model, one deterministic output.
    b1 = pd.read_csv(ROOT / "baselines" / "runs" / "b1" / "b1_alerts.csv")
    for p in PARTICIPANTS:
        w.add("B1", "fixed", p, b1.loc[b1.participant == p, "alert_s"].to_numpy(), np.nan, False, "none")

    for run in SEEDS:
        outer = {p: lstm_output(RUNS / "loo_final" / f"test_{p}" / run) for p in PARTICIPANTS}
        # LSTM primary: strict nesting.
        for p in PARTICIPANTS:
            inner = [lstm_output(RUNS / "nested_final" / f"outer_{p}" / f"test_{q}" / run) for q in PARTICIPANTS if q != p]
            if outer[p] is None or any(v is None for v in inner):
                print(f"  LSTM {run} {p}: nested runs incomplete, skipped")
                continue
            others = [q for q in PARTICIPANTS if q != p]
            sweeps = [sweep(lambda t, o=o: lstm_alerts(o, t), q, THRESHOLDS) for o, q in zip(inner, others)]
            theta, fallback = choose_threshold(pooled(sweeps, THRESHOLDS))
            w.add("LSTM", run, p, lstm_alerts(outer[p], theta), theta, fallback, "strict nested")
        other_participants_rule(w, "LSTM_approx", run, outer, lstm_alerts, THRESHOLDS)
        # Sensitivity analysis (§6): pressure scaled over the whole recording instead of the first 120 s.
        fullnorm = {p: lstm_output(RUNS / "loo_fullnorm" / f"test_{p}" / run) for p in PARTICIPANTS}
        if all(v is not None for v in fullnorm.values()):
            other_participants_rule(w, "LSTM_fullnorm_approx", run, fullnorm, lstm_alerts, THRESHOLDS)
        for config in CONFIGS:
            outputs = {p: lstm_output(RUNS / f"ablation_{config}" / f"test_{p}" / run) for p in PARTICIPANTS}
            other_participants_rule(w, f"LSTM_{config}", run, outputs, lstm_alerts, THRESHOLDS)
        # Exploratory: B2's "inside" target; window lengths (alert = end of the block, first block = whole window).
        inside = {p: lstm_output(RUNS / "loo_inside" / f"test_{p}" / run) for p in PARTICIPANTS}
        if all(v is not None for v in inside.values()):
            other_participants_rule(w, "LSTM_inside_approx", run, inside, lstm_alerts, THRESHOLDS)
        for seconds in WINDOWS:
            outputs = {p: lstm_output(RUNS / f"window_{seconds}s" / f"test_{p}" / run) for p in PARTICIPANTS}
            if all(v is not None for v in outputs.values()):
                other_participants_rule(w, f"LSTM_W{seconds}s_approx", run, outputs,
                                        lambda o, t, n=seconds * RATE: lstm_alert_times(o, t, window=n), THRESHOLDS)
        b2 = {}
        for p in PARTICIPANTS:
            path = ROOT / "baselines" / "runs" / "b2" / f"test_{p}" / f"{run}.npz"
            b2[p] = dict(np.load(path)) if path.exists() else None
        b2_alerts = lambda o, t: window_alert_times(o["probability"], o["window_end_s"], t)  # noqa: E731
        other_participants_rule(w, "B2", run, b2, b2_alerts, THRESHOLDS)
        # Exploratory E3: same rule, threshold grid extended to 0.99 (the prespecified grid ends at 0.95).
        other_participants_rule(w, "B2_ext", run, b2, b2_alerts, THRESHOLDS_EXT)
        other_participants_rule(w, "LSTM_approx_ext", run, outer, lstm_alerts, THRESHOLDS_EXT)
        if all(v is not None for v in inside.values()):
            other_participants_rule(w, "LSTM_inside_ext", run, inside, lstm_alerts, THRESHOLDS_EXT)
        w4 = {p: lstm_output(RUNS / "window_4s" / f"test_{p}" / run) for p in PARTICIPANTS}
        if all(v is not None for v in w4.values()):
            other_participants_rule(w, "LSTM_W4s_ext", run, w4, lambda o, t: lstm_alert_times(o, t, window=4 * RATE),
                                    THRESHOLDS_EXT)
        print(f"{run} scored", flush=True)

    b3 = {p: np.load(ROOT / "baselines" / "runs" / "b3" / f"{p}.npy") for p in PARTICIPANTS}
    other_participants_rule(w, "B3", "fixed", b3, b3_alert_times, B3_THRESHOLDS)

    # Secondary: averaged runs (ensembles), fixed 0.5 and threshold from the other participants.
    for name, members in (("ens3a", SEEDS[:3]), ("ens3b", SEEDS[3:]), ("ens6", SEEDS)):
        outputs = {}
        for p in PARTICIPANTS:
            curves = [lstm_output(RUNS / "loo_final" / f"test_{p}" / r) for r in members]
            outputs[p] = None if any(c is None for c in curves) else np.mean(curves, axis=0)
        if any(v is None for v in outputs.values()):
            continue
        for p in PARTICIPANTS:
            w.add(f"LSTM_{name}_fixed", "avg", p, lstm_alerts(outputs[p], 0.5), 0.5, False, "fixed 0.5")
        other_participants_rule(w, f"LSTM_{name}_sel", "avg", outputs, lstm_alerts, THRESHOLDS)

    pd.DataFrame(w.participants).to_csv(OUT / "participants.csv", index=False)
    pd.DataFrame(w.events).to_csv(OUT / "events.csv", index=False)
    pd.DataFrame(w.alerts).to_csv(OUT / "alerts.csv", index=False)
    table = pd.DataFrame(w.participants)
    summary = table.groupby("detector").agg(runs=("run", "nunique"), events=("events", "sum"), hits=("hits", "sum"),
                                            false=("false", "sum"), minutes=("minutes", "sum"))
    summary["recall"] = summary.hits / summary.events
    summary["false_per_hour"] = summary.false / summary.minutes * 60
    print(summary[["runs", "recall", "false_per_hour"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"Wrote {OUT / 'participants.csv'} and {OUT / 'events.csv'}")


if __name__ == "__main__":
    main()
