"""Score the models of the study protocol, §9b with the same event rule and the same threshold rule for all.

Threshold for participant i: other participants' outputs, same model and run (fewest false alarms with >= 95 % caught,
ties -> lowest), on GRID. HGB = B2's stored outputs, re-thresholded on GRID. B1, B3 and LSTM_ramp (= LSTM_approx) are
copied from results/ as context. A model with missing outputs is reported and left out, never scored partially.

Writes results/models_participants.csv, results/models_events.csv (same columns as participants.csv / events.csv).

  python comparison/score_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from events import PARTICIPANTS, window_alert_times  # noqa: E402
from score_detectors import Writer, other_participants_rule  # noqa: E402

RES = ROOT / "results"
RUNS = ["seed0", "seed1", "seed2", "seed345", "seed4", "seed5"]
ML, DL = ["logreg", "svm_rbf", "rf", "xgb", "HGB"], ["cnn", "tcn", "lstm", "gru", "cnn_gru"]   # transformer dropped 2026-11-24
CONTEXT = {"B1": "B1", "B3": "B3", "LSTM_approx": "LSTM_ramp"}
GRID = np.round(np.r_[0.02, np.arange(0.05, 0.951, 0.05), 0.96, 0.97, 0.98, 0.99, 0.995, 0.999], 3)


def folder(model: str) -> Path:
    return ROOT / "baselines" / "runs" / "b2" if model == "HGB" else ROOT / "comparison" / "runs" / model


def outputs(model: str, run: str) -> dict[str, dict] | None:
    paths = {p: folder(model) / f"test_{p}" / f"{run}.npz" for p in PARTICIPANTS}
    if not all(path.exists() for path in paths.values()):
        return None
    return {p: dict(np.load(path)) for p, path in paths.items()}


def runs_of(model: str) -> list[str]:
    return [r for r in RUNS if outputs(model, r) is not None]


def main() -> None:
    w = Writer()
    for model in ML + DL:
        runs = runs_of(model)
        expected = 1 if model in ("logreg", "svm_rbf") else len(RUNS)
        if len(runs) != expected:
            print(f"{model}: {len(runs)}/{expected} complete runs, LEFT OUT")
            continue
        for run in runs:
            other_participants_rule(w, model, run, outputs(model, run),
                                    lambda o, t: window_alert_times(o["probability"], o["window_end_s"], t), GRID)
    participants, events = pd.DataFrame(w.participants), pd.DataFrame(w.events)
    old_p, old_e = pd.read_csv(RES / "participants.csv"), pd.read_csv(RES / "events.csv")
    for source, name in CONTEXT.items():
        participants = pd.concat([participants, old_p[old_p.detector == source].assign(detector=name)])
        events = pd.concat([events, old_e[old_e.detector == source].assign(detector=name)])
    participants.to_csv(RES / "models_participants.csv", index=False)
    events.to_csv(RES / "models_events.csv", index=False)
    g = participants.groupby("detector", sort=False)
    table = pd.DataFrame({"runs": g.run.nunique(), "recall": g.hits.sum() / g.events.sum(),
                          "fa_per_hour": g.false.sum() / g.minutes.sum() * 60,
                          "theta_min": g.theta.min(), "theta_max": g.theta.max(),
                          "fallback": g.fallback_below_floor.sum()})
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
