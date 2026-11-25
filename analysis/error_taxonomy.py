"""Error taxonomy (the study protocol, §8.5), descriptive, for every scored detector.

False alarms (results/alerts.csv, kind == "false"), first matching category:
  anticipatory      a labelled stand-up starts 1-3 s after the alert (just before its association window)
  stand_to_sit      HGB state (paper-1 model, baselines/runs/b1 windows) changes from Standing/Walking/Tandem
                    to Sitting within +-2 s of the alert
  in_chair          HGB state Sitting and, within +-2 s, feet moving (peak gyro > 50 deg/s) or one foot unloaded
                    (load < 3 % of its 95th percentile while the other > 5 %)
  seated_other      HGB state Sitting, no foot activity
  standing_gait     HGB state Standing / Walking / Tandem away from a sit transition
Misses (results/events.csv, tau empty):
  blocked_by_anticipatory   an anticipatory false alarm of the same detector/run came <= 5 s (lock-out) before the start
  genuine                   otherwise

Writes results/error_taxonomy.csv (counts per detector, run, category) and prints the mean count per run.

  python analysis/error_taxonomy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from events import LOCKOUT_S, PARTICIPANTS, RATE

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CAL_S = 120


def context(participant: str) -> dict:
    folder = ROOT / "data_final" / participant
    gyro, load = [], []
    for side in ("left", "right"):
        imu = np.loadtxt(folder / f"imu_{side}.csv", delimiter=",", skiprows=1)
        pressure = np.loadtxt(folder / f"pressure_{side}.csv", delimiter=",", skiprows=1)[:, 1:].sum(axis=1)
        gyro.append(np.linalg.norm(imu[:, 4:7], axis=1))
        load.append(pressure / np.percentile(pressure[:CAL_S * RATE], 95))
    windows = pd.read_csv(ROOT / "baselines" / "runs" / "b1" / f"{participant}_windows.csv")
    return {"gyro": np.maximum(*gyro), "load": np.array(load),
            "centre": ((windows.start_s + windows.end_s) / 2).to_numpy(), "state": windows.label.to_numpy()}


def state_at(ctx: dict, t: float) -> str:
    return ctx["state"][np.argmin(np.abs(ctx["centre"] - t))]


def classify_false(ctx: dict, t: float, starts: np.ndarray) -> str:
    if np.any((starts - t > 1.0) & (starts - t <= 3.0)):
        return "anticipatory"
    near = (ctx["centre"] >= t - 2) & (ctx["centre"] <= t + 2)
    states = ctx["state"][near]
    if any(a != "Sitting" and b == "Sitting" for a, b in zip(states[:-1], states[1:])):
        return "stand_to_sit"
    if state_at(ctx, t) == "Sitting":
        a, b = max(0, int((t - 2) * RATE)), int((t + 2) * RATE)
        moving = ctx["gyro"][a:b].max(initial=0) > 50
        left, right = ctx["load"][0][a:b], ctx["load"][1][a:b]
        one_foot = np.any(((left < 0.03) & (right > 0.05)) | ((right < 0.03) & (left > 0.05)))
        return "in_chair" if moving or one_foot else "seated_other"
    return "standing_gait"


def main() -> None:
    alerts = pd.read_csv(RES / "alerts.csv")
    events = pd.read_csv(RES / "events.csv")
    contexts = {p: context(p) for p in PARTICIPANTS}
    starts = {p: events[(events.participant == p)].drop_duplicates("stand_up").sort_values("stand_up").start_s.to_numpy()
              for p in PARTICIPANTS}
    rows = []
    false = alerts[alerts.kind == "false"]
    for (detector, run, participant), group in false.groupby(["detector", "run", "participant"]):
        for t in group.alert_s:
            rows.append((detector, run, participant, "false_alarm", classify_false(contexts[participant], t, starts[participant]), t))
    anticipatory = pd.DataFrame([r for r in rows if r[4] == "anticipatory"],
                                columns=["detector", "run", "participant", "type", "category", "t"])
    missed = events[events.tau.isna()]
    for r in missed.itertuples():
        a = anticipatory[(anticipatory.detector == r.detector) & (anticipatory.run == r.run)
                         & (anticipatory.participant == r.participant)]
        blocked = np.any((r.start_s - a.t > 0) & (r.start_s - a.t <= LOCKOUT_S))
        rows.append((r.detector, r.run, r.participant, "miss", "blocked_by_anticipatory" if blocked else "genuine", r.start_s))
    table = pd.DataFrame(rows, columns=["detector", "run", "participant", "type", "category", "t"])
    table.to_csv(RES / "error_list.csv", index=False)
    counts = table.groupby(["detector", "run", "type", "category"]).size().rename("count").reset_index()
    counts.to_csv(RES / "error_taxonomy.csv", index=False)
    runs = events.groupby("detector").run.nunique()
    summary = counts.groupby(["detector", "type", "category"])["count"].sum().unstack(["type", "category"]).fillna(0)
    summary = summary.div(runs.reindex(summary.index), axis=0)
    print("Mean count per run (false alarms by category; misses genuine / blocked by an anticipatory alarm):\n")
    print(summary.round(1).to_string())
    print(f"\nWrote {RES / 'error_taxonomy.csv'} and {RES / 'error_list.csv'}")


if __name__ == "__main__":
    main()
