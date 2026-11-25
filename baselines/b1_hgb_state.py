"""B1: the paper-1 bilateral HGB state classifier, used exactly as deployed in the insole viewer.

Per participant of data_final/: 160-sample (2 s) windows every 80 samples (1 s), raw CSV values in the
model's column order; alert = first window classified Standing or Walking after a run of Sitting
(sitting_transition_events of GUI/insole_video_viewer.py). Live alert time = END of that window.
B1 is a fixed, pre-trained model: one deterministic output per participant (no runs, no threshold).

Writes baselines/runs/b1/<ID>_windows.csv (window start/end, class, probabilities) and b1_alerts.csv.

  python baselines/b1_hgb_state.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from events import PARTICIPANTS, RATE  # noqa: E402

WINDOW, STRIDE = 160, 80
SENSORS = ("accel_x_g", "accel_y_g", "accel_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
           *(f"pressure_{i}" for i in range(18)))


def features(left: np.ndarray, right: np.ndarray, columns: list[str]) -> np.ndarray:
    index = []
    for column in columns:
        m = re.fullmatch(r"(left|right)_(.+)_(\d+)", column)
        index.append((m.group(1) == "right", SENSORS.index(m.group(2)), int(m.group(3))))
    foot, sensor, sample = np.array(index, dtype=np.intp).T
    return np.stack((left, right))[foot, :, sensor, sample].T


def transitions(labels: np.ndarray) -> list[int]:
    """Window indices where a Sitting run directly enters Standing or Walking."""
    events, sitting = [], False
    for index, label in enumerate(labels):
        if label == "Sitting":
            sitting = True
        elif sitting:
            if label in ("Standing", "Walking"):
                events.append(index)
            sitting = False
    return events


def main() -> None:
    model = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet.joblib")
    meta = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet_meta.joblib")
    classes = np.asarray(model.classes_)
    out = ROOT / "baselines" / "runs" / "b1"
    out.mkdir(parents=True, exist_ok=True)
    alerts = []
    for participant in PARTICIPANTS:
        folder = ROOT / "data_final" / participant
        feet = [np.hstack([np.loadtxt(folder / f"imu_{side}.csv", delimiter=",", skiprows=1)[:, 1:],
                           np.loadtxt(folder / f"pressure_{side}.csv", delimiter=",", skiprows=1)[:, 1:]])
                for side in ("left", "right")]
        n = len(feet[0])
        starts = np.arange(0, n - WINDOW + 1, STRIDE)
        offsets = starts[:, None] + np.arange(WINDOW)
        windows = [foot.T.astype(np.float32)[:, offsets].transpose(1, 0, 2) for foot in feet]
        proba = model.predict_proba(features(*windows, meta["feature_cols"]))
        labels = classes[proba.argmax(axis=1)]
        with (out / f"{participant}_windows.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start_s", "end_s", "label", *[f"p_{c}" for c in classes]])
            for a, lab, p in zip(starts, labels, proba):
                writer.writerow([a / RATE, (a + WINDOW) / RATE, lab, *[f"{x:.4f}" for x in p]])
        events = transitions(labels)
        alerts += [(participant, (starts[i] + WINDOW) / RATE) for i in events]
        print(f"{participant}: {len(starts)} windows, {len(events)} sit->stand/walk alerts")
    with (out / "b1_alerts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "alert_s"])
        writer.writerows(alerts)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
