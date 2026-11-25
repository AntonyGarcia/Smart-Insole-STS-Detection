"""Shared event-level scoring for every detector (the study protocol, §8.2-8.4).

Times are seconds on the data_final file axis (sample i is at i / 80 s).

Alert time tau = when a live system could raise the alert:
  LSTM            end of the 0.5 s processing block in which the output crosses theta (lstm_alert_times)
  window models   end of the window whose output crosses theta (B1 2 s windows, B2 2 s windows every 0.5 s)
  sample models   the crossing sample itself, +1 sample (B3)

Matching (one rule for all detectors):
  association window of stand-up i: [s_i - 1 s, e_i + 3 s]   (s = start of rise, e = fully standing)
  each alert goes to the unmatched stand-up whose window contains it, nearest start first;
  the first alert matched to a stand-up is its alert; a further alert inside an already matched
  stand-up's window is a "duplicate" (reported, not a false alarm); any other alert is a false alarm.
Outcomes per stand-up: detected, onset latency tau - s, lead time e - tau (> 0: alert before upright).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 80
BEFORE_S, AFTER_S = 1.0, 3.0
LOCKOUT_S = 5.0
THRESHOLDS = np.round(np.arange(0.30, 0.951, 0.05), 2)
ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = [f"S{i:02d}" for i in range(1, 12)]


def labels(participant: str, data: Path = ROOT / "data_final") -> np.ndarray:
    """(n, 2) start/end seconds of the labelled stand-ups."""
    with (data / participant / "labels.csv").open(encoding="utf-8") as handle:
        return np.array([(float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(handle)])


def minutes(participant: str, data: Path = ROOT / "data_final") -> float:
    with (data / participant / "imu_left.csv").open(encoding="utf-8") as handle:
        return (sum(1 for _ in handle) - 1) / RATE / 60


def rising_crossings(values: np.ndarray, threshold: float, lockout: float, times: np.ndarray) -> np.ndarray:
    """Indices where values goes above threshold (first sample counts if already above), then lock-out."""
    above = values > threshold
    candidates = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    fired, blocked_until = [], -np.inf
    for index in candidates:
        if times[index] >= blocked_until:
            fired.append(index)
            blocked_until = times[index] + lockout
    return np.array(fired, dtype=int)


def lstm_alert_times(signal: np.ndarray, threshold: float, window: int = 160, step: int = 40,
                     lockout: float = LOCKOUT_S) -> np.ndarray:
    """LSTM per-sample output -> live alert times (end of the block that produced the crossing sample).
    Lock-out counts from the crossing sample, as in the trigger rule."""
    crossings = rising_crossings(signal, threshold, lockout, np.arange(len(signal)) / RATE)
    available = np.maximum(window, np.ceil((crossings + 1) / step) * step)
    return np.minimum(available, len(signal)) / RATE


def window_alert_times(probability: np.ndarray, window_end_s: np.ndarray, threshold: float,
                       lockout: float = LOCKOUT_S) -> np.ndarray:
    """Window-level output (one value per window, available at the window end) -> alert times."""
    fired = rising_crossings(probability, threshold, lockout, window_end_s)
    return window_end_s[fired]


@dataclass
class Score:
    events: int
    hits: int
    false: int
    duplicates: int
    tau: np.ndarray          # (events,) alert time per stand-up, NaN if missed
    alerts: list             # (time, kind) for every alert: kind = "hit" | "false" | "duplicate"

    @property
    def onset_latency(self) -> np.ndarray:
        return self.tau - self._starts

    @property
    def lead_time(self) -> np.ndarray:
        return self._ends - self.tau


def score(alerts: np.ndarray, stand_ups: np.ndarray) -> Score:
    starts, ends = stand_ups[:, 0], stand_ups[:, 1]
    lo, hi = starts - BEFORE_S, ends + AFTER_S
    tau = np.full(len(stand_ups), np.nan)
    false = duplicates = 0
    kinds = []
    for t in np.sort(alerts):
        inside = np.flatnonzero((lo <= t) & (t <= hi))
        if inside.size == 0:
            false += 1
            kinds.append((float(t), "false"))
            continue
        open_ = inside[np.isnan(tau[inside])]
        if open_.size == 0:
            duplicates += 1
            kinds.append((float(t), "duplicate"))
            continue
        tau[open_[np.argmin(np.abs(starts[open_] - t))]] = t
        kinds.append((float(t), "hit"))
    result = Score(len(stand_ups), int(np.sum(~np.isnan(tau))), false, duplicates, tau, kinds)
    result._starts, result._ends = starts, ends
    return result


def choose_threshold(pooled: list[tuple[float, int, int, int]], min_recall: float = 0.95) -> tuple[float, bool]:
    """the study protocol, §6 rule on (theta, hits, events, false) pooled over the selection participants:
    fewest false alarms among thetas catching >= min_recall; ties -> lowest theta.
    If none reaches the floor: the theta catching the most (ties -> lowest theta). Returns (theta, fallback)."""
    feasible = [p for p in pooled if p[1] / p[2] >= min_recall]
    if feasible:
        return min(feasible, key=lambda p: (p[3], p[0]))[0], False
    return max(pooled, key=lambda p: (p[1], -p[0]))[0], True


def read_lstm_output(run_dir: Path) -> np.ndarray:
    """Per-sample LSTM output from a training run folder (data_final: every sample is labelled)."""
    path = next(run_dir.glob("*_predictions.csv"))
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(3, 4))
    assert values[:, 1].all(), f"{path}: data_final runs should be fully labelled"
    return values[:, 0]
