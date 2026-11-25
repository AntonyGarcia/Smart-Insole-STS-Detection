"""The Movement signal: shared by the viewer (drawing) and the LSTM trainer (targets).

Keep this the single definition so the model learns exactly the curve the viewer shows.
"""

from __future__ import annotations

import math

import numpy as np


MOVEMENT_SIGNAL_HOLD = 0.5  # Seconds the signal stays at 1 after each end, before dropping to 0.
MOVEMENT_SIGMOID_STEEPNESS = 10.0  # Logistic slope over the start→end interval (rescaled to hit exactly 0 and 1).


def movement_signal(times: np.ndarray, movements: list[tuple[float, float]]) -> np.ndarray:
    """0 outside movements; a sigmoid from 0 at each start to 1 at its end, 1 for
    MOVEMENT_SIGNAL_HOLD seconds after the end, then 0 again. Overlaps take the maximum.

    `times` and the (start, end) pairs must use the same clock (e.g. video seconds).
    """
    times = np.asarray(times, dtype=float)
    signal = np.zeros_like(times)
    k = MOVEMENT_SIGMOID_STEEPNESS
    low, high = 1 / (1 + math.exp(k / 2)), 1 / (1 + math.exp(-k / 2))
    for start, end in movements:
        progress = np.clip((times - start) / (end - start), 0.0, 1.0)
        rise = (1 / (1 + np.exp(-k * (progress - 0.5))) - low) / (high - low)
        inside = (times >= start) & (times < end + MOVEMENT_SIGNAL_HOLD)
        signal = np.maximum(signal, np.where(inside, rise, 0.0))
    return signal


def movement_active(times: np.ndarray, movements: list[tuple[float, float]]) -> np.ndarray:
    """True wherever the signal belongs to a movement: from each start to end + hold."""
    times = np.asarray(times, dtype=float)
    active = np.zeros(times.shape, dtype=bool)
    for start, end in movements:
        active |= (times >= start) & (times < end + MOVEMENT_SIGNAL_HOLD)
    return active
