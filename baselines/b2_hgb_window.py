"""B2: HGB windowed sit-to-stand classifier trained on the SAME data, folds and runs as the LSTM (H5).

Input: the same 48 channels as the LSTM, pressure divided per foot by its 95th-percentile total load over the
first 120 s (as the LSTM; trees need no z-scoring). Windows: 2 s (160 samples), one every 0.5 s (40 samples),
ending at E = 160, 200, ...; the output for a window is available at E (live alert time = window end).
Features (paper-1 style): per channel mean, SD, min, max of the window + mean and SD of its four 0.5 s segments
-> 48 x 12 = 576 features. Target: the newest 0.5 s [E - 0.5 s, E) overlaps a stand-up (start .. end + 0.5 s hold),
the same "inside" target as the LSTM's auxiliary output.
Model: HistGradientBoostingClassifier(max_iter=200, random_state=run seed) as in paper 1.
Leave-one-participant-out, runs = seeds 0 1 2 345 4 5. Output per (test participant, run):
  baselines/runs/b2/test_<ID>/seed<k>.npz   window_end_s, probability

  python baselines/b2_hgb_window.py -j 6
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from events import PARTICIPANTS, RATE, labels  # noqa: E402

WINDOW, STEP, SEGMENTS = 160, 40, 4
HOLD_S = 0.5
CALIBRATION_S = 120
SEEDS = [0, 1, 2, 345, 4, 5]
OUT = ROOT / "baselines" / "runs" / "b2"


def load(participant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = ROOT / "data_final" / participant
    parts = []
    for side in ("left", "right"):
        imu = np.loadtxt(folder / f"imu_{side}.csv", delimiter=",", skiprows=1)[:, 1:]
        pressure = np.loadtxt(folder / f"pressure_{side}.csv", delimiter=",", skiprows=1)[:, 1:]
        pressure = pressure / (np.percentile(pressure[:CALIBRATION_S * RATE].sum(axis=1), 95) or 1.0)
        parts += [imu, pressure]
    x = np.hstack(parts).astype(np.float32)                                  # (T, 48), LSTM channel order
    ends = np.arange(WINDOW, len(x) + 1, STEP)
    windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW, axis=0)[ends - WINDOW]   # (W, 48, 160)
    segments = windows.reshape(len(ends), x.shape[1], SEGMENTS, WINDOW // SEGMENTS)
    feats = np.concatenate([windows.mean(-1), windows.std(-1), windows.min(-1), windows.max(-1),
                            segments.mean(-1).reshape(len(ends), -1), segments.std(-1).reshape(len(ends), -1)], axis=1)
    active = np.zeros(len(x), dtype=bool)
    for s, e in labels(participant):
        active[int(round(s * RATE)):int(round((e + HOLD_S) * RATE))] = True
    target = np.array([active[e - STEP:e].any() for e in ends])
    return feats, target, ends / RATE


def cache(participant: str) -> str:
    path = OUT / "features" / f"{participant}.npz"
    if not path.exists():
        feats, target, end_s = load(participant)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, feats=feats, target=target, end_s=end_s)
    return participant


def cached(participant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(OUT / "features" / f"{participant}.npz") as data:
        return data["feats"], data["target"], data["end_s"]


def fit_one(args: tuple[str, int]) -> str:
    from sklearn.ensemble import HistGradientBoostingClassifier
    test, seed = args
    path = OUT / f"test_{test}" / f"seed{seed}.npz"
    if path.exists():
        return f"{test} seed{seed}: exists"
    data = {p: cached(p) for p in PARTICIPANTS}
    x = np.vstack([data[p][0] for p in PARTICIPANTS if p != test])
    y = np.concatenate([data[p][1] for p in PARTICIPANTS if p != test])
    model = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x, y)
    probability = model.predict_proba(data[test][0])[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, window_end_s=data[test][2], probability=probability.astype(np.float32))
    return f"{test} seed{seed}: {len(y):,} training windows ({y.mean():.1%} positive), {model.n_iter_} iterations"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-j", "--jobs", type=int, default=6)
    args = parser.parse_args()
    tasks = [(p, s) for s in SEEDS for p in PARTICIPANTS]
    with ProcessPoolExecutor(args.jobs) as pool:
        print("features:", ", ".join(pool.map(cache, PARTICIPANTS)), flush=True)
        for message in pool.map(fit_one, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
