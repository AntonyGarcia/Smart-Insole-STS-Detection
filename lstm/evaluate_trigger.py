"""Score LSTM predictions as a sit-to-stand TRIGGER: output crosses a threshold -> one event.

Reads one or more <test>_predictions.csv files written by train_movement_lstm.py.
Several files (e.g. different seeds) are averaged into one ensemble signal.

Rules
  trigger        the output rises above `threshold` and stays above it for `min_above_s`;
                 after a trigger, no new one for `refractory_s` seconds
  movement span  labelled start s ... end e + hold (read back from the `expected` column)
  hit            a trigger inside [s - tolerance, e + hold + tolerance]; the first one counts
  miss           a movement with no trigger in its span
  false trigger  a trigger outside every movement span
  latency        trigger time - s  (negative = before the start mark)

Only samples the video covers (where labels exist) are scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 80


def movement_spans(expected: np.ndarray) -> list[tuple[int, int]]:
    """(start index, end+hold index) of each run where the labelled signal is non-zero."""
    on = np.r_[0, (expected > 0).astype(int), 0]
    edges = np.flatnonzero(np.diff(on))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def triggers(signal: np.ndarray, threshold: float, min_above_s: float, refractory_s: float) -> np.ndarray:
    need = max(1, round(min_above_s * SAMPLE_RATE))
    refractory = round(refractory_s * SAMPLE_RATE)
    fired, run, blocked_until = [], 0, -1
    for index, value in enumerate(signal):
        run = run + 1 if value > threshold else 0
        if run == need and index >= blocked_until:
            fired.append(index)
            blocked_until = index + refractory
    return np.array(fired, dtype=int)


def score(signal, expected, threshold, min_above_s, refractory_s, tolerance_s):
    spans = movement_spans(expected)
    fired = triggers(signal, threshold, min_above_s, refractory_s)
    tol = round(tolerance_s * SAMPLE_RATE)
    used = np.zeros(len(fired), dtype=bool)
    latencies = []
    for start, stop in spans:
        inside = np.flatnonzero((fired >= start - tol) & (fired < stop + tol))
        used[inside] = True
        if inside.size:
            latencies.append((fired[inside[0]] - start) / SAMPLE_RATE)
    latencies = np.array(latencies)
    return {
        "threshold": threshold,
        "movements": len(spans),
        "hits": len(latencies),
        "missed": len(spans) - len(latencies),
        "false": int((~used).sum()),
        "latency_median_s": float(np.median(latencies)) if latencies.size else np.nan,
        "latency_p90_s": float(np.percentile(latencies, 90)) if latencies.size else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", type=Path, help="<test>_predictions.csv (several = averaged)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--min-above-s", type=float, default=0.0, help="Output must stay above the threshold this long")
    parser.add_argument("--refractory-s", type=float, default=3.0, help="No new trigger for this long after one")
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.predictions]
    covered = frames[0]["video_covered"].astype(bool).to_numpy()
    expected = frames[0]["expected"].to_numpy()[covered]
    signal = np.mean([frame["lstm"].to_numpy()[covered] for frame in frames], axis=0)
    minutes = covered.sum() / SAMPLE_RATE / 60

    print(f"{len(frames)} prediction file(s), {minutes:.1f} min scored, min above {args.min_above_s:g} s, "
          f"refractory {args.refractory_s:g} s")
    print(f"{'thr':>5} {'hits':>9} {'missed':>6} {'false':>5} {'false/10min':>11} {'latency med':>11} {'p90':>6}")
    for threshold in args.thresholds:
        r = score(signal, expected, threshold, args.min_above_s, args.refractory_s, args.tolerance_s)
        print(f"{threshold:5.2f} {r['hits']:>4}/{r['movements']:<4} {r['missed']:>6} {r['false']:>5} "
              f"{r['false'] / minutes * 10:>11.1f} {r['latency_median_s']:>+10.2f}s {r['latency_p90_s']:>+5.2f}s")


if __name__ == "__main__":
    main()
