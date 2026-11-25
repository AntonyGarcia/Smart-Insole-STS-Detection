"""B3: plantar-load rule, the insole analogue of a chair-exit alarm (no learning except its threshold).

Signal: total load of both feet (36 cells), divided by its 95th percentile over the first 120 s (as the LSTM),
smoothed with a CAUSAL 0.25 s moving average (only past samples). Rule (applied in analysis/score_detectors.py):
alert when the signal rises above theta after having been at or below theta for >= 1 s (seated),
then 5 s lock-out. Live alert time = the crossing sample (+1 sample). Theta is chosen like the other
secondary detectors (fewest false alarms with >= 95 % caught on the other participants).

Writes baselines/runs/b3/<ID>.npy (per-sample signal).

  python baselines/b3_load_rule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from events import PARTICIPANTS, RATE  # noqa: E402

CALIBRATION_S = 120
SMOOTH = RATE // 4


def main() -> None:
    out = ROOT / "baselines" / "runs" / "b3"
    out.mkdir(parents=True, exist_ok=True)
    for participant in PARTICIPANTS:
        folder = ROOT / "data_final" / participant
        load = sum(np.loadtxt(folder / f"pressure_{side}.csv", delimiter=",", skiprows=1)[:, 1:].sum(axis=1)
                   for side in ("left", "right"))
        load = load / np.percentile(load[:CALIBRATION_S * RATE], 95)
        causal = np.convolve(load, np.ones(SMOOTH) / SMOOTH)[:len(load)]       # sample i uses i-19 .. i
        np.save(out / f"{participant}.npy", causal.astype(np.float32))
        print(f"{participant}: {len(load)} samples, median load {np.median(causal):.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
