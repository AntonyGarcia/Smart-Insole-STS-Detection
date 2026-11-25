"""the study protocol, §9: simulated power of the H1 test BEFORE B1 is scored.

Between-participant spread: SD of the LSTM's per-participant median onset latency (single models, theta 0.5,
averaged over the available runs of lstm/runs/loo_final), on the new event scoring (analysis/events.py).
The SD of the per-participant LSTM - B1 difference is assumed to be sqrt(2) x that SD (B1 assumed to vary as much,
independently) -- a conservative assumption, since it ignores the pairing within participants.
Power of the exact two-sided signed-rank test at the strictest Holm level 0.05/3, n = 11, for true mean
differences of 0.25 ... 1.5 s, 20,000 simulated studies each.

  python analysis/power_h1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from events import PARTICIPANTS, labels, lstm_alert_times, read_lstm_output, score
from stats_tools import signed_rank_test

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    medians = []
    for participant in PARTICIPANTS:
        stand_ups = labels(participant)
        per_run = []
        for run in sorted((ROOT / "lstm" / "runs" / "loo_final" / f"test_{participant}").glob("seed*")):
            if (run / "metrics.json").exists():
                result = score(lstm_alert_times(read_lstm_output(run), 0.5), stand_ups)
                per_run.append(np.nanmedian(result.onset_latency))
        medians.append(np.mean(per_run))
    medians = np.array(medians)
    sd_lstm = medians.std(ddof=1)
    sd_diff = np.sqrt(2) * sd_lstm
    print(f"LSTM per-participant median onset latency: mean {medians.mean():+.2f} s, SD {sd_lstm:.2f} s "
          f"-> assumed SD of differences {sd_diff:.2f} s")
    rng = np.random.default_rng(0)
    alpha = 0.05 / 3
    # Exact p depends only on the sign/rank pattern; precompute by simulation directly.
    for delta in (0.25, 0.5, 0.75, 1.0, 1.5):
        rejections = 0
        trials = 20_000
        for _ in range(trials):
            x = rng.normal(-delta, sd_diff, len(PARTICIPANTS))
            rejections += signed_rank_test(x)["p"] < alpha
        print(f"  true difference {-delta:+.2f} s: power {rejections / trials:.2f}")


if __name__ == "__main__":
    main()
