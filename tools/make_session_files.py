"""Write the session files that the LSTM scripts read, from the public dataset.

The dataset (https://doi.org/10.34740/KAGGLE/DSV/22827342) has one folder per participant:
  data_final/S01/imu_left.csv, imu_right.csv, pressure_left.csv, pressure_right.csv, labels.csv
This script adds, next to those folders:
  data_final/S01_session.json ... S11_session.json
Each file names the four signal files and lists the stand-ups (start_s, end_s, in seconds on the file axis).
The whole file is labelled, so video_duration_s is the length of the recording and the offset is 0.

  python tools/make_session_files.py                 (default: data_final/ in the repository root)
  python tools/make_session_files.py --data <folder>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 80
PARTICIPANTS = [f"S{i:02d}" for i in range(1, 12)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data_final")
    args = parser.parse_args()

    for participant in PARTICIPANTS:
        folder = args.data / participant
        with (folder / "imu_left.csv").open(encoding="utf-8") as handle:
            samples = sum(1 for _ in handle) - 1
        with (folder / "labels.csv").open(encoding="utf-8") as handle:
            stand_ups = [(float(row["start_s"]), float(row["end_s"])) for row in csv.DictReader(handle)]
        session = {
            "app": "insole-video-viewer-session",
            "version": 1,
            "note": "data_final: no video; files start seated before the first stand-up; times are file seconds",
            "video": None,
            "video_offset_s": 0.0,
            "video_duration_s": round(samples / RATE, 4),
            "imu": {side: f"{participant}/imu_{side}.csv" for side in ("left", "right")},
            "pressure": {side: f"{participant}/pressure_{side}.csv" for side in ("left", "right")},
            "movements": [{"name": f"movement {k}", "start_s": s, "end_s": e}
                          for k, (s, e) in enumerate(stand_ups, start=1)],
        }
        (args.data / f"{participant}_session.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
        print(f"{participant}: {len(stand_ups)} stand-ups, {samples / RATE / 60:.1f} min")


if __name__ == "__main__":
    main()
