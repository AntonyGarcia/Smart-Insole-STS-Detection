"""Run the LSTM training jobs of the study protocol on one GPU, several at a time (cross-platform, no shell quoting).

Job sets (participants S01-S11 of data_final/, 6 runs each with seeds 0 1 2 345 4 5, pressure calibration 120 s):
  loo       leave-one-participant-out, all 48 channels            -> lstm/runs/loo_final/test_<i>/seed<k>        (66)
  nested    strict nested: outer i, inner test j, train the other 9 -> lstm/runs/nested_final/outer_<i>/test_<j>/seed<k> (660)
  ablation  leave-one-out with --channels pressure|imu|left|right   -> lstm/runs/ablation_<v>/test_<i>/seed<k>  (264)
  fullnorm  leave-one-out, pressure scale over the WHOLE recording (sensitivity analysis, §6)
                                                                    -> lstm/runs/loo_fullnorm/test_<i>/seed<k>    (66)
  window    leave-one-out with 1, 3 and 4 s windows (exploratory W) -> lstm/runs/window_<W>s/test_<i>/seed<k>  (198)
  inside    leave-one-out, primary target = B2's "inside" indicator (exploratory)
                                                                    -> lstm/runs/loo_inside/test_<i>/seed<k>      (66)

The validation tail (early stopping) is the last training participant in ORDER,.
Finished jobs (metrics.json present) are skipped, so the runner can be restarted at any time.

  python tools/run_lstm_jobs.py loo nested ablation -j 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LSTM = ROOT / "lstm"
DATA = ROOT / "data_final"
ORDER = ["S01", "S03", "S05", "S04", "S06", "S07", "S08", "S09", "S10", "S11", "S02"]  # S02 last = validation tail
SEEDS = [0, 1, 2, 345, 4, 5]
VARIANTS = ["pressure", "imu", "left", "right"]
WINDOWS = [1, 3, 4]


def job(test: str, train: list[str], seed: int, out: Path, extra: list[str], note: str) -> tuple[Path, list[str]]:
    command = [sys.executable, str(LSTM / "train_movement_lstm.py"),
               "--train", *[str(DATA / f"{t}_session.json") for t in train],
               "--test", str(DATA / f"{test}_session.json"), "--seed", str(seed), "--out", str(out),
               "--pressure-calibration-s", "120", "--note", note, *extra]
    return out, command


def build(sets: list[str]) -> list[tuple[Path, list[str]]]:
    jobs = []
    for seed in SEEDS:                      # seed-major: every set becomes usable run by run
        if "loo" in sets:
            for i in ORDER:
                jobs.append(job(i, [p for p in ORDER if p != i], seed, LSTM / "runs" / "loo_final" / f"test_{i}" / f"seed{seed}",
                                [], f"fold: test {i}"))
        if "fullnorm" in sets:              # sensitivity analysis: pressure scale over the whole recording
            for i in ORDER:
                out, command = job(i, [p for p in ORDER if p != i], seed,
                                   LSTM / "runs" / "loo_fullnorm" / f"test_{i}" / f"seed{seed}",
                                   ["--no-charts"], f"fullnorm: test {i}")
                command[command.index("--pressure-calibration-s") + 1] = "0"
                jobs.append((out, command))
        if "inside" in sets:                # exploratory: B2's on/off target instead of the Movement ramp
            for i in ORDER:
                jobs.append(job(i, [p for p in ORDER if p != i], seed,
                                LSTM / "runs" / "loo_inside" / f"test_{i}" / f"seed{seed}",
                                ["--target", "inside", "--no-charts"], f"inside target: test {i}"))
        if "window" in sets:                # exploratory W: window length (2 s = loo_final)
            for w in WINDOWS:
                for i in ORDER:
                    jobs.append(job(i, [p for p in ORDER if p != i], seed,
                                    LSTM / "runs" / f"window_{w}s" / f"test_{i}" / f"seed{seed}",
                                    ["--window-s", str(w), "--no-charts"], f"window {w} s: test {i}"))
        if "ablation" in sets:
            for v in VARIANTS:
                for i in ORDER:
                    jobs.append(job(i, [p for p in ORDER if p != i], seed,
                                    LSTM / "runs" / f"ablation_{v}" / f"test_{i}" / f"seed{seed}",
                                    ["--channels", v, "--no-charts"], f"ablation {v}: test {i}"))
        if "nested" in sets:
            for i in ORDER:
                for j in ORDER:
                    if j != i:
                        jobs.append(job(j, [p for p in ORDER if p not in (i, j)], seed,
                                        LSTM / "runs" / "nested_final" / f"outer_{i}" / f"test_{j}" / f"seed{seed}",
                                        ["--no-charts"], f"nested: outer {i}, inner test {j}"))
    return jobs


def run(out: Path, command: list[str]) -> tuple[Path, int, float]:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / "train.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=LSTM).returncode
    return out, code, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", choices=["loo", "nested", "ablation", "fullnorm", "window", "inside"])
    parser.add_argument("-j", "--jobs", type=int, default=6, help="Trainings running at the same time")
    args = parser.parse_args()

    todo = [(out, command) for out, command in build(args.sets) if not (out / "metrics.json").exists()]
    print(f"{len(todo)} jobs to run, {args.jobs} at a time", flush=True)
    started, failed = time.time(), []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run, out, command) for out, command in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            out, code, seconds = future.result()
            if code != 0:
                failed.append(out)
            elapsed = (time.time() - started) / 3600
            print(f"{done}/{len(todo)}  exit {code}  {seconds:4.0f} s  {elapsed:5.2f} h  "
                  f"(eta {elapsed / done * (len(todo) - done):5.2f} h)  {out.relative_to(LSTM)}", flush=True)
    print(f"ALL DONE, {len(failed)} failed" + (": " + ", ".join(str(f) for f in failed) if failed else ""), flush=True)


if __name__ == "__main__":
    main()
