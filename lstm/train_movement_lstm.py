"""Train an LSTM that turns raw insole IMU + pressure into the viewer's Movement signal.

Data
    Session files data_final/S<ID>_session.json, written by tools/make_session_files.py from the
    public dataset. Each one names the IMU and pressure CSVs (80 Hz) and lists the stand-up marks
    (start, end). Targets are built with movement_signal() (movement_labels.py) on the file timeline.
    Default split (a quick check only): S01 + S02 train, S05 test. The last 15% of the last training
    recording is held out to choose the stopping epoch, so the test participant is only used once.
    The leave-one-participant-out runs of the paper are started by tools/run_lstm_jobs.py.

Input / output scheme ("A")
    A 2 s window (160 samples x 48 channels) goes in; the LSTM outputs one value per
    sample. A new window starts every 0.5 s and only its last 0.5 s (40 outputs) is
    kept, so every output has 1.5-2 s of history and the curve is continuous at 80 Hz.
    The very first window keeps all of its outputs (warm-up). Training scores exactly
    those last-0.5 s outputs.

Channels (48): per foot (left, then right): acc x/y/z, gyro x/y/z, pressure 0-17.
    Pressure is divided per recording and foot by the 95th percentile of that foot's
    total pressure (body-weight normalisation; for live use, measure it once in a
    calibration period). Then every channel is z-scored with training statistics.

Run (from anywhere):
    python train_movement_lstm.py
    ... --window-s 3      # try a longer window
Outputs go to runs/window_<W>s/: model.pt, metrics.json, predictions CSV, charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from movement_labels import MOVEMENT_SIGNAL_HOLD, movement_active, movement_signal  # noqa: E402

SAMPLE_RATE = 80
DATASET = HERE.parent / "data_final"
IMU_NAMES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
PRESSURE_NAMES = tuple(f"pressure_{index}" for index in range(18))
CHANNELS = tuple(f"{side}_{name}" for side in ("left", "right") for name in (*IMU_NAMES, *PRESSURE_NAMES))
PRESSURE_COLUMNS = np.array([i for i, name in enumerate(CHANNELS) if "pressure" in name])
CHANNEL_SETS = {  # column indices into CHANNELS for each sensor configuration (H4)
    "all": list(range(len(CHANNELS))),
    "pressure": [i for i, name in enumerate(CHANNELS) if "pressure" in name],
    "imu": [i for i, name in enumerate(CHANNELS) if "pressure" not in name],
    "left": [i for i, name in enumerate(CHANNELS) if name.startswith("left_")],
    "right": [i for i, name in enumerate(CHANNELS) if name.startswith("right_")],
}


# ----------------------------------------------------------------------------- data

@dataclass
class Recording:
    name: str
    x: np.ndarray                 # (T, 48) raw sensor values
    y: np.ndarray                 # (T,) Movement signal target
    active: np.ndarray            # (T,) True from each start to end + hold
    covered: tuple[int, int]      # sample range the video covers (only there are labels trustworthy)
    movements: list[tuple[float, float]]  # (start, end) in sensor seconds
    offset: float
    pressure_scale: dict[str, float] = field(default_factory=dict)


def mp4_duration(path: Path) -> float:
    """Duration from the MP4 movie header (no video decoding needed)."""
    with path.open("rb") as source:
        source.seek(0, 2)
        end = source.tell()
        position = 0
        while position < end:
            source.seek(position)
            size, kind = struct.unpack(">I4s", source.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", source.read(8))[0]
                header = 16
            elif size == 0:
                size = end - position
            if kind == b"moov":
                end, position = position + size, position + header
                continue
            if kind == b"mvhd":
                version = source.read(1)[0]
                source.read(3)
                if version == 1:
                    _created, _modified, timescale, duration = struct.unpack(">QQIQ", source.read(28))
                else:
                    _created, _modified, timescale, duration = struct.unpack(">IIII", source.read(16))
                return duration / timescale
            position += size
    raise ValueError(f"No movie header in {path.name}")


def load_csv(path: Path, columns: int) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != columns:
        raise ValueError(f"{path} should have {columns} columns")
    if not np.allclose(data[:, 0], np.arange(len(data)) / SAMPLE_RATE, atol=1e-6):
        raise ValueError(f"{path}: timestamps are not 1/{SAMPLE_RATE} s apart from 0")
    return data[:, 1:]


def load_recording(session_path: Path) -> Recording:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    base = session_path.parent

    def resolve(file: str) -> Path:
        candidate = Path(file)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    parts = []
    for side in ("left", "right"):
        parts.append(load_csv(resolve(session["imu"][side]), 7))
        parts.append(load_csv(resolve(session["pressure"][side]), 19))
    if len({len(part) for part in parts}) != 1:
        raise ValueError(f"{session_path.name}: IMU and pressure files have different lengths")
    x = np.concatenate(parts, axis=1).astype(np.float32)

    offset = float(session.get("video_offset_s", 0.0))
    marks = [(float(m["start_s"]), float(m["end_s"])) for m in session.get("movements", []) if m.get("end_s") is not None]
    if not marks:
        raise ValueError(f"{session_path.name} has no complete movements")
    video_times = np.arange(len(x)) / SAMPLE_RATE - offset          # sensor time -> video time
    y = movement_signal(video_times, marks).astype(np.float32)
    active = movement_active(video_times, marks)

    # The videos are not shipped with the dataset; their length is stored in the session file.
    # data_final sessions have no video at all ("video": null): the whole file is labelled.
    video = resolve(session["video"]) if session.get("video") else None
    duration = mp4_duration(video) if video is not None and video.is_file() else float(session["video_duration_s"])
    first = max(0, math.ceil(offset * SAMPLE_RATE))
    last = min(len(x), math.floor((offset + duration) * SAMPLE_RATE))
    name = Path(session["video"]).stem if session.get("video") else session_path.name.removesuffix("_session.json")
    return Recording(name, x, y, active, (first, last), [(s + offset, e + offset) for s, e in marks], offset)


def normalise_pressure(recording: Recording, calibration_s: float = 0.0) -> None:
    """Divide each foot's 18 pressure channels by that foot's 95th-percentile total load.
    calibration_s > 0: take the percentile over the first calibration_s seconds of the labelled part only
    (what a live system can know); 0: over the whole recording (uses the future)."""
    first = recording.covered[0]
    last = first + round(calibration_s * SAMPLE_RATE) if calibration_s > 0 else len(recording.x)
    for side, columns in (("left", PRESSURE_COLUMNS[:18]), ("right", PRESSURE_COLUMNS[18:])):
        total = recording.x[first:last, columns].sum(axis=1)
        scale = float(np.percentile(total, 95)) or 1.0
        recording.x[:, columns] /= scale
        recording.pressure_scale[side] = scale


# ----------------------------------------------------------------------------- model

class MovementLSTM(nn.Module):
    """Per-sample input projection -> LSTM -> two outputs per sample:
    the Movement signal (0..1) and an 'inside a movement' logit (auxiliary, helps training)."""

    def __init__(self, inputs: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(self.project(x))
        heads = self.head(output)
        return torch.sigmoid(heads[..., 0]), heads[..., 1]


@torch.no_grad()
def stream_predict(model: MovementLSTM, x: torch.Tensor, window: int, step: int, batch: int = 512) -> np.ndarray:
    """Scheme A: windows end at window, window+step, ... (and finally at T); each window is
    processed from scratch and contributes only its newest outputs. Returns (T,) predictions."""
    model.eval()
    total = len(x)
    if total < window:
        raise ValueError("Recording shorter than one window")
    ends = list(range(window, total + 1, step))
    if ends[-1] != total:
        ends.append(total)
    prediction = np.full(total, np.nan, dtype=np.float32)
    offsets = torch.arange(-window, 0, device=x.device)
    previous_end = 0
    for first in range(0, len(ends), batch):
        chunk = torch.tensor(ends[first:first + batch], device=x.device)
        signal, _ = model(x[chunk[:, None] + offsets])
        signal = signal.cpu().numpy()
        for row, end in enumerate(ends[first:first + batch]):
            keep = end - previous_end                      # all of the first window, then the new samples
            prediction[end - keep:end] = signal[row, window - keep:]
            previous_end = end
    return prediction


# ----------------------------------------------------------------------------- metrics

def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def evaluate(prediction: np.ndarray, recording: Recording, tolerance_s: float = 0.5) -> dict:
    """Sample errors over the video-covered part, plus event timing from the 0.5 crossings:
    rise = where the curve passes 0.5 going up (label: half-way through the movement),
    drop = where it falls back (label: end + hold)."""
    first, last = recording.covered
    p, y = prediction[first:last], recording.y[first:last]
    inside = recording.active[first:last]
    result = {
        "samples": int(last - first),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae_inside_movements": float(np.mean(np.abs(p[inside] - y[inside]))),
        "mae_outside_movements": float(np.mean(np.abs(p[~inside] - y[~inside]))),
        "correlation": float(np.corrcoef(p, y)[0, 1]),
    }
    truth, found = segments(y > 0.5), segments(p > 0.5)
    found = [(a, b) for a, b in found if (b - a) >= 0.1 * SAMPLE_RATE]  # ignore flickers < 0.1 s
    rise_errors, drop_errors, used = [], [], set()
    for a, b in truth:
        candidates = [(min(b, d) - max(a, c), i) for i, (c, d) in enumerate(found) if i not in used]
        overlap, best = max(candidates, default=(0, None))
        if best is None or overlap <= 0:
            continue
        c, d = found[best]
        rise, drop = (c - a) / SAMPLE_RATE, (d - b) / SAMPLE_RATE
        if abs(rise) <= tolerance_s and abs(drop) <= tolerance_s:
            used.add(best)
            rise_errors.append(rise)
            drop_errors.append(drop)
    detected = len(rise_errors)
    # Looser view: did the output pass 0.5 anywhere in [start, end + hold + 0.5 s] at all, and how many
    # output bursts are nowhere near a marked movement (pure false alarms)?
    span = segments(inside)
    near = np.zeros(len(p), dtype=bool)
    for a, b in span:
        near[max(0, a - SAMPLE_RATE // 2):b + SAMPLE_RATE // 2] = True
    above = p > 0.5
    responded = sum(bool(above[a:b + SAMPLE_RATE // 2].any()) for a, b in span)
    false_alarms = sum(not near[c:d].any() for c, d in found)
    result.update({
        "movements": len(truth),
        "responded": responded,
        "false_alarms": false_alarms,
        "detected_within_tolerance": detected,
        "tolerance_s": tolerance_s,
        "false_detections": len(found) - detected,
        "recall": detected / max(1, len(truth)),
        "precision": detected / max(1, len(found)),
        "rise_error_median_abs_ms": float(np.median(np.abs(rise_errors)) * 1000) if detected else None,
        "rise_error_median_signed_ms": float(np.median(rise_errors) * 1000) if detected else None,
        "drop_error_median_abs_ms": float(np.median(np.abs(drop_errors)) * 1000) if detected else None,
        "drop_error_median_signed_ms": float(np.median(drop_errors) * 1000) if detected else None,
    })
    return result


# ----------------------------------------------------------------------------- charts

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
EXPECTED_COLOR, LSTM_COLOR = "#2a78d6", "#eb6834"   # categorical slots 1-2, validated light


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for name, spine in axis.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)


def plot_test(recording: Recording, prediction: np.ndarray, metrics: dict, title: str, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "axes.labelcolor": INK_2})
    first, last = recording.covered
    t = np.arange(len(recording.y)) / SAMPLE_RATE
    paths = []

    # 1. Whole recording.
    figure, axis = plt.subplots(figsize=(16, 3.6), facecolor=SURFACE)
    _style(axis)
    axis.plot(t[first:last], recording.y[first:last], color=EXPECTED_COLOR, linewidth=2, label="Expected (labels)")
    axis.plot(t[first:last], prediction[first:last], color=LSTM_COLOR, linewidth=1.4, label="LSTM output")
    axis.set_xlim(t[first], t[last - 1])
    axis.set_ylim(-0.05, 1.12)
    axis.set_yticks([0, 0.5, 1])
    axis.set_xlabel("Sensor time (s)")
    axis.legend(loc="upper right", ncols=2, frameon=False, fontsize=9)
    axis.set_title(f"{title} — whole recording   ·   MAE {metrics['mae']:.3f}   ·   responds to "
                   f"{metrics['responded']}/{metrics['movements']} movements, "
                   f"{metrics['detected_within_tolerance']} with rise and drop within ±{metrics['tolerance_s']:g} s   ·   "
                   f"{metrics['false_alarms']} false alarms",
                   loc="left", fontsize=11, color=INK)
    figure.tight_layout()
    paths.append(out_dir / f"{recording.name}_overview.png")
    figure.savefig(paths[-1], dpi=130, facecolor=SURFACE)
    plt.close(figure)

    # 2. The whole recording, continuous, wrapped into stacked rows of `row_s` seconds.
    row_s = 120.0
    rows = math.ceil((last - first) / SAMPLE_RATE / row_s)
    figure, axes = plt.subplots(rows, 1, figsize=(16, 1.55 * rows + 0.9), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for row, axis in enumerate(axes):
        _style(axis)
        start_s = t[first] + row * row_s
        a = first + int(row * row_s * SAMPLE_RATE)
        b = min(last, first + int((row + 1) * row_s * SAMPLE_RATE))
        for number, (s, e) in enumerate(recording.movements, start=1):
            if start_s <= s < start_s + row_s:
                axis.axvspan(s, e, color=GRID, alpha=0.7, linewidth=0)
                axis.text(s, 1.13, f"#{number}", fontsize=7, color=MUTED, va="bottom")
        axis.plot(t[a:b], recording.y[a:b], color=EXPECTED_COLOR, linewidth=1.8)
        axis.plot(t[a:b], prediction[a:b], color=LSTM_COLOR, linewidth=1.3)
        axis.set_xlim(start_s, start_s + row_s)  # Same scale on every row, including the last.
        axis.set_ylim(-0.05, 1.12)
        axis.set_yticks([0, 1])
        axis.set_xticks(np.arange(math.ceil(start_s / 10) * 10, start_s + row_s + 1e-9, 10))
    axes[-1].set_xlabel("Sensor time (s)")
    handles = [plt.Line2D([], [], color=EXPECTED_COLOR, linewidth=2), plt.Line2D([], [], color=LSTM_COLOR, linewidth=2),
               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.7)]
    figure.legend(handles, ["Expected (labels)", "LSTM output", "s → e (marked movement)"],
                  loc="upper right", ncols=3, frameon=False, fontsize=9)
    figure.suptitle(f"{title} — continuous, {row_s:g} s per row", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (1.55 * rows + 0.9)))
    paths.append(out_dir / f"{recording.name}_continuous.png")
    figure.savefig(paths[-1], dpi=110, facecolor=SURFACE)
    plt.close(figure)

    # 3. Every movement, 2 s before the start to 2 s after the drop.
    columns = 6
    rows = math.ceil(len(recording.movements) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2.0 * rows + 0.8), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for number, ((start, end), axis) in enumerate(zip(recording.movements, axes), start=1):
        _style(axis)
        a = max(0, int((start - 2.0) * SAMPLE_RATE))
        b = min(len(t), int((end + MOVEMENT_SIGNAL_HOLD + 2.0) * SAMPLE_RATE))
        axis.axvspan(start, end, color=GRID, alpha=0.6, linewidth=0)
        axis.plot(t[a:b], recording.y[a:b], color=EXPECTED_COLOR, linewidth=2)
        axis.plot(t[a:b], prediction[a:b], color=LSTM_COLOR, linewidth=1.6)
        axis.set_ylim(-0.05, 1.12)
        axis.set_yticks([0, 1])
        axis.set_title(f"#{number}  ·  {int(start // 60):02d}:{start % 60:05.2f}  ·  {end - start:.2f} s",
                       loc="left", fontsize=8.5, color=INK_2)
    for axis in axes[len(recording.movements):]:
        axis.set_visible(False)
    handles = [plt.Line2D([], [], color=EXPECTED_COLOR, linewidth=2), plt.Line2D([], [], color=LSTM_COLOR, linewidth=2),
               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6)]
    figure.legend(handles, ["Expected (labels)", "LSTM output", "s → e (marked movement)"],
                  loc="upper right", ncols=3, frameon=False, fontsize=9)
    figure.suptitle(f"{title} — each movement (x axis: sensor seconds)", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.0 * rows + 0.8)))
    paths.append(out_dir / f"{recording.name}_movements.png")
    figure.savefig(paths[-1], dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return paths


def plot_history(history: list[dict], out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 3), facecolor=SURFACE)
    _style(axis)
    epochs = [h["epoch"] for h in history]
    axis.plot(epochs, [h["train_loss"] for h in history], color=EXPECTED_COLOR, linewidth=2, label="Training loss")
    axis.plot(epochs, [h["val_mae"] for h in history], color=LSTM_COLOR, linewidth=2, label="Validation MAE (last 15% of the last training recording)")
    axis.set_xlabel("Epoch")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path = out_dir / "training_curve.png"
    figure.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(figure)
    return path


# ----------------------------------------------------------------------------- training

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", type=Path,
                        default=[DATASET / "S01_session.json", DATASET / "S02_session.json"])
    parser.add_argument("--test", type=Path, default=DATASET / "S05_session.json")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Tail of the last training file held out")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--windows-per-epoch", type=int, default=16384)
    parser.add_argument("--positive-fraction", type=float, default=0.5,
                        help="Share of training windows whose scored part touches a movement")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-charts", action="store_true", help="Skip the PNG charts (bulk runs)")
    parser.add_argument("--channels", choices=sorted(CHANNEL_SETS), default="all",
                        help="Sensor configuration (H4 ablation): all 48, pressure only, IMU only, left foot, right foot")
    parser.add_argument("--pressure-calibration-s", type=float, default=0.0,
                        help="Pressure scale from the first N s of each recording (0 = whole recording)")
    parser.add_argument("--target", choices=("movement", "inside"), default="movement",
                        help="Regression target of the primary output: the Movement signal (sigmoid ramp from start "
                             "to end, then hold) or 'inside' (1 from start to end + hold, as B2; exploratory)")
    parser.add_argument("--out", type=Path, help="Output folder (default: runs/window_<W>s)")
    parser.add_argument("--note", default="", help="Extra text for the chart titles, e.g. 'fold: test S05'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = round(args.window_s * SAMPLE_RATE)
    step = round(args.step_s * SAMPLE_RATE)
    out_dir = args.out or HERE / "runs" / f"window_{args.window_s:g}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recordings = [load_recording(path) for path in args.train]
    test = load_recording(args.test)
    selected = CHANNEL_SETS[args.channels]
    channels = tuple(CHANNELS[i] for i in selected)
    for recording in (*train_recordings, test):
        normalise_pressure(recording, args.pressure_calibration_s)
        recording.x = np.ascontiguousarray(recording.x[:, selected])     # sensor configuration (after normalising)
        if args.target == "inside":      # same model and loss; only the target of the primary output changes
            recording.y = recording.active.astype(np.float32)
        print(f"{recording.name}: {len(recording.x):,} samples, video covers {recording.covered[0] / SAMPLE_RATE:.1f}–"
              f"{recording.covered[1] / SAMPLE_RATE:.1f} s, {len(recording.movements)} movements, offset {recording.offset:+.3f} s")

    # Scored ranges: train everywhere the video covers, except the held-out tail of the last file.
    train_ranges = {r.name: [r.covered] for r in train_recordings}
    last = train_recordings[-1]
    cut = last.covered[0] + int((1 - args.val_fraction) * (last.covered[1] - last.covered[0]))
    train_ranges[last.name] = [(last.covered[0], cut)]
    validation = Recording(last.name + " (validation tail)", last.x, last.y, last.active,
                           (cut, last.covered[1]), [m for m in last.movements if m[0] * SAMPLE_RATE >= cut], last.offset)

    # Channel statistics from the training part only.
    stacked = np.concatenate([r.x[a:b] for r in train_recordings for a, b in train_ranges[r.name]])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6

    def to_device(recording: Recording) -> torch.Tensor:
        return torch.tensor((recording.x - mean) / std, dtype=torch.float32, device=device)

    # One long tensor for all training recordings; candidate window ends never cross a file boundary.
    xs, ys, actives, ends, positive = [], [], [], [], []
    offset = 0
    for recording in train_recordings:
        xs.append(to_device(recording))
        ys.append(torch.tensor(recording.y, device=device))
        actives.append(torch.tensor(recording.active.astype(np.float32), device=device))
        for a, b in train_ranges[recording.name]:
            # The scored part [end-step, end) must lie inside [a, b); the input needs `window` samples.
            candidate = np.arange(max(a + step, window), b + 1)
            ends.append(candidate + offset)
            touched = np.convolve(recording.active.astype(np.int32), np.ones(step, dtype=np.int32))[candidate - 1] > 0
            positive.append(touched)
        offset += len(recording.x)
    x_all, y_all, active_all = torch.cat(xs), torch.cat(ys), torch.cat(actives)
    ends, positive = np.concatenate(ends), np.concatenate(positive)
    weights = np.where(positive, args.positive_fraction / positive.sum(), (1 - args.positive_fraction) / (~positive).sum())
    print(f"Training windows to sample from: {len(ends):,} ({positive.mean():.1%} touch a movement); device {device}")

    model = MovementLSTM(len(channels), args.hidden, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    window_offsets = torch.arange(-window, 0, device=device)
    scored = slice(window - step, window)
    x_validation = to_device(last)
    best_mae, best_state, best_epoch, history = math.inf, None, 0, []
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters; window {window} samples, step {step}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        chosen = rng.choice(ends, size=args.windows_per_epoch, p=weights)
        for first in range(0, len(chosen), args.batch):
            batch_ends = torch.tensor(chosen[first:first + args.batch], device=device)
            index = batch_ends[:, None] + window_offsets
            x = x_all[index]
            # Light augmentation: per-window channel gain and sensor noise.
            x = x * (1 + 0.1 * torch.randn(len(x), 1, x.shape[2], device=device)) + 0.05 * torch.randn_like(x)
            signal, inside_logit = model(x)
            target, inside = y_all[index][:, scored], active_all[index][:, scored]
            loss = F.mse_loss(signal[:, scored], target) + 0.5 * F.binary_cross_entropy_with_logits(inside_logit[:, scored], inside)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        prediction = stream_predict(model, x_validation, window, step)
        val = evaluate(prediction, validation)
        scheduler.step(val["mae"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val["mae"],
                        "val_detected": f"{val['detected_within_tolerance']}/{val['movements']}"})
        improved = val["mae"] < best_mae
        if improved:
            best_mae, best_epoch = val["mae"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  val MAE {val['mae']:.4f}  "
              f"val detected {history[-1]['val_detected']}  {time.time() - started:.1f}s{'  *' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            print(f"Stopping: no validation improvement for {args.patience} epochs")
            break

    model.load_state_dict(best_state)
    prediction = stream_predict(model, to_device(test), window, step)
    metrics = {"test": evaluate(prediction, test), "validation": evaluate(stream_predict(model, x_validation, window, step), validation),
               "best_epoch": best_epoch, "window_s": args.window_s, "step_s": args.step_s,
               "train": [r.name for r in train_recordings], "test_recording": test.name}

    torch.save({"state_dict": model.state_dict(), "channels": channels, "mean": mean, "std": std,
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("train", "test", "out")},
                "sample_rate": SAMPLE_RATE,
                "pressure_normalisation": "per recording and foot: / 95th percentile of total, over "
                                          + (f"the first {args.pressure_calibration_s:g} s" if args.pressure_calibration_s > 0
                                             else "the whole recording")},
               out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps({**metrics, "history": history}, indent=2), encoding="utf-8")
    with (out_dir / f"{test.name}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sensor_time_s", "video_time_s", "expected", "lstm", "video_covered"))
        for i in range(len(prediction)):
            writer.writerow((f"{i / SAMPLE_RATE:.4f}", f"{i / SAMPLE_RATE - test.offset:.4f}", f"{test.y[i]:.4f}",
                             f"{prediction[i]:.4f}", int(test.covered[0] <= i < test.covered[1])))

    note = f", {args.note}" if args.note else ""
    charts = [] if args.no_charts else plot_test(
        test, prediction, metrics["test"], f"{test.name}: expected vs LSTM ({args.window_s:g} s window, A{note})", out_dir)
    if not args.no_charts:
        charts.append(plot_history(history, out_dir))
    print(json.dumps(metrics["test"], indent=2))
    print("Wrote", out_dir, *[path.name for path in charts], sep="\n  ")


if __name__ == "__main__":
    main()
    # Everything is written and closed by now. On Windows with torch 2.15 cu132 / cuDNN 9.26, once a
    # cuDNN LSTM with dropout between layers has run in training mode, unloading the DLLs at process
    # exit aborts (0xC0000409, fault in ucrtbase.dll) -- after the run has finished, results unaffected.
    # os._exit() still unloads the DLLs, so end the process directly instead.
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)
