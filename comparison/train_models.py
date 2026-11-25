"""Train the models of the study protocol, §9b: same windows, target, folds and runs as B2 (HGB), for every model.

Machine learning on B2's 576 cached features (baselines/runs/b2/features): logreg, svm_rbf, rf (CPU), xgb (GPU).
Deep learning on the raw 2 s windows (160 x 48, z-scored with training statistics): cnn, tcn, lstm, gru, cnn_gru,
transformer (GPU). One fixed setting per model; no early stopping and no tuning. HGB itself is reused from B2.
Models without randomness (logreg, svm_rbf) run once, as "seed0".

Output, same format as B2:  comparison/runs/<model>/test_<ID>/seed<k>.npz   window_end_s, probability
Finished outputs are skipped, so the script can be restarted.

  python comparison/train_models.py dl ml -j 6            (families; or model names)
  python comparison/train_models.py all --smoke           (1 fold, 1 run, 1 epoch, to /tmp)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "baselines"))
from b2_hgb_window import CALIBRATION_S, HOLD_S, STEP, WINDOW, cached  # noqa: E402
from events import PARTICIPANTS, RATE, labels  # noqa: E402

SEEDS = [0, 1, 2, 345, 4, 5]
ML = ["logreg", "svm_rbf", "rf", "xgb"]
DL = ["cnn", "tcn", "lstm", "gru", "cnn_gru"]   # transformer dropped from the study 2026-11-24 (class kept below)
DETERMINISTIC = {"logreg", "svm_rbf"}
EPOCHS, BATCH, LR, WEIGHT_DECAY, JITTER = 40, 256, 1e-3, 1e-2, 20
CPU_THREADS = 4


# ----------------------------------------------------------------------------- data

@lru_cache(maxsize=None)
def raw(participant: str) -> tuple[np.ndarray, np.ndarray]:
    """(T, 48) channels in LSTM order with B2's pressure scaling, and the (T,) active mask."""
    folder = ROOT / "data_final" / participant
    parts = []
    for side in ("left", "right"):
        imu = np.loadtxt(folder / f"imu_{side}.csv", delimiter=",", skiprows=1)[:, 1:]
        pressure = np.loadtxt(folder / f"pressure_{side}.csv", delimiter=",", skiprows=1)[:, 1:]
        parts += [imu, pressure / (np.percentile(pressure[:CALIBRATION_S * RATE].sum(axis=1), 95) or 1.0)]
    x = np.hstack(parts).astype(np.float32)
    active = np.zeros(len(x), dtype=bool)
    for s, e in labels(participant):
        active[int(round(s * RATE)):int(round((e + HOLD_S) * RATE))] = True
    return x, active


# ----------------------------------------------------------------------------- deep models

def build(name: str):
    import torch
    import torch.nn as nn

    class Last(nn.Module):                                    # (B, T, C) -> newest step
        def forward(self, x):
            return x[:, -1]

    class Recurrent(nn.Module):
        def __init__(self, kind, inputs=48, hidden=64, project=True):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(0.2)) if project else nn.Identity()
            self.rnn = kind(hidden, hidden, num_layers=2, batch_first=True, dropout=0.2)

        def forward(self, x):
            return self.rnn(self.project(x))[0][:, -1]

    class ResBlock(nn.Module):
        def __init__(self, a, b, stride):
            super().__init__()
            self.body = nn.Sequential(nn.Conv1d(a, b, 5, stride, 2), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(0.1),
                                      nn.Conv1d(b, b, 5, 1, 2), nn.BatchNorm1d(b))
            self.skip = nn.Conv1d(a, b, 1, stride) if (a != b or stride != 1) else nn.Identity()

        def forward(self, x):
            return nn.functional.gelu(self.body(x) + self.skip(x))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(48, 48, 7, 2, 3), nn.BatchNorm1d(48), nn.GELU(),
                                     ResBlock(48, 48, 1), ResBlock(48, 64, 2), ResBlock(64, 64, 2))

        def forward(self, x):
            h = self.net(x.transpose(1, 2))
            return torch.cat([h.mean(-1), h.amax(-1)], dim=1)

    class Causal(nn.Module):
        def __init__(self, a, b, dilation):
            super().__init__()
            self.pad = 4 * dilation
            self.conv1, self.conv2 = nn.Conv1d(a, b, 5, dilation=dilation), nn.Conv1d(b, b, 5, dilation=dilation)
            self.norm1, self.norm2, self.drop = nn.BatchNorm1d(b), nn.BatchNorm1d(b), nn.Dropout(0.1)
            self.skip = nn.Conv1d(a, b, 1) if a != b else nn.Identity()

        def forward(self, x):
            pad = nn.functional.pad
            h = self.drop(nn.functional.gelu(self.norm1(self.conv1(pad(x, (self.pad, 0))))))
            return nn.functional.gelu(self.norm2(self.conv2(pad(h, (self.pad, 0)))) + self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(Causal(48, 48, 1), Causal(48, 48, 2), Causal(48, 48, 4), Causal(48, 48, 8),
                                     Causal(48, 48, 16))

        def forward(self, x):
            return self.net(x.transpose(1, 2))[:, :, -1]

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(48, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
                                      nn.Conv1d(64, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU())
            self.rnn = Recurrent(nn.GRU, 64, 64, project=False)

        def forward(self, x):
            return self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))

    class Transformer(nn.Module):
        def __init__(self, d=64, tokens=40):
            super().__init__()
            self.patch = nn.Conv1d(48, d, 4, 4)
            self.position = nn.Parameter(torch.randn(1, tokens, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 128, 0.1, "gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            h = self.norm(self.encoder(self.patch(x.transpose(1, 2)).transpose(1, 2) + self.position))
            return torch.cat([h.mean(1), h[:, -1]], dim=1)

    body, width = {"cnn": (CNN, 128), "tcn": (TCN, 48), "lstm": (lambda: Recurrent(nn.LSTM), 64),
                   "gru": (lambda: Recurrent(nn.GRU), 64), "cnn_gru": (CNNGRU, 64),
                   "transformer": (Transformer, 128)}[name]
    return nn.Sequential(body(), nn.Linear(width, 1))


def train_deep(name: str, test: str, seed: int, epochs: int) -> tuple[np.ndarray, np.ndarray, str]:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    train = [p for p in PARTICIPANTS if p != test]
    stacked = np.concatenate([raw(p)[0] for p in train])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    x_all = torch.tensor((stacked - mean) / std, device=device)
    active = torch.tensor(np.concatenate([raw(p)[1] for p in train]).astype(np.float32), device=device)
    cumulative = torch.cat([torch.zeros(1, device=device), active.cumsum(0)])        # exact: counts < 2^24
    ends, low, high, offset = [], [], [], 0
    for p in train:                                                                  # ends never leave their file
        n = len(raw(p)[0])
        e = np.arange(WINDOW, n + 1, STEP)
        ends.append(e + offset)
        low.append(np.full(len(e), offset + WINDOW))
        high.append(np.full(len(e), offset + n))
        offset += n
    ends, low, high = (torch.tensor(np.concatenate(v), device=device) for v in (ends, low, high))
    offsets = torch.arange(-WINDOW, 0, device=device)

    model = build(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batches = len(ends) // BATCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=epochs * batches)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(ends), device=device)
        for b in range(batches):
            pick = order[b * BATCH:(b + 1) * BATCH]
            e = ends[pick] + torch.randint(-JITTER, JITTER + 1, (len(pick),), device=device)
            e = torch.minimum(torch.maximum(e, low[pick]), high[pick])
            y = ((cumulative[e] - cumulative[e - STEP]) > 0).float()                 # label of the jittered window
            x = x_all[e[:, None] + offsets]
            x = x * (1 + 0.1 * torch.randn(len(x), 1, x.shape[2], device=device)) + 0.05 * torch.randn_like(x)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    xt = torch.tensor((raw(test)[0] - mean) / std, device=device)
    test_ends = np.arange(WINDOW, len(xt) + 1, STEP)
    out = []
    with torch.no_grad():
        for i in range(0, len(test_ends), 1024):
            e = torch.tensor(test_ends[i:i + 1024], device=device)
            out.append(torch.sigmoid(model(xt[e[:, None] + offsets]).squeeze(-1).double()).cpu().numpy())
    params = sum(p.numel() for p in model.parameters())
    return test_ends / RATE, np.concatenate(out), f"{params:,} parameters, last loss {loss.item():.3f}"


# ----------------------------------------------------------------------------- feature models

def train_features(name: str, test: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    data = {p: cached(p) for p in PARTICIPANTS}
    x = np.vstack([data[p][0] for p in PARTICIPANTS if p != test])
    y = np.concatenate([data[p][1] for p in PARTICIPANTS if p != test]).astype(int)
    xt = data[test][0]
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        probability = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(x, y).predict_proba(xt)[:, 1]
    elif name == "svm_rbf":                    # decision value through a logistic: monotone, not calibrated
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        value = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=2000)).fit(x, y).decision_function(xt)
        probability = 1 / (1 + np.exp(-value))
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        probability = RandomForestClassifier(500, random_state=seed, n_jobs=CPU_THREADS).fit(x, y).predict_proba(xt)[:, 1]
    else:
        from xgboost import XGBClassifier
        probability = XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
                                    random_state=seed, n_jobs=CPU_THREADS).fit(x, y).predict_proba(xt)[:, 1]
    return data[test][2], np.asarray(probability, dtype=np.float64), f"{len(y):,} windows"


# ----------------------------------------------------------------------------- runner

def job(args: tuple[str, str, int, int, str]) -> str:
    name, test, seed, epochs, out = args
    path = Path(out) / name / f"test_{test}" / f"seed{seed}.npz"
    if path.exists():
        return f"{name} {test} seed{seed}: exists"
    started = time.time()
    try:
        end_s, probability, note = (train_deep(name, test, seed, epochs) if name in DL else train_features(name, test, seed))
        if not np.all(np.isfinite(probability)):
            raise ValueError("non-finite probabilities")
    except Exception as exc:                   # reported in the log; the output stays missing
        return f"{name} {test} seed{seed}: FAILED {type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, window_end_s=end_s, probability=probability)
    temporary.replace(path)
    return f"{name} {test} seed{seed}: {note}, {time.time() - started:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="dl, ml, all or model names")
    parser.add_argument("-j", "--jobs", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="one fold, one run, one epoch, to /tmp/sts_models_smoke")
    args = parser.parse_args()
    names = []
    for m in args.models:
        names += {"dl": DL, "ml": ML, "all": ML + DL}.get(m, [m])
    unknown = [n for n in names if n not in ML + DL]
    if unknown:
        parser.error(f"unknown models: {unknown}")
    out = str(Path(tempfile.gettempdir(), "sts_models_smoke")) if args.smoke else str(ROOT / "comparison" / "runs")
    epochs = 1 if args.smoke else EPOCHS
    tasks = [(n, p, s, epochs, out) for s in (SEEDS[:1] if args.smoke else SEEDS) for n in names
             if s == SEEDS[0] or n not in DETERMINISTIC for p in (PARTICIPANTS[:1] if args.smoke else PARTICIPANTS)]
    print(f"{len(tasks)} jobs, {args.jobs} at a time", flush=True)
    failed, started = 0, time.time()
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for i, message in enumerate(pool.imap_unordered(job, tasks), start=1):
            failed += "FAILED" in message
            print(f"{i}/{len(tasks)}  {(time.time() - started) / 60:5.1f} min  {message}", flush=True)
    print(f"ALL DONE, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
