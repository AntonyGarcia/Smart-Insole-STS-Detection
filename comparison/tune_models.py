"""Tuned, strictly nested search (the study protocol, §9c): training.

inner  every model x 12 settings x 55 pairs {i, q}: trained on the other 9 participants (seed 0), predicts i and q
       -> comparison/tuning/inner/<model>/c<k>/<i>_<q>.npz          end_<p>, prob_<p> for both
final  settings chosen by tune_select.py (comparison/tuning/selected.json), trained on the 10 others, all runs
       -> comparison/tuning/final/<model>/test_<ID>/seed<k>.npz     window_end_s, probability

Setting 0 is the §9b setting. Finished outputs are skipped, so the script can be restarted.

  python comparison/tune_models.py inner dl -j 8
  python comparison/tune_models.py inner ml -j 5
  python comparison/tune_models.py final all -j 8
  python comparison/tune_models.py smoke               (checks; a few jobs of every model, timed, to /tmp)
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))
from train_models import BATCH, DETERMINISTIC, JITTER, PARTICIPANTS, RATE, SEEDS, STEP, WINDOW, cached, raw  # noqa: E402

ML = ["logreg", "svm_rbf", "rf", "xgb", "HGB"]
DL = ["cnn", "tcn", "lstm", "gru", "cnn_gru"]   # transformer dropped from the study 2026-11-24 (class kept below)
MODELS = ML + DL
SETTINGS, THREADS = 12, 4
OUT = ROOT / "comparison" / "tuning"
PAIRS = list(itertools.combinations(PARTICIPANTS, 2))

# First value of every list = §9b setting.
DEEP = {"lr": [1e-3, 3e-4, 3e-3], "weight_decay": [1e-2, 1e-3, 1e-1], "epochs": [40, 20, 60], "width": [1.0, 0.5, 2.0],
        "depth": [0, -1, 1], "dropout": [1, 2, 3], "augment": [1, 0, 2], "pos_weight": [1, 3]}
SPACES = {
    "logreg": {"C": [1.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 3.0], "class_weight": [None, "balanced"]},
    "svm_rbf": {"C": [1.0, 0.1, 0.3, 3.0, 10.0, 30.0], "gamma_factor": [1.0, 0.1, 0.3, 3.0], "class_weight": [None, "balanced"]},
    "rf": {"max_features": ["sqrt", "log2", 0.1], "min_samples_leaf": [1, 2, 5, 10], "max_depth": [None, 10, 20],
           "class_weight": [None, "balanced_subsample"]},
    "xgb": {"n_estimators": [200, 100, 400, 800], "learning_rate": [0.3, 0.03, 0.1], "max_depth": [6, 3, 4, 8],
            "min_child_weight": [1, 5, 10], "subsample": [0.8, 0.6, 1.0], "colsample_bytree": [0.8, 0.3, 0.5], "reg_lambda": [1, 5, 10]},
    "HGB": {"max_iter": [200, 100, 400], "learning_rate": [0.1, 0.03, 0.2], "max_leaf_nodes": [31, 15, 63],
            "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0, 10.0], "max_features": [1.0, 0.3, 0.1]},
    **{m: DEEP for m in DL},
}


def settings(model: str) -> list[dict]:
    space = SPACES[model]
    rng = np.random.default_rng([20261122, MODELS.index(model)])
    result = [{k: v[0] for k, v in space.items()}]
    while len(result) < SETTINGS:
        draw = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        if draw not in result:
            result.append(draw)
    return result


# ----------------------------------------------------------------------------- deep models (§9b, with sizes)

def build(name: str, s: dict):
    import torch
    import torch.nn as nn
    n = lambda base: int(base * s["width"])              # noqa: E731
    d1, d2 = 0.1 * s["dropout"], 0.2 * s["dropout"]
    layers = 2 + s["depth"]                              # recurrent layers: 1, 2, 3

    class Recurrent(nn.Module):
        def __init__(self, kind, inputs, hidden, project):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(d2)) if project else nn.Identity()
            self.rnn = kind(hidden, hidden, num_layers=layers, batch_first=True, dropout=d2 if layers > 1 else 0.0)

        def forward(self, x):
            return self.rnn(self.project(x))[0][:, -1]

    class ResBlock(nn.Module):
        def __init__(self, a, b, stride):
            super().__init__()
            self.body = nn.Sequential(nn.Conv1d(a, b, 5, stride, 2), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(d1),
                                      nn.Conv1d(b, b, 5, 1, 2), nn.BatchNorm1d(b))
            self.skip = nn.Conv1d(a, b, 1, stride) if (a != b or stride != 1) else nn.Identity()

        def forward(self, x):
            return nn.functional.gelu(self.body(x) + self.skip(x))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            a, b = n(48), n(64)
            blocks = [ResBlock(a, a, 1), ResBlock(a, b, 2), ResBlock(b, b, 2), ResBlock(b, b, 1)][:3 + s["depth"]]
            self.net = nn.Sequential(nn.Conv1d(48, a, 7, 2, 3), nn.BatchNorm1d(a), nn.GELU(), *blocks)

        def forward(self, x):
            h = self.net(x.transpose(1, 2))
            return torch.cat([h.mean(-1), h.amax(-1)], dim=1)

    class Causal(nn.Module):
        def __init__(self, a, b, dilation):
            super().__init__()
            self.pad = 4 * dilation
            self.conv1, self.conv2 = nn.Conv1d(a, b, 5, dilation=dilation), nn.Conv1d(b, b, 5, dilation=dilation)
            self.norm1, self.norm2, self.drop = nn.BatchNorm1d(b), nn.BatchNorm1d(b), nn.Dropout(d1)
            self.skip = nn.Conv1d(a, b, 1) if a != b else nn.Identity()

        def forward(self, x):
            pad = nn.functional.pad
            h = self.drop(nn.functional.gelu(self.norm1(self.conv1(pad(x, (self.pad, 0))))))
            return nn.functional.gelu(self.norm2(self.conv2(pad(h, (self.pad, 0)))) + self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            c = n(48)
            self.net = nn.Sequential(*[Causal(48 if i == 0 else c, c, 2 ** i) for i in range(5 + s["depth"])])

        def forward(self, x):
            return self.net(x.transpose(1, 2))[:, :, -1]

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            c = n(64)
            self.conv = nn.Sequential(nn.Conv1d(48, c, 5, 2, 2), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(d1),
                                      nn.Conv1d(c, c, 5, 2, 2), nn.BatchNorm1d(c), nn.GELU())
            self.rnn = Recurrent(nn.GRU, c, c, False)

        def forward(self, x):
            return self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))

    class Transformer(nn.Module):
        def __init__(self, tokens=40):
            super().__init__()
            d = n(64)
            self.patch = nn.Conv1d(48, d, 4, 4)
            self.position = nn.Parameter(torch.randn(1, tokens, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, d1, "gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 3 + s["depth"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            h = self.norm(self.encoder(self.patch(x.transpose(1, 2)).transpose(1, 2) + self.position))
            return torch.cat([h.mean(1), h[:, -1]], dim=1)

    h = n(64)
    body, width = {"cnn": (CNN, 2 * (n(64) if s["depth"] > -1 else n(64))), "tcn": (TCN, n(48)),
                   "lstm": (lambda: Recurrent(nn.LSTM, 48, h, True), h), "gru": (lambda: Recurrent(nn.GRU, 48, h, True), h),
                   "cnn_gru": (CNNGRU, h), "transformer": (Transformer, 2 * h)}[name]
    return nn.Sequential(body(), nn.Linear(width, 1))


def fit_deep(name: str, s: dict, train: list[str], test: list[str], seed: int) -> dict[str, np.ndarray]:
    """As train_models.train_deep, with the setting s. Returns participant -> window probabilities."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    stacked = np.concatenate([raw(p)[0] for p in train])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    x_all = torch.tensor((stacked - mean) / std, device=device)
    active = torch.tensor(np.concatenate([raw(p)[1] for p in train]).astype(np.float32), device=device)
    cumulative = torch.cat([torch.zeros(1, device=device), active.cumsum(0)])
    ends, low, high, offset = [], [], [], 0
    for p in train:
        n = len(raw(p)[0])
        e = np.arange(WINDOW, n + 1, STEP)
        ends.append(e + offset)
        low.append(np.full(len(e), offset + WINDOW))
        high.append(np.full(len(e), offset + n))
        offset += n
    ends, low, high = (torch.tensor(np.concatenate(v), device=device) for v in (ends, low, high))
    offsets = torch.arange(-WINDOW, 0, device=device)
    model = build(name, s).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    batches = len(ends) // BATCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=s["lr"], total_steps=s["epochs"] * batches)
    weight = torch.tensor(float(s["pos_weight"]), device=device)
    model.train()
    for _ in range(s["epochs"]):
        order = torch.randperm(len(ends), device=device)
        for b in range(batches):
            pick = order[b * BATCH:(b + 1) * BATCH]
            e = ends[pick] + torch.randint(-JITTER, JITTER + 1, (len(pick),), device=device)
            e = torch.minimum(torch.maximum(e, low[pick]), high[pick])
            y = ((cumulative[e] - cumulative[e - STEP]) > 0).float()
            x = x_all[e[:, None] + offsets]
            x = x * (1 + 0.1 * s["augment"] * torch.randn(len(x), 1, x.shape[2], device=device)) + 0.05 * s["augment"] * torch.randn_like(x)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y, pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()
    result = {}
    with torch.no_grad():
        for p in test:
            xt = torch.tensor((raw(p)[0] - mean) / std, device=device)
            test_ends = np.arange(WINDOW, len(xt) + 1, STEP)
            out = []
            for i in range(0, len(test_ends), 1024):
                e = torch.tensor(test_ends[i:i + 1024], device=device)
                out.append(torch.sigmoid(model(xt[e[:, None] + offsets]).squeeze(-1).double()).cpu().numpy())
            result[p] = np.concatenate(out)
    return result


def fit_features(name: str, s: dict, train: list[str], test: list[str], seed: int) -> dict[str, np.ndarray]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.vstack([cached(p)[0] for p in train])
    y = np.concatenate([cached(p)[1] for p in train]).astype(int)
    xt = {p: cached(p)[0] for p in test}
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **s)).fit(x, y)
    elif name == "svm_rbf":                      # decision value through a logistic, as in §9b
        from sklearn.svm import SVC
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        svc = SVC(C=s["C"], kernel="rbf", gamma=s["gamma_factor"] / (z.shape[1] * z.var()), class_weight=s["class_weight"],
                  cache_size=2000).fit(z, y)
        return {p: 1 / (1 + np.exp(-svc.decision_function(scaler.transform(v)))) for p, v in xt.items()}
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(500, random_state=seed, n_jobs=THREADS, **s).fit(x, y)
    elif name == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(tree_method="hist", device="cuda", random_state=seed, n_jobs=THREADS, **s).fit(x, y)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=seed, **s).fit(x, y)
    return {p: np.asarray(model.predict_proba(v)[:, 1], dtype=np.float64) for p, v in xt.items()}


def fit(name, s, train, test, seed):
    out = (fit_deep if name in DL else fit_features)(name, s, train, test, seed)
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError("non-finite probabilities")
    return out


# ----------------------------------------------------------------------------- jobs

def save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def job(args) -> str:
    kind, name, setting, held, seed, path = args
    if Path(path).exists():
        return f"{name} {path}: exists"
    started = time.time()
    train = [p for p in PARTICIPANTS if p not in held]
    try:
        out = fit(name, setting, train, list(held), seed)
    except Exception as exc:                     # reported in the log; the output stays missing
        return f"{name} {path}: FAILED {type(exc).__name__}: {exc}"
    end = {p: cached(p)[2] for p in held}
    for p in held:
        assert len(end[p]) == len(out[p]), (name, p)
    if kind == "inner":
        save(Path(path), **{f"end_{p}": end[p] for p in held}, **{f"prob_{p}": out[p] for p in held})
    else:
        save(Path(path), window_end_s=end[held[0]], probability=out[held[0]])
    return f"{name} {Path(path).parent.name}/{Path(path).name}: {time.time() - started:.0f} s"


def inner_path(model: str, k: int, pair: tuple[str, str], base: Path = OUT) -> Path:
    return base / "inner" / model / f"c{k}" / f"{pair[0]}_{pair[1]}.npz"


def final_path(model: str, p: str, seed: int) -> Path:
    return OUT / "final" / model / f"test_{p}" / f"seed{seed}.npz"


def run(tasks: list, jobs: int) -> int:
    print(f"{len(tasks)} jobs, {jobs} at a time", flush=True)
    failed, started = 0, time.time()
    with mp.get_context("spawn").Pool(jobs) as pool:
        for i, message in enumerate(pool.imap_unordered(job, tasks), start=1):
            failed += "FAILED" in message
            if "FAILED" in message or i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)}  {(time.time() - started) / 60:6.1f} min  {message}", flush=True)
    print(f"ALL DONE, {failed} failed", flush=True)
    return failed


def smoke() -> None:
    import torch
    from train_models import build as build_9b
    for m in DL:
        new, old = (sum(p.numel() for p in b.parameters()) for b in (build(m, settings(m)[0]), build_9b(m)))
        assert new == old, (m, new, old)
        for s in settings(m):
            build(m, s)(torch.zeros(2, WINDOW, 48))
    print("setting 0 has the §9b parameter counts; all deep settings run forward", flush=True)
    base = Path(tempfile.gettempdir(), "sts_tuning_smoke")
    tasks = [("inner", m, settings(m)[k], PAIRS[0], 0, str(inner_path(m, k, PAIRS[0], base))) for m in MODELS for k in (0, 1, 2)]
    with mp.get_context("spawn").Pool(8) as pool:
        for message in pool.imap_unordered(job, tasks):
            print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["inner", "final", "smoke"])
    parser.add_argument("models", nargs="*", default=["all"])
    parser.add_argument("-j", "--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "smoke":
        return smoke()
    names = [n for m in args.models for n in {"dl": DL, "ml": ML, "all": MODELS}.get(m, [m])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs.json").write_text(json.dumps({m: settings(m) for m in MODELS}, indent=1), encoding="utf-8")
    if args.stage == "inner":
        tasks = [("inner", m, settings(m)[k], pair, 0, str(inner_path(m, k, pair)))
                 for k in range(SETTINGS) for m in names for pair in PAIRS]
    else:
        chosen = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
        tasks = [("final", m, settings(m)[chosen[m][p]["setting"]], (p,), seed, str(final_path(m, p, seed)))
                 for seed in SEEDS for m in names if seed == SEEDS[0] or m not in DETERMINISTIC for p in PARTICIPANTS]
    sys.exit(1 if run(tasks, args.jobs) else 0)


if __name__ == "__main__":
    main()
