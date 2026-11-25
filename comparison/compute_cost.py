"""Computational cost of each detector with its base setting (Section 2.6): size and time to classify one window.

ML models are fitted on ten participants (all but S01) with the settings of train_models.py / b2_hgb_window.py.
DL models and the ramp LSTM are built untrained: size and time do not depend on the weights.
Timing: one window at a time (batch of 1), one CPU thread, median of 300 calls after 30 warm-up calls.
The 576 window features of the ML models are timed separately, from a raw 48 x 160 window.

  OMP_NUM_THREADS=1 python comparison/compute_cost.py      writes results/compute_cost.md and .csv
"""

from __future__ import annotations

import os
import pickle
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for folder in ("analysis", "baselines", "comparison", "lstm"):
    sys.path.insert(0, str(ROOT / folder))
from b2_hgb_window import cached  # noqa: E402
from events import PARTICIPANTS  # noqa: E402

RES = ROOT / "results"
REPEAT, WARM = 300, 30


def timed(fn) -> float:
    for _ in range(WARM):
        fn()
    t = []
    for _ in range(REPEAT):
        a = time.perf_counter()
        fn()
        t.append(time.perf_counter() - a)
    return 1000 * float(np.median(t))


def features(w: np.ndarray) -> np.ndarray:
    """The 576 features of one (48, 160) window, as in b2_hgb_window.load."""
    seg = w.reshape(48, 4, 40)
    return np.concatenate([w.mean(-1), w.std(-1), w.min(-1), w.max(-1), seg.mean(-1).ravel(), seg.std(-1).ravel()])


def main() -> None:
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    from train_models import build
    from train_movement_lstm import MovementLSTM

    torch.set_num_threads(1)
    data = {p: cached(p) for p in PARTICIPANTS}
    x = np.vstack([data[p][0] for p in PARTICIPANTS[1:]])
    y = np.concatenate([data[p][1] for p in PARTICIPANTS[1:]]).astype(int)
    one = data[PARTICIPANTS[0]][0][:1]
    raw = np.random.default_rng(0).standard_normal((48, 160)).astype(np.float32)
    rows = []

    ml = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
          "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=2000)),
          "Random forest": RandomForestClassifier(500, random_state=0, n_jobs=8),
          "XGBoost": XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=0, n_jobs=8),
          "HGB": HistGradientBoostingClassifier(max_iter=200, random_state=0)}
    for name, model in ml.items():
        model.fit(x, y)
        if name == "Random forest":
            model.n_jobs = 1
            size = sum(t.tree_.node_count for t in model.estimators_)
            unit = "nodes"
        elif name == "XGBoost":
            model.set_params(n_jobs=1)
            size = len(model.get_booster().trees_to_dataframe())
            unit = "nodes"
        elif name == "HGB":
            size = sum(len(pred.nodes) for it in model._predictors for pred in it)
            unit = "nodes"
        elif name == "SVM":
            svc = model[-1]
            size = int(svc.support_vectors_.size + svc.dual_coef_.size + 1)
            unit = f"values ({len(svc.support_)} support vectors)"
        else:
            lr = model[-1]
            size = int(lr.coef_.size + lr.intercept_.size)
            unit = "parameters"
        kb = len(pickle.dumps(model)) / 1024
        ms = timed(lambda m=model: m.predict_proba(one) if hasattr(m, "predict_proba") else m.decision_function(one)) \
            if name != "SVM" else timed(lambda m=model: m.decision_function(one))
        rows.append({"model": name, "family": "ML", "size": size, "unit": unit, "kb": kb, "ms": ms})
        print(rows[-1], flush=True)

    feature_ms = timed(lambda: features(raw))
    dl = {"1D-CNN": "cnn", "TCN": "tcn", "LSTM": "lstm", "GRU": "gru", "CNN-GRU": "cnn_gru"}
    window = torch.from_numpy(raw.T.copy()).unsqueeze(0)                       # (1, 160, 48)
    nets = {name: build(key) for name, key in dl.items()}
    nets["LSTM trained on the ramp"] = MovementLSTM(48, 64, 2, 0.2)
    for name, net in nets.items():
        net.eval()
        params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            ms = timed(lambda n=net: n(window))
        rows.append({"model": name, "family": "DL" if name in dl else "ramp", "size": params, "unit": "parameters",
                     "kb": 4 * params / 1024, "ms": ms})
        print(rows[-1], flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RES / "compute_cost.csv", index=False)
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    lines = [f"# Computational cost (base settings; one window, one CPU thread, {cpu})\n",
             f"Feature extraction for the ML models (576 features from one 48 x 160 window): {feature_ms:.3f} ms\n",
             "| Model | Family | Size | Stored, kB | Time per window, ms |", "|---|---|---|---|---|"]
    lines += [f"| {r.model} | {r.family} | {r.size:,} {r.unit} | {r.kb:,.0f} | {r.ms:.3f} |" for r in df.itertuples()]
    (RES / "compute_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
