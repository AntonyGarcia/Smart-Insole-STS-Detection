"""Computational cost (the study protocol, §8.6): size and CPU time per update of each detector, single CPU thread.

LSTM  one forward pass of a 2 s window (1 x 160 x 48) = one 0.5 s update (lstm/runs/loo_final, S01 seed0 model)
B1    paper-1 HGB: features + predict_proba for one 2 s window = one 1 s update
B2    HGB windowed classifier: predict_proba for one feature vector = one 0.5 s update (refit once on cached features)
B3    load rule: causal moving average + comparison over one 0.5 s block
Median of 1,000 timed calls after 50 warm-up calls. Writes results/compute_cost.csv.

  python analysis/compute_cost.py
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lstm"))
sys.path.insert(0, str(ROOT / "baselines"))
torch.set_num_threads(1)


def timed(fn, repeats: int = 1000) -> float:
    for _ in range(50):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    rows = []
    from train_movement_lstm import MovementLSTM
    path = ROOT / "lstm" / "runs" / "loo_final" / "test_S01" / "seed0" / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MovementLSTM(len(checkpoint["channels"]), config["hidden"], config["layers"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.randn(1, 160, len(checkpoint["channels"]))
    with torch.no_grad():
        ms = timed(lambda: model(x))
    params = sum(p.numel() for p in model.parameters())
    rows.append({"detector": "LSTM", "parameters": params, "size_kB": params * 4 / 1024,
                 "update_every_s": 0.5, "cpu_ms_per_update": ms})

    import b1_hgb_state as b1
    hgb = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet.joblib")
    meta = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet_meta.joblib")
    window = [np.random.rand(1, 24, 160).astype(np.float32) for _ in range(2)]
    ms = timed(lambda: hgb.predict_proba(b1.features(*window, meta["feature_cols"])), 300)
    rows.append({"detector": "B1", "parameters": np.nan,
                 "size_kB": (ROOT / "baselines" / "model" / "hgb_both_feet.joblib").stat().st_size / 1024,
                 "update_every_s": 1.0, "cpu_ms_per_update": ms})

    from sklearn.ensemble import HistGradientBoostingClassifier
    cache = ROOT / "baselines" / "runs" / "b2" / "features"
    data = [np.load(cache / f"S{i:02d}.npz") for i in range(2, 12)]
    b2 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(
        np.vstack([d["feats"] for d in data]), np.concatenate([d["target"] for d in data]))
    one = data[0]["feats"][:1]
    ms = timed(lambda: b2.predict_proba(one), 300)
    joblib.dump(b2, Path(tempfile.gettempdir(), "b2_size.joblib"))
    rows.append({"detector": "B2", "parameters": np.nan, "size_kB": Path(Path(tempfile.gettempdir(), "b2_size.joblib")).stat().st_size / 1024,
                 "update_every_s": 0.5, "cpu_ms_per_update": ms})

    block = np.random.rand(60).astype(np.float32)
    kernel = np.ones(20, dtype=np.float32) / 20
    ms = timed(lambda: np.convolve(block, kernel, mode="valid") > 0.5)
    rows.append({"detector": "B3", "parameters": 0, "size_kB": 0.0, "update_every_s": 0.5, "cpu_ms_per_update": ms})

    table = pd.DataFrame(rows)
    table["cpu"] = cpu_name()
    table["threads"] = 1
    table.to_csv(ROOT / "results" / "compute_cost.csv", index=False)
    print(f"CPU: {cpu_name()}, 1 thread\n")
    print(table.drop(columns=["cpu", "threads"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
