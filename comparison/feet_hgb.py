"""Exploratory, post hoc (2026-11-25): one foot against both feet with the reference HGB.

Prompted by RQ8 (M_ramp): one foot raised about 12 false alarms/h against 4.85 with both feet, but false alarms were not
part of the planned test. Same data as RQ8, so this is exploratory, not a confirmation.

Channel sets  all 48 channels, left foot (channels 0-23: IMU + 18 pressure cells), right foot (24-47).
Per outer fold (leave-one-participant-out, comparison/ga_channels.Fold):
  - operating point: threshold from the inner out-of-fold outputs of the 10 training participants with the paper's rule
    (fewest false alarms at >= 95 % caught), HGB trained on the 10, scored on the held-out participant (Fold.evaluate);
  - threshold-free: hits and false alarms of the held-out participant at the 128 thresholds of the tuned search.
Tests (all 48 minus one foot; Holm over the two feet within each metric)
  - FROC area over 0-5 false alarms/h from counts pooled over participants: exact permutation test swapping the two
    channel sets within participants (2^11), BCa CI over participants;
  - false alarms/h and F1 at the operating point: exact signed-rank (Pratt), Hodges-Lehmann estimate and 95 % CI.

  python comparison/feet_hgb.py [-j 4]      writes results/feet_hgb.md, feet_hgb_participants.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("analysis", "baselines", "comparison"):
    sys.path.insert(0, str(ROOT / folder))
from b2_hgb_window import cached  # noqa: E402
from events import PARTICIPANTS, score, window_alert_times  # noqa: E402
from ga_channels import THETA, Fold, columns, froc_area, hgb  # noqa: E402
from score_detectors import MINUTES, STAND_UPS  # noqa: E402

RES = ROOT / "results"
SETS = {"all": np.arange(48), "left": np.arange(24), "right": np.arange(24, 48)}


def run_fold(test: str) -> dict:
    fold = Fold(test, {p: cached(p) for p in PARTICIPANTS})
    out = {"participant": test}
    for name, channels in SETS.items():
        op = fold.evaluate(channels)
        cols = columns(channels)
        x, y = fold.stack(fold.train, cols)
        prob = hgb().fit(x, y).predict_proba(fold.data[test][0][:, cols])[:, 1]
        s = [score(window_alert_times(prob, fold.data[test][2], t), STAND_UPS[test]) for t in THETA]
        out[name] = {**op, "hits_t": [r.hits for r in s], "false_t": [r.false for r in s]}
    print(f"{test} done", flush=True)
    return out


def main() -> None:
    import pandas as pd
    from stats_tools import bca, hodges_lehmann, holm, signed_rank_test

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with get_context("spawn").Pool(args.jobs) as pool:
        folds = sorted(pool.map(run_fold, PARTICIPANTS), key=lambda f: f["participant"])

    n = len(PARTICIPANTS)
    events = np.array([len(STAND_UPS[p]) for p in PARTICIPANTS], float)
    hours = np.array([MINUTES[p] for p in PARTICIPANTS]) / 60
    H = {k: np.array([f[k]["hits_t"] for f in folds], float) for k in SETS}      # (participant, threshold)
    F = {k: np.array([f[k]["false_t"] for f in folds], float) for k in SETS}

    def area(k_of_p, idx):
        h = sum(H[k_of_p[i]][i] for i in idx)
        f = sum(F[k_of_p[i]][i] for i in idx)
        return 100 * froc_area(h / events[idx].sum(), f / hours[idx].sum())

    rows = []
    for f in folds:
        for k in SETS:
            o = f[k]
            rows.append({"participant": f["participant"], "channels": k, "theta": o["theta"], "fallback": o["fallback"],
                         "events": o["events"], "hits": o["hits"], "false": o["false"], "minutes": o["minutes"],
                         "fa_per_hour": 60 * o["false"] / o["minutes"], "f1": o["f1"], "onset": o["onset"]})
    per = pd.DataFrame(rows)
    per.to_csv(RES / "feet_hgb_participants.csv", index=False)
    wide = {m: per.pivot(index="participant", columns="channels", values=m).reindex(PARTICIPANTS)
            for m in ("hits", "false", "events", "minutes", "f1", "fa_per_hour", "onset")}

    lines = ["# One foot against both feet, HGB (exploratory, post hoc)\n",
             "| Channels | Caught % | False alarms/h | F1 | Onset s | FROC area % |", "|---|---|---|---|---|---|"]
    every = np.arange(n)
    for k in SETS:
        g = {m: wide[m][k].to_numpy(float) for m in wide}
        fns = {"caught": lambda i: 100 * g["hits"][i].sum() / g["events"][i].sum(),
               "fa": lambda i: 60 * g["false"][i].sum() / g["minutes"][i].sum(),
               "f1": lambda i: 2 * g["hits"][i].sum() / (g["events"][i].sum() + g["hits"][i].sum() + g["false"][i].sum()),
               "onset": lambda i: np.nanmedian(g["onset"][i])}
        c = {m: bca(fn, n) for m, fn in fns.items()}
        a = bca(lambda i, k=k: area([k] * n, i), n, resamples=2000)
        ci = lambda t, fmt: f"{t[0]:{fmt}} [{t[1]:{fmt}}, {t[2]:{fmt}}]"  # noqa: E731
        lines.append(f"| {k} | {ci(c['caught'], '.1f')} | {ci(c['fa'], '.2f')} | {ci(c['f1'], '.3f')} | "
                     f"{ci(c['onset'], '.2f')} | {ci(a, '.1f')} |")

    lines += ["", "## All 48 channels minus one foot (Holm over the two feet within each metric)", ""]
    for foot in ("left", "right"):
        diff = lambda i, foot=foot: area(["all"] * n, i) - area([foot] * n, i)  # noqa: E731
        observed = diff(every)
        null = [area(["all" if s else foot for s in swap], every) - area([foot if s else "all" for s in swap], every)
                for swap in itertools.product([True, False], repeat=n)]
        p_area = float(np.mean(np.abs(null) >= abs(observed) - 1e-9))
        _, lo, hi = bca(diff, n, resamples=2000)
        tests = {"FROC area, points": (observed, lo, hi, p_area, "permutation")}
        for m, label in (("fa_per_hour", "false alarms/h"), ("f1", "F1"), ("onset", "onset latency s")):
            d = (wide[m]["all"] - wide[m][foot]).to_numpy(float)
            hl = hodges_lehmann(d, 0.95, "two-sided")
            tests[label] = (hl["estimate"], hl["lo"], hl["hi"], signed_rank_test(d, "two-sided")["p"], "signed-rank")
        fa_d = (wide["false"]["all"] - wide["false"][foot]).to_numpy()
        tests["_count"] = (int((fa_d < 0).sum()), int((fa_d > 0).sum()), int((fa_d == 0).sum()))
        globals().setdefault("RESULTS", {})[foot] = tests
    for label in ("FROC area, points", "false alarms/h", "F1", "onset latency s"):
        adj = holm({foot: RESULTS[foot][label][3] for foot in ("left", "right")})
        for foot in ("left", "right"):
            est, lo, hi, p, test = RESULTS[foot][label]
            lines.append(f"- {label}, all − {foot}: {est:+.3f} [{lo:+.3f}, {hi:+.3f}], {test} p = {p:.4f}, Holm p = {adj[foot]:.4f}")
    for foot in ("left", "right"):
        more, fewer, same = RESULTS[foot]["_count"]
        lines.append(f"- false alarms at the operating point, {foot} foot vs all 48: more in {more}, fewer in {fewer}, "
                     f"equal in {same} participants")

    # M_ramp (first stage, RQ8 inputs): false alarms/h per participant, mean over runs; both feet minus one foot
    lines += ["", "## M_ramp, both feet minus one foot: false alarms/h (first-stage outputs; Holm over the two feet)", ""]
    parts = pd.read_csv(RES / "participants.csv")
    parts["fa_h"] = 60 * parts["false"] / parts["minutes"]
    fa = parts.groupby(["detector", "participant"]).fa_h.mean().unstack().reindex(columns=PARTICIPANTS)
    ramp = {}
    for foot, det in (("left", "LSTM_left"), ("right", "LSTM_right")):
        d = (fa.loc["LSTM_approx"] - fa.loc[det]).to_numpy(float)
        hl = hodges_lehmann(d, 0.95, "two-sided")
        ramp[foot] = (hl["estimate"], hl["lo"], hl["hi"], signed_rank_test(d, "two-sided")["p"],
                      int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()))
    adj = holm({f: ramp[f][3] for f in ramp})
    for foot, (est, lo, hi, p, more, fewer, same) in ramp.items():
        lines.append(f"- both − {foot}: {est:+.2f} /h [{lo:+.2f}, {hi:+.2f}], signed-rank p = {p:.4f}, Holm p = {adj[foot]:.4f}; "
                     f"{foot} foot more in {more}, fewer in {fewer}, equal in {same} participants")
    (RES / "feet_hgb.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
