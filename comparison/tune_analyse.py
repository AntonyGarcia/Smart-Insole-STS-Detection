"""Tuned, strictly nested search (the study protocol, §9c): scoring and statistics of the final models.

Reads comparison/tuning/selected.json and comparison/tuning/final/. Writes results/tuned_participants.csv,
tuned_events.csv, tuned_descriptive.csv, tuned_tests.csv, tuned_nested.csv and results/tuned_comparison.md.

  python comparison/tune_analyse.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for folder in ("analysis", "baselines", "comparison"):
    sys.path.insert(0, str(ROOT / folder))
from b2_hgb_window import cached  # noqa: E402
from events import PARTICIPANTS, score, window_alert_times  # noqa: E402
from score_detectors import STAND_UPS, Writer  # noqa: E402
from froc import EVENTS, HOURS, POINTS, area  # noqa: E402
from stats import paired_timing, participant_metrics, run_mean  # noqa: E402
from stats_tools import bca, hodges_lehmann, holm, signed_rank_test  # noqa: E402
from test_tools import sign_flip  # noqa: E402
from tune_models import DETERMINISTIC, DL, ML, MODELS, OUT, SEEDS, final_path, settings  # noqa: E402

RES = ROOT / "results"
REFERENCE, N = "HGB", len(PARTICIPANTS)
CONTEXT = {"B1": "B1", "B3": "B3", "LSTM": "LSTM_ramp"}
METRICS = {"f1": "F1", "sensitivity": "caught", "fa_per_hour": "FA/h", "onset": "onset latency s"}
LOWER_IS_BETTER = {"fa_per_hour", "onset"}
lines: list[str] = []
tests: list[dict] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def seeds(model: str) -> list[int]:
    return SEEDS[:1] if model in DETERMINISTIC else SEEDS


def compare(title: str, differences: dict[str, np.ndarray], unit: str = "", family: str = "") -> None:
    raw, info = {}, {}
    for m, d in differences.items():
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        raw[m] = signed_rank_test(d, "two-sided")["p"]
        info[m] = (hodges_lehmann(d, 0.95, "two-sided"), len(d))
    adjusted = holm(raw)
    say(f"\n### {title} (exact signed-rank; HL estimate [95 % CI]; Holm over {len(raw)})\n")
    for m, (c, n) in info.items():
        say(f"- {m}: {c['estimate']:+.3f}{unit} [{c['lo']:+.3f}, {c['hi']:+.3f}], p = {raw[m]:.4f}, Holm p = {adjusted[m]:.4f}"
            + (" *" if adjusted[m] < 0.05 else "") + ("" if n == N else f" (n = {n})"))
        tests.append({"family": family or title, "item": m, "estimate": c["estimate"], "lo": c["lo"], "hi": c["hi"], "p": raw[m],
                      "holm_p": adjusted[m], "n": n})


def main() -> None:
    selected = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    w, outputs = Writer(), {}
    for m in MODELS:
        for seed in seeds(m):
            for p in PARTICIPANTS:
                d = np.load(final_path(m, p, seed))                      # a missing file stops the analysis
                outputs[m, seed, p] = (d["probability"], d["window_end_s"])
                w.add(m, f"seed{seed}", p, window_alert_times(d["probability"], d["window_end_s"], selected[m][p]["theta"]),
                      selected[m][p]["theta"], selected[m][p]["fallback"], "strict nested, tuned")
    parts, events = pd.DataFrame(w.participants), pd.DataFrame(w.events)
    old_p, old_e = pd.read_csv(RES / "participants.csv"), pd.read_csv(RES / "events.csv")
    for source, name in CONTEXT.items():
        parts = pd.concat([parts, old_p[old_p.detector == source].assign(detector=name)])
        events = pd.concat([events, old_e[old_e.detector == source].assign(detector=name)])
    parts.to_csv(RES / "tuned_participants.csv", index=False)
    events.to_csv(RES / "tuned_events.csv", index=False)
    present = MODELS + list(CONTEXT.values())
    metrics = {m: participant_metrics(parts, events, m) for m in present}

    # --- settings -------------------------------------------------------------------------------------
    say("# Tuned, strictly nested search (the study protocol, §9c, exploratory)\n")
    say("## Chosen settings (number = how many of the 11 folds; c0 = untuned §9b setting)\n")
    for m in MODELS:
        c = Counter(selected[m][p]["setting"] for p in PARTICIPANTS)
        top = c.most_common(1)[0][0]
        say(f"- {m}: " + ", ".join(f"c{k} × {v}" for k, v in c.most_common()) + f"; most common: {json.dumps(settings(m)[top])}")
    fallback = {m: sum(selected[m][p]["fallback"] for p in PARTICIPANTS) for m in MODELS}
    say(f"\nFolds where no threshold reached 95 % on the inner outputs: {fallback}")

    # --- AUC and FROC of the final outputs --------------------------------------------------------------
    target = {p: cached(p)[1].astype(bool) for p in PARTICIPANTS}
    auc = {m: np.array([np.mean([roc_auc_score(target[p], outputs[m, s, p][0]) for s in seeds(m)]) for p in PARTICIPANTS]) for m in MODELS}
    froc = {}
    for m in MODELS:
        hs, fs = [], []
        for s in seeds(m):
            thresholds = np.quantile(np.concatenate([outputs[m, s, p][0] for p in PARTICIPANTS]), 1 - np.geomspace(5e-4, 0.6, POINTS))
            h, f = np.zeros((POINTS, N)), np.zeros((POINTS, N))
            for j, p in enumerate(PARTICIPANTS):
                for i, t in enumerate(thresholds):
                    r = score(window_alert_times(outputs[m, s, p][0], outputs[m, s, p][1], t), STAND_UPS[p])
                    h[i, j], f[i, j] = r.hits, r.false
            hs.append(h)
            fs.append(f)
        if len(hs) == 1:
            hs, fs = hs * len(SEEDS), fs * len(SEEDS)
        froc[m] = (np.array(hs), np.array(fs))

    # --- performance ------------------------------------------------------------------------------------
    say("\n## Performance: mean over runs, pooled over participants [95 % participant-bootstrap BCa CI]\n")
    say("| Model | Family | Runs | Caught % | FA/h | FA/h range over runs | F1 | Onset s | Before upright % | AUC | FROC area % |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    descriptive = []
    weights = lambda i: np.bincount(i, minlength=N).astype(float)  # noqa: E731
    for m in present:
        per = metrics[m].groupby("participant")[["events", "hits", "false", "minutes", "onset", "before_upright"]].mean().reindex(PARTICIPANTS)
        a = {c: per[c].to_numpy() for c in per}
        fns = {"caught": lambda i, a=a: 100 * a["hits"][i].sum() / a["events"][i].sum(),
               "fa_per_hour": lambda i, a=a: 60 * a["false"][i].sum() / a["minutes"][i].sum(),
               "f1": lambda i, a=a: 2 * a["hits"][i].sum() / (a["events"][i].sum() + a["hits"][i].sum() + a["false"][i].sum()),
               "onset": lambda i, a=a: np.nanmedian(a["onset"][i]),
               "before_upright": lambda i, a=a: 100 * np.nanmedian(a["before_upright"][i])}
        row = {"model": m, "family": "ML" if m in ML else "DL" if m in DL else "context", "runs": metrics[m].run.nunique()}
        for name, fn in fns.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bca(fn, N)
        g = metrics[m].groupby("run")
        by_run = 60 * g.false.sum() / g.minutes.sum()
        row["fa_min"], row["fa_max"] = by_run.min(), by_run.max()
        if m in auc:
            row["auc"], row["auc_lo"], row["auc_hi"] = bca(lambda i, v=auc[m]: v[i].mean(), N)
            row["area"], row["area_lo"], row["area_hi"] = bca(lambda i, m=m: 100 * area(*froc[m], weights(i)), N, resamples=2000)
        descriptive.append(row)
        ci = lambda k, f: f"{row[k]:{f}} [{row[k + '_lo']:{f}}, {row[k + '_hi']:{f}}]" if k in row else "–"  # noqa: E731
        say(f"| {m} | {row['family']} | {row['runs']} | {ci('caught', '.1f')} | {ci('fa_per_hour', '.2f')} | {row['fa_min']:.2f}–{row['fa_max']:.2f} | "
            f"{ci('f1', '.3f')} | {ci('onset', '.2f')} | {ci('before_upright', '.0f')} | {ci('auc', '.3f')} | {ci('area', '.1f')} |")
    pd.DataFrame(descriptive).to_csv(RES / "tuned_descriptive.csv", index=False)
    say("\nLSTM_ramp: strictly nested thresholds, one untuned setting. B1 and B3 as before.")

    # --- nested selection ---------------------------------------------------------------------------------
    say("\n## What the search delivers: winner chosen without the left-out participant, scored on that participant\n")
    winners = selected["_winners"]

    def held_out(names: dict[str, str]) -> dict[str, np.ndarray]:
        rows = []
        for p in PARTICIPANTS:
            d = metrics[names[p]]
            d = d[d.participant == p]
            rows.append([d.events.mean(), d.hits.mean(), d.false.mean(), d.minutes.mean(), d.onset.mean(), d.f1.mean(), d.fa_per_hour.mean(),
                         d.lead.mean()])
        return dict(zip(["events", "hits", "false", "minutes", "onset", "f1", "fa", "lead"], np.array(rows).T))
    held = {}
    say("| Rule | Chosen (folds) | Caught % | FA/h | F1 | Onset s |")
    say("|---|---|---|---|---|---|")
    nested = []
    candidates = len(MODELS) + 2                                             # tuned models + B1 + B3
    labels = {"A": f"A: fewest FA at ≥ 95 % caught, all {candidates}", "A_ML": "A within ML", "A_DL": "A within DL",
              "F1": f"highest F1, all {candidates}", "area": "largest FROC area"}
    for rule, label in labels.items():
        a = held[rule] = held_out({p: winners[p][rule] for p in PARTICIPANTS})
        c = Counter(winners[p][rule] for p in PARTICIPANTS)
        row = {"rule": rule, "chosen": ", ".join(f"{k} × {v}" for k, v in c.most_common()),
               "by_fold": " ".join(f"{p}:{winners[p][rule]}" for p in PARTICIPANTS)}
        row["caught"], row["caught_lo"], row["caught_hi"] = bca(lambda i: 100 * a["hits"][i].sum() / a["events"][i].sum(), N)
        row["fa"], row["fa_lo"], row["fa_hi"] = bca(lambda i: 60 * a["false"][i].sum() / a["minutes"][i].sum(), N)
        row["f1"], row["f1_lo"], row["f1_hi"] = bca(lambda i: 2 * a["hits"][i].sum() / (a["events"][i].sum() + a["hits"][i].sum() + a["false"][i].sum()), N)
        row["onset"], row["onset_lo"], row["onset_hi"] = bca(lambda i: np.nanmedian(a["onset"][i]), N)
        row["lead"], row["lead_lo"], row["lead_hi"] = bca(lambda i: np.nanmedian(a["lead"][i]), N)   # RQ1: lead time e - tau
        nested.append(row)
        ci = lambda k, f: f"{row[k]:{f}} [{row[k + '_lo']:{f}}, {row[k + '_hi']:{f}}]"  # noqa: E731
        say(f"| {label} | {row['chosen']} | {ci('caught', '.1f')} | {ci('fa', '.2f')} | {ci('f1', '.3f')} | {ci('onset', '.2f')} |")
    pd.DataFrame(nested).to_csv(RES / "tuned_nested.csv", index=False)
    say("\nBy fold:")
    for row in nested:
        say(f"- {row['rule']}: {row['by_fold']}")
    b1 = held_out({p: "B1" for p in PARTICIPANTS})
    for title, x, y in (("Best DL − best ML, both chosen by rule A", held["A_DL"], held["A_ML"]), ("Search winner (rule A) − B1", held["A"], b1)):
        compare(title, {"F1": x["f1"] - y["f1"], "FA/h": x["fa"] - y["fa"], "onset latency s": x["onset"] - y["onset"]})

    # --- omnibus -------------------------------------------------------------------------------------------
    say("\n## Omnibus (participants as blocks): Friedman, Kendall's W, average ranks (1 = best)\n")
    for metric, label in METRICS.items():
        table = np.column_stack([run_mean(metrics[m], metric).to_numpy() for m in present])
        if not np.isfinite(table).all():
            say(f"- **{label}**: not computed, a participant has no value")
            continue
        stat, p = friedmanchisquare(*table.T)
        ranks = np.mean([rankdata(r if metric in LOWER_IS_BETTER else -r) for r in table], axis=0)
        k = len(present)
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
        say(f"- **{label}**: χ²({k - 1}) = {stat:.1f}, p = {p:.2g}, W = {stat / (N * (k - 1)):.2f}; Nemenyi critical difference {cd:.2f}; ranks "
            + ", ".join(f"{present[i]} {ranks[i]:.1f}" for i in np.argsort(ranks)))
        tests.append({"family": "Friedman", "item": metric, "estimate": stat, "p": p})

    # --- vs HGB ----------------------------------------------------------------------------------------------
    others = [m for m in present if m != REFERENCE]
    say("\n## Comparison with HGB (reference kept from §9b)")
    for metric, label in (("f1", "F1"), ("fa_per_hour", "False alarms per hour"), ("onset", "Onset latency")):   # onset: RQ3
        ref = run_mean(metrics[REFERENCE], metric)
        compare(f"{label}: model − HGB", {m: (run_mean(metrics[m], metric) - ref).to_numpy() for m in others})
    for name in ("median", "mean"):
        d = {m: -getattr(paired_timing(events, REFERENCE, m).groupby(["participant", "run"]).d, name)().groupby("participant").mean()
             .reindex(PARTICIPANTS).to_numpy() for m in others}
        compare(f"Alert time, stand-ups caught by both, participant {name}: model − HGB", d, " s")
    compare("AUC per participant: model − HGB", {m: auc[m] - auc[REFERENCE] for m in MODELS if m != REFERENCE})
    share = {m: run_mean(metrics[m], "sensitivity").to_numpy() for m in present}
    flip = {m: sign_flip(share[m] - share[REFERENCE]) for m in others}
    adjusted = holm(flip)
    say("\n### Caught share vs HGB, sign-flip over participants (Holm)\n")
    say("- " + ", ".join(f"{m} {flip[m]:.3f} ({adjusted[m]:.3f})" for m in others))
    say("\n### FROC area vs HGB (permutation swapping models within participants; BCa CI; Holm)\n")
    swaps = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(bool)[: 2 ** (N - 1)]
    hr, fr = froc[REFERENCE]
    one, raw, info = np.ones(N), {}, {}
    for m in MODELS:
        if m == REFERENCE:
            continue
        h, f = froc[m]
        observed = area(h, f, one) - area(hr, fr, one)
        extreme = sum(abs(area(np.where(s, hr, h), np.where(s, fr, f), one) - area(np.where(s, h, hr), np.where(s, f, fr), one))
                      >= abs(observed) - 1e-12 for s in swaps)
        raw[m] = extreme / len(swaps)
        info[m] = bca(lambda i: 100 * (area(h, f, weights(i)) - area(hr, fr, weights(i))), N, resamples=2000)
    adjusted = holm(raw)
    for m, (e, lo, hi) in info.items():
        say(f"- {m}: {e:+.1f} points [{lo:+.1f}, {hi:+.1f}], p = {raw[m]:.4f}, Holm p = {adjusted[m]:.4f}" + (" *" if adjusted[m] < 0.05 else ""))
        tests.append({"family": "FROC area vs HGB", "item": m, "estimate": e, "lo": lo, "hi": hi, "p": raw[m], "holm_p": adjusted[m]})

    # --- tuned vs untuned --------------------------------------------------------------------------------------
    say("\n## Tuned (nested) − untuned (§9b)")
    up, ue = pd.read_csv(RES / "models_participants.csv"), pd.read_csv(RES / "models_events.csv")
    before = {m: participant_metrics(up, ue, m) for m in MODELS}
    for metric, label in (("f1", "F1"), ("fa_per_hour", "False alarms per hour")):
        compare(f"{label}: tuned − untuned", {m: (run_mean(metrics[m], metric) - run_mean(before[m], metric)).to_numpy() for m in MODELS})

    # --- families ------------------------------------------------------------------------------------------------
    say(f"\n## Machine learning ({len(ML)}) vs deep learning ({len(DL)}): per-participant family means, DL − ML\n")
    for metric, label in METRICS.items():
        d = (np.mean([run_mean(metrics[m], metric).to_numpy() for m in DL], axis=0)
             - np.mean([run_mean(metrics[m], metric).to_numpy() for m in ML], axis=0))
        d = d[np.isfinite(d)]
        c, p = hodges_lehmann(d, 0.95, "two-sided"), signed_rank_test(d, "two-sided")["p"]
        say(f"- {label}: {c['estimate']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}], p = {p:.4f}")
        tests.append({"family": "DL − ML family means", "item": metric, "estimate": c["estimate"], "lo": c["lo"], "hi": c["hi"], "p": p})
    say("\nThe family comparison describes these models only; it is not evidence about all ML or all DL methods.")
    pd.DataFrame(tests).to_csv(RES / "tuned_tests.csv", index=False)
    (RES / "tuned_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
