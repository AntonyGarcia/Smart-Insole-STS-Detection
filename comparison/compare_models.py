"""Statistical comparison of the models of the study protocol, §9b (exploratory; reference = HGB).

Main results use the participant (n = 11) as the unit. Pooled tests (Cochran's Q, exact McNemar, DeLong) are a
supplement: they treat stand-ups / windows as independent. Holm within each test type, over all models vs HGB.

Reads results/models_*.csv and the window outputs; writes results/model_comparison.md, model_descriptive.csv,
model_tests.csv.

  python comparison/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "comparison"))
from b2_hgb_window import cached  # noqa: E402
from events import PARTICIPANTS  # noqa: E402
from score_models import DL, ML, RES, outputs, runs_of  # noqa: E402
from stats import paired_timing, participant_metrics, run_mean  # noqa: E402
from stats_tools import bca, hodges_lehmann, holm, signed_rank_test  # noqa: E402
from test_tools import cochran_q, delong, delong_clustered, mcnemar_clustered, mcnemar_exact, sign_flip  # noqa: E402

REFERENCE, MAIN_RUN = "HGB", "seed0"
METRICS = {"f1": "F1", "sensitivity": "caught", "fa_per_hour": "FA/h", "onset": "onset latency s"}
LOWER_IS_BETTER = {"fa_per_hour", "onset"}
lines: list[str] = []
tests: list[dict] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def main() -> None:
    parts, events = pd.read_csv(RES / "models_participants.csv"), pd.read_csv(RES / "models_events.csv")
    present = list(dict.fromkeys(parts.detector))
    trained = [m for m in ML + DL if m in present]
    others = [m for m in present if m != REFERENCE]
    metrics = {m: participant_metrics(parts, events, m) for m in present}
    n = len(PARTICIPANTS)
    target = {p: cached(p)[1].astype(bool) for p in PARTICIPANTS}
    y = np.concatenate([target[p] for p in PARTICIPANTS])
    cluster = np.repeat(np.arange(n), [len(target[p]) for p in PARTICIPANTS])

    # window scores: auc / average precision per participant (mean over runs)
    scores, auc = {}, {}
    for m in trained:
        rows = []
        for run in runs_of(m):
            out = outputs(m, run)
            for p in PARTICIPANTS:
                assert len(out[p]["probability"]) == len(target[p]), (m, run, p)
            scores[m, run] = np.concatenate([out[p]["probability"] for p in PARTICIPANTS])
            rows += [{"run": run, "participant": p, "auc": roc_auc_score(target[p], out[p]["probability"]),
                      "ap": average_precision_score(target[p], out[p]["probability"])} for p in PARTICIPANTS]
        auc[m] = pd.DataFrame(rows).groupby("participant")[["auc", "ap"]].mean().reindex(PARTICIPANTS)

    say("# Model comparison (the study protocol, §9b, exploratory; reference HGB = B2)\n")
    say("## Performance: mean over runs, pooled over participants [95 % participant-bootstrap BCa CI]\n")
    say("| Model | Family | Runs | Caught % | FA/h | F1 | Onset s | Before upright % | AUC | AP |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    descriptive = []
    for m in present:
        family = "ML" if m in ML else "DL" if m in DL else "context"
        per = metrics[m].groupby("participant")[["events", "hits", "false", "minutes", "onset", "before_upright"]].mean().reindex(PARTICIPANTS)
        a = {c: per[c].to_numpy() for c in per}
        fns = {"caught": lambda i, a=a: 100 * a["hits"][i].sum() / a["events"][i].sum(),
               "fa_per_hour": lambda i, a=a: 60 * a["false"][i].sum() / a["minutes"][i].sum(),
               "f1": lambda i, a=a: 2 * a["hits"][i].sum() / (a["events"][i].sum() + a["hits"][i].sum() + a["false"][i].sum()),
               "onset": lambda i, a=a: np.nanmedian(a["onset"][i]),
               "before_upright": lambda i, a=a: 100 * np.nanmedian(a["before_upright"][i])}
        row = {"model": m, "family": family, "runs": metrics[m].run.nunique()}
        for name, fn in fns.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bca(fn, n)
        if m in auc:
            for name in ("auc", "ap"):
                v = auc[m][name].to_numpy()
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = bca(lambda i, v=v: v[i].mean(), n)
        descriptive.append(row)
        ci = lambda k, f: f"{row[k]:{f}} [{row[k + '_lo']:{f}}, {row[k + '_hi']:{f}}]" if k in row else "–"  # noqa: E731
        say(f"| {m} | {family} | {row['runs']} | {ci('caught', '.1f')} | {ci('fa_per_hour', '.2f')} | {ci('f1', '.3f')} | "
            f"{ci('onset', '.2f')} | {ci('before_upright', '.0f')} | {ci('auc', '.3f')} | {ci('ap', '.3f')} |")
    pd.DataFrame(descriptive).to_csv(RES / "model_descriptive.csv", index=False)
    say("\nAUC and AP: mean of the per-participant values. HGB here uses the §9b threshold grid.")

    # --- omnibus + ranks -----------------------------------------------------------------------------
    say("\n## Omnibus (participants as blocks): Friedman, Kendall's W, average ranks (1 = best)\n")
    for metric, label in METRICS.items():
        table = np.column_stack([run_mean(metrics[m], metric).to_numpy() for m in present])
        stat, p = friedmanchisquare(*table.T)
        ranks = np.mean([rankdata(r if metric in LOWER_IS_BETTER else -r) for r in table], axis=0)
        k = len(present)
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
        say(f"- **{label}**: χ²({k - 1}) = {stat:.1f}, p = {p:.2g}, W = {stat / (n * (k - 1)):.2f}; Nemenyi critical difference {cd:.2f}; ranks "
            + ", ".join(f"{present[i]} {ranks[i]:.1f}" for i in np.argsort(ranks)))
        tests.append({"test": "Friedman", "metric": metric, "statistic": stat, "p": p})

    # --- participant-level tests vs HGB --------------------------------------------------------------
    level = 1 - 0.05 / len(others)

    def compare(name: str, differences: dict[str, np.ndarray], unit: str) -> None:
        raw, info = {}, {}
        for m, d in differences.items():
            d = d[np.isfinite(d)]
            raw[m] = signed_rank_test(d, "two-sided")["p"]
            info[m] = (hodges_lehmann(d, 0.95, "two-sided"), hodges_lehmann(d, level, "two-sided"), len(d))
        adjusted = holm(raw)
        say(f"\n### {name}: model − HGB (exact signed-rank; HL estimate [95 % CI]; Holm over {len(raw)} models)\n")
        for m in differences:
            c, wide, count = info[m]
            say(f"- {m}: {c['estimate']:+.3f}{unit} [{c['lo']:+.3f}, {c['hi']:+.3f}], p = {raw[m]:.4f}, Holm p = {adjusted[m]:.4f}"
                + (" *" if adjusted[m] < 0.05 else "") + ("" if count == n else f" (n = {count})"))
            tests.append({"test": "signed-rank", "metric": name, "model": m, "estimate": c["estimate"], "lo": c["lo"], "hi": c["hi"],
                          "adj_lo": wide["lo"], "adj_hi": wide["hi"], "p": raw[m], "holm_p": adjusted[m]})

    say("\n## Participant-level comparisons with HGB (main results)")
    for metric, label in (("f1", "F1"), ("fa_per_hour", "False alarms per hour")):
        ref = run_mean(metrics[REFERENCE], metric)
        compare(label, {m: (run_mean(metrics[m], metric) - ref).to_numpy() for m in others}, "")
    timing = {}
    for m in others:
        d = paired_timing(events, REFERENCE, m)       # HGB run k vs the model's run k (or its only run)
        timing[m] = -d.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
    compare("Alert time, stand-ups caught by both (negative = earlier than HGB)", timing, " s")
    compare("AUC per participant", {m: (auc[m].auc - auc[REFERENCE].auc).to_numpy() for m in trained if m != REFERENCE}, "")

    # --- caught / missed ------------------------------------------------------------------------------
    caught = {}
    for (m, run), g in events.groupby(["detector", "run"]):
        g = g.sort_values(["participant", "stand_up"])
        caught[m, run] = (g.tau.notna().to_numpy(), g.participant.to_numpy())

    def outcome(m: str, run: str) -> np.ndarray:
        return caught[m, run if (m, run) in caught else next(r for mm, r in caught if mm == m)][0]
    who = caught[REFERENCE, MAIN_RUN][1]
    x = np.column_stack([outcome(m, MAIN_RUN) for m in present])
    q = cochran_q(x)
    say(f"\n## Caught / missed per stand-up ({len(x)} stand-ups, {MAIN_RUN})\n")
    say(f"Cochran's Q (pooled, stand-ups treated as independent): Q({q['df']}) = {q['q']:.1f}, p = {q['p']:.2g}\n")
    runs = runs_of(REFERENCE)
    exact = {m: mcnemar_exact(outcome(m, MAIN_RUN), outcome(REFERENCE, MAIN_RUN)) for m in others}
    clustered = {m: mcnemar_clustered(outcome(m, MAIN_RUN), outcome(REFERENCE, MAIN_RUN), who) for m in others}
    ref_share = run_mean(metrics[REFERENCE], "sensitivity")
    flip = {m: sign_flip((run_mean(metrics[m], "sensitivity") - ref_share).to_numpy()) for m in others}
    h = [holm({m: v for m, v in d.items()}) for d in ({m: exact[m]["p"] for m in others}, {m: clustered[m]["p"] for m in others}, flip)]
    say("| Model | caught only by model / only by HGB | clustered McNemar p (Holm) | sign-flip on participants, all runs p (Holm) | exact McNemar p, pooled (Holm) | exact McNemar p over runs |")
    say("|---|---|---|---|---|---|")
    for m in others:
        ps = [mcnemar_exact(outcome(m, r), outcome(REFERENCE, r))["p"] for r in runs]
        say(f"| {m} | {exact[m]['only_a']} / {exact[m]['only_b']} | {clustered[m]['p']:.4f} ({h[1][m]:.4f}) | {flip[m]:.4f} ({h[2][m]:.4f}) | "
            f"{exact[m]['p']:.2g} ({h[0][m]:.2g}) | {min(ps):.2g}–{max(ps):.2g} |")
        tests.append({"test": "McNemar", "metric": "caught", "model": m, **{k: exact[m][k] for k in ("only_a", "only_b")},
                      "p_exact": exact[m]["p"], "holm_exact": h[0][m], "p_clustered": clustered[m]["p"], "holm_clustered": h[1][m],
                      "p_signflip": flip[m], "holm_signflip": h[2][m], "p_runs_min": min(ps), "p_runs_max": max(ps)})

    # --- DeLong ---------------------------------------------------------------------------------------
    say(f"\n## Pooled window AUC vs HGB ({len(y)} windows, {y.mean():.1%} positive, {MAIN_RUN})\n")
    rest = [m for m in trained if m != REFERENCE]
    pooled = {m: delong(scores[m, MAIN_RUN], scores[REFERENCE, MAIN_RUN], y) for m in rest}
    robust = {m: delong_clustered(scores[m, MAIN_RUN], scores[REFERENCE, MAIN_RUN], y, cluster) for m in rest}
    hp, hr = holm({m: pooled[m]["p"] for m in rest}), holm({m: robust[m]["p"] for m in rest})
    say("| Model | AUC model / HGB | difference [95 % CI, clustered] | clustered DeLong p (Holm) | DeLong p, pooled (Holm) | clustered p over runs |")
    say("|---|---|---|---|---|---|")
    for m in rest:
        ps = [delong_clustered(scores[m, r if (m, r) in scores else MAIN_RUN], scores[REFERENCE, r], y, cluster)["p"] for r in runs]
        r = robust[m]
        say(f"| {m} | {r['auc_a']:.4f} / {r['auc_b']:.4f} | {r['difference']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.4f} ({hr[m]:.4f}) | "
            f"{pooled[m]['p']:.2g} ({hp[m]:.2g}) | {min(ps):.2g}–{max(ps):.2g} |")
        tests.append({"test": "DeLong", "metric": "auc", "model": m, "estimate": r["difference"], "lo": r["lo"], "hi": r["hi"],
                      "p_clustered": r["p"], "holm_clustered": hr[m], "p_exact": pooled[m]["p"], "holm_exact": hp[m],
                      "p_runs_min": min(ps), "p_runs_max": max(ps)})

    # --- families -------------------------------------------------------------------------------------
    ml, dl = [m for m in trained if m in ML], [m for m in trained if m in DL]
    say(f"\n## Machine learning ({len(ml)} models) vs deep learning ({len(dl)} models): per-participant family means, DL − ML\n")
    for metric, label in METRICS.items():
        d = (np.mean([run_mean(metrics[m], metric).to_numpy() for m in dl], axis=0)
             - np.mean([run_mean(metrics[m], metric).to_numpy() for m in ml], axis=0))
        c, p = hodges_lehmann(d, 0.95, "two-sided"), signed_rank_test(d, "two-sided")["p"]
        say(f"- {label}: {c['estimate']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}], p = {p:.4f}")
        tests.append({"test": "family signed-rank", "metric": metric, "estimate": c["estimate"], "lo": c["lo"], "hi": c["hi"], "p": p})
    say("\nThe family comparison describes these models only; it is not evidence about all ML or all DL methods.")

    pd.DataFrame(tests).to_csv(RES / "model_tests.csv", index=False)
    (RES / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
