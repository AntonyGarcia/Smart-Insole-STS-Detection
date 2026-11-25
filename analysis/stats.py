"""Hypothesis tests, confidence intervals and corrections of the study protocol, §9, from results/*.csv
(written by analysis/score_detectors.py).

Unit = participant (n = 11). Multi-run detectors: each participant's value is computed per run, then averaged
over runs (metrics are averaged, never model outputs). Tests: exact signed-rank (stats_tools), Friedman (H4).
Effects: Hodges-Lehmann with exact CIs at the nominal and the multiplicity-adjusted level. Descriptive metrics:
participant-level BCa bootstrap (10,000). Supporting: linear mixed model (H1, H5) and GEE logistic (H2).

Writes results/stats_report.md, results/descriptive.csv, results/hypotheses.csv, results/per_run.csv.

  python analysis/stats.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from events import PARTICIPANTS
from stats_tools import bca, hodges_lehmann, holm, signed_rank_test

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALPHA = 0.05
DELTA_S, DELTA_F = 0.05, 1.0          # H3 margins (the study protocol, §4)
CONFIRMATORY_M, H4_M, H5_M = 3, 4, 2
AFTER_S = 3.0                         # end of the association window, for the censored sensitivity analysis

lines: list[str] = []
hyp_rows: list[dict] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


# ----------------------------------------------------------------------------- per-participant values

def participant_metrics(parts: pd.DataFrame, events: pd.DataFrame, detector: str) -> pd.DataFrame:
    """One row per (run, participant): sensitivity, FA/h, F1, precision, median onset latency, median lead time,
    share of detected stand-ups alerted before upright."""
    p = parts[parts.detector == detector].copy()
    e = events[events.detector == detector].copy()
    e["onset"] = e.tau - e.start_s
    e["lead"] = e.end_s - e.tau
    agg = e.dropna(subset=["tau"]).groupby(["run", "participant"]).agg(
        onset=("onset", "median"), lead=("lead", "median"), before_upright=("lead", lambda v: np.mean(v > 0)))
    p = p.merge(agg, on=["run", "participant"], how="left")
    p["sensitivity"] = p.hits / p.events
    p["fa_per_hour"] = p.false / p.minutes * 60
    p["precision"] = p.hits / (p.hits + p.false).replace(0, np.nan)
    p["f1"] = 2 * p.hits / (p.events + p.hits + p.false)
    return p


def run_mean(m: pd.DataFrame, column: str) -> pd.Series:
    return m.groupby("participant")[column].mean().reindex(PARTICIPANTS)


def paired_timing(events: pd.DataFrame, a: str, b: str, censored: bool = False) -> pd.DataFrame:
    """Per (run of a, participant, stand-up): tau_a - tau_b for stand-ups detected by both (or censored)."""
    ea = events[events.detector == a]
    eb = events[events.detector == b]
    b_runs = eb.run.unique()
    rows = []
    for run in ea.run.unique():
        run_b = run if run in b_runs else b_runs[0]          # B1 has one fixed run
        x = ea[ea.run == run].merge(eb[eb.run == run_b], on=["participant", "stand_up"], suffixes=("_a", "_b"))
        if censored:
            x = x.assign(tau_a=x.tau_a.fillna(x.end_s_a + AFTER_S), tau_b=x.tau_b.fillna(x.end_s_a + AFTER_S))
        x = x.dropna(subset=["tau_a", "tau_b"])
        rows.append(pd.DataFrame({"run": run, "participant": x.participant, "d": x.tau_a - x.tau_b}))
    return pd.concat(rows)


# ----------------------------------------------------------------------------- reporting helpers

def fmt_ci(ci: dict, unit: str = "", pct: bool = False) -> str:
    f = (lambda v: f"{100 * v:+.1f}") if pct else (lambda v: f"{v:+.2f}")
    lo = "−∞" if not np.isfinite(ci["lo"]) else f(ci["lo"])
    hi = "+∞" if not np.isfinite(ci["hi"]) else f(ci["hi"])
    return f"{f(ci['estimate'])}{unit} [{lo}, {hi}] (achieved {100 * ci['achieved_level']:.1f} %)"


def record(hypothesis: str, test: str, x: np.ndarray, alternative: str, adjusted_level: float, side: str,
           unit: str = " s", pct: bool = False, note: str = "") -> dict:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        say(f"  ⚠ {int(np.sum(~np.isfinite(x)))} participant(s) without a value dropped from {hypothesis}")
        x = x[np.isfinite(x)]
    t = signed_rank_test(x, alternative)
    ci95 = hodges_lehmann(x, 0.95 if side == "two-sided" else 0.975, side)
    ci_adj = hodges_lehmann(x, adjusted_level, side)
    row = {"hypothesis": hypothesis, "test": test, "alternative": alternative, "n": int(np.sum(np.isfinite(x))),
           "p": t["p"], "T": t["T"], "rank_biserial": t["rank_biserial"], "hl_estimate": ci95["estimate"],
           "ci_lo": ci95["lo"], "ci_hi": ci95["hi"], "ci_level": ci95["achieved_level"],
           "adj_ci_lo": ci_adj["lo"], "adj_ci_hi": ci_adj["hi"], "adj_ci_level": ci_adj["achieved_level"], "note": note}
    say(f"  {test}: p = {t['p']:.4f} ({alternative}), rank-biserial r = {t['rank_biserial']:+.2f}")
    say(f"    HL estimate and {'95 %' if side == 'two-sided' else 'one-sided 97.5 %'} CI: {fmt_ci(ci95, unit, pct)}")
    say(f"    multiplicity-adjusted CI: {fmt_ci(ci_adj, unit, pct)}")
    hyp_rows.append(row)
    return row


def wilson(successes: float, total: float, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def mixed_model(d: pd.DataFrame, label: str) -> None:
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # crossed random intercepts (participant, run): one all-encompassing group + two variance components
            d = d.reset_index(drop=True).assign(all=1)
            fit = smf.mixedlm("d ~ 1", d, groups="all", re_formula="0",
                              vc_formula={"participant": "0 + C(participant)", "run": "0 + C(run)"}).fit(reml=True)
            lo, hi = fit.conf_int().loc["Intercept"]
            say(f"  supporting LMM ({label}; random intercepts participant + run, {len(d)} paired stand-ups): "
                f"β0 = {fit.params['Intercept']:+.3f} s [{lo:+.3f}, {hi:+.3f}], Wald p = {fit.pvalues['Intercept']:.2g}; "
                f"converged: {fit.converged}; variance components {dict(zip(fit.model.exog_vc.names, np.round(fit.vcomp, 4)))}")
        except Exception as exc:          # singular fits etc. are reported, not hidden
            say(f"  supporting LMM ({label}) failed: {exc}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    parts = pd.read_csv(RES / "participants.csv")
    events = pd.read_csv(RES / "events.csv")
    n = len(PARTICIPANTS)
    say("# Statistical results (the study protocol, §9)\n")
    say(f"Participants: {n}. Runs per multi-run detector are averaged per participant (metrics, not outputs).\n")

    # --- Descriptive table with BCa CIs -----------------------------------------------------------
    say("## Descriptive performance (95 % participant-bootstrap BCa CIs; run range = min–max of the pooled value over runs)\n")
    desc, wilson_lines = [], []
    for detector in sorted(parts.detector.unique(), key=lambda d: (not d.startswith("LSTM"), d)):
        m = participant_metrics(parts, events, detector)
        per = m.groupby("participant")[["events", "hits", "false", "minutes", "onset", "lead", "before_upright"]].mean() \
               .reindex(PARTICIPANTS)
        arrays = {c: per[c].to_numpy() for c in per.columns}

        def pooled(fn):
            return lambda idx: fn({c: v[idx] for c, v in arrays.items()})
        stats = {
            "sensitivity": pooled(lambda a: a["hits"].sum() / a["events"].sum()),
            "precision": pooled(lambda a: a["hits"].sum() / (a["hits"].sum() + a["false"].sum())),
            "f1": pooled(lambda a: 2 * a["hits"].sum() / (a["events"].sum() + a["hits"].sum() + a["false"].sum())),
            "fa_per_hour": pooled(lambda a: a["false"].sum() / a["minutes"].sum() * 60),
            "onset_latency_s": pooled(lambda a: np.nanmedian(a["onset"])),
            "lead_time_s": pooled(lambda a: np.nanmedian(a["lead"])),
            "share_before_upright": pooled(lambda a: np.nanmedian(a["before_upright"])),
        }
        row = {"detector": detector, "runs": m.run.nunique()}
        for name, fn in stats.items():
            est, lo, hi = bca(fn, n)
            row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        if m.run.nunique() > 1:
            by_run = m.groupby("run").agg(h=("hits", "sum"), e=("events", "sum"), f=("false", "sum"), mi=("minutes", "sum"))
            row["sensitivity_run_range"] = f"{(by_run.h / by_run.e).min():.3f}-{(by_run.h / by_run.e).max():.3f}"
            row["fa_per_hour_run_range"] = f"{(by_run.f / by_run.mi * 60).min():.2f}-{(by_run.f / by_run.mi * 60).max():.2f}"
            row["theta_range"] = f"{m.theta.min():.2f}-{m.theta.max():.2f}"
        # Supplement (§9): Wilson intervals on the pooled counts (mean over runs); stand-ups treated as independent.
        by_run = m.groupby("run")[["hits", "events", "false"]].sum()
        hits, events_n, false_n = by_run.hits.mean(), by_run.events.mean(), by_run.false.mean()
        row["sensitivity_wilson_lo"], row["sensitivity_wilson_hi"] = wilson(hits, events_n)
        row["precision_wilson_lo"], row["precision_wilson_hi"] = wilson(hits, hits + false_n)
        wilson_lines.append(f"- {detector}: sensitivity {100 * hits / events_n:.1f} % [{100 * row['sensitivity_wilson_lo']:.1f}, "
                            f"{100 * row['sensitivity_wilson_hi']:.1f}], precision {100 * hits / (hits + false_n):.1f} % "
                            f"[{100 * row['precision_wilson_lo']:.1f}, {100 * row['precision_wilson_hi']:.1f}]")
        desc.append(row)
        say(f"- **{detector}** ({row['runs']} run{'s' if row['runs'] > 1 else ''}): "
            f"sensitivity {100 * row['sensitivity']:.1f} % [{100 * row['sensitivity_lo']:.1f}, {100 * row['sensitivity_hi']:.1f}], "
            f"false alarms {row['fa_per_hour']:.2f}/h [{row['fa_per_hour_lo']:.2f}, {row['fa_per_hour_hi']:.2f}], "
            f"F1 {row['f1']:.3f} [{row['f1_lo']:.3f}, {row['f1_hi']:.3f}], "
            f"onset latency {row['onset_latency_s']:+.2f} s [{row['onset_latency_s_lo']:+.2f}, {row['onset_latency_s_hi']:+.2f}], "
            f"lead time {row['lead_time_s']:+.2f} s [{row['lead_time_s_lo']:+.2f}, {row['lead_time_s_hi']:+.2f}], "
            f"alerted before upright {100 * row['share_before_upright']:.0f} % "
            f"[{100 * row['share_before_upright_lo']:.0f}, {100 * row['share_before_upright_hi']:.0f}]"
            + (f"; run ranges: sensitivity {row['sensitivity_run_range']}, FA/h {row['fa_per_hour_run_range']}, "
               f"θ {row['theta_range']}" if "theta_range" in row else ""))
    pd.DataFrame(desc).to_csv(RES / "descriptive.csv", index=False)
    say("\n### Supplement: 95 % Wilson intervals on pooled counts (mean over runs). They treat stand-ups as independent, "
        "so they are narrower than the participant-bootstrap intervals above\n")
    for text in wilson_lines:
        say(text)

    # Variation across training runs (§9): each run pooled over participants, then median [min–max] over runs.
    say("\n### Multi-run detectors: median [min–max] over the training runs (a range, not a CI)\n")
    run_rows = []
    for detector in sorted(parts.detector.unique()):
        m = participant_metrics(parts, events, detector)
        if m.run.nunique() < 2:
            continue
        g = m.groupby("run")
        counts = g[["hits", "events", "false", "minutes"]].sum()
        per_run_values = pd.DataFrame({
            "sensitivity_pct": 100 * counts.hits / counts.events, "fa_per_hour": 60 * counts.false / counts.minutes,
            "f1": 2 * counts.hits / (counts.events + counts.hits + counts.false),
            "onset_s": g.onset.median(), "lead_s": g.lead.median(), "before_upright_pct": 100 * g.before_upright.median()})
        run_rows.append({"detector": detector, **{f"{c}_{k}": getattr(per_run_values[c], k)()
                                                  for c in per_run_values for k in ("median", "min", "max")}})
        say(f"- {detector}: " + ", ".join(f"{c} {v.median():.3g} [{v.min():.3g}–{v.max():.3g}]"
                                          for c, v in per_run_values.items()))
    pd.DataFrame(run_rows).to_csv(RES / "run_variation.csv", index=False)

    lstm = participant_metrics(parts, events, "LSTM")
    b1 = participant_metrics(parts, events, "B1")
    adj_two = 1 - ALPHA / CONFIRMATORY_M
    adj_one = 1 - (ALPHA / 2) / CONFIRMATORY_M

    # --- H1 ------------------------------------------------------------------------------------------
    say("\n## Confirmatory hypotheses (Holm across H1–H3, α = 0.05)\n")
    say("### H1 — the LSTM alerts earlier than B1 (τ_LSTM − τ_B1; negative = LSTM earlier)")
    d = paired_timing(events, "LSTM", "B1")
    x1 = d.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
    h1 = record("H1", "exact signed-rank on per-participant median difference", x1, "two-sided", adj_two, "two-sided")
    mixed_model(d, "H1")
    dc = paired_timing(events, "LSTM", "B1", censored=True)
    xc = dc.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
    say(f"  sensitivity analysis, misses censored at e + {AFTER_S:g} s: "
        f"p = {signed_rank_test(xc)['p']:.4f}, HL {hodges_lehmann(xc)['estimate']:+.2f} s")

    # --- H2 ------------------------------------------------------------------------------------------
    say("\n### H2 — for most stand-ups the LSTM alerts before the person is fully upright (share − 0.5)")
    share = run_mean(lstm, "before_upright").to_numpy()
    say(f"  per-participant shares: {', '.join(f'{v:.2f}' for v in share)}")
    h2 = record("H2", "exact one-sided signed-rank, share − 0.5", share - 0.5, "greater", adj_one, "lower",
                unit=" (share − 0.5)", pct=False)
    try:
        import statsmodels.api as sm
        e = events[(events.detector == "LSTM")].dropna(subset=["tau"])
        y = (e.end_s - e.tau > 0).astype(float).to_numpy()
        gee = sm.GEE(y, np.ones((len(y), 1)), groups=e.participant.to_numpy(), family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable()).fit()
        b, se = gee.params[0], gee.bse[0]
        from scipy.stats import norm
        say(f"  supporting GEE logistic (participant clusters, exchangeable; {len(y)} detected stand-ups over runs): "
            f"P = {1 / (1 + np.exp(-b)):.3f} [{1 / (1 + np.exp(-(b - 1.96 * se))):.3f}, {1 / (1 + np.exp(-(b + 1.96 * se))):.3f}], "
            f"one-sided p = {1 - norm.cdf(b / se):.2g}")
    except Exception as exc:
        say(f"  supporting GEE failed: {exc}")

    # --- H3 ------------------------------------------------------------------------------------------
    say(f"\n### H3 — not worse than B1 (margins: sensitivity {100 * DELTA_S:.0f} percentage points, false alarms +{DELTA_F:g}/h)")
    sens_diff = (run_mean(lstm, "sensitivity") - run_mean(b1, "sensitivity")).to_numpy()
    fa_diff = (run_mean(lstm, "fa_per_hour") - run_mean(b1, "fa_per_hour")).to_numpy()
    say("  H3a sensitivity difference (LSTM − B1) + margin > 0:")
    h3a = record("H3a", "exact one-sided signed-rank on (Δsens + 0.05)", sens_diff + DELTA_S, "greater", adj_one, "lower",
                 unit=" (Δsens + margin)", note="CI is for Δsens + margin; non-inferior if lower bound > 0")
    say("  H3b false-alarm difference: margin − (FA/h LSTM − FA/h B1) > 0:")
    h3b = record("H3b", "exact one-sided signed-rank on (1/h − ΔFA/h)", DELTA_F - fa_diff, "greater", adj_one, "lower",
                 unit=" /h (margin − ΔFA)", note="CI is for margin − ΔFA/h; non-inferior if lower bound > 0")
    p3 = max(h3a["p"], h3b["p"])

    # --- Holm ----------------------------------------------------------------------------------------
    family = {"H1": h1["p"], "H2": min(1.0, 2 * h2["p"]), "H3": min(1.0, 2 * p3)}
    adjusted = holm(family)
    say("\n### Holm step-down across H1–H3 (one-sided p doubled to the two-sided scale; H3 uses max(H3a, H3b))")
    for name in ("H1", "H2", "H3"):
        say(f"  {name}: p = {family[name]:.4f}, Holm-adjusted p = {adjusted[name]:.4f} → "
            f"{'REJECT H0' if adjusted[name] < ALPHA else 'do not reject H0'}")
    for row in hyp_rows:
        key = row["hypothesis"][:2]
        row["holm_p"] = adjusted.get(key, np.nan)
    if adjusted["H3"] < ALPHA:
        say("  H3 holds → superiority tests (fixed sequence, two-sided, α = 0.05):")
        record("H3-sup-sens", "exact two-sided signed-rank, Δsensitivity", sens_diff, "two-sided", 0.95, "two-sided",
               unit=" (Δsens)")
        record("H3-sup-FA", "exact two-sided signed-rank, ΔFA/h", fa_diff, "two-sided", 0.95, "two-sided", unit=" /h")

    # --- Per-run robustness ---------------------------------------------------------------------------
    say("\n### Robustness: confirmatory tests on each training run separately (unadjusted p)")
    per_run = []
    b1_sens = run_mean(b1, "sensitivity")
    b1_fa = run_mean(b1, "fa_per_hour")
    for run in sorted(lstm.run.unique()):
        r = lstm[lstm.run == run].set_index("participant").reindex(PARTICIPANTS)
        xr = d[d.run == run].groupby("participant").d.median().reindex(PARTICIPANTS).to_numpy()
        res = {"run": run,
               "H1_p": signed_rank_test(xr)["p"], "H1_hl": hodges_lehmann(xr)["estimate"],
               "H2_p": signed_rank_test(r.before_upright.to_numpy() - 0.5, "greater")["p"],
               "H3a_p": signed_rank_test((r.sensitivity - b1_sens).to_numpy() + DELTA_S, "greater")["p"],
               "H3b_p": signed_rank_test(DELTA_F - (r.fa_per_hour - b1_fa).to_numpy(), "greater")["p"]}
        per_run.append(res)
        say(f"  {run}: H1 p = {res['H1_p']:.4f} (HL {res['H1_hl']:+.2f} s), H2 p = {res['H2_p']:.4f}, "
            f"H3a p = {res['H3a_p']:.4f}, H3b p = {res['H3b_p']:.4f}")
    pr = pd.DataFrame(per_run)
    pr.to_csv(RES / "per_run.csv", index=False)
    say(f"  runs reaching the main decision at 0.05: H1 {int((pr.H1_p < ALPHA).sum())}/{len(pr)}, "
        f"H2 {int((pr.H2_p < ALPHA / 2).sum())}/{len(pr)}, "
        f"H3 {int(((pr.H3a_p < ALPHA / 2) & (pr.H3b_p < ALPHA / 2)).sum())}/{len(pr)}")

    # --- H4 ------------------------------------------------------------------------------------------
    say("\n## Secondary hypotheses\n")
    say("### H4 — sensor configuration (thresholds from the other participants, same rule)")
    configs = ["LSTM_approx", "LSTM_pressure", "LSTM_imu", "LSTM_left", "LSTM_right"]
    if all(c in parts.detector.unique() for c in configs):
        for metric, unit in (("f1", ""), ("onset", " s")):
            table = np.column_stack([run_mean(participant_metrics(parts, events, c), metric).to_numpy() for c in configs])
            stat, p = friedmanchisquare(*table.T)
            k = len(configs)
            w_fn = lambda idx, t=table: friedmanchisquare(*t[idx].T).statistic / (len(idx) * (k - 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w, wlo, whi = bca(w_fn, n)
            say(f"  {metric}: medians " + ", ".join(f"{c.removeprefix('LSTM_')} {np.median(table[:, i]):.3f}"
                                                    for i, c in enumerate(configs)))
            say(f"  Friedman χ²({k - 1}) = {stat:.2f}, p = {p:.4f}; Kendall's W = {w:.2f} [{wlo:.2f}, {whi:.2f}]")
            hyp_rows.append({"hypothesis": f"H4-{metric}", "test": "Friedman", "p": p, "T": stat, "hl_estimate": w,
                             "ci_lo": wlo, "ci_hi": whi, "note": "Kendall's W with BCa CI"})
            if p < ALPHA:
                post = {}
                for i, c in enumerate(configs[1:], start=1):
                    say(f"  full vs {c.removeprefix('LSTM_')} ({metric}, full − other):")
                    post[c] = record(f"H4-{metric}-{c}", "exact two-sided signed-rank", table[:, 0] - table[:, i],
                                     "two-sided", 1 - ALPHA / H4_M, "two-sided", unit=unit)["p"]
                for c, ap in holm(post).items():
                    say(f"    Holm within H4 ({metric}), full vs {c.removeprefix('LSTM_')}: adjusted p = {ap:.4f}")
    else:
        say("  ablation results incomplete; skipped")

    # --- H5 ------------------------------------------------------------------------------------------
    say("\n### H5 — LSTM vs B2 (same data, folds and threshold rule)")
    if "B2" in parts.detector.unique():
        d5 = paired_timing(events, "LSTM_approx", "B2")
        x5 = d5.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
        r5a = record("H5-timing", "exact two-sided signed-rank, τ_LSTM − τ_B2", x5, "two-sided", 1 - ALPHA / H5_M, "two-sided")
        mixed_model(d5, "H5")
        f1_diff = (run_mean(participant_metrics(parts, events, "LSTM_approx"), "f1")
                   - run_mean(participant_metrics(parts, events, "B2"), "f1")).to_numpy()
        r5b = record("H5-F1", "exact two-sided signed-rank, F1_LSTM − F1_B2", f1_diff, "two-sided", 1 - ALPHA / H5_M,
                     "two-sided", unit="")
        for name, ap in holm({"timing": r5a["p"], "F1": r5b["p"]}).items():
            say(f"  Holm within H5, {name}: adjusted p = {ap:.4f}")
    else:
        say("  B2 results missing; skipped")

    # --- Exploratory (not part of the confirmatory family; unadjusted 95 % CIs) -----------------------
    detectors = set(parts.detector.unique())
    say("\n## Exploratory analyses (added after the confirmatory results; unadjusted, hypothesis-generating)\n")
    if "LSTM_inside_approx" in detectors:
        say("### E1 — LSTM trained on B2's 'inside' target instead of the Movement ramp (same model, loss and threshold rule)")
        for other in ("LSTM_approx", "B2"):
            de = paired_timing(events, "LSTM_inside_approx", other)
            xe = de.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
            say(f"  timing, τ_inside − τ_{other} (negative = inside-target LSTM earlier):")
            record(f"E1-timing-vs-{other}", "exact two-sided signed-rank", xe, "two-sided", 0.95, "two-sided",
                   note="exploratory")
            f1e = (run_mean(participant_metrics(parts, events, "LSTM_inside_approx"), "f1")
                   - run_mean(participant_metrics(parts, events, other), "f1")).to_numpy()
            say(f"  F1, inside − {other}:")
            record(f"E1-F1-vs-{other}", "exact two-sided signed-rank", f1e, "two-sided", 0.95, "two-sided", unit="",
                   note="exploratory")
    else:
        say("  E1 (inside target): results missing; skipped")
    windows = ["LSTM_W1s_approx", "LSTM_approx", "LSTM_W3s_approx", "LSTM_W4s_approx"]
    if all(c in detectors for c in windows):
        say("\n### E2 — window length 1 / 2 / 3 / 4 s (threshold from the other participants)")
        for metric, unit in (("f1", ""), ("onset", " s"), ("fa_per_hour", " /h")):
            table = np.column_stack([run_mean(participant_metrics(parts, events, c), metric).to_numpy() for c in windows])
            stat, p = friedmanchisquare(*table.T)
            say(f"  {metric}: medians " + ", ".join(f"{w} s {np.median(table[:, i]):.3f}" for i, w in enumerate((1, 2, 3, 4)))
                + f"; Friedman χ²(3) = {stat:.2f}, p = {p:.4f}")
            hyp_rows.append({"hypothesis": f"E2-{metric}", "test": "Friedman", "p": p, "T": stat, "note": "exploratory"})
    else:
        say("  E2 (window length): results missing; skipped")
    ext = ["LSTM_inside_ext", "B2_ext", "LSTM_approx_ext", "LSTM_W4s_ext"]
    if all(c in detectors for c in ext):
        say("\n### E3 — threshold grid extended to 0.99 (prespecified grid ends at 0.95; same selection rule)")
        for c in ext:
            m = participant_metrics(parts, events, c)
            say(f"  {c}: θ chosen " + ", ".join(f"{t:.2f} ×{n}" for t, n in m.theta.value_counts().sort_index().items()))
        for a, b in (("LSTM_inside_ext", "B2_ext"), ("LSTM_inside_ext", "LSTM_approx_ext"), ("LSTM_W4s_ext", "B2_ext")):
            de = paired_timing(events, a, b)
            xe = de.groupby(["participant", "run"]).d.median().groupby("participant").mean().reindex(PARTICIPANTS).to_numpy()
            say(f"  timing, τ_{a} − τ_{b} (negative = {a} earlier):")
            record(f"E3-timing-{a}-vs-{b}", "exact two-sided signed-rank", xe, "two-sided", 0.95, "two-sided",
                   note="exploratory")
            for metric, unit in (("f1", ""), ("fa_per_hour", " /h")):
                diff = (run_mean(participant_metrics(parts, events, a), metric)
                        - run_mean(participant_metrics(parts, events, b), metric)).to_numpy()
                say(f"  {metric}, {a} − {b}:")
                record(f"E3-{metric}-{a}-vs-{b}", "exact two-sided signed-rank", diff, "two-sided", 0.95, "two-sided",
                       unit=unit, note="exploratory")
    else:
        say("  E3 (extended grid): results missing; skipped")

    pd.DataFrame(hyp_rows).to_csv(RES / "hypotheses.csv", index=False)
    (RES / "stats_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'stats_report.md'}")


if __name__ == "__main__":
    main()
