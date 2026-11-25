"""Operating-point analysis (the study protocol, §8.4, secondary), on the final LSTM leave-one-out outputs (loo_final, 6 runs).

1. Sweep: threshold θ ∈ {0.30, ..., 0.95} × minimum time above θ ∈ {0, 0.05, 0.1, 0.2, 0.3} s, lock-out 5 s.
   An alert needs the output above θ for the minimum time without interruption; it is raised at the end of the
   0.5 s block that contains the confirming sample (the same live timing as the primary detector), and the
   lock-out counts from the confirming sample. Pooled hits, false alarms, F1 and median onset latency per run.
2. Anatomy of the false alarms of the primary detector (LSTM, strict nested θ): peak output and time above θ
   of each false burst, i.e. what rule would have been needed to suppress it.
3. Cross-validated rule choice: for each held-out participant, the rule is chosen on the other 10 participants'
   out-of-fold outputs and applied to the held-out one; summed over participants.
     F1 rule        (protocol §8.4)  maximise pooled F1; ties -> shorter minimum time, then lower θ
     §6 rule        fewest false alarms with ≥ 95 % caught; ties -> lower θ, then shorter minimum time
     θ only         the §6 rule with minimum time 0 (= LSTM_approx), the reference for what the minimum time adds
   The best rule chosen on all 11 participants is reported only as an optimistic upper bound.

Writes results/operating_point_sweep.csv, operating_point_cv.csv, operating_point_false_alarms.csv and
results/operating_point.md.

  python analysis/operating_point.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from events import LOCKOUT_S, PARTICIPANTS, RATE, THRESHOLDS, labels, lstm_alert_times, minutes, rising_crossings, score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = ROOT / "lstm" / "runs" / "loo_final"
SEEDS = ["seed0", "seed1", "seed2", "seed345", "seed4", "seed5"]
MIN_TIMES = [0.0, 0.05, 0.1, 0.2, 0.3]
WINDOW, STEP = 160, 40

STAND_UPS = {p: labels(p) for p in PARTICIPANTS}
MINUTES = {p: minutes(p) for p in PARTICIPANTS}
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def output(participant: str, run: str) -> np.ndarray:
    path = next((RUNS / f"test_{participant}" / run).glob("*_predictions.csv"))
    return pd.read_csv(path, usecols=["lstm"])["lstm"].to_numpy()


def alerts_min_time(signal: np.ndarray, theta: float, min_time: float) -> np.ndarray:
    m = int(round(min_time * RATE))
    above = signal > theta
    if m == 0:
        confirmed = above
    else:     # above θ for the last m + 1 samples without interruption
        count = np.convolve(above.astype(np.int32), np.ones(m + 1, dtype=np.int32))[:len(above)]
        confirmed = count == m + 1
    fired = rising_crossings(confirmed.astype(float), 0.5, LOCKOUT_S, np.arange(len(signal)) / RATE)
    available = np.maximum(WINDOW, np.ceil((fired + 1) / STEP) * STEP)
    return np.minimum(available, len(signal)) / RATE


def f1(h: int, e: int, f: int) -> float:
    return 2 * h / (e + h + f) if e + h + f else 0.0


def main() -> None:
    rules = [(t, d) for d in MIN_TIMES for t in THRESHOLDS]
    sweep_rows, cv_rows = [], []
    say("# Operating-point analysis (the study protocol, §8.4, secondary; LSTM leave-one-out outputs, 6 runs)\n")
    for run in SEEDS:
        outs = {p: output(p, run) for p in PARTICIPANTS}
        # per participant and rule: hits, events, false, onset latencies
        cell = {}
        for p in PARTICIPANTS:
            for t, d in rules:
                s = score(alerts_min_time(outs[p], t, d), STAND_UPS[p])
                cell[p, t, d] = (s.hits, s.events, s.false, s.onset_latency[np.isfinite(s.onset_latency)])
        for t, d in rules:
            h = sum(cell[p, t, d][0] for p in PARTICIPANTS)
            e = sum(cell[p, t, d][1] for p in PARTICIPANTS)
            f = sum(cell[p, t, d][2] for p in PARTICIPANTS)
            onset = np.concatenate([cell[p, t, d][3] for p in PARTICIPANTS])
            sweep_rows.append({"run": run, "theta": t, "min_time_s": d, "hits": h, "events": e, "false": f,
                               "sensitivity": h / e, "fa_per_hour": f / sum(MINUTES.values()) * 60, "f1": f1(h, e, f),
                               "median_onset_s": float(np.median(onset)) if onset.size else np.nan})

        def pooled(group: list[str], rule) -> tuple[int, int, int]:
            return tuple(sum(cell[(p, *rule)][k] for p in group) for k in range(3))

        choosers = {
            "F1 rule": (rules, lambda r, g: (-f1(*pooled(g, r)), r[1], r[0])),
            "§6 rule": (rules, lambda r, g: (0 if pooled(g, r)[0] / pooled(g, r)[1] >= 0.95 else 1, pooled(g, r)[2], r[0], r[1])),
            "θ only (§6 rule, min time 0)": ([r for r in rules if r[1] == 0],
                                             lambda r, g: (0 if pooled(g, r)[0] / pooled(g, r)[1] >= 0.95 else 1,
                                                           pooled(g, r)[2], r[0])),
        }
        for name, (candidates, key) in choosers.items():
            h = e = f = 0
            onset, chosen = [], []
            for p in PARTICIPANTS:
                others = [q for q in PARTICIPANTS if q != p]
                rule = min(candidates, key=lambda r: key(r, others))
                chosen.append(rule)
                hp, ep, fp, op = cell[(p, *rule)]
                h, e, f = h + hp, e + ep, f + fp
                onset.append(op)
            onset = np.concatenate(onset)
            best_all = min(candidates, key=lambda r: key(r, PARTICIPANTS))
            hb, eb, fb = pooled(PARTICIPANTS, best_all)
            cv_rows.append({"run": run, "chooser": name, "hits": h, "events": e, "false": f, "sensitivity": h / e,
                            "fa_per_hour": f / sum(MINUTES.values()) * 60, "f1": f1(h, e, f),
                            "median_onset_s": float(np.median(onset)),
                            "rules_chosen": ";".join(f"{t:.2f}/{d:g}" for t, d in chosen),
                            "in_sample_rule": f"{best_all[0]:.2f}/{best_all[1]:g}", "in_sample_hits": hb,
                            "in_sample_false": fb, "in_sample_f1": f1(hb, eb, fb)})
        print(f"{run} done", flush=True)

    sweep = pd.DataFrame(sweep_rows)
    cv = pd.DataFrame(cv_rows)
    sweep.to_csv(RES / "operating_point_sweep.csv", index=False)
    cv.to_csv(RES / "operating_point_cv.csv", index=False)
    hours = sum(MINUTES.values()) / 60

    say("## Cross-validated rule choice (rule chosen on the other 10 participants; median [min–max] over 6 runs)\n")
    for name, g in cv.groupby("chooser", sort=False):
        def rng(c, fmt):
            return f"{fmt(g[c].median())} [{fmt(g[c].min())}–{fmt(g[c].max())}]"
        say(f"- **{name}**: caught {rng('sensitivity', lambda v: f'{100 * v:.1f} %')}, "
            f"false alarms/h {rng('fa_per_hour', lambda v: f'{v:.2f}')}, F1 {rng('f1', lambda v: f'{v:.3f}')}, "
            f"median onset {rng('median_onset_s', lambda v: f'{v:+.2f} s')}")
        say(f"  in-sample best rule (optimistic upper bound), per run: "
            + ", ".join(f"{r.in_sample_rule} â†’ {r.in_sample_hits}/{r.events} caught, {r.in_sample_false} false"
                        for r in g.itertuples()))

    say("\n## Sweep: pooled over participants, median over 6 runs (full table in operating_point_sweep.csv)\n")
    med = sweep.groupby(["min_time_s", "theta"])[["sensitivity", "fa_per_hour", "f1", "median_onset_s"]].median().reset_index()
    for d, g in med.groupby("min_time_s"):
        best = g.loc[g.f1.idxmax()]
        say(f"- min time {d:g} s: best median F1 {best.f1:.3f} at θ {best.theta:.2f} "
            f"({100 * best.sensitivity:.1f} % caught, {best.fa_per_hour:.2f} false/h, onset {best.median_onset_s:+.2f} s)")

    # --- anatomy of the primary detector's false alarms -------------------------------------------------
    alerts = pd.read_csv(RES / "alerts.csv")
    parts = pd.read_csv(RES / "participants.csv")
    errors = pd.read_csv(RES / "error_list.csv") if (RES / "error_list.csv").exists() else None
    false = alerts[(alerts.detector == "LSTM") & (alerts.kind == "false")]
    rows = []
    for (run, p), group in false.groupby(["run", "participant"]):
        theta = float(parts[(parts.detector == "LSTM") & (parts.run == run) & (parts.participant == p)].theta.iloc[0])
        signal = output(p, run)
        crossings = rising_crossings(signal, theta, LOCKOUT_S, np.arange(len(signal)) / RATE)
        alert_at = lstm_alert_times(signal, theta)
        for t in group.alert_s:
            i = crossings[np.argmin(np.abs(alert_at - t))]
            above = signal[i:] > theta
            length = int(np.argmin(above)) if not above.all() else len(above)
            category = ""
            if errors is not None:
                match = errors[(errors.detector == "LSTM") & (errors.run == run) & (errors.participant == p)
                               & (errors.type == "false_alarm") & (np.abs(errors.t - t) < 1e-6)]
                category = match.category.iloc[0] if len(match) else ""
            rows.append({"run": run, "participant": p, "alert_s": t, "theta": theta,
                         "peak": float(signal[i:i + length].max()), "time_above_s": length / RATE, "category": category})
    fa = pd.DataFrame(rows)
    fa.to_csv(RES / "operating_point_false_alarms.csv", index=False)
    say(f"\n## False alarms of the primary detector (LSTM, strict nested θ): {len(fa)} over 6 runs "
        f"({len(fa) / 6:.1f} per run, {len(fa) / 6 / hours:.2f}/h)\n")
    if len(fa):
        say(f"- time above θ: median {fa.time_above_s.median():.2f} s [IQR {fa.time_above_s.quantile(.25):.2f}–"
            f"{fa.time_above_s.quantile(.75):.2f}]; peak output: median {fa.peak.median():.2f}")
        for d in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            say(f"- a minimum time of {d:g} s would have suppressed {int((fa.time_above_s < d).sum())} "
                f"({100 * (fa.time_above_s < d).mean():.0f} %) of them (before any effect on hits)")
        if fa.category.astype(bool).any():
            say("- by category: " + ", ".join(f"{c} {n}" for c, n in fa.category.value_counts().items())
                + "; median time above θ by category: "
                + ", ".join(f"{c} {v:.2f} s" for c, v in fa.groupby("category").time_above_s.median().items()))
    (RES / "operating_point.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'operating_point.md'} and the CSVs")


if __name__ == "__main__":
    main()
