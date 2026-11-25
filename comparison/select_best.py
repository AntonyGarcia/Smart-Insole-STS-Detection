"""Which detector is best, and how stable is that answer? (the study protocol, §12, 2026-11-22; exploratory.)

Candidates: the 14 detectors of §9b. All numbers are pooled over participants and averaged over runs.
  Rule A  among detectors catching >= 95 %: fewest false alarms/h, ties -> earliest onset; none -> highest caught
  Rule B  highest F1 (ties -> fewest false alarms/h)
(a) leave-one-participant-out selection, (b) participant bootstrap of the winner and of the Pareto front.

Reads results/models_participants.csv, results/models_events.csv. Writes results/select_best.md, results/select_best.csv.

  python comparison/select_best.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "models"              # "tuned": outer results of §9c
NAME = "select_best" if PREFIX == "models" else f"select_best_{PREFIX}"
FLOOR, RESAMPLES = 0.95, 10_000
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def load():
    d = pd.read_csv(RES / f"{PREFIX}_participants.csv")
    e = pd.read_csv(RES / f"{PREFIX}_events.csv")
    names, people = list(d.detector.unique()), sorted(d.participant.unique())
    e["latency"] = e.tau - e.start_s
    onset = e.groupby(["detector", "run", "participant"]).latency.median().groupby(["detector", "participant"]).mean().unstack()
    onset = onset.loc[names, people].to_numpy()
    runs = {}                                                       # detector -> (runs, people, [events, hits, false, minutes])
    for n in names:
        g = d[d.detector == n].set_index(["run", "participant"])
        keys = list(dict.fromkeys(g.index.get_level_values(0)))
        runs[n] = np.array([[g.loc[(r, p), ["events", "hits", "false", "minutes"]].to_numpy(float) for p in people] for r in keys])
    return names, people, runs, onset


def metrics(runs, onset, names, w):
    """Arrays over detectors for the participants weighted by w (resample counts)."""
    caught, fa, f1 = [], [], []
    for n in names:
        s = np.einsum("rpk,p->rk", runs[n], w)
        events, hits, false, minutes = s.T
        caught.append((hits / events).mean())
        fa.append((60 * false / minutes).mean())
        f1.append((2 * hits / (events + hits + false)).mean())
    index = np.repeat(np.arange(len(w)), w.astype(int))
    return np.array(caught), np.array(fa), np.array(f1), np.nanmedian(onset[:, index], axis=1)


def rule_a(caught, fa, f1, onset):
    ok = np.flatnonzero(caught >= FLOOR - 1e-12)
    if len(ok) == 0:
        return int(np.argmax(caught))
    return int(ok[np.lexsort((onset[ok], np.round(fa[ok], 9)))[0]])


def rule_b(caught, fa, f1, onset):
    return int(np.lexsort((fa, -np.round(f1, 9)))[0])


def pareto(caught, fa, f1, onset):
    x = np.column_stack([-caught, fa, onset])
    return np.array([not any(np.all(x[j] <= x[i]) and np.any(x[j] < x[i]) for j in range(len(x))) for i in range(len(x))])


def main() -> None:
    names, people, runs, onset = load()
    n = len(people)
    m = metrics(runs, onset, names, np.ones(n))
    front = pareto(*m)
    rules = {"A (>= 95 % caught, fewest false alarms)": rule_a, "B (highest F1)": rule_b}

    say("# Selecting the best detector (exploratory)\n")
    say("## (a) Choose on ten participants, score on the eleventh\n")
    everything = {}
    for label, rule in rules.items():
        apparent = rule(*m)
        chosen, held = [], []
        for i in range(n):
            w = np.ones(n)
            w[i] = 0
            k = rule(*metrics(runs, onset, names, w))
            chosen.append(names[k])
            held.append(runs[names[k]][:, i, :].mean(axis=0))          # mean over the chosen detector's runs
        events, hits, false, minutes = np.sum(held, axis=0)
        lat = [onset[names.index(c), i] for i, c in enumerate(chosen)]
        counts = pd.Series(chosen).value_counts()
        say(f"### Rule {label}\n")
        say(f"- Winner on all 11: **{names[apparent]}** (apparent {100 * m[0][apparent]:.1f} % caught, {m[1][apparent]:.2f} FA/h, "
            f"F1 {m[2][apparent]:.3f}, onset {m[3][apparent]:.2f} s)")
        say(f"- Chosen in the 11 folds: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        say(f"- By fold: " + ", ".join(f"{p} {c}" for p, c in zip(people, chosen)))
        say(f"- Held-out result of the selection: {100 * hits / events:.1f} % caught, {60 * false / minutes:.2f} FA/h, "
            f"F1 {2 * hits / (events + hits + false):.3f}, onset {np.median(lat):.2f} s\n")
        everything[label] = rule

    rng = np.random.default_rng(20261122)
    wins = {label: np.zeros(len(names)) for label in rules}
    on_front = np.zeros(len(names))
    none = 0
    for _ in range(RESAMPLES):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        b = metrics(runs, onset, names, w)
        none += not np.any(b[0] >= FLOOR)
        for label, rule in rules.items():
            wins[label][rule(*b)] += 1
        on_front += pareto(*b)
    table = pd.DataFrame({"detector": names, "caught_pct": 100 * m[0], "fa_per_hour": m[1], "f1": m[2], "onset_s": m[3],
                          "pareto": front, "win_A_pct": 100 * wins[list(rules)[0]] / RESAMPLES,
                          "win_B_pct": 100 * wins[list(rules)[1]] / RESAMPLES, "pareto_pct": 100 * on_front / RESAMPLES})
    table = table.sort_values("win_A_pct", ascending=False)
    table.to_csv(RES / f"{NAME}.csv", index=False)
    say(f"## (b) Participant bootstrap ({RESAMPLES:,} resamples)\n")
    say("| Detector | Caught % | FA/h | F1 | Onset s | Wins rule A % | Wins rule B % | On the Pareto front, all 11 | On the Pareto front % |")
    say("|---|---|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        say(f"| {r.detector} | {r.caught_pct:.1f} | {r.fa_per_hour:.2f} | {r.f1:.3f} | {r.onset_s:.2f} | {r.win_A_pct:.1f} | {r.win_B_pct:.1f} "
            f"| {'yes' if r.pareto else 'no'} | {r.pareto_pct:.0f} |")
    say(f"\nNo detector reached 95 % caught in {100 * none / RESAMPLES:.1f} % of the resamples.")
    (RES / f"{NAME}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
