"""Decisions on the research hypotheses (paper Table research-hypotheses, Sections 2.9.4 and 2.9.5).

Rule: H0 is rejected when its Holm-adjusted p < 0.05; where H1 states a direction, the Hodges-Lehmann estimate must
also point that way. RQ1 uses BCa lower limits instead of a test. RQ7 reads results/ga_channels_participants.csv.

Reads results/tuned_tests.csv, results/tuned_nested.csv (tuned search) and results/hypotheses.csv (first stage, H4 =
sensor configurations of the ramp LSTM). Writes results/hypothesis_decisions.md and results/hypothesis_decisions.csv.

  python comparison/decide_hypotheses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from stats_tools import hodges_lehmann, holm, signed_rank_test  # noqa: E402

RES = ROOT / "results"
ALPHA = 0.05
tests = pd.read_csv(RES / "tuned_tests.csv")
nested = pd.read_csv(RES / "tuned_nested.csv").set_index("rule")
first = pd.read_csv(RES / "hypotheses.csv").set_index("hypothesis")
rows: list[dict] = []


def test(family: str, item: str) -> pd.Series:
    r = tests[(tests.family == family) & (tests["item"] == item)]
    if len(r) != 1:
        raise SystemExit(f"expected one test for {family!r} / {item!r}, found {len(r)}")
    return r.iloc[0]


def paired(r: pd.Series, unit: str = "") -> str:
    return f"{r.estimate:+.3f}{unit} [{r.lo:+.3f}, {r.hi:+.3f}], p = {r.p:.4f}, Holm p = {r.holm_p:.4f}"


def add(rq: str, criterion: str, evidence: str, rejected: bool | None) -> None:
    rows.append({"rq": rq, "criterion": criterion, "evidence": evidence,
                 "decision": "pending" if rejected is None else "H0 rejected" if rejected else "H0 not rejected"})


# RQ1: level reached by the detector chosen without the test participant (main rule A)
a = nested.loc["A"]
add("RQ1", "BCa lower limits: caught > 95 % and lead time > 0 s",
    f"caught {a.caught:.1f} % [{a.caught_lo:.1f}, {a.caught_hi:.1f}]; lead time {a.lead:.2f} s [{a.lead_lo:.2f}, {a.lead_hi:.2f}]",
    bool(a.caught_lo > 95 and a.lead_lo > 0))

# RQ2: threshold rule (B3) against HGB on false alarms/h; H1: learned models have fewer, i.e. B3 - HGB > 0
fr = test("Friedman", "fa_per_hour")
r = test("False alarms per hour: model − HGB", "B3")
add("RQ2", "Holm p < 0.05 and B3 − HGB > 0 (false alarms/h)",
    f"Friedman p = {fr.p:.2g}; B3 − HGB {paired(r, ' /h')}", bool(r.holm_p < ALPHA and r.estimate > 0))

# RQ3: previous HAR system (B1) against HGB on median onset latency; H1: B1 alerts later, i.e. B1 - HGB > 0
r = test("Onset latency: model − HGB", "B1")
add("RQ3", "Holm p < 0.05 and B1 − HGB > 0 (onset latency)", f"B1 − HGB {paired(r, ' s')}",
    bool(r.holm_p < ALPHA and r.estimate > 0))

# RQ4: chosen detector against B1, one family of three; H1: the chosen detector alerts earlier
fam = "Search winner (rule A) − B1"
r = test(fam, "onset latency s")
others = "; ".join(f"{m}: {paired(test(fam, m))}" for m in ("F1", "FA/h"))
add("RQ4", "Holm p < 0.05 and chosen − B1 < 0 (onset latency)", f"onset {paired(r, ' s')}; {others}",
    bool(r.holm_p < ALPHA and r.estimate < 0))

# RQ5: best DL against best ML, any difference in either direction
fam = "Best DL − best ML, both chosen by rule A"
items = {m: test(fam, m) for m in ("F1", "FA/h", "onset latency s")}
add("RQ5", "Holm p < 0.05 for any of F1, false alarms/h, onset latency",
    "; ".join(f"{m}: {paired(r)}" for m, r in items.items()), any(r.holm_p < ALPHA for r in items.values()))

# RQ6 and RQ8: ramp LSTM sensor configurations (first-stage H4, F1). Post hoc tests run only if the Friedman test is
# significant; Holm over the four comparisons. Estimate = all sensors - subset, so > 0 means the subset is worse.
fried = first.loc["H4-f1"]
subsets = ["LSTM_pressure", "LSTM_imu", "LSTM_left", "LSTM_right"]
if fried.p < ALPHA and all(f"H4-f1-{c}" in first.index for c in subsets):
    adj = holm({c: first.loc[f"H4-f1-{c}", "p"] for c in subsets})
    post = {c: (first.loc[f"H4-f1-{c}", "hl_estimate"], first.loc[f"H4-f1-{c}", "p"], adj[c]) for c in subsets}
else:
    post = {}
show = lambda c: (f"{c.removeprefix('LSTM_')}: all − subset {post[c][0]:+.3f}, p = {post[c][1]:.4f}, Holm p = {post[c][2]:.4f}"  # noqa: E731
                  if c in post else f"{c.removeprefix('LSTM_')}: not tested (Friedman not significant)")
add("RQ6", "Friedman p < 0.05, then Holm p < 0.05 for pressure alone or IMU alone",
    f"Friedman p = {fried.p:.4f}, W = {fried.hl_estimate:.2f}; " + "; ".join(show(c) for c in subsets[:2]),
    any(post[c][2] < ALPHA for c in subsets[:2]) if post else False)
# RQ7: GA channel panel (comparison/ga_channels.py) against all 48 channels, F1, two-sided, a family of one
ga = pd.read_csv(RES / "ga_channels_participants.csv").pivot(index="participant", columns="arm", values="f1")
d = (ga["panel"] - ga["all"]).to_numpy(float)
t, hl = signed_rank_test(d, "two-sided"), hodges_lehmann(d, 0.95, "two-sided")
add("RQ7", "p < 0.05 for GA panel − all 48 (F1), either direction",
    f"panel − all {hl['estimate']:+.4f} [{hl['lo']:+.4f}, {hl['hi']:+.4f}], p = {t['p']:.4f}", bool(t["p"] < ALPHA))
add("RQ8", "Holm p < 0.05 and all − single foot > 0 (F1), within the RQ6 family",
    "; ".join(show(c) for c in subsets[2:]),
    any(post[c][2] < ALPHA and post[c][0] > 0 for c in subsets[2:]) if post else False)

out = pd.DataFrame(rows)
out.to_csv(RES / "hypothesis_decisions.csv", index=False)
md = ["# Decisions on the research hypotheses\n", "| RQ | Criterion | Evidence | Decision |", "|---|---|---|---|"]
md += [f"| {r.rq} | {r.criterion} | {r.evidence} | {r.decision} |" for r in out.itertuples()]
(RES / "hypothesis_decisions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
