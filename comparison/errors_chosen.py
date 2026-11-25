"""Errors of the detector chosen without the test participant (main rule A of the tuned search).

Misses: duration of missed and caught stand-ups, plantar load just before the rise, and whether HGB missed the same ones.
False alarms: where they happened. Only stand-ups are labelled, so the situation is inferred:
  - "just before a rise": within 3 s before the association window of the next stand-up (s - 4 s to s - 1 s);
  - "seated": total plantar load below 0.5 of its 95th percentile (data_final/README.md: seated ~0.1-0.45,
    standing or walking ~0.6-1), smoothed over 0.25 s;
  - "upright": load at or above 0.5 (standing, walking to the other chair or sitting down).
Counts are means over runs (the SVM has one run).

  python comparison/errors_chosen.py        writes results/errors_chosen.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ("analysis", "baselines", "comparison"):
    sys.path.insert(0, str(ROOT / folder))
from events import PARTICIPANTS, RATE, score, window_alert_times  # noqa: E402
from score_detectors import STAND_UPS  # noqa: E402
from tune_models import DETERMINISTIC, OUT, SEEDS, final_path  # noqa: E402

RES = ROOT / "results"
SEATED_BELOW = 0.5


def load_fraction(p: str) -> np.ndarray:
    """Total plantar load of both feet, smoothed over 0.25 s, as a fraction of its 95th percentile."""
    total = sum(np.loadtxt(ROOT / "data_final" / p / f"pressure_{s}.csv", delimiter=",", skiprows=1)[:, 1:].sum(1)
                for s in ("left", "right"))
    k = int(0.25 * RATE)
    smooth = np.convolve(total, np.ones(k) / k, mode="same")
    return smooth / (np.percentile(smooth, 95) or 1.0)


def at(load: np.ndarray, t: float) -> float:
    return float(load[min(len(load) - 1, max(0, int(round(t * RATE))))])


def main() -> None:
    selected = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    nested = pd.read_csv(RES / "tuned_nested.csv").set_index("rule")
    pick = dict(item.split(":") for item in nested.loc["A", "by_fold"].split())
    seeds = lambda m: SEEDS[:1] if m in DETERMINISTIC else SEEDS  # noqa: E731

    def alerts(m: str, p: str) -> list:
        out = []
        for s in seeds(m):
            d = np.load(final_path(m, p, s))
            out.append(score(window_alert_times(d["probability"], d["window_end_s"], selected[m][p]["theta"]), STAND_UPS[p]))
        return out

    fa_rows, ev_rows = [], []
    for p in PARTICIPANTS:
        m, su, load = pick[p], STAND_UPS[p], load_fraction(p)
        chosen, ref = alerts(m, p), alerts("HGB", p)
        runs = len(chosen)
        for r in chosen:
            for t, kind in r.alerts:
                if kind != "false":
                    continue
                before = (su[:, 0] - 4 <= t) & (t < su[:, 0] - 1)
                place = "just before a rise" if before.any() else ("seated" if at(load, t) < SEATED_BELOW else "upright")
                prev, nxt = su[su[:, 1] < t], su[su[:, 0] > t]
                fa_rows.append({"participant": p, "model": m, "t": t, "place": place, "load": at(load, t), "weight": 1 / runs,
                                "after_end": t - prev[-1, 1] if len(prev) else np.nan,
                                "before_next": nxt[0, 0] - t if len(nxt) else np.nan})
        missed_share = np.mean([np.isnan(r.tau) for r in chosen], axis=0)
        hgb_missed = np.mean([np.isnan(r.tau) for r in ref], axis=0)
        for i, (s, e) in enumerate(su):
            ev_rows.append({"participant": p, "model": m, "stand_up": i + 1, "duration": e - s, "missed": missed_share[i],
                            "hgb_missed": hgb_missed[i], "load_before": at(load, s - 0.5)})

    fa, ev = pd.DataFrame(fa_rows), pd.DataFrame(ev_rows)
    fa.to_csv(RES / "errors_false_alarms.csv", index=False)
    ev.to_csv(RES / "errors_stand_ups.csv", index=False)
    lines = ["# Errors of the detector chosen without the test participant (rule A)\n", "## Misses\n"]
    ev["w_missed"] = ev.missed
    total = ev.w_missed.sum()
    lines.append(f"- missed (mean over runs): {total:.1f} of {len(ev)}")
    by_p = ev.groupby("participant").w_missed.sum()
    lines.append("- by participant: " + ", ".join(f"{p} {v:.1f}" for p, v in by_p.items() if v > 0))
    miss = ev[ev.missed >= 0.5]
    hit = ev[ev.missed < 0.5]
    lines.append(f"- stand-ups missed in at least half of the runs: {len(miss)}; "
                 f"duration median {miss.duration.median():.2f} s (range {miss.duration.min():.2f}-{miss.duration.max():.2f}) "
                 f"against {hit.duration.median():.2f} s for caught ones")
    lines.append(f"- load 0.5 s before the rise (fraction of 95th percentile): missed median {miss.load_before.median():.2f}, "
                 f"caught median {hit.load_before.median():.2f}")
    lines.append(f"- of these, also missed by HGB in at least half of its runs: {(miss.hgb_missed >= 0.5).sum()}")
    for p, g in miss.groupby("participant"):
        lines.append(f"  - {p} ({g.model.iloc[0]}): stand-ups {', '.join(str(x) for x in g.stand_up)}; durations "
                     f"{', '.join(f'{x:.2f}' for x in g.duration)}; load before {', '.join(f'{x:.2f}' for x in g.load_before)}; "
                     f"HGB missed {', '.join(f'{x:.2f}' for x in g.hgb_missed)}")
    s01 = ev[ev.participant == "S01"]
    lines.append(f"- S01 all stand-ups: duration median {s01.duration.median():.2f} s; load before median {s01.load_before.median():.2f}")

    lines += ["", "## False alarms (mean over runs)\n"]
    n_fa = fa.weight.sum()
    lines.append(f"- total: {n_fa:.1f}")
    for place, g in fa.groupby("place"):
        lines.append(f"- {place}: {g.weight.sum():.1f} ({100 * g.weight.sum() / n_fa:.0f} %)")
    lines.append("- by participant: " + ", ".join(f"{p} {v:.1f}" for p, v in fa.groupby("participant").weight.sum().items()))
    lines.append("- each false alarm (participant, model, time s, place, load, s after the previous stand-up ended, "
                 "s before the next one starts):")
    for r in fa.drop_duplicates(["participant", "t"]).itertuples():
        lines.append(f"  - {r.participant} {r.model} {r.t:.1f} {r.place} {r.load:.2f} {r.after_end:.1f} {r.before_next:.1f}")
    (RES / "errors_chosen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
