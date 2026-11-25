"""Does the update rate explain H1? (the study protocol, §12, exploratory.)

B1 updates every 1 s, the LSTM every 0.5 s. Variants, all scored with events.score:
  B1 1 s            as deployed (must reproduce baselines/runs/b1/b1_alerts.csv exactly: self-check)
  B1 1 s shifted    same, windows starting 0.5 s later
  B1 0.5 s          same model and alert rule, window every 0.5 s (the two grids above interleaved)
  *_lock            the same with a 5 s lock-out
  LSTM 0.5 s        primary detector (strict nested thresholds; must reproduce results/events.csv: self-check)
  LSTM 1 s          same outputs and thresholds; alerts available only every 1 s, on B1's grid
  LSTM 1 s shifted  same, grid shifted by 0.5 s
Differences: stand-ups caught by both; per participant the median (as in H1) and the mean, averaged over the LSTM runs.

Writes results/update_rate.md and results/update_rate.csv.

  python analysis/update_rate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
from b1_hgb_state import WINDOW, features, transitions  # noqa: E402
from events import LOCKOUT_S, PARTICIPANTS, RATE, rising_crossings, score  # noqa: E402
from score_detectors import MINUTES, RUNS, SEEDS, STAND_UPS, lstm_output  # noqa: E402
from stats_tools import hodges_lehmann, holm, signed_rank_test  # noqa: E402

RES = ROOT / "results"
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def b1_labels() -> dict[str, np.ndarray]:
    """Labels of windows every 0.5 s (index k: samples 40k .. 40k + 160)."""
    model = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet.joblib")
    meta = joblib.load(ROOT / "baselines" / "model" / "hgb_both_feet_meta.joblib")
    classes, result = np.asarray(model.classes_), {}
    for p in PARTICIPANTS:
        folder = ROOT / "data_final" / p
        feet = [np.hstack([np.loadtxt(folder / f"{kind}_{side}.csv", delimiter=",", skiprows=1)[:, 1:] for kind in ("imu", "pressure")])
                for side in ("left", "right")]
        offsets = np.arange(0, len(feet[0]) - WINDOW + 1, 40)[:, None] + np.arange(WINDOW)
        windows = [foot.T.astype(np.float32)[:, offsets].transpose(1, 0, 2) for foot in feet]
        result[p] = classes[model.predict_proba(features(*windows, meta["feature_cols"])).argmax(axis=1)]
    return result


def lockout(times: np.ndarray) -> np.ndarray:
    kept, until = [], -np.inf
    for t in times:
        if t >= until:
            kept.append(t)
            until = t + LOCKOUT_S
    return np.array(kept)


def lstm_alerts(signal: np.ndarray, theta: float, step: int, phase: int) -> np.ndarray:
    crossings = rising_crossings(signal, theta, LOCKOUT_S, np.arange(len(signal)) / RATE)
    available = np.ceil((crossings + 1 - phase) / step) * step + phase
    first = WINDOW if (WINDOW - phase) % step == 0 else WINDOW + phase
    return np.minimum(np.maximum(available, first), len(signal)) / RATE


def main() -> None:
    labels = b1_labels()
    participants = pd.read_csv(RES / "participants.csv")
    theta = participants[participants.detector == "LSTM"].set_index(["run", "participant"]).theta
    alerts: dict[tuple[str, str, str], np.ndarray] = {}
    for p in PARTICIPANTS:
        ends = (np.arange(len(labels[p])) * 40 + WINDOW) / RATE
        for name, pick in (("B1 1 s", slice(0, None, 2)), ("B1 1 s shifted", slice(1, None, 2)), ("B1 0.5 s", slice(None))):
            times = ends[pick][transitions(labels[p][pick])]
            alerts[name, "fixed", p], alerts[name + " lock", "fixed", p] = times, lockout(times)
        for run in SEEDS:
            signal = lstm_output(RUNS / "loo_final" / f"test_{p}" / run)
            for name, step, phase in (("LSTM 0.5 s", 40, 0), ("LSTM 1 s", 80, 0), ("LSTM 1 s shifted", 80, 40)):
                alerts[name, run, p] = lstm_alerts(signal, float(theta[run, p]), step, phase)

    deployed = pd.read_csv(ROOT / "baselines" / "runs" / "b1" / "b1_alerts.csv")
    old = pd.read_csv(RES / "alerts.csv")
    for p in PARTICIPANTS:
        assert np.allclose(alerts["B1 1 s", "fixed", p], deployed[deployed.participant == p].alert_s), p
        for run in SEEDS:
            stored = old[(old.detector == "LSTM") & (old.run == run) & (old.participant == p)].alert_s.to_numpy()
            assert np.allclose(np.sort(alerts["LSTM 0.5 s", run, p]), np.sort(stored)), (run, p)
    print("self-checks passed: B1 1 s and LSTM 0.5 s reproduce the stored alerts")

    tau, rows = {}, {}
    for (name, run, p), times in alerts.items():
        s = score(times, STAND_UPS[p])
        tau[name, run, p] = s.tau
        rows.setdefault(name, []).append((run, p, s.hits, s.events, s.false, s.duplicates, np.nanmedian(s.onset_latency)))
    say("# Update-rate check of H1 (exploratory)\n")
    say("| Variant | Caught % | False alarms/h | Duplicates per run | Onset latency s (median participant) |")
    say("|---|---|---|---|---|")
    table = []
    for name, r in rows.items():
        d = pd.DataFrame(r, columns=["run", "participant", "hits", "events", "false", "duplicates", "onset"])
        runs = d.run.nunique()
        values = {"variant": name, "caught_pct": 100 * d.hits.sum() / d.events.sum(),
                  "fa_per_hour": 60 * d.false.sum() / (runs * sum(MINUTES.values())), "duplicates": d.duplicates.sum() / runs,
                  "onset_s": d.groupby("participant").onset.mean().median()}
        table.append(values)
        say(f"| {name} | {values['caught_pct']:.1f} | {values['fa_per_hour']:.2f} | {values['duplicates']:.1f} | {values['onset_s']:+.2f} |")
    pd.DataFrame(table).to_csv(RES / "update_rate.csv", index=False)

    def differences(a: str, b: str, summary) -> np.ndarray:
        out = []
        for p in PARTICIPANTS:
            values = []
            for run in SEEDS:
                d = tau[a, run, p] - tau[b, "fixed", p]
                if np.isfinite(d).any():
                    values.append(summary(d[np.isfinite(d)]))
            out.append(np.mean(values))
        return np.array(out)

    pairs = [("planned H1", "LSTM 0.5 s", "B1 1 s"), ("main (a)", "LSTM 0.5 s", "B1 0.5 s"), ("main (b)", "LSTM 1 s", "B1 1 s"),
             ("descriptive", "LSTM 1 s shifted", "B1 1 s shifted"), ("descriptive", "LSTM 0.5 s", "B1 1 s shifted"),
             ("descriptive", "LSTM 0.5 s", "B1 0.5 s lock")]
    for label, summary in (("median (as in H1)", np.median), ("mean", np.mean)):
        say(f"\n## Alert time, LSTM − B1 (negative = LSTM earlier), participant {label}\n")
        x = {(role, a, b): differences(a, b, summary) for role, a, b in pairs}
        p = {k: signed_rank_test(v, "two-sided")["p"] for k, v in x.items()}
        adjusted = holm({k: v for k, v in p.items() if k[0].startswith("main")})
        for k, v in x.items():
            c = hodges_lehmann(v, 0.95, "two-sided")
            say(f"- {k[0]}: {k[1]} vs {k[2]}: {c['estimate']:+.3f} s [{c['lo']:+.3f}, {c['hi']:+.3f}], p = {p[k]:.4f}"
                + (f", Holm p = {adjusted[k]:.4f}" if k in adjusted else "") + f"; LSTM earlier in {int((v < 0).sum())}/11 participants")
    (RES / "update_rate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
