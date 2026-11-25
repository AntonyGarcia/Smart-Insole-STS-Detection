"""Per-participant results (supplementary table, the study protocol, §10 item 5).

For each participant: recording length, stand-ups, and for the main detectors the caught share, false alarms per
hour, median onset latency, median lead time and the share alerted before upright. Multi-run detectors: mean over
the 6 runs (the analysis unit of §9), with the run range of the caught share and false alarms for the LSTM.

Writes results/participant_table.csv and results/participant_table.md.

  python analysis/participant_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from events import PARTICIPANTS
from stats import participant_metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DETECTORS = ["LSTM", "B1", "B2", "B3", "LSTM_inside_approx", "LSTM_ens6_sel"]
METRICS = {"sensitivity": "caught %", "fa_per_hour": "FA/h", "onset": "onset s", "lead": "lead s",
           "before_upright": "before upright %"}


def main() -> None:
    parts = pd.read_csv(RES / "participants.csv")
    events = pd.read_csv(RES / "events.csv")
    base = parts[parts.detector == "B1"].set_index("participant").reindex(PARTICIPANTS)
    table = pd.DataFrame({"participant": PARTICIPANTS, "minutes": base.minutes.round(1).to_numpy(),
                          "stand_ups": base.events.astype(int).to_numpy()})
    present = [d for d in DETECTORS if d in set(parts.detector)]
    for detector in present:
        m = participant_metrics(parts, events, detector)
        mean = m.groupby("participant")[list(METRICS)].mean().reindex(PARTICIPANTS)
        for column in METRICS:
            table[f"{detector}_{column}"] = mean[column].to_numpy()
        if detector == "LSTM":
            g = m.groupby("participant")
            for column in ("sensitivity", "fa_per_hour"):
                lo, hi = g[column].min().reindex(PARTICIPANTS), g[column].max().reindex(PARTICIPANTS)
                table[f"LSTM_{column}_range"] = [f"{a:.3g}–{b:.3g}" for a, b in zip(lo, hi)]
            table["LSTM_theta_range"] = [f"{a:.2f}–{b:.2f}" for a, b in
                                         zip(g.theta.min().reindex(PARTICIPANTS), g.theta.max().reindex(PARTICIPANTS))]
    table.to_csv(RES / "participant_table.csv", index=False)

    out = ["# Per-participant results (mean over runs for multi-run detectors)\n"]
    for detector in present:
        out.append(f"\n## {detector}\n")
        header = ["participant", "minutes", "stand-ups", *METRICS.values()]
        if detector == "LSTM":
            header += ["caught range", "FA/h range", "θ range"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for _, r in table.iterrows():
            cells = [r.participant, f"{r.minutes:.1f}", str(r.stand_ups),
                     f"{100 * r[f'{detector}_sensitivity']:.1f}", f"{r[f'{detector}_fa_per_hour']:.2f}",
                     f"{r[f'{detector}_onset']:+.2f}", f"{r[f'{detector}_lead']:+.2f}",
                     f"{100 * r[f'{detector}_before_upright']:.0f}"]
            if detector == "LSTM":
                cells += [r.LSTM_sensitivity_range, r.LSTM_fa_per_hour_range, r.LSTM_theta_range]
            out.append("| " + " | ".join(cells) + " |")
    (RES / "participant_table.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote {RES / 'participant_table.csv'} and {RES / 'participant_table.md'}")


if __name__ == "__main__":
    main()
