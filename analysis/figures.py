"""Paper figures (the study protocol, §10) from results/*.csv and the run outputs. PNG (200 dpi) + PDF in results/figures/.

fig1_example_traces    S01, run seed0 (pre-specified): plantar load per foot; labelled Movement signal vs LSTM output,
                       its threshold and alerts; paper-1 HGB states with B1 alerts. Stand-ups shaded.
fig2_timing            per-participant median onset latency and lead time for LSTM, B1, B2, B3 (run-averaged).
fig3_threshold_sweep   primary LSTM outer outputs: sensitivity, false alarms/h and onset latency vs threshold, one thin
                       line per run + the median over runs; the range of thresholds chosen by strict nesting shaded.
fig4_confusion         stand-up-level confusion matrix of the primary LSTM (mean over runs); negatives = stretches
                       between association windows (>= 1 s) - a stretch with any false alarm is FP, else TN.
Also writes results/confusion_lstm.csv.

  python analysis/figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from events import AFTER_S, BEFORE_S, PARTICIPANTS, RATE, THRESHOLDS, labels, lstm_alert_times, read_lstm_output, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
SURFACE, GRID, INK, INK_2, MUTED, AXIS, SPAN = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#ecebe5"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical slots 1-4 (fixed order)
DETECTORS = {"LSTM": SLOT[0], "B1": SLOT[1], "B2": SLOT[2], "B3": SLOT[3]}
STATE = {"Sitting": "#b9d3f3", "Standing": "#8fd3b6", "Walking": "#f3c08f", "Tandem": "#d9d7cf"}
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig1(parts: pd.DataFrame, alerts: pd.DataFrame) -> None:
    participant, run, t0, t1 = "S01", "seed0", 0.0, 60.0
    stand_ups = labels(participant)
    a, b = int(t0 * RATE), int(t1 * RATE)
    t = np.arange(a, b) / RATE
    folder = ROOT / "data_final" / participant
    load = [np.loadtxt(folder / f"pressure_{s}.csv", delimiter=",", skiprows=1)[:, 1:].sum(axis=1) for s in ("left", "right")]
    load = [x / np.percentile(x[:120 * RATE], 95) for x in load]
    output = read_lstm_output(ROOT / "lstm" / "runs" / "loo_final" / f"test_{participant}" / run)
    from movement_labels import movement_signal  # noqa: E402  (lstm/ on sys.path below)
    expected = movement_signal(t, [tuple(x) for x in stand_ups])
    theta = parts[(parts.detector == "LSTM") & (parts.run == run) & (parts.participant == participant)].theta.iloc[0]
    lstm_alerts = alerts[(alerts.detector == "LSTM") & (alerts.run == run) & (alerts.participant == participant)]
    b1_alerts = alerts[(alerts.detector == "B1") & (alerts.participant == participant)]
    windows = pd.read_csv(ROOT / "baselines" / "runs" / "b1" / f"{participant}_windows.csv")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.1, 1.4, 0.45]})
    for ax in axes:
        style(ax)
        for s, e in stand_ups:
            if s < t1 and e > t0:
                ax.axvspan(s, e, color=SPAN, linewidth=0, zorder=0)
    for x, color, name in ((load[0], SLOT[0], "left foot"), (load[1], SLOT[1], "right foot")):
        axes[0].plot(t, x[a:b], color=color, linewidth=1.2, label=name)
    axes[0].set_ylabel("plantar load\n(× standing level)")
    axes[0].legend(frameon=False, ncols=2, loc="upper right", fontsize=8)
    axes[1].fill_between(t, expected, color="#d9d7cf", linewidth=0, label="labelled (target)")
    axes[1].plot(t, output[a:b], color=SLOT[0], linewidth=1.6, label="LSTM output")
    axes[1].axhline(theta, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].text(t1, theta, f"  θ = {theta:.2f}", va="center", ha="left", fontsize=8, color=INK_2)
    for row, color, marker in ((lstm_alerts, SLOT[0], "v"),):
        for r in row.itertuples():
            if t0 <= r.alert_s <= t1:
                axes[1].plot(r.alert_s, 1.12, marker=marker, color=color if r.kind != "false" else "#d03b3b",
                             markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
    axes[1].set_ylim(-0.03, 1.2)
    axes[1].set_ylabel("Movement signal")
    axes[1].legend(frameon=False, ncols=2, loc="upper left", fontsize=8, bbox_to_anchor=(0, 0.92))
    for w in windows.itertuples():
        centre = (w.start_s + w.end_s) / 2
        if t0 - 0.5 <= centre <= t1 + 0.5:
            axes[2].axvspan(centre - 0.5, centre + 0.5, ymin=0.2, ymax=0.8, color=STATE[w.label], linewidth=0)
    for r in b1_alerts.itertuples():
        if t0 <= r.alert_s <= t1:
            axes[2].plot(r.alert_s, 0.95, marker="v", color=SLOT[1], markersize=8, markeredgecolor=SURFACE,
                         markeredgewidth=1.5, clip_on=False, transform=axes[2].get_xaxis_transform())
    axes[2].set_yticks([])
    axes[2].set_ylabel("paper-1\nHGB state")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE.values()]
    axes[2].legend(handles, STATE.keys(), frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.45), fontsize=8)
    axes[2].set_xlabel("time (s)")
    axes[2].set_xlim(t0, t1)
    fig.suptitle(f"{participant}, run {run}: ▼ LSTM alerts (blue; red = false) and B1 alerts (orange); grey bands = labelled stand-ups",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig1_example_traces")


def fig2(events: pd.DataFrame) -> None:
    e = events[events.detector.isin(DETECTORS)].dropna(subset=["tau"]).copy()
    e["onset"] = e.tau - e.start_s
    e["lead"] = e.end_s - e.tau
    per = e.groupby(["detector", "run", "participant"])[["onset", "lead"]].median() \
           .groupby(["detector", "participant"]).mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), facecolor=SURFACE)
    for ax, column, title in ((axes[0], "onset", "onset latency: alert − start of rise (s)"),
                              (axes[1], "lead", "lead time: fully upright − alert (s)")):
        style(ax)
        for k, (detector, color) in enumerate(DETECTORS.items()):
            values = per[per.detector == detector][column].to_numpy()
            jitter = (np.arange(len(values)) - len(values) / 2) * 0.025
            ax.scatter(k + jitter, values, s=16, color=color, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            if len(values):
                m = np.median(values)
                ax.plot([k - 0.3, k + 0.3], [m, m], color=INK, linewidth=1.6, zorder=4)
        ax.axhline(0, color=AXIS, linewidth=1)
        top = ax.get_ylim()[1]
        for k, detector in enumerate(DETECTORS):      # median value above each column, clear of the dots
            values = per[per.detector == detector][column].to_numpy()
            if len(values):
                ax.text(k, top, f"{np.median(values):+.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(top=top + 0.08 * (top - ax.get_ylim()[0]))
        ax.set_xticks(range(len(DETECTORS)), DETECTORS.keys())
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=14)
        ax.set_xlim(-0.5, len(DETECTORS) - 0.5)
    axes[1].annotate("alert before upright ↑", xy=(1, 0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)
    fig.text(0.01, -0.02, "Dots: participants (median over their stand-ups, averaged over runs). Bar: median of participants.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig2_timing")


def fig3(parts: pd.DataFrame) -> None:
    runs = sorted(parts[parts.detector == "LSTM"].run.unique())
    curves = {"sens": [], "fa": [], "lat": []}
    for run in runs:
        h = np.zeros(len(THRESHOLDS)); n = np.zeros(len(THRESHOLDS)); f = np.zeros(len(THRESHOLDS)); minutes = 0.0
        lat = [[] for _ in THRESHOLDS]
        for p in PARTICIPANTS:
            out = read_lstm_output(ROOT / "lstm" / "runs" / "loo_final" / f"test_{p}" / run)
            stand_ups = labels(p)
            minutes += len(out) / RATE / 60
            for k, theta in enumerate(THRESHOLDS):
                s = score(lstm_alert_times(out, theta), stand_ups)
                h[k] += s.hits; n[k] += s.events; f[k] += s.false
                lat[k].append(np.nanmedian(s.onset_latency) if s.hits else np.nan)
        curves["sens"].append(100 * h / n)
        curves["fa"].append(f / minutes * 60)
        curves["lat"].append([np.nanmedian(v) for v in lat])
    chosen = parts[parts.detector == "LSTM"].theta
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), facecolor=SURFACE)
    for ax, key, title in zip(axes, ("sens", "fa", "lat"),
                              ("stand-ups caught (%)", "false alarms per hour", "onset latency (s)")):
        style(ax)
        ax.axvspan(chosen.min() - 0.01, chosen.max() + 0.01, color=SPAN, linewidth=0, zorder=0)
        for c in curves[key]:
            ax.plot(THRESHOLDS, c, color=AXIS, linewidth=0.9)
        ax.plot(THRESHOLDS, np.median(curves[key], axis=0), color=SLOT[0], linewidth=2, marker="o", markersize=3.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("threshold θ")
    fig.text(0.01, -0.04, f"Grey lines: each of {len(runs)} runs; blue: median over runs; band: thresholds chosen by strict "
             f"nesting ({chosen.min():.2f}–{chosen.max():.2f}).", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig3_threshold_sweep")


def fig4(events: pd.DataFrame, alerts: pd.DataFrame) -> None:
    rows = []
    for run in sorted(events[events.detector == "LSTM"].run.unique()):
        tp = fn = fp = tn = 0
        for p in PARTICIPANTS:
            e = events[(events.detector == "LSTM") & (events.run == run) & (events.participant == p)]
            if e.empty:
                continue
            tp += int(e.tau.notna().sum()); fn += int(e.tau.isna().sum())
            false_t = alerts[(alerts.detector == "LSTM") & (alerts.run == run) & (alerts.participant == p)
                             & (alerts.kind == "false")].alert_s.to_numpy()
            n = sum(1 for _ in open(ROOT / "data_final" / p / "imu_left.csv")) - 1
            edges = [0.0] + [x for s, e_ in zip(e.start_s, e.end_s) for x in (s - BEFORE_S, e_ + AFTER_S)] + [n / RATE]
            for a, b in zip(edges[0::2], edges[1::2]):
                if b - a >= 1.0:
                    if np.any((false_t >= a) & (false_t < b)):
                        fp += 1
                    else:
                        tn += 1
        rows.append({"run": run, "TP": tp, "FN": fn, "FP": fp, "TN": tn})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "confusion_lstm.csv", index=False)
    mean = table[["TP", "FN", "FP", "TN"]].mean()
    fig, ax = plt.subplots(figsize=(3.6, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cells = [[mean.TP, mean.FN], [mean.FP, mean.TN]]
    shares = [[mean.TP / (mean.TP + mean.FN), mean.FN / (mean.TP + mean.FN)],
              [mean.FP / (mean.FP + mean.TN), mean.TN / (mean.FP + mean.TN)]]
    ramp = plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#e8f0fb", "#2a78d6", "#184f95"])
    for i in range(2):
        for j in range(2):
            color = ramp(shares[i][j])
            ax.add_patch(plt.Rectangle((j + 0.03, 1 - i + 0.03), 0.94, 0.94, color=color))
            ink = "white" if shares[i][j] > 0.5 else INK
            value = cells[i][j]
            text = f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
            ax.text(j + 0.5, 1 - i + 0.58, text, ha="center", va="center", fontsize=15, color=ink)
            ax.text(j + 0.5, 1 - i + 0.3, f"{100 * shares[i][j]:.1f} % of row", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["alert", "no alert"]); ax.set_yticks([1.5, 0.5], ["stand-up", "no stand-up"])
    ax.xaxis.set_ticks_position("top")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.01, 0.0, f"Primary LSTM, mean over {len(table)} runs. Negatives: stretches between association windows.",
             fontsize=7.5, color=MUTED)
    save(fig, "fig4_confusion")


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "lstm"))
    FIG.mkdir(parents=True, exist_ok=True)
    parts = pd.read_csv(RES / "participants.csv")
    events = pd.read_csv(RES / "events.csv")
    alerts = pd.read_csv(RES / "alerts.csv")
    print("Writing figures:")
    fig1(parts, alerts)
    fig2(events)
    fig3(parts)
    fig4(events, alerts)
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
