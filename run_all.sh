#!/usr/bin/env bash
# Full analysis of the paper, in order. Run from the repository root after the set-up in README.md.
# Trainings are long (about 1000 LSTM runs and the nested search of ten models); a CUDA GPU is strongly recommended.
# Every training step skips runs that have already finished, so the script can be stopped and started again.
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
JOBS="${JOBS:-8}"
step() { echo; echo "$(date '+%F %T') == $*"; }
mkdir -p results

step "session files from the dataset"
"$PY" tools/make_session_files.py

step "threshold rule (M_thr)";                       "$PY" baselines/b3_load_rule.py
step "HAR classifier of the previous study (M_HAR)"; "$PY" baselines/b1_hgb_state.py
step "HGB stand-up classifier (M_ML5)";              "$PY" baselines/b2_hgb_window.py -j "$JOBS"

step "LSTM with the continuous target (M_ramp): leave-one-out, nested thresholds, sensor subsets"
"$PY" tools/run_lstm_jobs.py loo nested ablation -j "$JOBS"
step "LSTM: sensitivity and exploratory runs (whole-recording pressure scale, window length, other target)"
"$PY" tools/run_lstm_jobs.py fullnorm window inside -j "$JOBS"

step "event-level scoring and statistics of the first stage"
"$PY" analysis/score_detectors.py
"$PY" analysis/stats.py
"$PY" analysis/error_taxonomy.py
"$PY" analysis/operating_point.py
"$PY" analysis/participant_table.py
"$PY" analysis/compute_cost.py
"$PY" analysis/figures.py

step "ML and DL models with base settings (Appendix A)"
"$PY" comparison/train_models.py dl ml -j "$JOBS"
"$PY" comparison/score_models.py
"$PY" comparison/test_tools.py
"$PY" comparison/compare_models.py
"$PY" comparison/froc.py -j "$JOBS"
"$PY" comparison/select_best.py

step "tuned, strictly nested search of the ten ML and DL models"
"$PY" comparison/tune_models.py inner dl -j "$JOBS"
"$PY" comparison/tune_models.py inner ml -j "$JOBS"
"$PY" comparison/tune_select.py -j "$JOBS"
"$PY" comparison/tune_models.py final all -j "$JOBS"
"$PY" comparison/tune_analyse.py
"$PY" comparison/select_best.py tuned
"$PY" comparison/decide_hypotheses.py

step "channel search (genetic algorithm, nested)"
"$PY" comparison/ga_channels.py run -j "$JOBS"
"$PY" comparison/ga_channels.py analyse

step "one foot against both feet (post hoc), errors of the chosen detector, computational cost"
"$PY" comparison/feet_hgb.py -j "$JOBS"
"$PY" comparison/errors_chosen.py
OMP_NUM_THREADS=1 "$PY" comparison/compute_cost.py

step "ALL DONE: results/"
