# Sit-to-Stand Detection with Bilateral Smart Insoles

Analysis code for the paper *Sit-to-Stand Detection with Bilateral Smart Insoles Using Artificial Intelligence*.

Thirteen detectors of stand-ups were compared on continuous recordings from smart insoles worn in both shoes: a threshold on plantar pressure, the activity classifier of a previous study, an LSTM with a continuous target, and five machine learning (ML) and five deep learning (DL) models. Every detector was tested on participants it had not been trained or tuned on (leave-one-participant-out), and was scored on the share of stand-ups caught, the false alarms per hour and the time of the alert.

## Data

This repository contains code only. No data, trained models or results are included.

The dataset is public on Kaggle: **Smart Insole: Sitting to Standing**, https://doi.org/10.34740/KAGGLE/DSV/22827342.

It has 11 participants, 514 video-labelled stand-ups and 142.1 min of recordings, with 48 channels at 80 Hz. Each foot gives 18 plantar pressure channels and a 6-axis IMU.

Download it and unzip it so that the participant folders are in `data_final/`:

```
data_final/
  S01/  imu_left.csv  imu_right.csv  pressure_left.csv  pressure_right.csv  labels.csv
  ...
  S11/
```

## Set-up

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/make_session_files.py # writes data_final/S01_session.json ... S11_session.json, read by the LSTM scripts
```

The activity classifier of the previous study (M_HAR) is not included. It is needed only by `baselines/b1_hgb_state.py` and `analysis/compute_cost.py`. Where to place it is described in [baselines/model/README.md](baselines/model/README.md).

## Running the analysis

`run_all.sh` runs every step of the analysis in order:

```bash
bash run_all.sh                    # PYTHON=... and JOBS=... can be set
```

The trainings are long: about 1000 LSTM runs, the base-setting runs of ten models, a strictly nested search over their settings, and a nested genetic-algorithm search over channels. A CUDA GPU is strongly recommended. Finished runs are skipped, so the script can be stopped and started again. Each step can also be run on its own. The usage of every script is given at the top of the file.

Outputs go to `results/` (tables in Markdown and CSV, figures in `results/figures/`) and to the `runs/` folders next to each training script.

## Detectors

| Paper | Code | Script |
|---|---|---|
| M_thr, threshold on plantar pressure | B3 | `baselines/b3_load_rule.py` |
| M_HAR, activity classifier of the previous study | B1 | `baselines/b1_hgb_state.py` |
| M_ramp, LSTM with a continuous target | LSTM | `lstm/train_movement_lstm.py`, run by `tools/run_lstm_jobs.py` |
| M_ML5, histogram gradient boosting (reference model) | B2, HGB | `baselines/b2_hgb_window.py`, `comparison/train_models.py` |
| Logistic regression, SVM, random forest, XGBoost | ML | `comparison/train_models.py`, `comparison/tune_models.py` |
| 1D-CNN, TCN, LSTM, GRU, CNN-GRU | DL | `comparison/train_models.py`, `comparison/tune_models.py` |

## Repository layout

```
analysis/      event-level scoring (events.py), statistics, error analysis, computational cost, figures
baselines/     threshold rule, activity classifier of the previous study, HGB stand-up classifier
lstm/          LSTM with the continuous target, and its threshold and seed analyses
comparison/    ML and DL models: training, nested tuning, scoring, statistical comparison, FROC,
               channel search (NSGA-II), one foot against both feet, errors of the chosen detector
tools/         session files from the dataset, LSTM job runner
run_all.sh     the whole analysis in order
```

## Evaluation rules

All detectors are scored by `analysis/events.py` with the same rules:

- **Alert time:** an alert is timed when a live system could raise it. That is the end of the 0.5 s block for the ML and DL models and the LSTM, the end of the 2 s window for the activity classifier, and the next sample for the threshold rule.
- **Lock-out:** further alerts are ignored for 5 s.
- **Matching:** an alert is matched to a stand-up if it falls between 1 s before the start of the rise and 3 s after the person is upright. Any other alert is a false alarm.
- **Threshold:** the alert threshold of each detector is chosen on the other participants. The rule is the fewest false alarms among thresholds that catch at least 95% of stand-ups.

## Citation

If you use this code, please cite the paper (citation to be added on publication) and the dataset:

> A. Garcia, "Smart Insole: Sitting to Standing," Kaggle, 2026. doi:10.34740/KAGGLE/DSV/22827342.

## License

MIT, see [LICENSE](LICENSE).
