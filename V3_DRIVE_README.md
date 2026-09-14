CFB Prediction Centre V3 — Drive Engine

V3 is an additive challenger. It does not replace or mutate frozen V2.

Core design

Observed production is adjusted for opponent quality before it updates team state. A 40-point / 400-yard type performance against a weak defense receives less credit than the same performance against an elite defense. Defensive production is adjusted symmetrically for quality of offenses faced.

V3 models the path to the final output:

chronological opponent-adjusted drive offense/defense state

expected possessions/drives

starting field position

points per drive and drive outcome shape

success/PPA/explosiveness/turnover interaction

score-state / garbage-time behavior

Monte Carlo score distribution

winner probability, projected team scores, margin and total

Files

Place v3_drive_engine.py and v3_drive_simulator.py in engine/.
Place all other .py files in jobs/.

First historical build

python -m jobs.fetch_v3_cfbd_data
python -m jobs.build_v3_drive_dataset
python -m jobs.train_v3_drive_models
python -m jobs.evaluate_v3_drive
python -m jobs.run_v3_drive_2026

Historical /plays requests are cached by season/week. Re-running does not spend calls on cached weeks.

Output

data/predictions/2026_v3_drive_predictions.csv

V2 remains the production/control model until V3 wins out-of-sample validation.