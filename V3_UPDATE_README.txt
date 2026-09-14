CFB PREDICTION CENTRE 2026 - V3.1 UPDATE
========================================

Copy this update over the root of CFB_Prediction_Centre2026_V3_FINAL and allow
Windows to replace files with the same names. Keep your existing .env and data.

FIRST RUN AFTER COPYING THE UPDATE
----------------------------------
python -m jobs.run_pipeline --refresh-current --rebuild-history

NORMAL WEEKLY UPDATE
--------------------
python -m jobs.run_pipeline --refresh-current

FAST TEST RUN (10,000 simulations)
----------------------------------
python -m jobs.run_pipeline --refresh-current --quick

The normal command refreshes the V2 control, current CFBD results and PBP,
2026 opponent-adjusted state, ELO, 50,000-simulation forecasts, immutable
pregame locks, betting/app data, and the V3 app overlay.

Historical 2021-2025 drive reconstruction is frozen after the first run. V3
uses the official structural engine. The older jobs named v3_drive_* remain
research utilities and are not the app's production V3 data path.
