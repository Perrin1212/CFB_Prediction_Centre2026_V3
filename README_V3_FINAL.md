# CFB Prediction Centre 2026 — V3 Final Architecture

This project is the full feature-rich Prediction Centre, not the stripped prototype. The existing ~2,000-line Streamlit application and mature V2-era engines are retained where they provide useful production/UI functionality. V3 now owns its CFBD acquisition and its new predictive pipeline.

## Authoritative V3 pipeline

`CFBD -> data/raw/cfbd -> v3_build_canonical -> chronological opponent-adjusted states -> matchup/possession engine -> structural residual models -> drive Monte Carlo -> 2026 predictions`

### Core new V3 files
- `engine/cfbd_client.py` — cache-first API client.
- `engine/v3_possession_engine.py` — play-to-drive reconstruction, field position, down conversion, PPA/success/explosiveness and garbage-time weighting.
- `engine/v3_team_state.py` — chronological offense/defense/ELO state and opponent adjustment.
- `engine/v3_matchup_engine.py` — current-opponent interaction and expected possessions.
- `engine/v3_game_simulator.py` — possession-level Monte Carlo with game script and correlated scoring environment.
- `jobs/v3_acquire_history.py` — V3-owned games, PBP, team box and optional advanced box acquisition.
- `jobs/v3_build_canonical.py` — authoritative game universe, drives and same-kickoff-safe pregame rows.
- `jobs/v3_train_and_validate.py` — 2021-24 training, untouched 2025 holdout, market leakage guard.
- `jobs/v3_predict_2026.py` — forward 2026 predictions.
- `jobs/v3_audit_project.py` — syntax/feature/UI audit.

## Data rules
1. Market data is never a predictive feature.
2. Pregame state is computed before a game's result is applied.
3. Games sharing a kickoff timestamp are snapshotted as a batch before any of them update state.
4. 2025 is holdout-only for the V3 promotion exam.
5. Raw API payloads are cached locally and are not committed.
6. `.env` is local-only and excluded from Git.

## Production commands

The first run after installing this update rebuilds and freezes the validated
2021-2025 base, refits the production model through 2025, refreshes 2026, runs
the simulations and rebuilds the app overlay:

```powershell
python -m jobs.run_pipeline --refresh-current --rebuild-history
```

Normal weekly updates use one command. Historical drives are not rebuilt and
the 2021-2025 model is not re-tuned on 2026 outcomes:

```powershell
python -m jobs.run_pipeline --refresh-current
```

Production uses 50,000 simulations per new matchup. For a faster local check:

```powershell
python -m jobs.run_pipeline --refresh-current --quick
```

Use `--skip-v2` only when deliberately refreshing the V3 challenger without
running the frozen V2 production control.

## First environment setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```
Add the CFBD key to `.env`, then run the first production command above.
Do not promote V3 over frozen V2 until the 2025 validation report justifies it.

## Current-season weighting

Pregame team identity uses 35% current season after one valid game, 55% after
two, 70% after three, 80% after four, 90% after five and 92% thereafter. Every
observation is opponent-adjusted. ELO, expected possessions, drive efficiency,
field position, success, explosiveness and turnovers all enter the matchup.

Pregame predictions that have crossed kickoff are copied to
`data/predictions/v3_2026_predictions_locked.csv` and are never regenerated.

## Existing application
The original `app/app.py` is intentionally retained rather than replaced by a miniature shell. Existing tracker, locks, market-display, betting-performance and application-building jobs are also retained so working Prediction Centre functionality is not discarded. Their data adapters can be switched to V3 outputs after V3 passes validation.
