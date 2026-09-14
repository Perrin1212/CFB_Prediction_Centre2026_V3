from __future__ import annotations

"""Evaluate the exact production blend on the untouched 2025 holdout."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.v3_matchup_engine import V3_MATCHUP_FEATURES
from engine.v3_prediction import predict_from_matchup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=SETTINGS.validation_simulations)
    args = parser.parse_args()
    data = pd.read_csv(SETTINGS.processed_root / "chronological_training_rows.csv", low_memory=False)
    holdout = data[pd.to_numeric(data.season, errors="coerce").eq(SETTINGS.holdout_season)].copy()
    bundle = joblib.load(SETTINGS.model_root / "v3_structural_models.joblib")
    if holdout.empty:
        raise RuntimeError("Untouched 2025 holdout is empty")

    rows: list[dict] = []
    for row in holdout.itertuples(index=False):
        matchup = {key: getattr(row, key) for key in V3_MATCHUP_FEATURES}
        result = predict_from_matchup(matchup, bundle, args.simulations,
                                      SETTINGS.seed + int(row.game_id) % 100_000,
                                      validation=True)
        simulation = result.pop("simulation")
        rows.append({"game_id": int(row.game_id), "home_points": float(row.home_points),
                     "away_points": float(row.away_points), "home_win": float(row.home_win),
                     **result, **{f"sim_{key}": value for key, value in simulation.items()}})
    p = pd.DataFrame(rows)
    actual_margin = p.home_points - p.away_points
    actual_total = p.home_points + p.away_points
    probability = p.home_win_probability.clip(1e-6, 1 - 1e-6)
    model_probability = p.model_home_win_probability.clip(1e-6, 1 - 1e-6)
    simulation_probability = p.sim_home_win_probability.clip(1e-6, 1 - 1e-6)
    actual_scores = np.concatenate([p.home_points, p.away_points])
    metrics = {
        "games": int(len(p)), "simulations_per_game": int(args.simulations),
        "winner_accuracy": float(accuracy_score(p.home_win, probability >= .5)),
        "brier": float(brier_score_loss(p.home_win, probability)),
        "log_loss": float(log_loss(p.home_win, probability)),
        "model_only_brier": float(brier_score_loss(p.home_win, model_probability)),
        "simulation_only_brier": float(brier_score_loss(p.home_win, simulation_probability)),
        "home_mae": float(mean_absolute_error(p.home_points, p.projected_home_points)),
        "away_mae": float(mean_absolute_error(p.away_points, p.projected_away_points)),
        "margin_mae": float(mean_absolute_error(actual_margin, p.projected_margin)),
        "total_mae": float(mean_absolute_error(actual_total, p.projected_total)),
        "simulated_40_plus_rate": float(pd.concat([p.sim_home_40_plus_probability, p.sim_away_40_plus_probability]).mean()),
        "actual_40_plus_rate": float((actual_scores >= 40).mean()),
        "simulated_10_or_less_rate": float(pd.concat([p.sim_home_10_or_less_probability, p.sim_away_10_or_less_probability]).mean()),
        "actual_10_or_less_rate": float((actual_scores <= 10).mean()),
        "home_80_interval_coverage": float(((p.home_points >= p.sim_home_p10) & (p.home_points <= p.sim_home_p90)).mean()),
        "away_80_interval_coverage": float(((p.away_points >= p.sim_away_p10) & (p.away_points <= p.sim_away_p90)).mean()),
        "margin_80_interval_coverage": float(((actual_margin >= p.sim_margin_p10) & (actual_margin <= p.sim_margin_p90)).mean()),
        "total_80_interval_coverage": float(((actual_total >= p.sim_total_p10) & (actual_total <= p.sim_total_p90)).mean()),
    }
    probability_bins = pd.cut(probability, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = p.assign(probability=probability, bucket=probability_bins).groupby(
        "bucket", observed=False
    ).agg(games=("home_win", "size"), predicted=("probability", "mean"), actual=("home_win", "mean")).reset_index()
    calibration["bucket"] = calibration.bucket.astype(str)
    p.to_csv(SETTINGS.processed_root / "v3_2025_production_predictions.csv", index=False)
    calibration.to_csv(SETTINGS.audit_root / "v3_2025_calibration.csv", index=False)
    (SETTINGS.audit_root / "v3_2025_production_validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
