from __future__ import annotations

"""Shared inference logic for V3 validation and live production."""

import numpy as np
import pandas as pd

from engine.v3_game_simulator import simulate_game
from engine.v3_matchup_engine import V3_MATCHUP_FEATURES


# The structural classifier is better calibrated than the possession simulator
# on the holdout.  Simulations therefore provide score ranges and tail risk;
# they do not dilute the validated winner probability.
PROBABILITY_MODEL_WEIGHT = 1.00
SCORE_MODEL_WEIGHT = 0.70


def _logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(value, -30, 30)))))


def predict_from_matchup(
    matchup: dict,
    bundle: dict,
    simulations: int,
    seed: int,
    validation: bool = False,
) -> dict[str, object]:
    """Generate the structural, simulation and blended prediction for one game."""
    if validation:
        models = bundle.get("validation_models")
        medians = bundle.get("validation_impute_values", {})
        if not models:
            raise RuntimeError("Model bundle does not contain validation models")
    else:
        models = {
            "home_residual_model": bundle["home_residual_model"],
            "away_residual_model": bundle["away_residual_model"],
            "winner_model": bundle["winner_model"],
        }
        medians = bundle.get("impute_values", {})

    X = pd.DataFrame([{key: matchup.get(key, np.nan) for key in V3_MATCHUP_FEATURES}])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(medians)).fillna(0.0)

    home_base = (
        float(matchup.get("home_drives", 11.5)) * float(matchup.get("home_ppd", 2.25))
        + float(matchup.get("home_field_points", 0.0)) / 2.0
    )
    away_base = (
        float(matchup.get("away_drives", 11.5)) * float(matchup.get("away_ppd", 2.25))
        - float(matchup.get("home_field_points", 0.0)) / 2.0
    )
    model_home = float(
        np.clip(home_base + models["home_residual_model"].predict(X)[0], 0, 80)
    )
    model_away = float(
        np.clip(away_base + models["away_residual_model"].predict(X)[0], 0, 80)
    )
    model_probability = float(
        np.clip(models["winner_model"].predict_proba(X)[0, 1], 1e-6, 1 - 1e-6)
    )

    # Anchor the simulated scoring rates to the structural score model while
    # retaining the drive-outcome shape, turnover state and possession variance.
    simulation_matchup = dict(matchup)
    simulation_matchup["home_ppd"] = float(
        np.clip(model_home / max(float(matchup.get("home_drives", 11.5)), 6.0), 0.15, 5.50)
    )
    simulation_matchup["away_ppd"] = float(
        np.clip(model_away / max(float(matchup.get("away_drives", 11.5)), 6.0), 0.15, 5.50)
    )
    simulation = simulate_game(
        simulation_matchup,
        simulations=int(simulations),
        seed=int(seed),
    )

    simulation_probability = float(simulation["home_win_probability"])
    probability = _sigmoid(
        PROBABILITY_MODEL_WEIGHT * _logit(model_probability)
        + (1 - PROBABILITY_MODEL_WEIGHT) * _logit(simulation_probability)
    )
    projected_home = (
        SCORE_MODEL_WEIGHT * model_home
        + (1 - SCORE_MODEL_WEIGHT) * float(simulation["projected_home_points"])
    )
    projected_away = (
        SCORE_MODEL_WEIGHT * model_away
        + (1 - SCORE_MODEL_WEIGHT) * float(simulation["projected_away_points"])
    )

    return {
        "home_win_probability": probability,
        "away_win_probability": 1.0 - probability,
        "projected_home_points": float(projected_home),
        "projected_away_points": float(projected_away),
        "projected_margin": float(projected_home - projected_away),
        "projected_total": float(projected_home + projected_away),
        "model_home_win_probability": model_probability,
        "model_projected_home_points": model_home,
        "model_projected_away_points": model_away,
        "probability_model_weight": PROBABILITY_MODEL_WEIGHT,
        "score_model_weight": SCORE_MODEL_WEIGHT,
        "simulation": simulation,
    }
