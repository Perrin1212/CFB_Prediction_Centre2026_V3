from __future__ import annotations

"""Validate on untouched 2025 data, then refit the production model through 2025."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.v3_matchup_engine import V3_MATCHUP_FEATURES


def forbidden(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if any(token in column.lower() for token in SETTINGS.market_forbidden_tokens)
    ]


def feature_frame(
    frame: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    values = frame.reindex(columns=V3_MATCHUP_FEATURES).apply(pd.to_numeric, errors="coerce")
    if medians is None:
        raw_medians = values.median(numeric_only=True)
        medians = {
            column: float(raw_medians.get(column, 0.0))
            if pd.notna(raw_medians.get(column, np.nan))
            else 0.0
            for column in V3_MATCHUP_FEATURES
        }
    return values.fillna(pd.Series(medians)).fillna(0.0), medians


def score_models(seed: int) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    common = dict(
        loss="absolute_error",
        max_iter=300,
        max_leaf_nodes=23,
        l2_regularization=3,
        min_samples_leaf=35,
        learning_rate=0.045,
    )
    return (
        HistGradientBoostingRegressor(random_state=seed, **common),
        HistGradientBoostingRegressor(random_state=seed + 1, **common),
    )


def winner_model() -> LogisticRegression:
    # Preserve the validated coefficient geometry while allowing enough
    # iterations for the unscaled structural feature set to converge.
    return LogisticRegression(C=0.35, max_iter=10_000, random_state=SETTINGS.seed)


def structural_bases(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    home = frame.home_drives * frame.home_ppd + frame.home_field_points / 2.0
    away = frame.away_drives * frame.away_ppd - frame.home_field_points / 2.0
    return home, away


def fit_models(frame: pd.DataFrame, seed: int) -> tuple[dict[str, object], dict[str, float]]:
    X, medians = feature_frame(frame)
    home_base, away_base = structural_bases(frame)
    home_model, away_model = score_models(seed)
    home_model.fit(X, frame.home_points - home_base)
    away_model.fit(X, frame.away_points - away_base)
    win_model = winner_model()
    win_model.fit(X, frame.home_win)
    return {
        "home_residual_model": home_model,
        "away_residual_model": away_model,
        "winner_model": win_model,
    }, medians


def main() -> None:
    training_path = SETTINGS.processed_root / "chronological_training_rows.csv"
    if not training_path.exists():
        raise FileNotFoundError(f"Canonical training data not found: {training_path}")

    data = pd.read_csv(training_path, low_memory=False)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    data = data.dropna(subset=["home_points", "away_points", "season"]).copy()
    train = data[data.season.isin(SETTINGS.train_seasons)].copy()
    holdout = data[data.season.eq(SETTINGS.holdout_season)].copy()

    if train.empty:
        raise RuntimeError("No 2021-2024 training games were found")
    if not 400 <= len(holdout) <= SETTINGS.max_reasonable_games_per_season:
        raise RuntimeError(f"2025 holdout suspiciously sized: {len(holdout)}")
    violations = forbidden(V3_MATCHUP_FEATURES)
    if violations:
        raise RuntimeError(f"Market leakage in feature list: {violations}")

    print(f"Training seasons: {SETTINGS.train_seasons} ({len(train):,} games)")
    print(f"Untouched holdout: {SETTINGS.holdout_season} ({len(holdout):,} games)")
    print(f"Predictive features: {len(V3_MATCHUP_FEATURES)}; market leakage check: PASS")

    validation_models, validation_medians = fit_models(train, SETTINGS.seed)
    X_holdout, _ = feature_frame(holdout, validation_medians)
    home_base, away_base = structural_bases(holdout)
    projected_home = np.clip(
        home_base + validation_models["home_residual_model"].predict(X_holdout), 0, 80
    )
    projected_away = np.clip(
        away_base + validation_models["away_residual_model"].predict(X_holdout), 0, 80
    )
    probability = np.clip(
        validation_models["winner_model"].predict_proba(X_holdout)[:, 1], 1e-6, 1 - 1e-6
    )
    margin = projected_home - projected_away
    total = projected_home + projected_away
    actual_team_scores = np.concatenate([holdout.home_points, holdout.away_points])
    predicted_team_scores = np.concatenate([projected_home, projected_away])

    metrics = {
        "train_games": int(len(train)),
        "holdout_games": int(len(holdout)),
        "winner_accuracy": float(accuracy_score(holdout.home_win, probability >= 0.5)),
        "brier": float(brier_score_loss(holdout.home_win, probability)),
        "log_loss": float(log_loss(holdout.home_win, probability)),
        "home_mae": float(mean_absolute_error(holdout.home_points, projected_home)),
        "away_mae": float(mean_absolute_error(holdout.away_points, projected_away)),
        "margin_mae": float(mean_absolute_error(holdout.actual_margin, margin)),
        "total_mae": float(mean_absolute_error(holdout.actual_total, total)),
        "point_prediction_40_plus_rate": float((predicted_team_scores >= 40).mean()),
        "actual_40_plus_rate": float((actual_team_scores >= 40).mean()),
        "point_prediction_10_or_less_rate": float((predicted_team_scores <= 10).mean()),
        "actual_10_or_less_rate": float((actual_team_scores <= 10).mean()),
    }

    predictions = holdout[
        ["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points"]
    ].copy()
    predictions["v3_home_win_probability"] = probability
    predictions["v3_projected_home_points"] = projected_home
    predictions["v3_projected_away_points"] = projected_away
    predictions["v3_projected_margin"] = margin
    predictions["v3_projected_total"] = total
    predictions.to_csv(
        SETTINGS.processed_root / "v3_2025_holdout_predictions.csv", index=False
    )

    # Only after the untouched exam has been recorded do we refit a separate
    # production model on every eligible game through 2025.
    production = data[data.season.le(SETTINGS.holdout_season)].copy()
    production_models, production_medians = fit_models(production, SETTINGS.seed + 100)
    bundle = {
        **production_models,
        "features": V3_MATCHUP_FEATURES,
        "impute_values": production_medians,
        "production_seasons": tuple(sorted(production.season.astype(int).unique())),
        "production_games": int(len(production)),
        "validation_models": validation_models,
        "validation_impute_values": validation_medians,
        "metrics": metrics,
        "train_seasons": SETTINGS.train_seasons,
        "holdout_season": SETTINGS.holdout_season,
        "score_loss": "absolute_error",
    }
    SETTINGS.model_root.mkdir(parents=True, exist_ok=True)
    SETTINGS.audit_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, SETTINGS.model_root / "v3_structural_models.joblib")
    (SETTINGS.audit_root / "v3_2025_validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(
        f"Production refit complete: {len(production):,} games from "
        f"{bundle['production_seasons']}"
    )
    print(f"Model saved: {SETTINGS.model_root / 'v3_structural_models.joblib'}")


if __name__ == "__main__":
    main()
