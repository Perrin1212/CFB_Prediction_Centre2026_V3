from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SCORING_HISTORY_PATH = (
    PROCESSED_DIR
    / "scoring_history.csv"
)

CANDIDATE_RESULTS_PATH = (
    PROCESSED_DIR
    / "score_model_candidate_results.csv"
)

HOLDOUT_RESULTS_PATH = (
    PROCESSED_DIR
    / "score_model_2025_results.csv"
)

HOLDOUT_PREDICTIONS_PATH = (
    PROCESSED_DIR
    / "score_model_2025_predictions.csv"
)

METADATA_PATH = (
    PROCESSED_DIR
    / "score_model_selection_metadata.json"
)


# ============================================================
# MODEL SPLITS
# ============================================================

SELECTION_TRAIN_SEASONS = [
    2023,
]

SELECTION_VALIDATION_SEASON = 2024

DEVELOPMENT_REFIT_SEASONS = [
    2023,
    2024,
]

DEVELOPMENT_EVALUATION_SEASON = 2025


# ============================================================
# RIDGE CONFIG
# ============================================================

RIDGE_ALPHA = 1.0


# ============================================================
# HELPERS
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_history() -> pd.DataFrame:

    if not SCORING_HISTORY_PATH.exists():

        raise FileNotFoundError(
            "Missing scoring history:\n"
            f"{SCORING_HISTORY_PATH}\n\n"
            "Run the historical scoring build first."
        )

    frame = pd.read_csv(
        SCORING_HISTORY_PATH,
        low_memory=False,
    )

    if "start_date" in frame.columns:

        frame["start_date"] = pd.to_datetime(
            frame["start_date"],
            errors="coerce",
            utc=True,
        )

    return frame


def numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def mae(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual_array)
        &
        np.isfinite(predicted_array)
    )

    if not valid.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(
                actual_array[valid]
                -
                predicted_array[valid]
            )
        )
    )


def rmse(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual_array)
        &
        np.isfinite(predicted_array)
    )

    if not valid.any():
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                (
                    actual_array[valid]
                    -
                    predicted_array[valid]
                )
                ** 2
            )
        )
    )


def correlation(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual_array)
        &
        np.isfinite(predicted_array)
    )

    if valid.sum() < 3:
        return float("nan")

    return float(
        np.corrcoef(
            actual_array[valid],
            predicted_array[valid],
        )[0, 1]
    )


def score_winner_accuracy(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
) -> float:

    actual_margin = np.asarray(
        actual_margin,
        dtype=float,
    )

    predicted_margin = np.asarray(
        predicted_margin,
        dtype=float,
    )

    valid = (
        np.isfinite(actual_margin)
        &
        np.isfinite(predicted_margin)
        &
        (actual_margin != 0)
    )

    if not valid.any():
        return float("nan")

    actual_home_win = (
        actual_margin[valid]
        >
        0
    )

    predicted_home_win = (
        predicted_margin[valid]
        >
        0
    )

    return float(
        np.mean(
            actual_home_win
            ==
            predicted_home_win
        )
    )


# ============================================================
# PREPARE HISTORICAL TARGETS
# ============================================================

def prepare_history(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "season",
        "home_points",
        "away_points",
        "expected_home_points",
        "expected_away_points",
        "expected_home_margin",
        "expected_total_points",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise ValueError(
            "scoring_history.csv is missing "
            f"required columns: {missing}"
        )

    data = frame.copy()

    for column in required:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[
        "actual_home_margin"
    ] = (
        data["home_points"]
        -
        data["away_points"]
    )

    data[
        "actual_total_points"
    ] = (
        data["home_points"]
        +
        data["away_points"]
    )

    data[
        "actual_home_win"
    ] = (
        data[
            "actual_home_margin"
        ]
        >
        0
    ).astype(int)

    data = data[
        data["season"].isin(
            [
                2023,
                2024,
                2025,
            ]
        )
    ].copy()

    data = data[
        data[
            [
                "home_points",
                "away_points",
                "expected_home_margin",
                "expected_total_points",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    ].copy()

    return data


# ============================================================
# CANDIDATE ARCHITECTURES
# ============================================================

def available_features(
    data: pd.DataFrame,
    requested: list[str],
) -> list[str]:

    return [
        feature
        for feature in requested
        if feature in data.columns
    ]


def candidate_definitions(
    data: pd.DataFrame,
) -> dict[
    str,
    dict[
        str,
        list[str],
    ],
]:

    candidates: dict[
        str,
        dict[
            str,
            list[str],
        ],
    ] = {}

    # --------------------------------------------------------
    # 1. RAW SCORING ENGINE
    # --------------------------------------------------------

    candidates[
        "raw_scoring"
    ] = {
        "margin_features": [
            "expected_home_margin",
        ],
        "total_features": [
            "expected_total_points",
        ],
    }

    # --------------------------------------------------------
    # 2. RAW + ELO
    # --------------------------------------------------------

    margin_features = available_features(
        data,
        [
            "expected_home_margin",
            "elo_difference",
        ],
    )

    if len(
        margin_features
    ) >= 2:

        candidates[
            "scoring_elo"
        ] = {
            "margin_features": (
                margin_features
            ),
            "total_features": [
                "expected_total_points",
            ],
        }

    # --------------------------------------------------------
    # 3. RAW + MATCHUP
    # --------------------------------------------------------

    margin_features = available_features(
        data,
        [
            "expected_home_margin",
            "offensive_matchup_difference",
        ],
    )

    if len(
        margin_features
    ) >= 2:

        candidates[
            "scoring_matchup"
        ] = {
            "margin_features": (
                margin_features
            ),
            "total_features": [
                "expected_total_points",
            ],
        }

    # --------------------------------------------------------
    # 4. RAW + ELO + MATCHUP
    # --------------------------------------------------------

    margin_features = available_features(
        data,
        [
            "expected_home_margin",
            "elo_difference",
            "offensive_matchup_difference",
        ],
    )

    if len(
        margin_features
    ) >= 3:

        candidates[
            "scoring_elo_matchup"
        ] = {
            "margin_features": (
                margin_features
            ),
            "total_features": [
                "expected_total_points",
            ],
        }

    # --------------------------------------------------------
    # 5. COMPACT FULL MARGIN
    # --------------------------------------------------------

    margin_features = available_features(
        data,
        [
            "expected_home_margin",
            "elo_difference",
            "offensive_matchup_difference",
            "adjusted_strength_difference",
        ],
    )

    if len(
        margin_features
    ) >= 3:

        candidates[
            "compact_margin"
        ] = {
            "margin_features": (
                margin_features
            ),
            "total_features": [
                "expected_total_points",
            ],
        }

    # --------------------------------------------------------
    # 6. COMPACT MARGIN + ENVIRONMENT TOTAL
    # --------------------------------------------------------

    margin_features = available_features(
        data,
        [
            "expected_home_margin",
            "elo_difference",
            "offensive_matchup_difference",
            "adjusted_strength_difference",
        ],
    )

    total_features = available_features(
        data,
        [
            "expected_total_points",
            "expected_total_plays",
            "expected_total_drives",
            "game_pace_index",
        ],
    )

    if (
        len(
            margin_features
        )
        >=
        3
        and
        len(
            total_features
        )
        >=
        2
    ):

        candidates[
            "compact_full"
        ] = {
            "margin_features": (
                margin_features
            ),
            "total_features": (
                total_features
            ),
        }

    return candidates


# ============================================================
# PIPELINE
# ============================================================

def build_ridge_pipeline(
    features: list[str],
) -> Pipeline:

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                features,
            ),
        ],
        remainder="drop",
    )

    model = Ridge(
        alpha=RIDGE_ALPHA,
    )

    return Pipeline(
        steps=[
            (
                "features",
                transformer,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# FIT / PREDICT ONE CANDIDATE
# ============================================================

def fit_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    definition: dict[
        str,
        list[str],
    ],
) -> dict[str, Any]:

    margin_features = (
        definition[
            "margin_features"
        ]
    )

    total_features = (
        definition[
            "total_features"
        ]
    )

    # --------------------------------------------------------
    # RAW BASELINE
    # --------------------------------------------------------

    if (
        margin_features
        ==
        [
            "expected_home_margin",
        ]
        and
        total_features
        ==
        [
            "expected_total_points",
        ]
    ):

        predicted_margin = (
            test[
                "expected_home_margin"
            ]
            .to_numpy(
                dtype=float
            )
        )

        predicted_total = (
            test[
                "expected_total_points"
            ]
            .to_numpy(
                dtype=float
            )
        )

        return {
            "margin_model": None,
            "total_model": None,
            "predicted_margin": (
                predicted_margin
            ),
            "predicted_total": (
                predicted_total
            ),
        }

    # --------------------------------------------------------
    # MARGIN MODEL
    # --------------------------------------------------------

    margin_train = train[
        margin_features
        +
        [
            "actual_home_margin",
        ]
    ].dropna()

    margin_model = build_ridge_pipeline(
        margin_features
    )

    margin_model.fit(
        margin_train[
            margin_features
        ],
        margin_train[
            "actual_home_margin"
        ],
    )

    predicted_margin = (
        margin_model.predict(
            test[
                margin_features
            ]
        )
    )

    # --------------------------------------------------------
    # TOTAL MODEL
    # --------------------------------------------------------

    if (
        total_features
        ==
        [
            "expected_total_points",
        ]
    ):

        predicted_total = (
            test[
                "expected_total_points"
            ]
            .to_numpy(
                dtype=float
            )
        )

        total_model = None

    else:

        total_train = train[
            total_features
            +
            [
                "actual_total_points",
            ]
        ].dropna()

        total_model = (
            build_ridge_pipeline(
                total_features
            )
        )

        total_model.fit(
            total_train[
                total_features
            ],
            total_train[
                "actual_total_points"
            ],
        )

        predicted_total = (
            total_model.predict(
                test[
                    total_features
                ]
            )
        )

    return {
        "margin_model": (
            margin_model
        ),
        "total_model": (
            total_model
        ),
        "predicted_margin": (
            predicted_margin
        ),
        "predicted_total": (
            predicted_total
        ),
    }


# ============================================================
# SCORE RECONSTRUCTION
# ============================================================

def scores_from_margin_total(
    predicted_margin: np.ndarray,
    predicted_total: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    predicted_margin = np.asarray(
        predicted_margin,
        dtype=float,
    )

    predicted_total = np.asarray(
        predicted_total,
        dtype=float,
    )

    home_score = (
        predicted_total
        +
        predicted_margin
    ) / 2.0

    away_score = (
        predicted_total
        -
        predicted_margin
    ) / 2.0

    home_score = np.clip(
        home_score,
        0.0,
        70.0,
    )

    away_score = np.clip(
        away_score,
        0.0,
        70.0,
    )

    return (
        home_score,
        away_score,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    frame: pd.DataFrame,
    predicted_margin: np.ndarray,
    predicted_total: np.ndarray,
) -> dict[str, float]:

    (
        predicted_home,
        predicted_away,
    ) = scores_from_margin_total(
        predicted_margin,
        predicted_total,
    )

    actual_margin = (
        frame[
            "actual_home_margin"
        ]
        .to_numpy(
            dtype=float
        )
    )

    actual_total = (
        frame[
            "actual_total_points"
        ]
        .to_numpy(
            dtype=float
        )
    )

    actual_home = (
        frame[
            "home_points"
        ]
        .to_numpy(
            dtype=float
        )
    )

    actual_away = (
        frame[
            "away_points"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "games": int(
            len(frame)
        ),
        "margin_mae": (
            mae(
                actual_margin,
                predicted_margin,
            )
        ),
        "margin_rmse": (
            rmse(
                actual_margin,
                predicted_margin,
            )
        ),
        "margin_correlation": (
            correlation(
                actual_margin,
                predicted_margin,
            )
        ),
        "total_mae": (
            mae(
                actual_total,
                predicted_total,
            )
        ),
        "total_rmse": (
            rmse(
                actual_total,
                predicted_total,
            )
        ),
        "total_correlation": (
            correlation(
                actual_total,
                predicted_total,
            )
        ),
        "home_score_mae": (
            mae(
                actual_home,
                predicted_home,
            )
        ),
        "away_score_mae": (
            mae(
                actual_away,
                predicted_away,
            )
        ),
        "winner_accuracy": (
            score_winner_accuracy(
                actual_margin,
                predicted_margin,
            )
        ),
    }


# ============================================================
# SELECTION SCORE
# ============================================================

def selection_score(
    metrics: dict[str, float],
) -> float:
    """
    Primary focus is margin accuracy.

    Total accuracy matters too, but slightly less because
    relative team strength / winner consistency is currently
    our largest weakness.

    Lower is better.
    """

    return float(
        (
            metrics[
                "margin_mae"
            ]
            *
            0.65
        )
        +
        (
            metrics[
                "total_mae"
            ]
            *
            0.35
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- SCORE MODEL SELECTION"
    )

    print(
        "Goal:"
    )

    print(
        "Determine whether the raw scoring engine "
        "benefits from a compact historically validated "
        "relative-strength correction."
    )

    print()

    print(
        "NO 2026 games are used for model selection."
    )

    print()

    print(
        "Selection:"
    )

    print(
        "2023 -> train"
    )

    print(
        "2024 -> architecture validation"
    )

    print()

    print(
        "Then:"
    )

    print(
        "2023 + 2024 -> refit"
    )

    print(
        "2025 -> development evaluation"
    )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "1. LOAD SCORING HISTORY"
    )

    raw_history = load_history()

    history = prepare_history(
        raw_history
    )

    print(
        f"Rows loaded:       "
        f"{len(raw_history):,}"
    )

    print(
        f"Usable rows:       "
        f"{len(history):,}"
    )

    for season in [
        2023,
        2024,
        2025,
    ]:

        count = int(
            (
                history[
                    "season"
                ]
                ==
                season
            )
            .sum()
        )

        print(
            f"{season}:             "
            f"{count:,}"
        )

    # ========================================================
    # CANDIDATES
    # ========================================================

    section(
        "2. AVAILABLE CANDIDATE ARCHITECTURES"
    )

    candidates = candidate_definitions(
        history
    )

    for name, definition in (
        candidates.items()
    ):

        print()

        print(
            name
        )

        print(
            "  Margin:"
        )

        for feature in (
            definition[
                "margin_features"
            ]
        ):

            print(
                f"    - {feature}"
            )

        print(
            "  Total:"
        )

        for feature in (
            definition[
                "total_features"
            ]
        ):

            print(
                f"    - {feature}"
            )

    if len(
        candidates
    ) < 2:

        raise RuntimeError(
            "Not enough candidate architectures "
            "could be constructed from scoring_history.csv."
        )

    # ========================================================
    # SELECTION SPLIT
    # ========================================================

    section(
        "3. 2024 ARCHITECTURE SELECTION"
    )

    selection_train = history[
        history[
            "season"
        ]
        .isin(
            SELECTION_TRAIN_SEASONS
        )
    ].copy()

    selection_validation = history[
        history[
            "season"
        ]
        ==
        SELECTION_VALIDATION_SEASON
    ].copy()

    print(
        f"Training rows:   "
        f"{len(selection_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(selection_validation):,}"
    )

    candidate_rows: list[
        dict[str, Any]
    ] = []

    for (
        candidate_name,
        definition,
    ) in candidates.items():

        fit = fit_candidate(
            train=selection_train,
            test=selection_validation,
            definition=definition,
        )

        metrics = evaluate_predictions(
            frame=selection_validation,
            predicted_margin=(
                fit[
                    "predicted_margin"
                ]
            ),
            predicted_total=(
                fit[
                    "predicted_total"
                ]
            ),
        )

        combined_score = (
            selection_score(
                metrics
            )
        )

        row = {
            "candidate": (
                candidate_name
            ),
            "selection_score": (
                combined_score
            ),
            "margin_features": (
                "|".join(
                    definition[
                        "margin_features"
                    ]
                )
            ),
            "total_features": (
                "|".join(
                    definition[
                        "total_features"
                    ]
                )
            ),
        }

        row.update(
            metrics
        )

        candidate_rows.append(
            row
        )

    candidate_results = (
        pd.DataFrame(
            candidate_rows
        )
        .sort_values(
            [
                "selection_score",
                "margin_mae",
                "total_mae",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    print()

    display_columns = [
        "candidate",
        "selection_score",
        "margin_mae",
        "total_mae",
        "home_score_mae",
        "away_score_mae",
        "winner_accuracy",
        "margin_correlation",
    ]

    print(
        candidate_results[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "selection_score": (
                    lambda x: f"{x:.3f}"
                ),
                "margin_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "total_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "home_score_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "away_score_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "winner_accuracy": (
                    lambda x: f"{x:.2%}"
                ),
                "margin_correlation": (
                    lambda x: f"{x:.3f}"
                ),
            },
        )
    )

    candidate_results.to_csv(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    selected_name = str(
        candidate_results.iloc[0][
            "candidate"
        ]
    )

    selected_definition = (
        candidates[
            selected_name
        ]
    )

    print()

    print(
        "SELECTED ARCHITECTURE:"
    )

    print(
        f"  {selected_name}"
    )

    print()

    print(
        "Margin features:"
    )

    for feature in (
        selected_definition[
            "margin_features"
        ]
    ):

        print(
            f"  - {feature}"
        )

    print()

    print(
        "Total features:"
    )

    for feature in (
        selected_definition[
            "total_features"
        ]
    ):

        print(
            f"  - {feature}"
        )

    # ========================================================
    # REFIT DEVELOPMENT MODEL
    # ========================================================

    section(
        "4. REFIT ON 2023 + 2024"
    )

    development_train = history[
        history[
            "season"
        ]
        .isin(
            DEVELOPMENT_REFIT_SEASONS
        )
    ].copy()

    development_holdout = history[
        history[
            "season"
        ]
        ==
        DEVELOPMENT_EVALUATION_SEASON
    ].copy()

    print(
        f"Refit rows:    "
        f"{len(development_train):,}"
    )

    print(
        f"2025 rows:     "
        f"{len(development_holdout):,}"
    )

    selected_fit = fit_candidate(
        train=development_train,
        test=development_holdout,
        definition=selected_definition,
    )

    selected_metrics = (
        evaluate_predictions(
            frame=development_holdout,
            predicted_margin=(
                selected_fit[
                    "predicted_margin"
                ]
            ),
            predicted_total=(
                selected_fit[
                    "predicted_total"
                ]
            ),
        )
    )

    # ========================================================
    # RAW 2025 BENCHMARK
    # ========================================================

    raw_fit = fit_candidate(
        train=development_train,
        test=development_holdout,
        definition=(
            candidates[
                "raw_scoring"
            ]
        ),
    )

    raw_metrics = evaluate_predictions(
        frame=development_holdout,
        predicted_margin=(
            raw_fit[
                "predicted_margin"
            ]
        ),
        predicted_total=(
            raw_fit[
                "predicted_total"
            ]
        ),
    )

    # ========================================================
    # HOLDOUT COMPARISON
    # ========================================================

    section(
        "5. 2025 DEVELOPMENT EVALUATION"
    )

    evaluation_rows = []

    for (
        name,
        metrics,
    ) in [
        (
            "raw_scoring",
            raw_metrics,
        ),
        (
            selected_name,
            selected_metrics,
        ),
    ]:

        row = {
            "model": name,
        }

        row.update(
            metrics
        )

        evaluation_rows.append(
            row
        )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    print(
        evaluation[
            [
                "model",
                "games",
                "margin_mae",
                "total_mae",
                "home_score_mae",
                "away_score_mae",
                "winner_accuracy",
                "margin_correlation",
            ]
        ].to_string(
            index=False,
            formatters={
                "margin_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "total_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "home_score_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "away_score_mae": (
                    lambda x: f"{x:.3f}"
                ),
                "winner_accuracy": (
                    lambda x: f"{x:.2%}"
                ),
                "margin_correlation": (
                    lambda x: f"{x:.3f}"
                ),
            },
        )
    )

    evaluation.to_csv(
        HOLDOUT_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # SAVE 2025 PREDICTIONS
    # ========================================================

    predicted_margin = (
        selected_fit[
            "predicted_margin"
        ]
    )

    predicted_total = (
        selected_fit[
            "predicted_total"
        ]
    )

    (
        predicted_home_points,
        predicted_away_points,
    ) = scores_from_margin_total(
        predicted_margin,
        predicted_total,
    )

    holdout_predictions = (
        development_holdout.copy()
    )

    holdout_predictions[
        "selected_score_model"
    ] = selected_name

    holdout_predictions[
        "final_predicted_home_margin"
    ] = predicted_margin

    holdout_predictions[
        "final_predicted_total"
    ] = predicted_total

    holdout_predictions[
        "final_predicted_home_points"
    ] = predicted_home_points

    holdout_predictions[
        "final_predicted_away_points"
    ] = predicted_away_points

    holdout_predictions[
        "raw_margin_error_abs"
    ] = np.abs(
        holdout_predictions[
            "actual_home_margin"
        ]
        -
        holdout_predictions[
            "expected_home_margin"
        ]
    )

    holdout_predictions[
        "final_margin_error_abs"
    ] = np.abs(
        holdout_predictions[
            "actual_home_margin"
        ]
        -
        holdout_predictions[
            "final_predicted_home_margin"
        ]
    )

    holdout_predictions.to_csv(
        HOLDOUT_PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "selection_train_seasons": (
            SELECTION_TRAIN_SEASONS
        ),
        "selection_validation_season": (
            SELECTION_VALIDATION_SEASON
        ),
        "development_refit_seasons": (
            DEVELOPMENT_REFIT_SEASONS
        ),
        "development_evaluation_season": (
            DEVELOPMENT_EVALUATION_SEASON
        ),
        "ridge_alpha": (
            RIDGE_ALPHA
        ),
        "selected_candidate": (
            selected_name
        ),
        "margin_features": (
            selected_definition[
                "margin_features"
            ]
        ),
        "total_features": (
            selected_definition[
                "total_features"
            ]
        ),
        "selection_metric": (
            "0.65 * margin_mae + "
            "0.35 * total_mae"
        ),
        "selected_2025_metrics": (
            selected_metrics
        ),
        "raw_2025_metrics": (
            raw_metrics
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    section(
        "6. MODEL-SELECTION SUMMARY"
    )

    raw_margin = (
        raw_metrics[
            "margin_mae"
        ]
    )

    selected_margin = (
        selected_metrics[
            "margin_mae"
        ]
    )

    raw_winner = (
        raw_metrics[
            "winner_accuracy"
        ]
    )

    selected_winner = (
        selected_metrics[
            "winner_accuracy"
        ]
    )

    margin_change = (
        raw_margin
        -
        selected_margin
    )

    winner_change = (
        selected_winner
        -
        raw_winner
    )

    print(
        f"Selected architecture: "
        f"{selected_name}"
    )

    print()

    print(
        f"2025 raw margin MAE:      "
        f"{raw_margin:.3f}"
    )

    print(
        f"2025 selected margin MAE: "
        f"{selected_margin:.3f}"
    )

    print(
        f"Margin MAE improvement:   "
        f"{margin_change:+.3f}"
    )

    print()

    print(
        f"2025 raw winner accuracy: "
        f"{raw_winner:.2%}"
    )

    print(
        f"2025 selected accuracy:   "
        f"{selected_winner:.2%}"
    )

    print(
        f"Accuracy change:           "
        f"{winner_change:+.2%}"
    )

    print()

    if (
        selected_name
        ==
        "raw_scoring"
    ):

        print(
            "RESULT:"
        )

        print(
            "The historical evidence does NOT "
            "justify a score correction."
        )

        print()

        print(
            "Keep the current raw scoring engine."
        )

    elif (
        selected_margin
        <
        raw_margin
    ):

        print(
            "RESULT:"
        )

        print(
            "The selected correction improved "
            "historical margin accuracy."
        )

        print()

        print(
            "Next step: freeze this score architecture "
            "using all 2023-2025 history."
        )

    else:

        print(
            "RESULT:"
        )

        print(
            "The selected 2024 architecture did not "
            "improve 2025 margin MAE."
        )

        print()

        print(
            "Do NOT freeze it yet."
        )

        print(
            "We should retain the raw score model "
            "or reassess the candidate architecture."
        )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {CANDIDATE_RESULTS_PATH}"
    )

    print(
        f"  {HOLDOUT_RESULTS_PATH}"
    )

    print(
        f"  {HOLDOUT_PREDICTIONS_PATH}"
    )

    print(
        f"  {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()