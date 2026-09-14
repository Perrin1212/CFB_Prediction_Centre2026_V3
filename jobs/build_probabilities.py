from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


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
# IMPORTS
# ============================================================

from config.settings import PROCESSED_DATA_DIR

from engine.probabilities import (
    ProbabilityConfig,
    ProbabilityEngine,
)


# ============================================================
# FILES
# ============================================================

SIMULATION_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "simulation_history.csv"
)

PROBABILITY_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "probability_history.csv"
)

CANDIDATE_RESULTS_FILE = (
    PROCESSED_DATA_DIR
    / "probability_candidate_results.csv"
)

HOLDOUT_CALIBRATION_FILE = (
    PROCESSED_DATA_DIR
    / "probability_holdout_calibration.csv"
)

MODEL_METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "probability_model_metadata.json"
)


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


def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"✓ Saved {len(df):,} rows → {path}"
    )


def binary_metrics(
    actual: pd.Series,
    probability: pd.Series,
) -> dict[str, float]:

    y = (
        pd.to_numeric(
            actual,
            errors="coerce",
        )
        .astype(float)
        .to_numpy()
    )

    p = (
        pd.to_numeric(
            probability,
            errors="coerce",
        )
        .astype(float)
        .clip(
            1e-6,
            1.0 - 1e-6,
        )
        .to_numpy()
    )

    prediction = (
        p
        >
        0.50
    ).astype(int)

    accuracy = float(
        np.mean(
            prediction
            ==
            y.astype(int)
        )
    )

    brier = float(
        np.mean(
            (
                p
                -
                y
            )
            ** 2
        )
    )

    log_loss = float(
        -np.mean(
            y
            *
            np.log(
                p
            )
            +
            (
                1.0
                -
                y
            )
            *
            np.log(
                1.0
                -
                p
            )
        )
    )

    return {
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— PROBABILITY ARCHITECTURE VALIDATION"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    if not SIMULATION_HISTORY_FILE.exists():

        raise FileNotFoundError(
            "\nRequired file does not exist:\n"
            f"{SIMULATION_HISTORY_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING SIMULATION HISTORY"
    )

    history = pd.read_csv(
        SIMULATION_HISTORY_FILE,
        low_memory=False,
    )

    print(
        f"Simulation rows loaded: "
        f"{len(history):,}"
    )

    print(
        f"Unique games:           "
        f"{history['game_id'].nunique():,}"
    )

    print()

    print(
        "Season counts:"
    )

    print(
        history[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # CONFIG
    # ========================================================

    section(
        "PROBABILITY VALIDATION DESIGN"
    )

    config = ProbabilityConfig(
        train_season=2023,
        validation_season=2024,
        holdout_season=2025,

        logistic_c=1.0,
        max_iter=2000,

        probability_floor=0.001,
        probability_ceiling=0.999,
    )

    print(
        f"Candidate training season: "
        f"{config.train_season}"
    )

    print(
        f"Architecture validation:   "
        f"{config.validation_season}"
    )

    print(
        f"Forward holdout:           "
        f"{config.holdout_season}"
    )

    print()

    print(
        "Selection priority:"
    )

    print(
        "  1. Validation log loss"
    )

    print(
        "  2. Validation Brier score"
    )

    print(
        "  3. Validation accuracy"
    )

    print()

    print(
        "No hyperparameter grid is being run."
    )

    print(
        "All candidates use the same fixed logistic "
        "regularisation."
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    section(
        "PREPARING MODELLING DATA"
    )

    engine = ProbabilityEngine(
        config=config
    )

    data = (
        engine.prepare_dataset(
            history
        )
    )

    print(
        f"Usable non-tie games: "
        f"{len(data):,}"
    )

    print()

    print(
        "Usable games by season:"
    )

    print(
        data[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    train = (
        data[
            data[
                "season"
            ]
            ==
            config.train_season
        ]
        .copy()
    )

    validation = (
        data[
            data[
                "season"
            ]
            ==
            config.validation_season
        ]
        .copy()
    )

    holdout = (
        data[
            data[
                "season"
            ]
            ==
            config.holdout_season
        ]
        .copy()
    )

    if train.empty:

        raise RuntimeError(
            "2023 candidate-training dataset is empty."
        )

    if validation.empty:

        raise RuntimeError(
            "2024 architecture-validation dataset is empty."
        )

    if holdout.empty:

        raise RuntimeError(
            "2025 holdout dataset is empty."
        )

    print()

    print(
        f"Training games:   "
        f"{len(train):,}"
    )

    print(
        f"Validation games: "
        f"{len(validation):,}"
    )

    print(
        f"Holdout games:    "
        f"{len(holdout):,}"
    )

    # ========================================================
    # BASELINE METRICS
    # ========================================================

    section(
        "UNTRAINED BASELINE SIGNALS"
    )

    baseline_rows = []

    baseline_definitions = {
        "raw_elo": (
            "home_elo_win_probability"
        ),

        "raw_simulation": (
            "simulation_home_win_probability"
        ),
    }

    for name, column in baseline_definitions.items():

        metrics = binary_metrics(
            actual=(
                validation[
                    "actual_home_win"
                ]
            ),

            probability=(
                validation[
                    column
                ]
            ),
        )

        baseline_rows.append(
            {
                "model": name,
                "stage": "2024_validation",

                "features": column,

                "games": len(
                    validation
                ),

                **metrics,
            }
        )

        print(
            f"{name:<20} | "
            f"Accuracy {metrics['accuracy']:.2%} | "
            f"Brier {metrics['brier']:.4f} | "
            f"LogLoss {metrics['log_loss']:.4f}"
        )

    # ========================================================
    # CANDIDATE MODELS
    # ========================================================

    section(
        "CANDIDATE ARCHITECTURES — 2024 VALIDATION"
    )

    architectures = (
        engine.candidate_architectures()
    )

    candidate_results = []

    candidate_bundles = {}

    for (
        model_name,
        features,
    ) in architectures.items():

        bundle = engine.fit_model(
            data=train,

            model_name=model_name,

            features=features,
        )

        probability = (
            engine.predict(
                bundle=bundle,
                data=validation,
            )
        )

        metrics = (
            engine.evaluate_probabilities(
                data=validation,
                probability=probability,
            )
        )

        candidate_bundles[
            model_name
        ] = bundle

        candidate_results.append(
            {
                "model": model_name,

                "stage": "2024_validation",

                "features": (
                    ", ".join(
                        features
                    )
                ),

                **metrics,
            }
        )

        print(
            f"{model_name:<20} | "
            f"Accuracy {metrics['accuracy']:.2%} | "
            f"Brier {metrics['brier']:.4f} | "
            f"LogLoss {metrics['log_loss']:.4f}"
        )

    # ========================================================
    # SELECT ARCHITECTURE
    # ========================================================

    section(
        "ARCHITECTURE SELECTION"
    )

    candidates = pd.DataFrame(
        candidate_results
    )

    candidates = (
        candidates
        .sort_values(
            [
                "log_loss",
                "brier",
                "accuracy",
            ],

            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    winner_row = (
        candidates.iloc[0]
    )

    selected_name = str(
        winner_row[
            "model"
        ]
    )

    selected_features = (
        architectures[
            selected_name
        ]
    )

    print(
        "Ranked candidate models:"
    )

    print()

    for rank, row in enumerate(
        candidates.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.model:<20} | "
            f"Accuracy {row.accuracy:.2%} | "
            f"Brier {row.brier:.4f} | "
            f"LogLoss {row.log_loss:.4f}"
        )

    print()

    print(
        f"Selected architecture: "
        f"{selected_name}"
    )

    print()

    print(
        "Selected features:"
    )

    for feature in selected_features:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # REFIT SELECTED MODEL ON 2023 + 2024
    # ========================================================

    section(
        "REFITTING SELECTED ARCHITECTURE"
    )

    development = (
        data[
            data[
                "season"
            ].isin(
                [
                    config.train_season,
                    config.validation_season,
                ]
            )
        ]
        .copy()
    )

    print(
        f"Development games: "
        f"{len(development):,}"
    )

    final_bundle = (
        engine.fit_model(
            data=development,

            model_name=selected_name,

            features=selected_features,
        )
    )

    # ========================================================
    # 2025 HOLDOUT
    # ========================================================

    section(
        "2025 FORWARD HOLDOUT"
    )

    holdout_probability = (
        engine.predict(
            bundle=final_bundle,
            data=holdout,
        )
    )

    holdout[
        "v2_raw_home_win_probability"
    ] = (
        holdout_probability
    )

    holdout[
        "v2_raw_away_win_probability"
    ] = (
        1.0
        -
        holdout[
            "v2_raw_home_win_probability"
        ]
    )

    holdout_metrics = (
        engine.evaluate_probabilities(
            data=holdout,
            probability=holdout_probability,
        )
    )

    # --------------------------------------------------------
    # BASELINES ON SAME HOLDOUT
    # --------------------------------------------------------

    elo_holdout = binary_metrics(
        actual=(
            holdout[
                "actual_home_win"
            ]
        ),

        probability=(
            holdout[
                "home_elo_win_probability"
            ]
        ),
    )

    simulation_holdout = binary_metrics(
        actual=(
            holdout[
                "actual_home_win"
            ]
        ),

        probability=(
            holdout[
                "simulation_home_win_probability"
            ]
        ),
    )

    print(
        f"{'Metric':<20}"
        f"{'ELO':>12}"
        f"{'Simulation':>14}"
        f"{'V2 Model':>14}"
    )

    print(
        "-" * 60
    )

    print(
        f"{'Accuracy':<20}"
        f"{elo_holdout['accuracy']:>11.2%}"
        f"{simulation_holdout['accuracy']:>13.2%}"
        f"{holdout_metrics['accuracy']:>13.2%}"
    )

    print(
        f"{'Brier':<20}"
        f"{elo_holdout['brier']:>12.4f}"
        f"{simulation_holdout['brier']:>14.4f}"
        f"{holdout_metrics['brier']:>14.4f}"
    )

    print(
        f"{'Log loss':<20}"
        f"{elo_holdout['log_loss']:>12.4f}"
        f"{simulation_holdout['log_loss']:>14.4f}"
        f"{holdout_metrics['log_loss']:>14.4f}"
    )

    # ========================================================
    # MODEL COEFFICIENTS
    # ========================================================

    section(
        "SELECTED MODEL COEFFICIENTS"
    )

    coefficients = (
        final_bundle.model.coef_[
            0
        ]
    )

    coefficient_table = pd.DataFrame(
        {
            "feature": (
                selected_features
            ),

            "standardised_coefficient": (
                coefficients
            ),

            "absolute_coefficient": (
                np.abs(
                    coefficients
                )
            ),
        }
    )

    coefficient_table = (
        coefficient_table
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    for row in coefficient_table.itertuples(
        index=False
    ):

        print(
            f"{row.feature:<40} "
            f"{row.standardised_coefficient:>9.4f}"
        )

    print()

    print(
        "These coefficients are on standardised features."
    )

    print(
        "Large coefficients do not automatically imply "
        "independent causal importance."
    )

    # ========================================================
    # HOLDOUT CALIBRATION TABLE
    # ========================================================

    section(
        "2025 RAW PROBABILITY CALIBRATION"
    )

    calibration = (
        engine.calibration_table(
            actual=(
                holdout[
                    "actual_home_win"
                ]
            ),

            probability=(
                holdout[
                    "v2_raw_home_win_probability"
                ]
            ),

            bins=10,
        )
    )

    for row in calibration.itertuples(
        index=False
    ):

        if row.games == 0:

            continue

        print(
            f"{str(row.bin):<18} | "
            f"{int(row.games):>4} games | "
            f"Pred {row.average_probability:>6.2%} | "
            f"Actual {row.actual_home_win_rate:>6.2%} | "
            f"Error {row.calibration_error:>+7.2%}"
        )

    # ========================================================
    # CONFIDENCE BUCKETS
    # ========================================================

    section(
        "2025 CONFIDENCE PERFORMANCE"
    )

    holdout[
        "v2_predicted_probability"
    ] = np.maximum(
        holdout[
            "v2_raw_home_win_probability"
        ],

        holdout[
            "v2_raw_away_win_probability"
        ],
    )

    holdout[
        "v2_predicted_home"
    ] = (
        holdout[
            "v2_raw_home_win_probability"
        ]
        >
        0.50
    )

    holdout[
        "v2_correct"
    ] = (
        holdout[
            "v2_predicted_home"
        ]
        ==
        holdout[
            "actual_home_win"
        ].astype(bool)
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    for threshold in thresholds:

        subset = (
            holdout[
                holdout[
                    "v2_predicted_probability"
                ]
                >=
                threshold
            ]
        )

        if subset.empty:

            continue

        accuracy = (
            subset[
                "v2_correct"
            ]
            .mean()
        )

        print(
            f"≥ {threshold:.0%} confidence | "
            f"{len(subset):>4} games | "
            f"Accuracy {accuracy:>6.2%}"
        )

    # ========================================================
    # EXTREME PROBABILITY CHECK
    # ========================================================

    section(
        "EXTREME PROBABILITY CHECK"
    )

    p = (
        holdout[
            "v2_raw_home_win_probability"
        ]
    )

    print(
        f"Highest home win probability: "
        f"{p.max():.2%}"
    )

    print(
        f"Median home win probability:  "
        f"{p.median():.2%}"
    )

    print(
        f"Lowest home win probability:  "
        f"{p.min():.2%}"
    )

    extreme_95 = int(
        (
            (
                p
                >=
                0.95
            )
            |
            (
                p
                <=
                0.05
            )
        ).sum()
    )

    extreme_99 = int(
        (
            (
                p
                >=
                0.99
            )
            |
            (
                p
                <=
                0.01
            )
        ).sum()
    )

    print()

    print(
        f"2025 games ≥95% / ≤5%: "
        f"{extreme_95:,}"
    )

    print(
        f"2025 games ≥99% / ≤1%: "
        f"{extreme_99:,}"
    )

    # ========================================================
    # BUILD FULL HISTORY OUTPUT
    # ========================================================

    section(
        "BUILDING PROBABILITY HISTORY"
    )

    probability_history = (
        data.copy()
    )

    probability_history[
        "v2_raw_home_win_probability"
    ] = np.nan

    # --------------------------------------------------------
    # 2024 architecture-validation predictions
    #
    # Model trained only on 2023.
    # --------------------------------------------------------

    original_selected_bundle = (
        candidate_bundles[
            selected_name
        ]
    )

    validation_probability = (
        engine.predict(
            bundle=original_selected_bundle,
            data=validation,
        )
    )

    probability_history.loc[
        probability_history[
            "season"
        ]
        ==
        config.validation_season,

        "v2_raw_home_win_probability",
    ] = (
        validation_probability
    )

    # --------------------------------------------------------
    # 2025 holdout predictions
    #
    # Model trained on 2023 + 2024.
    # --------------------------------------------------------

    probability_history.loc[
        probability_history[
            "season"
        ]
        ==
        config.holdout_season,

        "v2_raw_home_win_probability",
    ] = (
        holdout_probability
    )

    probability_history[
        "v2_raw_away_win_probability"
    ] = (
        1.0
        -
        probability_history[
            "v2_raw_home_win_probability"
        ]
    )

    probability_history[
        "probability_architecture"
    ] = (
        selected_name
    )

    probability_history[
        "probability_is_out_of_sample"
    ] = (
        probability_history[
            "season"
        ]
        .isin(
            [
                config.validation_season,
                config.holdout_season,
            ]
        )
        &
        probability_history[
            "v2_raw_home_win_probability"
        ].notna()
    )

    print(
        f"Probability-history rows: "
        f"{len(probability_history):,}"
    )

    print(
        f"Out-of-sample probability rows: "
        f"{int(probability_history['probability_is_out_of_sample'].sum()):,}"
    )

    # ========================================================
    # RESULTS OUTPUT
    # ========================================================

    results_output = pd.concat(
        [
            pd.DataFrame(
                baseline_rows
            ),

            candidates,
        ],
        ignore_index=True,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "selected_architecture": (
            selected_name
        ),

        "selected_features": (
            selected_features
        ),

        "train_season": (
            config.train_season
        ),

        "validation_season": (
            config.validation_season
        ),

        "holdout_season": (
            config.holdout_season
        ),

        "logistic_c": (
            config.logistic_c
        ),

        "2025_holdout": {
            "games": int(
                holdout_metrics[
                    "games"
                ]
            ),

            "accuracy": float(
                holdout_metrics[
                    "accuracy"
                ]
            ),

            "brier": float(
                holdout_metrics[
                    "brier"
                ]
            ),

            "log_loss": float(
                holdout_metrics[
                    "log_loss"
                ]
            ),
        },

        "2025_elo": {
            "accuracy": float(
                elo_holdout[
                    "accuracy"
                ]
            ),

            "brier": float(
                elo_holdout[
                    "brier"
                ]
            ),

            "log_loss": float(
                elo_holdout[
                    "log_loss"
                ]
            ),
        },

        "2025_simulation": {
            "accuracy": float(
                simulation_holdout[
                    "accuracy"
                ]
            ),

            "brier": float(
                simulation_holdout[
                    "brier"
                ]
            ),

            "log_loss": float(
                simulation_holdout[
                    "log_loss"
                ]
            ),
        },

        "standardised_coefficients": {
            feature: float(
                coefficient
            )

            for feature, coefficient in zip(
                selected_features,
                final_bundle.model.coef_[
                    0
                ],
            )
        },

        "intercept": float(
            final_bundle.model.intercept_[
                0
            ]
        ),
    }

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING PROBABILITY OUTPUT"
    )

    save_csv(
        probability_history,
        PROBABILITY_HISTORY_FILE,
    )

    save_csv(
        results_output,
        CANDIDATE_RESULTS_FILE,
    )

    save_csv(
        calibration,
        HOLDOUT_CALIBRATION_FILE,
    )

    MODEL_METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MODEL_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"✓ Saved metadata → "
        f"{MODEL_METADATA_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "PROBABILITY ARCHITECTURE COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {PROBABILITY_HISTORY_FILE}"
    )

    print(
        f"  {CANDIDATE_RESULTS_FILE}"
    )

    print(
        f"  {HOLDOUT_CALIBRATION_FILE}"
    )

    print(
        f"  {MODEL_METADATA_FILE}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "  FOOTBALL MODEL"
    )

    print(
        "       ↓"
    )

    print(
        "  MONTE CARLO"
    )

    print(
        "       ↓"
    )

    print(
        "  COMPACT PROBABILITY MODEL"
    )

    print(
        "       ↓"
    )

    print(
        "  CALIBRATION"
    )

    print(
        "       ↓"
    )

    print(
        "  FINAL 2026 PREDICTIONS"
    )

    print()


if __name__ == "__main__":
    main()