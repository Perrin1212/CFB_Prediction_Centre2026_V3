from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import joblib
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

from engine.production_model import (
    ProductionProbabilityModel,
)


# ============================================================
# FILES
# ============================================================

SIMULATION_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "simulation_history.csv"
)

PRODUCTION_MODEL_FILE = (
    PROCESSED_DATA_DIR
    / "v2_probability_model.joblib"
)

MODEL_MANIFEST_FILE = (
    PROCESSED_DATA_DIR
    / "v2_model_manifest.json"
)

MODEL_TRAINING_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "v2_probability_training_rows.csv"
)


# ============================================================
# LOCKED ARCHITECTURE
# ============================================================

ARCHITECTURE_NAME = (
    "elo_matchup"
)

LOCKED_FEATURES = [
    "elo_logit",
    "offensive_matchup_difference",
]

CALIBRATION_METHOD = (
    "identity"
)

MODEL_VERSION = (
    "CFB-V2-2026.1"
)


# ============================================================
# HISTORICAL BENCHMARK
#
# IMPORTANT:
# These are the previously measured forward-holdout metrics.
# They are NOT recalculated after production refitting.
# ============================================================

HISTORICAL_BENCHMARK = {
    "season": 2025,

    "games": 956,

    "accuracy": 0.7280,

    "brier": 0.1754,

    "log_loss": 0.5227,

    "elo_accuracy": 0.7259,

    "elo_brier": 0.1855,

    "elo_log_loss": 0.5505,
}


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


def safe_float(
    value,
) -> float | None:

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):

            return None

        return value

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— PRODUCTION MODEL FREEZE"
    )

    # ========================================================
    # VALIDATE SOURCE
    # ========================================================

    if not SIMULATION_HISTORY_FILE.exists():

        raise FileNotFoundError(
            "\nRequired historical feature file missing:\n"
            f"{SIMULATION_HISTORY_FILE}"
        )

    # ========================================================
    # LOAD HISTORY
    # ========================================================

    section(
        "LOADING HISTORICAL MODEL DATA"
    )

    history = pd.read_csv(
        SIMULATION_HISTORY_FILE,
        low_memory=False,
    )

    print(
        f"Rows loaded:   "
        f"{len(history):,}"
    )

    print(
        f"Unique games:  "
        f"{history['game_id'].nunique():,}"
    )

    print()

    print(
        "Rows by season:"
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
    # PREPARE DATA
    # ========================================================

    section(
        "PREPARING LOCKED TRAINING DATA"
    )

    probability_engine = (
        ProbabilityEngine(
            ProbabilityConfig(
                train_season=2023,
                validation_season=2024,
                holdout_season=2025,

                logistic_c=1.0,
                max_iter=2000,

                probability_floor=0.001,
                probability_ceiling=0.999,
            )
        )
    )

    data = (
        probability_engine.prepare_dataset(
            history
        )
    )

    # --------------------------------------------------------
    # PRODUCTION TRAINING UNIVERSE
    #
    # Historical model architecture is already locked.
    # All 2023-2025 games are now allowed to train the
    # production model.
    # --------------------------------------------------------

    training = (
        data[
            data[
                "season"
            ]
            .isin(
                [
                    2023,
                    2024,
                    2025,
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if training.empty:

        raise RuntimeError(
            "Production training dataset is empty."
        )

    print(
        f"Production training games: "
        f"{len(training):,}"
    )

    print()

    print(
        "Training games by season:"
    )

    print(
        training[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    section(
        "LOCKED FEATURE VALIDATION"
    )

    missing_features = [
        feature
        for feature in LOCKED_FEATURES
        if feature not in training.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Locked production features missing: "
            f"{missing_features}"
        )

    missing_rows = (
        training[
            LOCKED_FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
    )

    print(
        f"Locked architecture: "
        f"{ARCHITECTURE_NAME}"
    )

    print()

    print(
        "Locked features:"
    )

    for feature in LOCKED_FEATURES:

        print(
            f"  - {feature}"
        )

    print()

    print(
        f"Rows missing locked features: "
        f"{int(missing_rows.sum()):,}"
    )

    if missing_rows.any():

        raise RuntimeError(
            "Production training data contains missing "
            "locked features."
        )

    # ========================================================
    # FIT FINAL PRODUCTION MODEL
    # ========================================================

    section(
        "TRAINING FINAL PRODUCTION PROBABILITY MODEL"
    )

    bundle = (
        probability_engine.fit_model(
            data=training,

            model_name=ARCHITECTURE_NAME,

            features=LOCKED_FEATURES,
        )
    )

    production_model = (
        ProductionProbabilityModel(
            architecture_name=(
                ARCHITECTURE_NAME
            ),

            features=list(
                LOCKED_FEATURES
            ),

            scaler=(
                bundle.scaler
            ),

            model=(
                bundle.model
            ),

            probability_floor=0.001,

            probability_ceiling=0.999,
        )
    )

    print(
        "Production model trained successfully."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "2025 is now part of production training."
    )

    print(
        "Therefore no post-refit 2025 metric will be "
        "reported as out-of-sample."
    )

    print()

    print(
        "The next genuine forward evaluation period is 2026."
    )

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    section(
        "FINAL PRODUCTION MODEL PARAMETERS"
    )

    coefficients = (
        bundle.model.coef_[
            0
        ]
    )

    for (
        feature,
        coefficient,
    ) in zip(
        LOCKED_FEATURES,
        coefficients,
    ):

        print(
            f"{feature:<40} "
            f"{coefficient:>10.4f}"
        )

    intercept = float(
        bundle.model.intercept_[
            0
        ]
    )

    print(
        f"{'intercept':<40} "
        f"{intercept:>10.4f}"
    )

    print()

    print(
        "Coefficients are based on standardised features."
    )

    # ========================================================
    # TRAINING DISTRIBUTION
    # ========================================================

    section(
        "PRODUCTION TRAINING DISTRIBUTION"
    )

    training_probability = (
        production_model.predict_home_probability(
            training
        )
    )

    print(
        f"Highest home probability: "
        f"{training_probability.max():.2%}"
    )

    print(
        f"Median home probability:  "
        f"{np.median(training_probability):.2%}"
    )

    print(
        f"Lowest home probability:  "
        f"{training_probability.min():.2%}"
    )

    extreme_95 = int(
        (
            (
                training_probability
                >=
                0.95
            )
            |
            (
                training_probability
                <=
                0.05
            )
        ).sum()
    )

    extreme_99 = int(
        (
            (
                training_probability
                >=
                0.99
            )
            |
            (
                training_probability
                <=
                0.01
            )
        ).sum()
    )

    print()

    print(
        f"Training games ≥95% / ≤5%: "
        f"{extreme_95:,}"
    )

    print(
        f"Training games ≥99% / ≤1%: "
        f"{extreme_99:,}"
    )

    # ========================================================
    # SAVE TRAINING AUDIT DATA
    # ========================================================

    section(
        "SAVING MODEL ARTIFACTS"
    )

    MODEL_TRAINING_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_output_columns = [
        column
        for column in [
            "game_id",
            "season",
            "week",
            "start_date",

            "home_team",
            "away_team",

            "home_points",
            "away_points",

            "actual_home_win",

            "home_elo_win_probability",
            "elo_logit",

            "offensive_matchup_difference",
        ]
        if column in training.columns
    ]

    training[
        training_output_columns
    ].to_csv(
        MODEL_TRAINING_DATA_FILE,
        index=False,
    )

    print(
        f"✓ Saved {len(training):,} training rows → "
        f"{MODEL_TRAINING_DATA_FILE}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    production_model.save(
        PRODUCTION_MODEL_FILE
    )

    print(
        f"✓ Saved production model → "
        f"{PRODUCTION_MODEL_FILE}"
    )

    # ========================================================
    # VERIFY MODEL RELOAD
    # ========================================================

    reloaded_model = (
        ProductionProbabilityModel.load(
            PRODUCTION_MODEL_FILE
        )
    )

    verification_sample = (
        training.tail(
            min(
                25,
                len(training),
            )
        )
        .copy()
    )

    original_probability = (
        production_model.predict_home_probability(
            verification_sample
        )
    )

    reloaded_probability = (
        reloaded_model.predict_home_probability(
            verification_sample
        )
    )

    maximum_reload_error = float(
        np.max(
            np.abs(
                original_probability
                -
                reloaded_probability
            )
        )
    )

    print(
        f"✓ Model reload verification error: "
        f"{maximum_reload_error:.12f}"
    )

    if maximum_reload_error > 1e-12:

        raise RuntimeError(
            "Production model reload verification failed."
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    frozen_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    manifest = {
        "model_version": (
            MODEL_VERSION
        ),

        "model_name": (
            "CFB Prediction Centre 2026 V2"
        ),

        "frozen_at_utc": (
            frozen_at
        ),

        "status": (
            "historical_architecture_frozen"
        ),

        "training_seasons": [
            2023,
            2024,
            2025,
        ],

        "production_training_games": int(
            len(training)
        ),

        "architecture": {
            "name": (
                ARCHITECTURE_NAME
            ),

            "features": list(
                LOCKED_FEATURES
            ),

            "calibration": (
                CALIBRATION_METHOD
            ),
        },

        "historical_forward_benchmark": (
            HISTORICAL_BENCHMARK
        ),

        "production_coefficients": {
            feature: float(
                coefficient
            )

            for (
                feature,
                coefficient,
            ) in zip(
                LOCKED_FEATURES,
                coefficients,
            )
        },

        "production_intercept": (
            intercept
        ),

        "scaler_mean": [
            float(
                value
            )
            for value in bundle.scaler.mean_
        ],

        "scaler_scale": [
            float(
                value
            )
            for value in bundle.scaler.scale_
        ],

        "probability_floor": (
            production_model.probability_floor
        ),

        "probability_ceiling": (
            production_model.probability_ceiling
        ),

        "next_true_forward_test": (
            2026
        ),

        "market_data_used_in_model": False,

        "notes": [
            (
                "Architecture selected using historical "
                "chronological validation."
            ),

            (
                "2025 benchmark was measured before "
                "production refit."
            ),

            (
                "After architecture freeze, 2025 was added "
                "to production training."
            ),

            (
                "No additional probability calibration "
                "is applied because identity calibration "
                "won development validation."
            ),

            (
                "2026 must be treated as genuine forward "
                "model evaluation."
            ),
        ],
    }

    with open(
        MODEL_MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        f"✓ Saved model manifest → "
        f"{MODEL_MANIFEST_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "V2 HISTORICAL MODEL FREEZE COMPLETE"
    )

    print(
        f"Model version:       "
        f"{MODEL_VERSION}"
    )

    print(
        f"Architecture:        "
        f"{ARCHITECTURE_NAME}"
    )

    print(
        f"Calibration:         "
        f"{CALIBRATION_METHOD}"
    )

    print(
        f"Training games:      "
        f"{len(training):,}"
    )

    print(
        "Training seasons:    "
        "2023, 2024, 2025"
    )

    print()

    print(
        "Historical architecture is now FROZEN."
    )

    print()

    print(
        "Do not tune the model against 2026 results."
    )

    print(
        "2026 is the genuine forward evaluation period."
    )

    print()

    print(
        "Next architecture step:"
    )

    print(
        "  Load 2026 schedule/results"
    )

    print(
        "       ↓"
    )

    print(
        "  Advance ELO + team states chronologically"
    )

    print(
        "       ↓"
    )

    print(
        "  Build 2026 game matchup features"
    )

    print(
        "       ↓"
    )

    print(
        "  Run frozen production probability model"
    )

    print(
        "       ↓"
    )

    print(
        "  Compare predictions against completed "
        "2026 games only after prediction"
    )


if __name__ == "__main__":

    main()