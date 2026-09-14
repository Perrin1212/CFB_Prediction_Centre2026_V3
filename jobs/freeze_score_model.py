from __future__ import annotations

import json
import sys
from pathlib import Path

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

SELECTION_METADATA_PATH = (
    PROCESSED_DIR
    / "score_model_selection_metadata.json"
)

MODEL_PATH = (
    PROCESSED_DIR
    / "v2_score_model.joblib"
)

MANIFEST_PATH = (
    PROCESSED_DIR
    / "v2_score_model_manifest.json"
)

TRAINING_ROWS_PATH = (
    PROCESSED_DIR
    / "v2_score_model_training_rows.csv"
)


# ============================================================
# FROZEN ARCHITECTURE
# ============================================================

MODEL_NAME = "compact_full"

TRAINING_SEASONS = [
    2023,
    2024,
    2025,
]

RIDGE_ALPHA = 1.0


MARGIN_FEATURES = [
    "expected_home_margin",
    "elo_difference",
    "offensive_matchup_difference",
    "adjusted_strength_difference",
]


TOTAL_FEATURES = [
    "expected_total_points",
    "expected_total_plays",
    "expected_total_drives",
    "game_pace_index",
]


# ============================================================
# DISPLAY
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# LOAD
# ============================================================

def load_scoring_history() -> pd.DataFrame:

    if not SCORING_HISTORY_PATH.exists():

        raise FileNotFoundError(
            "Missing scoring history:\n"
            f"{SCORING_HISTORY_PATH}"
        )

    frame = pd.read_csv(
        SCORING_HISTORY_PATH,
        low_memory=False,
    )

    return frame


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_pipeline(
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

    ridge = Ridge(
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
                ridge,
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- FREEZE SCORE MODEL"
    )

    print(
        "Frozen architecture:"
    )

    print(
        MODEL_NAME
    )

    print()

    print(
        "Training seasons:"
    )

    print(
        ", ".join(
            str(season)
            for season in TRAINING_SEASONS
        )
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Architecture selection is already complete."
    )

    print(
        "This job does NOT perform further tuning."
    )

    print(
        "It only refits the selected score architecture "
        "on all available 2023-2025 historical data."
    )

    # ========================================================
    # VERIFY SELECTION METADATA
    # ========================================================

    section(
        "1. VERIFY SELECTED ARCHITECTURE"
    )

    if not SELECTION_METADATA_PATH.exists():

        raise FileNotFoundError(
            "Missing score-model selection metadata:\n"
            f"{SELECTION_METADATA_PATH}"
        )

    with open(
        SELECTION_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        selection_metadata = json.load(
            file
        )

    selected_candidate = (
        selection_metadata.get(
            "selected_candidate"
        )
    )

    print(
        f"Selected candidate from validation: "
        f"{selected_candidate}"
    )

    if (
        selected_candidate
        !=
        MODEL_NAME
    ):

        raise RuntimeError(
            "Selected score architecture does not match "
            "the architecture this freeze job expects.\n\n"
            f"Expected: {MODEL_NAME}\n"
            f"Found:    {selected_candidate}"
        )

    selected_margin_features = (
        selection_metadata.get(
            "margin_features",
            [],
        )
    )

    selected_total_features = (
        selection_metadata.get(
            "total_features",
            [],
        )
    )

    if (
        selected_margin_features
        !=
        MARGIN_FEATURES
    ):

        raise RuntimeError(
            "Margin feature architecture mismatch.\n\n"
            f"Expected: {MARGIN_FEATURES}\n"
            f"Found:    {selected_margin_features}"
        )

    if (
        selected_total_features
        !=
        TOTAL_FEATURES
    ):

        raise RuntimeError(
            "Total feature architecture mismatch.\n\n"
            f"Expected: {TOTAL_FEATURES}\n"
            f"Found:    {selected_total_features}"
        )

    print(
        "PASS - selected architecture matches "
        "the frozen compact_full specification."
    )

    # ========================================================
    # LOAD HISTORY
    # ========================================================

    section(
        "2. LOAD HISTORICAL SCORING DATA"
    )

    history = load_scoring_history()

    print(
        f"Rows loaded: "
        f"{len(history):,}"
    )

    required_columns = (
        [
            "season",
            "home_points",
            "away_points",
        ]
        +
        MARGIN_FEATURES
        +
        TOTAL_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing:

        raise ValueError(
            "scoring_history.csv is missing "
            f"required score-model fields: {missing}"
        )

    # ========================================================
    # PREPARE TARGETS
    # ========================================================

    section(
        "3. PREPARE FINAL TRAINING DATA"
    )

    data = history[
        history[
            "season"
        ]
        .isin(
            TRAINING_SEASONS
        )
    ].copy()

    numeric_columns = list(
        dict.fromkeys(
            [
                "home_points",
                "away_points",
            ]
            +
            MARGIN_FEATURES
            +
            TOTAL_FEATURES
        )
    )

    for column in numeric_columns:

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

    training_columns = list(
        dict.fromkeys(
            MARGIN_FEATURES
            +
            TOTAL_FEATURES
            +
            [
                "actual_home_margin",
                "actual_total_points",
            ]
        )
    )

    complete_mask = (
        data[
            training_columns
        ]
        .notna()
        .all(
            axis=1
        )
    )

    training = (
        data[
            complete_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    dropped_rows = (
        len(data)
        -
        len(training)
    )

    print(
        f"Historical rows:      "
        f"{len(data):,}"
    )

    print(
        f"Complete training:    "
        f"{len(training):,}"
    )

    print(
        f"Dropped incomplete:   "
        f"{dropped_rows:,}"
    )

    if training.empty:

        raise RuntimeError(
            "No complete score-model training rows."
        )

    if dropped_rows:

        raise RuntimeError(
            "Historical scoring training data contains "
            "incomplete rows. Freeze stopped so we do not "
            "silently train on a different universe."
        )

    for season in TRAINING_SEASONS:

        season_rows = int(
            (
                training[
                    "season"
                ]
                ==
                season
            )
            .sum()
        )

        print(
            f"{season}:               "
            f"{season_rows:,}"
        )

    # ========================================================
    # FIT MARGIN MODEL
    # ========================================================

    section(
        "4. FIT FINAL MARGIN MODEL"
    )

    margin_model = build_pipeline(
        MARGIN_FEATURES
    )

    margin_model.fit(
        training[
            MARGIN_FEATURES
        ],
        training[
            "actual_home_margin"
        ],
    )

    print(
        "Margin model fitted."
    )

    print()

    print(
        "Margin features:"
    )

    for feature in MARGIN_FEATURES:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # FIT TOTAL MODEL
    # ========================================================

    section(
        "5. FIT FINAL TOTAL MODEL"
    )

    total_model = build_pipeline(
        TOTAL_FEATURES
    )

    total_model.fit(
        training[
            TOTAL_FEATURES
        ],
        training[
            "actual_total_points"
        ],
    )

    print(
        "Total model fitted."
    )

    print()

    print(
        "Total features:"
    )

    for feature in TOTAL_FEATURES:

        print(
            f"  - {feature}"
        )

    # ========================================================
    # STORE ARTIFACT
    # ========================================================

    section(
        "6. SAVE FROZEN SCORE MODEL"
    )

    artifact = {
        "model_name": (
            MODEL_NAME
        ),
        "model_version": (
            "v2_score_compact_full_1"
        ),
        "ridge_alpha": (
            RIDGE_ALPHA
        ),
        "training_seasons": (
            TRAINING_SEASONS
        ),
        "margin_features": (
            MARGIN_FEATURES
        ),
        "total_features": (
            TOTAL_FEATURES
        ),
        "margin_model": (
            margin_model
        ),
        "total_model": (
            total_model
        ),
        "score_reconstruction": {
            "home": (
                "(predicted_total + "
                "predicted_home_margin) / 2"
            ),
            "away": (
                "(predicted_total - "
                "predicted_home_margin) / 2"
            ),
        },
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print(
        f"Saved model:\n{MODEL_PATH}"
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "model_name": (
            MODEL_NAME
        ),
        "model_version": (
            "v2_score_compact_full_1"
        ),
        "architecture_locked": True,
        "architecture_selected_on": {
            "training_season": 2023,
            "validation_season": 2024,
        },
        "development_evaluation_season": (
            2025
        ),
        "final_training_seasons": (
            TRAINING_SEASONS
        ),
        "training_rows": int(
            len(training)
        ),
        "ridge_alpha": (
            RIDGE_ALPHA
        ),
        "margin_features": (
            MARGIN_FEATURES
        ),
        "total_features": (
            TOTAL_FEATURES
        ),
        "selection_2025_results": {
            "raw_margin_mae": (
                selection_metadata[
                    "raw_2025_metrics"
                ][
                    "margin_mae"
                ]
            ),
            "selected_margin_mae": (
                selection_metadata[
                    "selected_2025_metrics"
                ][
                    "margin_mae"
                ]
            ),
            "raw_winner_accuracy": (
                selection_metadata[
                    "raw_2025_metrics"
                ][
                    "winner_accuracy"
                ]
            ),
            "selected_winner_accuracy": (
                selection_metadata[
                    "selected_2025_metrics"
                ][
                    "winner_accuracy"
                ]
            ),
            "raw_total_mae": (
                selection_metadata[
                    "raw_2025_metrics"
                ][
                    "total_mae"
                ]
            ),
            "selected_total_mae": (
                selection_metadata[
                    "selected_2025_metrics"
                ][
                    "total_mae"
                ]
            ),
        },
        "notes": [
            (
                "No 2026 results were used to select "
                "or train this frozen score model."
            ),
            (
                "Official winner probability remains "
                "the separate frozen probability model."
            ),
            (
                "Monte Carlo remains a score-distribution "
                "layer and does not replace official "
                "winner probability."
            ),
        ],
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()

    print(
        f"Saved manifest:\n{MANIFEST_PATH}"
    )

    # ========================================================
    # SAVE TRAINING ROW AUDIT
    # ========================================================

    audit_columns = [
        column
        for column in [
            "game_id",
            "season",
            "week",
            "start_date",
            "away_team",
            "home_team",
            "home_points",
            "away_points",
            "actual_home_margin",
            "actual_total_points",
        ]
        +
        MARGIN_FEATURES
        +
        TOTAL_FEATURES
        if column in training.columns
    ]

    training[
        audit_columns
    ].to_csv(
        TRAINING_ROWS_PATH,
        index=False,
    )

    print()

    print(
        f"Saved training audit:\n"
        f"{TRAINING_ROWS_PATH}"
    )

    # ========================================================
    # SANITY PREDICTIONS
    #
    # Not used for tuning.
    # Only verifies artifact operation.
    # ========================================================

    section(
        "7. ARTIFACT SANITY CHECK"
    )

    loaded = joblib.load(
        MODEL_PATH
    )

    margin_predictions = (
        loaded[
            "margin_model"
        ]
        .predict(
            training[
                MARGIN_FEATURES
            ]
        )
    )

    total_predictions = (
        loaded[
            "total_model"
        ]
        .predict(
            training[
                TOTAL_FEATURES
            ]
        )
    )

    missing_margin_predictions = int(
        np.sum(
            ~np.isfinite(
                margin_predictions
            )
        )
    )

    missing_total_predictions = int(
        np.sum(
            ~np.isfinite(
                total_predictions
            )
        )
    )

    print(
        f"Margin predictions: "
        f"{len(margin_predictions):,}"
    )

    print(
        f"Total predictions:  "
        f"{len(total_predictions):,}"
    )

    print(
        f"Invalid margins:     "
        f"{missing_margin_predictions:,}"
    )

    print(
        f"Invalid totals:      "
        f"{missing_total_predictions:,}"
    )

    if (
        missing_margin_predictions
        or
        missing_total_predictions
    ):

        raise RuntimeError(
            "Frozen score artifact produced "
            "invalid predictions."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "SCORE MODEL FREEZE COMPLETE"
    )

    print(
        "✓ compact_full architecture verified."
    )

    print(
        "✓ Margin model trained on 2023-2025."
    )

    print(
        "✓ Total model trained on 2023-2025."
    )

    print(
        "✓ 2,880-game historical universe preserved."
    )

    print(
        "✓ No 2026 games used."
    )

    print(
        "✓ Frozen model artifact saved."
    )

    print(
        "✓ Manifest saved."
    )

    print()

    print(
        "Frozen score model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Apply this frozen correction inside "
        "the leakage-safe 2026 scoring runner."
    )


if __name__ == "__main__":
    main()