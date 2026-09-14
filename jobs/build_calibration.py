from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import PROCESSED_DATA_DIR

from engine.calibration import (
    CalibrationConfig,
    CalibrationEngine,
)


PROBABILITY_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "probability_history.csv"
)

CALIBRATED_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "calibrated_probability_history.csv"
)

CALIBRATION_RESULTS_FILE = (
    PROCESSED_DATA_DIR
    / "calibration_candidate_results.csv"
)

CALIBRATION_HOLDOUT_FILE = (
    PROCESSED_DATA_DIR
    / "calibration_2025_results.csv"
)

CALIBRATION_TABLE_FILE = (
    PROCESSED_DATA_DIR
    / "calibration_2025_table.csv"
)

CALIBRATION_METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "calibration_model_metadata.json"
)


def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def save_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )

    print(
        f"✓ Saved {len(frame):,} rows → {path}"
    )


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— CONSTRAINED PROBABILITY CALIBRATION"
    )

    if not PROBABILITY_HISTORY_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{PROBABILITY_HISTORY_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING PROBABILITY HISTORY"
    )

    history = pd.read_csv(
        PROBABILITY_HISTORY_FILE,
        low_memory=False,
    )

    print(
        f"Probability rows loaded: "
        f"{len(history):,}"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = CalibrationConfig(
        development_season=2024,
        holdout_season=2025,
        calibration_train_fraction=0.60,
        logistic_c=1.0,
        max_iter=2000,
        probability_floor=0.001,
        probability_ceiling=0.999,
    )

    section(
        "CALIBRATION DESIGN"
    )

    print(
        "2024 OOS predictions:"
    )

    print(
        "  First 60% -> calibration fit"
    )

    print(
        "  Last 40%  -> calibration selection"
    )

    print()

    print(
        "2025 remains untouched until final evaluation."
    )

    print()

    print(
        "Candidates:"
    )

    print(
        "  identity"
    )

    print(
        "  platt"
    )

    print(
        "  maturity_shrink_25"
    )

    print(
        "  maturity_shrink_50"
    )

    print(
        "  maturity_shrink_75"
    )

    print(
        "  maturity_shrink_100"
    )

    print()

    print(
        "All maturity candidates are mathematically constrained:"
    )

    print(
        "lower maturity can ONLY move probabilities toward 50%."
    )

    # ========================================================
    # PREPARE
    # ========================================================

    engine = CalibrationEngine(
        config=config
    )

    data = engine.prepare_dataset(
        history
    )

    development = (
        data[
            data[
                "season"
            ]
            ==
            config.development_season
        ]
        .copy()
        .reset_index(
            drop=True
        )
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
        .reset_index(
            drop=True
        )
    )

    section(
        "CALIBRATION DATA"
    )

    print(
        f"2024 development games: "
        f"{len(development):,}"
    )

    print(
        f"2025 holdout games:     "
        f"{len(holdout):,}"
    )

    split_index = int(
        len(development)
        *
        config.calibration_train_fraction
    )

    calibration_train = (
        development.iloc[
            :split_index
        ]
        .copy()
    )

    calibration_validation = (
        development.iloc[
            split_index:
        ]
        .copy()
    )

    print()

    print(
        f"2024 fit games:        "
        f"{len(calibration_train):,}"
    )

    print(
        f"2024 validation games: "
        f"{len(calibration_validation):,}"
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    section(
        "2024 CALIBRATION CANDIDATE VALIDATION"
    )

    architectures = (
        engine.candidate_architectures()
    )

    candidate_results = []

    for (
        name,
        shrink_alpha,
    ) in architectures.items():

        bundle = engine.fit_model(
            data=calibration_train,
            model_name=name,
            shrink_alpha=shrink_alpha,
        )

        probability = engine.predict(
            bundle=bundle,
            data=calibration_validation,
        )

        metrics = engine.evaluate(
            data=calibration_validation,
            probability=probability,
        )

        candidate_results.append(
            {
                "calibrator": name,

                "shrink_alpha": (
                    np.nan
                    if shrink_alpha is None
                    else shrink_alpha
                ),

                **metrics,
            }
        )

        print(
            f"{name:<22} | "
            f"Accuracy {metrics['accuracy']:.2%} | "
            f"Brier {metrics['brier']:.4f} | "
            f"LogLoss {metrics['log_loss']:.4f}"
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

    # ========================================================
    # SELECT
    # ========================================================

    section(
        "CALIBRATOR SELECTION"
    )

    for rank, row in enumerate(
        candidates.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.calibrator:<22} | "
            f"Accuracy {row.accuracy:.2%} | "
            f"Brier {row.brier:.4f} | "
            f"LogLoss {row.log_loss:.4f}"
        )

    selected_name = str(
        candidates.iloc[
            0
        ][
            "calibrator"
        ]
    )

    selected_alpha = (
        architectures[
            selected_name
        ]
    )

    print()

    print(
        f"Selected calibrator: "
        f"{selected_name}"
    )

    # ========================================================
    # REFIT
    # ========================================================

    section(
        "REFITTING SELECTED CALIBRATOR"
    )

    final_bundle = engine.fit_model(
        data=development,
        model_name=selected_name,
        shrink_alpha=selected_alpha,
    )

    print(
        f"Refit / reference games: "
        f"{len(development):,}"
    )

    # ========================================================
    # HOLDOUT
    # ========================================================

    section(
        "2025 FORWARD CALIBRATION HOLDOUT"
    )

    raw_probability = (
        holdout[
            "v2_raw_home_win_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    calibrated_probability = (
        engine.predict(
            bundle=final_bundle,
            data=holdout,
        )
    )

    raw_metrics = engine.evaluate(
        data=holdout,
        probability=raw_probability,
    )

    calibrated_metrics = (
        engine.evaluate(
            data=holdout,
            probability=calibrated_probability,
        )
    )

    print(
        f"{'Metric':<20}"
        f"{'Raw V2':>14}"
        f"{'Calibrated':>14}"
    )

    print(
        "-" * 48
    )

    print(
        f"{'Accuracy':<20}"
        f"{raw_metrics['accuracy']:>13.2%}"
        f"{calibrated_metrics['accuracy']:>13.2%}"
    )

    print(
        f"{'Brier':<20}"
        f"{raw_metrics['brier']:>14.4f}"
        f"{calibrated_metrics['brier']:>14.4f}"
    )

    print(
        f"{'Log loss':<20}"
        f"{raw_metrics['log_loss']:>14.4f}"
        f"{calibrated_metrics['log_loss']:>14.4f}"
    )

    # ========================================================
    # CALIBRATION DETAILS
    # ========================================================

    section(
        "SELECTED CALIBRATION PARAMETERS"
    )

    if selected_name == "identity":

        print(
            "Identity selected."
        )

        print(
            "No probability adjustment is applied."
        )

    elif selected_name == "platt":

        print(
            f"raw_logit coefficient: "
            f"{final_bundle.model.coef_[0][0]:.4f}"
        )

        print(
            f"intercept:             "
            f"{final_bundle.model.intercept_[0]:.4f}"
        )

    else:

        print(
            f"Maturity shrink alpha: "
            f"{final_bundle.shrink_alpha:.2f}"
        )

        print()

        print(
            "This transform cannot reverse the predicted winner"
        )

        print(
            "and cannot make a low-maturity game more extreme."
        )

    # ========================================================
    # CALIBRATION TABLE
    # ========================================================

    section(
        "2025 CALIBRATED PROBABILITY TABLE"
    )

    calibration_table = (
        engine.calibration_table(
            data=holdout,
            probability=calibrated_probability,
            bins=10,
        )
    )

    for row in calibration_table.itertuples(
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
    # CONFIDENCE
    # ========================================================

    section(
        "2025 CALIBRATED CONFIDENCE PERFORMANCE"
    )

    predicted_probability = np.maximum(
        calibrated_probability,
        1.0
        -
        calibrated_probability,
    )

    actual_home_win = (
        holdout[
            "actual_home_win"
        ]
        .astype(bool)
        .to_numpy()
    )

    predicted_home = (
        calibrated_probability
        >
        0.50
    )

    correct = (
        predicted_home
        ==
        actual_home_win
    )

    for threshold in [
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
    ]:

        mask = (
            predicted_probability
            >=
            threshold
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            continue

        accuracy = float(
            correct[
                mask
            ].mean()
        )

        print(
            f"≥ {threshold:.0%} confidence | "
            f"{count:>4} games | "
            f"Accuracy {accuracy:>6.2%}"
        )

    # ========================================================
    # EXTREMES
    # ========================================================

    section(
        "EXTREME PROBABILITY CHECK"
    )

    print(
        f"Highest home probability: "
        f"{calibrated_probability.max():.2%}"
    )

    print(
        f"Median home probability:  "
        f"{np.median(calibrated_probability):.2%}"
    )

    print(
        f"Lowest home probability:  "
        f"{calibrated_probability.min():.2%}"
    )

    extreme_95 = int(
        (
            (
                calibrated_probability
                >=
                0.95
            )
            |
            (
                calibrated_probability
                <=
                0.05
            )
        ).sum()
    )

    extreme_99 = int(
        (
            (
                calibrated_probability
                >=
                0.99
            )
            |
            (
                calibrated_probability
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
    # OUTPUT HISTORY
    # ========================================================

    section(
        "BUILDING CALIBRATED HISTORY"
    )

    output = data.copy()

    output[
        "v2_calibrated_home_win_probability"
    ] = np.nan

    holdout_mask = (
        output[
            "season"
        ]
        ==
        config.holdout_season
    )

    output.loc[
        holdout_mask,
        "v2_calibrated_home_win_probability",
    ] = calibrated_probability

    output[
        "v2_calibrated_away_win_probability"
    ] = (
        1.0
        -
        output[
            "v2_calibrated_home_win_probability"
        ]
    )

    output[
        "calibration_method"
    ] = selected_name

    output[
        "calibrated_probability_is_forward_holdout"
    ] = (
        holdout_mask
        &
        output[
            "v2_calibrated_home_win_probability"
        ].notna()
    )

    holdout_results = (
        holdout.copy()
    )

    holdout_results[
        "raw_home_probability"
    ] = raw_probability

    holdout_results[
        "calibrated_home_probability"
    ] = calibrated_probability

    holdout_results[
        "raw_away_probability"
    ] = (
        1.0
        -
        raw_probability
    )

    holdout_results[
        "calibrated_away_probability"
    ] = (
        1.0
        -
        calibrated_probability
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "selected_calibrator": (
            selected_name
        ),

        "selected_shrink_alpha": (
            selected_alpha
        ),

        "development_season": 2024,
        "holdout_season": 2025,

        "2025_raw": {
            "accuracy": float(
                raw_metrics[
                    "accuracy"
                ]
            ),

            "brier": float(
                raw_metrics[
                    "brier"
                ]
            ),

            "log_loss": float(
                raw_metrics[
                    "log_loss"
                ]
            ),
        },

        "2025_calibrated": {
            "accuracy": float(
                calibrated_metrics[
                    "accuracy"
                ]
            ),

            "brier": float(
                calibrated_metrics[
                    "brier"
                ]
            ),

            "log_loss": float(
                calibrated_metrics[
                    "log_loss"
                ]
            ),
        },
    }

    if (
        selected_name
        ==
        "platt"
    ):

        metadata[
            "platt_coefficient"
        ] = float(
            final_bundle.model.coef_[
                0
            ][
                0
            ]
        )

        metadata[
            "platt_intercept"
        ] = float(
            final_bundle.model.intercept_[
                0
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING CALIBRATION OUTPUT"
    )

    save_csv(
        output,
        CALIBRATED_HISTORY_FILE,
    )

    save_csv(
        candidates,
        CALIBRATION_RESULTS_FILE,
    )

    save_csv(
        holdout_results,
        CALIBRATION_HOLDOUT_FILE,
    )

    save_csv(
        calibration_table,
        CALIBRATION_TABLE_FILE,
    )

    with open(
        CALIBRATION_METADATA_FILE,
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
        f"{CALIBRATION_METADATA_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "CONSTRAINED CALIBRATION COMPLETE"
    )

    print(
        "Next step:"
    )

    print(
        "  Freeze the V2 historical architecture"
    )

    print(
        "  Train the production state through 2025"
    )

    print(
        "  Run genuinely forward 2026 predictions"
    )


if __name__ == "__main__":
    main()