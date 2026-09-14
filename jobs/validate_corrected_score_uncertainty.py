from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT / FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAINING_ROWS_FILE = (
    PROCESSED_DIR
    / "v2_score_model_training_rows.csv"
)

OUTPUT_HISTORY_FILE = (
    PROCESSED_DIR
    / "corrected_score_uncertainty_history.csv"
)

OUTPUT_STATE_FILE = (
    PROCESSED_DIR
    / "v2_corrected_score_uncertainty.json"
)


# ============================================================
# FROZEN SCORE ARCHITECTURE
# ============================================================

MODEL_NAME = "compact_full"
MODEL_VERSION = "v2_score_compact_full_1"
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

VALIDATION_SEASONS = [2024, 2025]


# ============================================================
# UNCERTAINTY CONFIG
# ============================================================

UPDATE_RATE = 0.05

STARTING_HOME_SCORE_SD = 14.0
STARTING_AWAY_SCORE_SD = 14.0
STARTING_RESIDUAL_CORRELATION = 0.10

MIN_SCORE_SD = 8.0
MAX_SCORE_SD = 22.0

MIN_CORRELATION = -0.35
MAX_CORRELATION = 0.60

INTERVAL_LEVEL = 0.80

# 80% central normal interval.
Z_80 = 1.2815515655446004


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:

    for candidate in candidates:

        if candidate in frame.columns:

            if frame[candidate].notna().any():
                return candidate

    raise RuntimeError(
        f"Could not resolve required column for {label}.\n"
        f"Tried: {candidates}"
    )


def numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def build_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA,
                ),
            ),
        ]
    )


def safe_corr_from_state(
    home_variance: float,
    away_variance: float,
    covariance: float,
) -> float:

    denominator = math.sqrt(
        max(home_variance, 0.0)
        *
        max(away_variance, 0.0)
    )

    if denominator <= 0.0:
        return 0.0

    corr = covariance / denominator

    return float(
        np.clip(
            corr,
            MIN_CORRELATION,
            MAX_CORRELATION,
        )
    )


def bounded_variance(
    variance: float,
) -> float:

    sd = math.sqrt(
        max(
            variance,
            0.0,
        )
    )

    sd = float(
        np.clip(
            sd,
            MIN_SCORE_SD,
            MAX_SCORE_SD,
        )
    )

    return sd ** 2


@dataclass
class UncertaintyState:

    games_seen: int = 0

    home_variance: float = (
        STARTING_HOME_SCORE_SD ** 2
    )

    away_variance: float = (
        STARTING_AWAY_SCORE_SD ** 2
    )

    covariance: float = (
        STARTING_RESIDUAL_CORRELATION
        *
        STARTING_HOME_SCORE_SD
        *
        STARTING_AWAY_SCORE_SD
    )

    def snapshot(self) -> dict:

        home_sd = math.sqrt(
            self.home_variance
        )

        away_sd = math.sqrt(
            self.away_variance
        )

        corr = safe_corr_from_state(
            self.home_variance,
            self.away_variance,
            self.covariance,
        )

        return {
            "games_seen": self.games_seen,
            "home_score_sd": home_sd,
            "away_score_sd": away_sd,
            "score_residual_correlation": corr,
            "home_variance": self.home_variance,
            "away_variance": self.away_variance,
            "covariance": self.covariance,
        }

    def update(
        self,
        home_residual: float,
        away_residual: float,
    ) -> None:

        alpha = UPDATE_RATE

        self.home_variance = (
            (1.0 - alpha)
            * self.home_variance
            +
            alpha
            * (home_residual ** 2)
        )

        self.away_variance = (
            (1.0 - alpha)
            * self.away_variance
            +
            alpha
            * (away_residual ** 2)
        )

        self.covariance = (
            (1.0 - alpha)
            * self.covariance
            +
            alpha
            * (
                home_residual
                *
                away_residual
            )
        )

        self.home_variance = bounded_variance(
            self.home_variance
        )

        self.away_variance = bounded_variance(
            self.away_variance
        )

        corr = safe_corr_from_state(
            self.home_variance,
            self.away_variance,
            self.covariance,
        )

        max_covariance = (
            corr
            *
            math.sqrt(
                self.home_variance
                *
                self.away_variance
            )
        )

        self.covariance = max_covariance

        self.games_seen += 1


# ============================================================
# OOS SCORE PREDICTIONS
# ============================================================

def prepare_training_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    required_features = (
        MARGIN_FEATURES
        +
        TOTAL_FEATURES
    )

    missing = [
        column
        for column in required_features
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Frozen compact_full features are missing "
            "from v2_score_model_training_rows.csv:\n"
            + "\n".join(
                missing
            )
        )

    season_column = resolve_column(
        frame,
        [
            "season",
            "year",
        ],
        "season",
    )

    home_points_column = resolve_column(
        frame,
        [
            "actual_home_points",
            "home_points",
            "home_score",
        ],
        "actual home points",
    )

    away_points_column = resolve_column(
        frame,
        [
            "actual_away_points",
            "away_points",
            "away_score",
        ],
        "actual away points",
    )

    prepared = frame.copy()

    prepared["season"] = numeric_series(
        prepared,
        season_column,
    )

    prepared[
        "_actual_home_points"
    ] = numeric_series(
        prepared,
        home_points_column,
    )

    prepared[
        "_actual_away_points"
    ] = numeric_series(
        prepared,
        away_points_column,
    )

    prepared[
        "_actual_home_margin"
    ] = (
        prepared[
            "_actual_home_points"
        ]
        -
        prepared[
            "_actual_away_points"
        ]
    )

    prepared[
        "_actual_total_points"
    ] = (
        prepared[
            "_actual_home_points"
        ]
        +
        prepared[
            "_actual_away_points"
        ]
    )

    for column in required_features:

        prepared[column] = numeric_series(
            prepared,
            column,
        )

    keep_columns = (
        [
            column
            for column in [
                "season",
                "week",
                "start_date",
                "game_id",
                "game_db_id",
                "cfbd_game_id",
                "away_team",
                "home_team",
            ]
            if column in prepared.columns
        ]
        +
        required_features
        +
        [
            "_actual_home_points",
            "_actual_away_points",
            "_actual_home_margin",
            "_actual_total_points",
        ]
    )

    prepared = prepared[
        keep_columns
    ].copy()

    complete_columns = (
        [
            "season",
            "_actual_home_points",
            "_actual_away_points",
            "_actual_home_margin",
            "_actual_total_points",
        ]
        +
        required_features
    )

    prepared = prepared.dropna(
        subset=complete_columns
    ).copy()

    prepared["season"] = (
        prepared["season"]
        .astype(int)
    )

    return prepared


def chronological_sort(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    work = frame.copy()

    sort_columns: list[str] = []

    if "season" in work.columns:
        sort_columns.append(
            "season"
        )

    if "week" in work.columns:

        work["_sort_week"] = pd.to_numeric(
            work["week"],
            errors="coerce",
        )

        sort_columns.append(
            "_sort_week"
        )

    if "start_date" in work.columns:

        work["_sort_start_date"] = pd.to_datetime(
            work["start_date"],
            errors="coerce",
            utc=True,
        )

        sort_columns.append(
            "_sort_start_date"
        )

    for candidate in [
        "cfbd_game_id",
        "game_db_id",
        "game_id",
    ]:

        if candidate in work.columns:

            work[
                f"_sort_{candidate}"
            ] = pd.to_numeric(
                work[candidate],
                errors="coerce",
            )

            sort_columns.append(
                f"_sort_{candidate}"
            )

            break

    if sort_columns:

        work = (
            work
            .sort_values(
                sort_columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    drop_columns = [
        column
        for column in work.columns
        if column.startswith(
            "_sort_"
        )
    ]

    return work.drop(
        columns=drop_columns,
        errors="ignore",
    )


def create_oos_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    outputs = []

    for validation_season in VALIDATION_SEASONS:

        train = frame[
            frame["season"]
            <
            validation_season
        ].copy()

        validate = frame[
            frame["season"]
            ==
            validation_season
        ].copy()

        if train.empty:

            raise RuntimeError(
                f"No training rows exist before "
                f"{validation_season}."
            )

        if validate.empty:

            raise RuntimeError(
                f"No validation rows exist for "
                f"{validation_season}."
            )

        margin_model = (
            build_pipeline()
        )

        total_model = (
            build_pipeline()
        )

        margin_model.fit(
            train[
                MARGIN_FEATURES
            ],
            train[
                "_actual_home_margin"
            ],
        )

        total_model.fit(
            train[
                TOTAL_FEATURES
            ],
            train[
                "_actual_total_points"
            ],
        )

        predicted_margin = (
            margin_model.predict(
                validate[
                    MARGIN_FEATURES
                ]
            )
        )

        predicted_total = (
            total_model.predict(
                validate[
                    TOTAL_FEATURES
                ]
            )
        )

        validate[
            "oos_expected_home_margin"
        ] = predicted_margin

        validate[
            "oos_expected_total_points"
        ] = predicted_total

        validate[
            "oos_expected_home_points"
        ] = (
            predicted_total
            +
            predicted_margin
        ) / 2.0

        validate[
            "oos_expected_away_points"
        ] = (
            predicted_total
            -
            predicted_margin
        ) / 2.0

        validate[
            "oos_home_residual"
        ] = (
            validate[
                "_actual_home_points"
            ]
            -
            validate[
                "oos_expected_home_points"
            ]
        )

        validate[
            "oos_away_residual"
        ] = (
            validate[
                "_actual_away_points"
            ]
            -
            validate[
                "oos_expected_away_points"
            ]
        )

        validate[
            "oos_margin_residual"
        ] = (
            validate[
                "_actual_home_margin"
            ]
            -
            validate[
                "oos_expected_home_margin"
            ]
        )

        validate[
            "oos_total_residual"
        ] = (
            validate[
                "_actual_total_points"
            ]
            -
            validate[
                "oos_expected_total_points"
            ]
        )

        validate[
            "score_model_training_seasons"
        ] = (
            f"{int(train['season'].min())}"
            f"-"
            f"{int(train['season'].max())}"
        )

        outputs.append(
            validate
        )

    combined = pd.concat(
        outputs,
        ignore_index=True,
    )

    return chronological_sort(
        combined
    )


# ============================================================
# SEQUENTIAL UNCERTAINTY VALIDATION
# ============================================================

def validate_uncertainty(
    oos: pd.DataFrame,
) -> pd.DataFrame:

    state = UncertaintyState()

    output_rows = []

    for _, row in oos.iterrows():

        snapshot = state.snapshot()

        home_sd = float(
            snapshot[
                "home_score_sd"
            ]
        )

        away_sd = float(
            snapshot[
                "away_score_sd"
            ]
        )

        corr = float(
            snapshot[
                "score_residual_correlation"
            ]
        )

        covariance = (
            corr
            *
            home_sd
            *
            away_sd
        )

        margin_variance = max(
            home_sd ** 2
            +
            away_sd ** 2
            -
            2.0
            *
            covariance,
            1e-9,
        )

        total_variance = max(
            home_sd ** 2
            +
            away_sd ** 2
            +
            2.0
            *
            covariance,
            1e-9,
        )

        margin_sd = math.sqrt(
            margin_variance
        )

        total_sd = math.sqrt(
            total_variance
        )

        expected_home = float(
            row[
                "oos_expected_home_points"
            ]
        )

        expected_away = float(
            row[
                "oos_expected_away_points"
            ]
        )

        expected_margin = float(
            row[
                "oos_expected_home_margin"
            ]
        )

        expected_total = float(
            row[
                "oos_expected_total_points"
            ]
        )

        actual_home = float(
            row[
                "_actual_home_points"
            ]
        )

        actual_away = float(
            row[
                "_actual_away_points"
            ]
        )

        actual_margin = float(
            row[
                "_actual_home_margin"
            ]
        )

        actual_total = float(
            row[
                "_actual_total_points"
            ]
        )

        home_low = (
            expected_home
            -
            Z_80
            *
            home_sd
        )

        home_high = (
            expected_home
            +
            Z_80
            *
            home_sd
        )

        away_low = (
            expected_away
            -
            Z_80
            *
            away_sd
        )

        away_high = (
            expected_away
            +
            Z_80
            *
            away_sd
        )

        margin_low = (
            expected_margin
            -
            Z_80
            *
            margin_sd
        )

        margin_high = (
            expected_margin
            +
            Z_80
            *
            margin_sd
        )

        total_low = (
            expected_total
            -
            Z_80
            *
            total_sd
        )

        total_high = (
            expected_total
            +
            Z_80
            *
            total_sd
        )

        result = row.to_dict()

        result.update(
            {
                "uncertainty_games_seen_before_game": (
                    snapshot[
                        "games_seen"
                    ]
                ),
                "pregame_home_score_sd": home_sd,
                "pregame_away_score_sd": away_sd,
                "pregame_residual_correlation": corr,
                "pregame_margin_sd": margin_sd,
                "pregame_total_sd": total_sd,
                "home_score_p10": home_low,
                "home_score_p90": home_high,
                "away_score_p10": away_low,
                "away_score_p90": away_high,
                "margin_p10": margin_low,
                "margin_p90": margin_high,
                "total_p10": total_low,
                "total_p90": total_high,
                "home_interval_hit": (
                    home_low
                    <=
                    actual_home
                    <=
                    home_high
                ),
                "away_interval_hit": (
                    away_low
                    <=
                    actual_away
                    <=
                    away_high
                ),
                "margin_interval_hit": (
                    margin_low
                    <=
                    actual_margin
                    <=
                    margin_high
                ),
                "total_interval_hit": (
                    total_low
                    <=
                    actual_total
                    <=
                    total_high
                ),
            }
        )

        output_rows.append(
            result
        )

        state.update(
            home_residual=float(
                row[
                    "oos_home_residual"
                ]
            ),
            away_residual=float(
                row[
                    "oos_away_residual"
                ]
            ),
        )

    history = pd.DataFrame(
        output_rows
    )

    final_state = state.snapshot()

    history.attrs[
        "final_uncertainty_state"
    ] = final_state

    return history


# ============================================================
# REPORTING
# ============================================================

def print_oos_accuracy(
    frame: pd.DataFrame,
) -> None:

    section(
        "OUT-OF-SAMPLE CORRECTED SCORE PERFORMANCE"
    )

    for season in VALIDATION_SEASONS:

        sample = frame[
            frame["season"]
            ==
            season
        ]

        if sample.empty:
            continue

        home_mae = (
            sample[
                "oos_home_residual"
            ]
            .abs()
            .mean()
        )

        away_mae = (
            sample[
                "oos_away_residual"
            ]
            .abs()
            .mean()
        )

        margin_mae = (
            sample[
                "oos_margin_residual"
            ]
            .abs()
            .mean()
        )

        total_mae = (
            sample[
                "oos_total_residual"
            ]
            .abs()
            .mean()
        )

        print(
            f"{season}: "
            f"games={len(sample):,} | "
            f"home MAE={home_mae:.3f} | "
            f"away MAE={away_mae:.3f} | "
            f"margin MAE={margin_mae:.3f} | "
            f"total MAE={total_mae:.3f}"
        )

    print()

    print(
        f"Combined OOS rows: "
        f"{len(frame):,}"
    )


def print_interval_validation(
    history: pd.DataFrame,
) -> None:

    section(
        "CORRECTED-SCORE UNCERTAINTY VALIDATION"
    )

    print(
        f"Target central interval: "
        f"{INTERVAL_LEVEL:.0%}"
    )

    print()

    metrics = [
        (
            "Home score",
            "home_interval_hit",
        ),
        (
            "Away score",
            "away_interval_hit",
        ),
        (
            "Margin",
            "margin_interval_hit",
        ),
        (
            "Total",
            "total_interval_hit",
        ),
    ]

    for label, column in metrics:

        coverage = (
            history[column]
            .astype(float)
            .mean()
        )

        print(
            f"{label:<14} coverage: "
            f"{coverage:>7.2%}"
        )

    print()

    for season in VALIDATION_SEASONS:

        sample = history[
            history["season"]
            ==
            season
        ]

        if sample.empty:
            continue

        print(
            f"{season} "
            f"home={sample['home_interval_hit'].mean():.2%} | "
            f"away={sample['away_interval_hit'].mean():.2%} | "
            f"margin={sample['margin_interval_hit'].mean():.2%} | "
            f"total={sample['total_interval_hit'].mean():.2%}"
        )


def print_final_state(
    history: pd.DataFrame,
) -> dict:

    final_state = history.attrs[
        "final_uncertainty_state"
    ]

    section(
        "END-2025 CORRECTED UNCERTAINTY STATE"
    )

    print(
        f"Games learned from:       "
        f"{final_state['games_seen']:,}"
    )

    print(
        f"Home score SD:            "
        f"{final_state['home_score_sd']:.4f}"
    )

    print(
        f"Away score SD:            "
        f"{final_state['away_score_sd']:.4f}"
    )

    print(
        f"Residual correlation:     "
        f"{final_state['score_residual_correlation']:.4f}"
    )

    return final_state


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- CORRECTED SCORE UNCERTAINTY"
    )

    print(
        "Purpose:"
    )

    print(
        "Build Monte Carlo uncertainty from "
        "OUT-OF-SAMPLE compact_full score residuals."
    )

    print()

    print(
        "2026 is NOT used to fit or calibrate "
        "this uncertainty layer."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    section(
        "1. LOAD FROZEN SCORE TRAINING ROWS"
    )

    if not TRAINING_ROWS_FILE.exists():

        raise FileNotFoundError(
            f"Required file missing:\n"
            f"{TRAINING_ROWS_FILE}"
        )

    raw = pd.read_csv(
        TRAINING_ROWS_FILE,
        low_memory=False,
    )

    print(
        f"Rows loaded: "
        f"{len(raw):,}"
    )

    prepared = prepare_training_rows(
        raw
    )

    print(
        f"Complete usable rows: "
        f"{len(prepared):,}"
    )

    print(
        "Seasons:"
    )

    print(
        prepared[
            "season"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # OOS SCORE PREDICTIONS
    # --------------------------------------------------------

    section(
        "2. BUILD WALK-FORWARD CORRECTED SCORES"
    )

    print(
        "2024: train compact_full on 2023 only."
    )

    print(
        "2025: train compact_full on 2023 + 2024."
    )

    print()

    print(
        "No game is evaluated using a correction model "
        "trained on that game's season."
    )

    oos = create_oos_predictions(
        prepared
    )

    print_oos_accuracy(
        oos
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    section(
        "3. REPLAY CORRECTED RESIDUAL UNCERTAINTY"
    )

    history = validate_uncertainty(
        oos
    )

    print_interval_validation(
        history
    )

    final_state = print_final_state(
        history
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    section(
        "4. SAVE CORRECTED UNCERTAINTY ARTIFACTS"
    )

    history.to_csv(
        OUTPUT_HISTORY_FILE,
        index=False,
    )

    manifest = {
        "artifact_type": (
            "corrected_score_uncertainty"
        ),
        "score_model_name": MODEL_NAME,
        "score_model_version": MODEL_VERSION,
        "ridge_alpha": RIDGE_ALPHA,
        "validation_seasons": (
            VALIDATION_SEASONS
        ),
        "margin_features": (
            MARGIN_FEATURES
        ),
        "total_features": (
            TOTAL_FEATURES
        ),
        "uncertainty_update_rate": (
            UPDATE_RATE
        ),
        "starting_home_score_sd": (
            STARTING_HOME_SCORE_SD
        ),
        "starting_away_score_sd": (
            STARTING_AWAY_SCORE_SD
        ),
        "starting_residual_correlation": (
            STARTING_RESIDUAL_CORRELATION
        ),
        "minimum_score_sd": (
            MIN_SCORE_SD
        ),
        "maximum_score_sd": (
            MAX_SCORE_SD
        ),
        "minimum_correlation": (
            MIN_CORRELATION
        ),
        "maximum_correlation": (
            MAX_CORRELATION
        ),
        "interval_level": (
            INTERVAL_LEVEL
        ),
        "oos_rows": int(
            len(history)
        ),
        "end_2025_state": (
            final_state
        ),
        "uses_2026_for_fitting": False,
    }

    OUTPUT_STATE_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved history -> "
        f"{OUTPUT_HISTORY_FILE}"
    )

    print(
        f"Saved state   -> "
        f"{OUTPUT_STATE_FILE}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    section(
        "CORRECTED SCORE UNCERTAINTY VALIDATION COMPLETE"
    )

    print(
        "Next decision:"
    )

    print(
        "If historical OOS interval coverage is sensible, "
        "replace the raw-score uncertainty state used by "
        "the 2026 Monte Carlo runner with this corrected "
        "end-2025 state."
    )

    print()

    print(
        "Do NOT retune the winner model or compact_full "
        "score architecture from this job."
    )


if __name__ == "__main__":
    main()
