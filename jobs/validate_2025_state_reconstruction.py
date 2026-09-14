from __future__ import annotations

from pathlib import Path
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
# ENGINE IMPORTS
# ============================================================

from engine.elo import EloConfig, EloEngine
from engine.ratings import RatingConfig, TeamRatingsEngine
from engine.opponent_adjustment import (
    OpponentAdjustmentConfig,
    OpponentAdjustmentEngine,
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_GAMES_PATH = (
    PROCESSED_DIR / "historical_games.csv"
)

HISTORICAL_TEAM_GAMES_PATH = (
    PROCESSED_DIR / "historical_team_games.csv"
)

ELO_HISTORY_PATH = (
    PROCESSED_DIR / "elo_history.csv"
)

RAW_HISTORY_PATH = (
    PROCESSED_DIR / "raw_rating_history.csv"
)

ADJUSTED_HISTORY_PATH = (
    PROCESSED_DIR / "opponent_adjusted_rating_history.csv"
)


# ============================================================
# TOLERANCE
# ============================================================

ABS_TOLERANCE = 1e-8


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def result_line(
    label: str,
    passed: bool,
    detail: str = "",
) -> None:

    status = "PASS" if passed else "FAIL"

    if detail:
        print(
            f"{status:<6} {label:<42} {detail}"
        )
    else:
        print(
            f"{status:<6} {label}"
        )


# ============================================================
# FILE CHECK
# ============================================================

def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_classification(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def normalise_bool(
    value,
) -> bool:

    if pd.isna(value):
        return False

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = str(value).strip().lower()

    return text in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


# ============================================================
# MODEL UNIVERSE
# ============================================================

def build_model_universe(
    games: pd.DataFrame,
) -> pd.DataFrame:

    data = games.copy()

    if (
        "home_classification"
        not in data.columns
        or
        "away_classification"
        not in data.columns
    ):

        raise ValueError(
            "historical_games.csv must contain "
            "home_classification and "
            "away_classification."
        )

    home_class = clean_classification(
        data["home_classification"]
    )

    away_class = clean_classification(
        data["away_classification"]
    )

    fbs_vs_fbs = (
        (home_class == "fbs")
        &
        (away_class == "fbs")
    )

    fbs_vs_fcs = (
        (
            (home_class == "fbs")
            &
            (away_class == "fcs")
        )
        |
        (
            (home_class == "fcs")
            &
            (away_class == "fbs")
        )
    )

    data = data[
        fbs_vs_fbs
        |
        fbs_vs_fcs
    ].copy()

    # Historical reconstruction must only use
    # completed games.
    data["home_points"] = pd.to_numeric(
        data["home_points"],
        errors="coerce",
    )

    data["away_points"] = pd.to_numeric(
        data["away_points"],
        errors="coerce",
    )

    data = data[
        data["home_points"].notna()
        &
        data["away_points"].notna()
    ].copy()

    data["start_date"] = pd.to_datetime(
        data["start_date"],
        errors="coerce",
        utc=True,
    )

    data = (
        data
        .sort_values(
            [
                "season",
                "week",
                "start_date",
                "cfbd_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# TEAM-GAME FILTER
# ============================================================

def filter_team_games_to_model_universe(
    team_games: pd.DataFrame,
    model_games: pd.DataFrame,
) -> pd.DataFrame:

    data = team_games.copy()

    if "id" not in model_games.columns:

        raise ValueError(
            "historical_games.csv does not "
            "contain internal game id column 'id'."
        )

    if "game_id" not in data.columns:

        raise ValueError(
            "historical_team_games.csv does "
            "not contain game_id."
        )

    valid_game_ids = set(
        pd.to_numeric(
            model_games["id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    data["game_id"] = pd.to_numeric(
        data["game_id"],
        errors="coerce",
    )

    data = data[
        data["game_id"].isin(
            valid_game_ids
        )
    ].copy()

    return data


# ============================================================
# DERIVE OPPONENT NAME
# ============================================================

def ensure_opponent_name(
    team_games: pd.DataFrame,
) -> pd.DataFrame:

    data = team_games.copy()

    if (
        "opponent_name" in data.columns
        and
        data["opponent_name"].notna().all()
    ):
        return data

    if "team_name" not in data.columns:

        raise ValueError(
            "team_games is missing team_name."
        )

    opponent_lookup = (
        data[
            [
                "game_id",
                "team_name",
            ]
        ]
        .rename(
            columns={
                "team_name":
                "opponent_name",
            }
        )
    )

    paired = data.merge(
        opponent_lookup,
        on="game_id",
        how="left",
    )

    paired = paired[
        paired["team_name"]
        !=
        paired["opponent_name"]
    ].copy()

    counts = (
        paired
        .groupby(
            [
                "game_id",
                "team_name",
            ],
            dropna=False,
        )
        .size()
    )

    if not (
        counts == 1
    ).all():

        bad = counts[
            counts != 1
        ]

        raise ValueError(
            "Could not uniquely derive opponent "
            "for every team-game row.\n"
            f"Problem rows:\n{bad.head(20)}"
        )

    return (
        paired
        .sort_values(
            [
                "season",
                "start_date",
                "game_id",
                "team_name",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ELO CONFIG
# ============================================================

def build_frozen_elo_config() -> EloConfig:

    return EloConfig(
        starting_rating=1500.0,
        home_field_advantage=75.0,
        k_factor=30.0,
        preseason_regression=0.45,
        mov_multiplier_enabled=True,
        fbs_over_fcs_weight=0.60,
        fcs_over_fbs_weight=0.70,
        fbs_fcs_mov_cap=1.80,
        minimum_rating=900.0,
        maximum_rating=2200.0,
    )


# ============================================================
# KEY RESOLUTION
# ============================================================

def resolve_single_key(
    reconstructed: pd.DataFrame,
    saved: pd.DataFrame,
    candidates: list[str],
) -> str:

    for candidate in candidates:

        if (
            candidate in reconstructed.columns
            and
            candidate in saved.columns
        ):
            return candidate

    raise ValueError(
        "Could not find common comparison key. "
        f"Tried: {candidates}"
    )


# ============================================================
# NUMERIC HISTORY COMPARISON
# ============================================================

def compare_numeric_history(
    name: str,
    reconstructed: pd.DataFrame,
    saved: pd.DataFrame,
    key_columns: list[str],
    numeric_columns: list[str],
) -> tuple[bool, dict]:

    left = reconstructed.copy()
    right = saved.copy()

    missing_keys_left = [
        column
        for column in key_columns
        if column not in left.columns
    ]

    missing_keys_right = [
        column
        for column in key_columns
        if column not in right.columns
    ]

    if (
        missing_keys_left
        or
        missing_keys_right
    ):

        return (
            False,
            {
                "reason":
                "missing comparison keys",
                "left_missing":
                missing_keys_left,
                "right_missing":
                missing_keys_right,
            },
        )

    common_numeric = [
        column
        for column in numeric_columns
        if (
            column in left.columns
            and
            column in right.columns
        )
    ]

    if not common_numeric:

        return (
            False,
            {
                "reason":
                "no requested numeric columns "
                "exist in both datasets",
            },
        )

    left = left[
        key_columns
        +
        common_numeric
    ].copy()

    right = right[
        key_columns
        +
        common_numeric
    ].copy()

    for column in key_columns:

        left[column] = (
            left[column]
            .astype("string")
            .str.strip()
        )

        right[column] = (
            right[column]
            .astype("string")
            .str.strip()
        )

    merged = left.merge(
        right,
        on=key_columns,
        how="outer",
        suffixes=(
            "_reconstructed",
            "_saved",
        ),
        indicator=True,
    )

    unmatched = merged[
        merged["_merge"]
        !=
        "both"
    ]

    max_diffs: dict[
        str,
        float,
    ] = {}

    failures: list[str] = []

    for column in common_numeric:

        reconstructed_column = (
            f"{column}_reconstructed"
        )

        saved_column = (
            f"{column}_saved"
        )

        a = pd.to_numeric(
            merged[
                reconstructed_column
            ],
            errors="coerce",
        )

        b = pd.to_numeric(
            merged[
                saved_column
            ],
            errors="coerce",
        )

        both_nan = (
            a.isna()
            &
            b.isna()
        )

        one_nan = (
            a.isna()
            ^
            b.isna()
        )

        numeric_mask = (
            ~a.isna()
            &
            ~b.isna()
        )

        if numeric_mask.any():

            diff = (
                a[numeric_mask]
                -
                b[numeric_mask]
            ).abs()

            max_diff = float(
                diff.max()
            )

        else:

            max_diff = 0.0

        max_diffs[column] = max_diff

        if one_nan.any():

            failures.append(
                f"{column}: NaN mismatch"
            )

        if max_diff > ABS_TOLERANCE:

            failures.append(
                f"{column}: "
                f"max diff {max_diff:.12g}"
            )

    passed = (
        len(unmatched) == 0
        and
        len(failures) == 0
        and
        len(left) == len(right)
    )

    return (
        passed,
        {
            "name": name,
            "reconstructed_rows":
            len(left),

            "saved_rows":
            len(right),

            "matched_rows":
            int(
                (
                    merged["_merge"]
                    ==
                    "both"
                ).sum()
            ),

            "unmatched_rows":
            len(unmatched),

            "columns_compared":
            common_numeric,

            "max_diffs":
            max_diffs,

            "failures":
            failures,
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— 2025 STATE RECONSTRUCTION VALIDATION"
    )

    print(
        "Purpose:"
    )

    print(
        "Rebuild the historical model from scratch "
        "using 2023-2025 data and verify that the "
        "result matches the already-saved official "
        "historical artifacts."
    )

    print()
    print(
        "No 2026 result is used."
    )

    print(
        "No model artifact is modified."
    )

    print(
        "No historical CSV is modified."
    )

    # ========================================================
    # FILE CHECKS
    # ========================================================

    section(
        "1. REQUIRED FILES"
    )

    required_files = [
        HISTORICAL_GAMES_PATH,
        HISTORICAL_TEAM_GAMES_PATH,
        ELO_HISTORY_PATH,
        RAW_HISTORY_PATH,
        ADJUSTED_HISTORY_PATH,
    ]

    for path in required_files:

        require_file(
            path
        )

        print(
            f"FOUND  {path.name}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    section(
        "2. LOAD HISTORICAL DATA"
    )

    historical_games = pd.read_csv(
        HISTORICAL_GAMES_PATH,
        low_memory=False,
    )

    historical_team_games = pd.read_csv(
        HISTORICAL_TEAM_GAMES_PATH,
        low_memory=False,
    )

    saved_elo_history = pd.read_csv(
        ELO_HISTORY_PATH,
        low_memory=False,
    )

    saved_raw_history = pd.read_csv(
        RAW_HISTORY_PATH,
        low_memory=False,
    )

    saved_adjusted_history = pd.read_csv(
        ADJUSTED_HISTORY_PATH,
        low_memory=False,
    )

    print(
        f"Historical games:            "
        f"{len(historical_games):,}"
    )

    print(
        f"Historical team-games:       "
        f"{len(historical_team_games):,}"
    )

    print(
        f"Saved ELO history:           "
        f"{len(saved_elo_history):,}"
    )

    print(
        f"Saved raw rating history:    "
        f"{len(saved_raw_history):,}"
    )

    print(
        f"Saved adjusted history:      "
        f"{len(saved_adjusted_history):,}"
    )

    # ========================================================
    # BUILD MODEL UNIVERSE
    # ========================================================

    section(
        "3. REBUILD MODEL UNIVERSE"
    )

    model_games = build_model_universe(
        historical_games
    )

    model_team_games = (
        filter_team_games_to_model_universe(
            historical_team_games,
            model_games,
        )
    )

    model_team_games[
        "start_date"
    ] = pd.to_datetime(
        model_team_games[
            "start_date"
        ],
        errors="coerce",
        utc=True,
    )

    model_team_games = (
        ensure_opponent_name(
            model_team_games
        )
    )

    print(
        f"Eligible games:              "
        f"{len(model_games):,}"
    )

    print(
        f"Eligible team-game rows:     "
        f"{len(model_team_games):,}"
    )

    exactly_two = (
        model_team_games
        .groupby(
            "game_id"
        )
        .size()
        .eq(2)
        .sum()
    )

    print(
        f"Games with exactly 2 rows:   "
        f"{exactly_two:,}"
    )

    # ========================================================
    # RECONSTRUCT ELO
    # ========================================================

    section(
        "4. RECONSTRUCT ELO"
    )

    elo_engine = EloEngine(
        config=build_frozen_elo_config()
    )

    reconstructed_elo = (
        elo_engine.process_games(
            model_games
        )
    )

    print(
        f"Reconstructed ELO rows:      "
        f"{len(reconstructed_elo):,}"
    )

    print(
        f"Current ELO teams:           "
        f"{len(elo_engine.current_ratings()):,}"
    )

    # ========================================================
    # RECONSTRUCT RAW RATINGS
    # ========================================================

    section(
        "5. RECONSTRUCT RAW OFFENCE / DEFENCE"
    )

    raw_engine = TeamRatingsEngine(
        config=RatingConfig()
    )

    reconstructed_raw = (
        raw_engine.process_games(
            model_team_games
        )
    )

    print(
        f"Reconstructed raw rows:      "
        f"{len(reconstructed_raw):,}"
    )

    print(
        f"Current raw teams:           "
        f"{len(raw_engine.current_ratings()):,}"
    )

    # ========================================================
    # RECONSTRUCT OPPONENT ADJUSTMENT
    # ========================================================

    section(
        "6. RECONSTRUCT OPPONENT-ADJUSTED RATINGS"
    )

    adjusted_engine = (
        OpponentAdjustmentEngine(
            config=(
                OpponentAdjustmentConfig()
            )
        )
    )

    reconstructed_adjusted = (
        adjusted_engine.process_history(
            reconstructed_raw
        )
    )

    print(
        f"Reconstructed adjusted rows: "
        f"{len(reconstructed_adjusted):,}"
    )

    print(
        f"Current adjusted teams:      "
        f"{len(adjusted_engine.current_ratings()):,}"
    )

    # ========================================================
    # ELO COMPARISON
    # ========================================================

    section(
        "7. COMPARE ELO HISTORY"
    )

    elo_key = resolve_single_key(
        reconstructed_elo,
        saved_elo_history,
        [
            "game_db_id",
            "cfbd_game_id",
            "game_id",
        ],
    )

    elo_passed, elo_details = (
        compare_numeric_history(
            name="ELO",
            reconstructed=(
                reconstructed_elo
            ),
            saved=(
                saved_elo_history
            ),
            key_columns=[
                elo_key
            ],
            numeric_columns=[
                "home_elo_pre",
                "away_elo_pre",
                "home_elo_win_probability",
                "away_elo_win_probability",
                "mov_multiplier",
                "effective_k",
                "elo_change",
                "home_elo_post",
                "away_elo_post",
            ],
        )
    )

    result_line(
        "ELO reconstruction",
        elo_passed,
        (
            f"matched="
            f"{elo_details.get('matched_rows', 0):,} "
            f"unmatched="
            f"{elo_details.get('unmatched_rows', 0):,}"
        ),
    )

    for (
        column,
        difference,
    ) in elo_details.get(
        "max_diffs",
        {},
    ).items():

        print(
            f"       {column:<36} "
            f"max diff = "
            f"{difference:.12g}"
        )

    if elo_details.get(
        "failures"
    ):

        print(
            "       Failures:"
        )

        for failure in (
            elo_details[
                "failures"
            ]
        ):

            print(
                f"       - {failure}"
            )

    # ========================================================
    # RAW RATING COMPARISON
    # ========================================================

    section(
        "8. COMPARE RAW RATING HISTORY"
    )

    raw_passed, raw_details = (
        compare_numeric_history(
            name="Raw ratings",
            reconstructed=(
                reconstructed_raw
            ),
            saved=(
                saved_raw_history
            ),
            key_columns=[
                "game_id",
                "team_name",
            ],
            numeric_columns=[
                "team_offense_rating_pre",
                "team_defense_rating_pre",
                "opponent_offense_rating_pre",
                "opponent_defense_rating_pre",
                "rating_state_games_pre",
                "rating_state_national_points_pre",
                "rating_state_national_ypp_pre",
                "rating_state_national_turnovers_pre",
                "rating_state_national_first_downs_pre",
            ],
        )
    )

    result_line(
        "Raw rating reconstruction",
        raw_passed,
        (
            f"matched="
            f"{raw_details.get('matched_rows', 0):,} "
            f"unmatched="
            f"{raw_details.get('unmatched_rows', 0):,}"
        ),
    )

    for (
        column,
        difference,
    ) in raw_details.get(
        "max_diffs",
        {},
    ).items():

        print(
            f"       {column:<36} "
            f"max diff = "
            f"{difference:.12g}"
        )

    if raw_details.get(
        "failures"
    ):

        print(
            "       Failures:"
        )

        for failure in (
            raw_details[
                "failures"
            ]
        ):

            print(
                f"       - {failure}"
            )

    # ========================================================
    # ADJUSTED RATING COMPARISON
    # ========================================================

    section(
        "9. COMPARE OPPONENT-ADJUSTED HISTORY"
    )

    adjusted_passed, adjusted_details = (
        compare_numeric_history(
            name="Opponent adjustment",
            reconstructed=(
                reconstructed_adjusted
            ),
            saved=(
                saved_adjusted_history
            ),
            key_columns=[
                "game_id",
                "team_name",
            ],
            numeric_columns=[
                "team_adjusted_offense_rating_pre",
                "team_adjusted_defense_rating_pre",
                "opponent_adjusted_offense_rating_pre",
                "opponent_adjusted_defense_rating_pre",
                "adjusted_games_pre",
                "offense_game_rating_adjusted",
                "defense_game_rating_adjusted",
                "opponent_raw_defense_factor",
                "opponent_raw_offense_factor",
            ],
        )
    )

    result_line(
        "Opponent-adjusted reconstruction",
        adjusted_passed,
        (
            f"matched="
            f"{adjusted_details.get('matched_rows', 0):,} "
            f"unmatched="
            f"{adjusted_details.get('unmatched_rows', 0):,}"
        ),
    )

    for (
        column,
        difference,
    ) in adjusted_details.get(
        "max_diffs",
        {},
    ).items():

        print(
            f"       {column:<36} "
            f"max diff = "
            f"{difference:.12g}"
        )

    if adjusted_details.get(
        "failures"
    ):

        print(
            "       Failures:"
        )

        for failure in (
            adjusted_details[
                "failures"
            ]
        ):

            print(
                f"       - {failure}"
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    section(
        "10. FINAL RECONSTRUCTION RESULT"
    )

    all_passed = (
        elo_passed
        and
        raw_passed
        and
        adjusted_passed
    )

    result_line(
        "ELO",
        elo_passed,
    )

    result_line(
        "Raw offence / defence",
        raw_passed,
    )

    result_line(
        "Opponent adjustment",
        adjusted_passed,
    )

    print()

    if all_passed:

        print(
            "✓ HISTORICAL STATE RECONSTRUCTION "
            "IS EXACT."
        )

        print()
        print(
            "The live runner can safely replay "
            "2023-2025 and continue those same "
            "engine objects into the 2026 season."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "Build jobs/run_2026_forward.py"
        )

    else:

        print(
            "✗ HISTORICAL STATE RECONSTRUCTION "
            "DOES NOT YET MATCH."
        )

        print()
        print(
            "Do NOT generate official 2026 "
            "predictions yet."
        )

        print()
        print(
            "The differences above must be "
            "resolved first."
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()