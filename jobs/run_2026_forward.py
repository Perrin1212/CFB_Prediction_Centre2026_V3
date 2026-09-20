from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENGINE IMPORTS
# ============================================================

from engine.elo import EloConfig, EloEngine
from engine.ratings import RatingConfig, TeamRatingsEngine
from engine.opponent_adjustment import (
    OpponentAdjustmentConfig,
    OpponentAdjustmentEngine,
)
from engine.matchup import MatchupConfig, MatchupEngine
from engine.production_model import ProductionProbabilityModel


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HISTORICAL_GAMES_PATH = (
    PROCESSED_DIR
    / "historical_games.csv"
)

HISTORICAL_TEAM_GAMES_PATH = (
    PROCESSED_DIR
    / "historical_team_games.csv"
)

FORWARD_GAMES_PATH = (
    PROCESSED_DIR
    / "2026_forward_games.csv"
)

FORWARD_TEAM_GAMES_PATH = (
    PROCESSED_DIR
    / "2026_forward_team_games.csv"
)

MODEL_PATH = (
    PROCESSED_DIR
    / "v2_probability_model.joblib"
)

MODEL_MANIFEST_PATH = (
    PROCESSED_DIR
    / "v2_probability_model_manifest.json"
)

OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_forward_predictions.csv"
)

COMPLETED_OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_completed_forward_predictions.csv"
)

UPCOMING_OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_upcoming_predictions.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

CURRENT_SEASON = 2026
PROBABILITY_EPSILON = 1e-9


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path: Path) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(
    value: Any,
    fallback: float = np.nan,
) -> float:

    try:

        numeric = float(value)

        if np.isfinite(numeric):
            return numeric

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(fallback)


def safe_int(
    value: Any,
    fallback: int = 0,
) -> int:

    value_float = safe_float(
        value,
        np.nan,
    )

    if not np.isfinite(value_float):
        return int(fallback)

    return int(value_float)


def normalise_name(
    value: Any,
) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_bool(
    value: Any,
) -> bool:

    if pd.isna(value):
        return False

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = (
        str(value)
        .strip()
        .lower()
    )

    return text in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def normalise_classification(
    value: Any,
) -> str:

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def probability_logit(
    probability: float,
) -> float:

    probability = float(
        np.clip(
            probability,
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
    )

    return float(
        math.log(
            probability
            /
            (
                1.0
                -
                probability
            )
        )
    )


# ============================================================
# CANONICAL 2026 GAME ID
# ============================================================

def canonical_schedule_game_id(
    row: pd.Series,
) -> int:

    """
    Return the external CFBD game ID for a 2026 schedule row.

    In 2026_forward_games.csv:
        cfbd_id = external CFBD ID
        game_id = also external CFBD ID
        id      = blank

    CFBD ID is therefore the authoritative forward-game key.
    """

    for column in (
        "cfbd_id",
        "cfbd_game_id",
        "game_id",
        "id",
    ):

        if column not in row.index:
            continue

        value = safe_float(
            row.get(column),
            np.nan,
        )

        if np.isfinite(value):
            return int(value)

    raise ValueError(
        "Could not determine canonical CFBD "
        "game ID for schedule row."
    )


def canonical_team_game_id_column(
    data: pd.DataFrame,
) -> str:

    """
    Determine the column in 2026 team-game data that contains
    the external CFBD game ID.

    2026_forward_team_games.csv contains:
        game_id      -> internal/local ID, e.g. 30
        cfbd_game_id -> external CFBD ID, e.g. 401856766

    So cfbd_game_id MUST be preferred.
    """

    if "cfbd_game_id" in data.columns:

        values = pd.to_numeric(
            data["cfbd_game_id"],
            errors="coerce",
        )

        if values.notna().any():
            return "cfbd_game_id"

    if "cfbd_id" in data.columns:

        values = pd.to_numeric(
            data["cfbd_id"],
            errors="coerce",
        )

        if values.notna().any():
            return "cfbd_id"

    if "game_id" in data.columns:

        return "game_id"

    raise ValueError(
        "Could not identify a game ID column "
        "in 2026 team-game data."
    )


# ============================================================
# FROZEN ELO CONFIGURATION
# ============================================================

def build_frozen_elo_config() -> EloConfig:

    return EloConfig(
        starting_rating=1500.0,
        fcs_starting_rating=1100.0,
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
# HISTORICAL MODEL UNIVERSE
# ============================================================

def build_historical_model_universe(
    games: pd.DataFrame,
) -> pd.DataFrame:

    data = games.copy()

    home_class = (
        data["home_classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    away_class = (
        data["away_classification"]
        .astype("string")
        .str.strip()
        .str.lower()
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

    sort_columns = [
        "season",
        "week",
        "start_date",
    ]

    if "cfbd_id" in data.columns:

        sort_columns.append(
            "cfbd_id"
        )

    return (
        data
        .sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HISTORICAL TEAM GAME PREPARATION
# ============================================================

def filter_historical_team_games(
    team_games: pd.DataFrame,
    model_games: pd.DataFrame,
) -> pd.DataFrame:

    if "id" not in model_games.columns:

        raise ValueError(
            "historical_games.csv is missing "
            "internal game id column 'id'."
        )

    data = team_games.copy()

    data["game_id"] = pd.to_numeric(
        data["game_id"],
        errors="coerce",
    )

    valid_ids = set(
        pd.to_numeric(
            model_games["id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    data = data[
        data["game_id"].isin(
            valid_ids
        )
    ].copy()

    data["start_date"] = pd.to_datetime(
        data["start_date"],
        errors="coerce",
        utc=True,
    )

    return data


# ============================================================
# DERIVE OPPONENT
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

        data["opponent_name"] = (
            data["opponent_name"]
            .map(normalise_name)
        )

        return data

    if (
        "opponent_team" in data.columns
        and
        data["opponent_team"].notna().all()
    ):

        data["opponent_name"] = (
            data["opponent_team"]
            .map(normalise_name)
        )

        return data

    if "team_name" not in data.columns:

        raise ValueError(
            "Team-game data is missing team_name."
        )

    original_columns = list(
        data.columns
    )

    opponent_lookup = (
        data[
            [
                "game_id",
                "team_name",
            ]
        ]
        .copy()
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

    paired["team_name"] = (
        paired["team_name"]
        .map(normalise_name)
    )

    paired["opponent_name"] = (
        paired["opponent_name"]
        .map(normalise_name)
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

    if not counts.eq(1).all():

        bad = counts[
            counts != 1
        ]

        raise ValueError(
            "Opponent identity could not be "
            "uniquely derived.\n"
            f"{bad.head(20)}"
        )

    desired_columns = (
        original_columns
        +
        ["opponent_name"]
    )

    paired = paired[
        [
            column
            for column
            in desired_columns
            if column in paired.columns
        ]
    ]

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
# LOAD MODEL VERSION
# ============================================================

def read_model_version() -> str:

    if not MODEL_MANIFEST_PATH.exists():

        return "CFB-V2-2026.1"

    try:

        with MODEL_MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:

            manifest = json.load(
                handle
            )

        for key in (
            "model_version",
            "version",
            "name",
        ):

            value = manifest.get(
                key
            )

            if value:
                return str(value)

    except Exception:
        pass

    return "CFB-V2-2026.1"


# ============================================================
# COMPLETED GAME
# ============================================================

def is_completed_game(
    row: pd.Series,
) -> bool:

    home_points = safe_float(
        row.get(
            "home_points"
        ),
        np.nan,
    )

    away_points = safe_float(
        row.get(
            "away_points"
        ),
        np.nan,
    )

    return bool(
        np.isfinite(
            home_points
        )
        and
        np.isfinite(
            away_points
        )
    )


# ============================================================
# TEAM GAME LOOKUP
# ============================================================

def build_team_game_lookup(
    forward_team_games: pd.DataFrame,
) -> tuple[
    dict[int, pd.DataFrame],
    str,
]:

    lookup: dict[
        int,
        pd.DataFrame,
    ] = {}

    if forward_team_games.empty:

        return (
            lookup,
            "",
        )

    key_column = (
        canonical_team_game_id_column(
            forward_team_games
        )
    )

    data = forward_team_games.copy()

    data[
        "_canonical_cfbd_game_id"
    ] = pd.to_numeric(
        data[key_column],
        errors="coerce",
    )

    invalid = data[
        "_canonical_cfbd_game_id"
    ].isna()

    if invalid.any():

        raise ValueError(
            "2026 team-game data contains "
            f"{int(invalid.sum())} rows without "
            "a valid canonical game ID."
        )

    data[
        "_canonical_cfbd_game_id"
    ] = (
        data[
            "_canonical_cfbd_game_id"
        ]
        .astype(int)
    )

    for game_id, rows in (
        data.groupby(
            "_canonical_cfbd_game_id",
            sort=False,
        )
    ):

        lookup[
            int(game_id)
        ] = (
            rows
            .drop(
                columns=[
                    "_canonical_cfbd_game_id"
                ],
                errors="ignore",
            )
            .copy()
        )

    return (
        lookup,
        key_column,
    )


# ============================================================
# PREGAME TEAM ROW FOR MATCHUP ENGINE
# ============================================================

def build_matchup_team_row(
    game_row: pd.Series,
    team_name: str,
    home_away: str,
    raw_snapshot: dict[str, Any],
    adjusted_snapshot: dict[str, Any],
) -> pd.Series:

    if home_away == "home":

        points = game_row.get(
            "home_points"
        )

    else:

        points = game_row.get(
            "away_points"
        )

    return pd.Series(
        {
            "game_id":
            canonical_schedule_game_id(
                game_row
            ),

            "season":
            safe_int(
                game_row.get(
                    "season"
                ),
                CURRENT_SEASON,
            ),

            "week":
            safe_int(
                game_row.get(
                    "week"
                ),
                0,
            ),

            "start_date":
            game_row.get(
                "start_date"
            ),

            "neutral_site":
            normalise_bool(
                game_row.get(
                    "neutral_site"
                )
            ),

            "conference_game":
            normalise_bool(
                game_row.get(
                    "conference_game"
                )
            ),

            "home_away":
            home_away,

            "team_name":
            team_name,

            "team_points":
            points,

            "team_offense_rating_pre":
            raw_snapshot[
                "offense_rating_pre"
            ],

            "team_defense_rating_pre":
            raw_snapshot[
                "defense_rating_pre"
            ],

            "team_adjusted_offense_rating_pre":
            adjusted_snapshot[
                "adjusted_offense_rating_pre"
            ],

            "team_adjusted_defense_rating_pre":
            adjusted_snapshot[
                "adjusted_defense_rating_pre"
            ],

            "adjusted_games_pre":
            adjusted_snapshot[
                "games_pre"
            ],
        }
    )


# ============================================================
# ACTUAL STAT HELPERS
# ============================================================

def actual_stat_value(
    row: pd.Series,
    candidates: list[str],
    fallback: float,
) -> float:

    for column in candidates:

        if column not in row.index:
            continue

        value = safe_float(
            row.get(
                column
            ),
            np.nan,
        )

        if np.isfinite(value):
            return value

    return float(fallback)


def find_team_row(
    rows: pd.DataFrame,
    team_name: str,
    home_away: str,
) -> pd.Series:

    team_name = normalise_name(
        team_name
    )

    if "team_name" in rows.columns:

        matches = rows[
            rows["team_name"]
            .map(normalise_name)
            ==
            team_name
        ]

        if len(matches) == 1:

            return matches.iloc[0]

    if "home_away" in rows.columns:

        matches = rows[
            rows["home_away"]
            .astype("string")
            .str.strip()
            .str.lower()
            ==
            home_away
        ]

        if len(matches) == 1:

            return matches.iloc[0]

    raise ValueError(
        f"Could not identify {home_away} "
        f"team row for {team_name}."
    )


# ============================================================
# BUILD ADJUSTMENT PERFORMANCE ROW
# ============================================================

def build_adjustment_row(
    team_row: pd.Series,
    opponent_row: pd.Series,
    raw_snapshot: dict[str, Any],
    opponent_raw_snapshot: dict[str, Any],
    team_points: float,
    opponent_points: float,
) -> pd.Series:

    team_ypp = actual_stat_value(
        team_row,
        [
            "yards_per_play_proxy",
            "yards_per_play",
        ],
        raw_snapshot[
            "national_ypp_pre"
        ],
    )

    team_turnovers = actual_stat_value(
        team_row,
        [
            "turnovers",
        ],
        raw_snapshot[
            "national_turnovers_pre"
        ],
    )

    team_first_downs = actual_stat_value(
        team_row,
        [
            "first_downs",
        ],
        raw_snapshot[
            "national_first_downs_pre"
        ],
    )

    opponent_ypp = actual_stat_value(
        opponent_row,
        [
            "yards_per_play_proxy",
            "yards_per_play",
        ],
        raw_snapshot[
            "national_ypp_pre"
        ],
    )

    opponent_turnovers = actual_stat_value(
        opponent_row,
        [
            "turnovers",
        ],
        raw_snapshot[
            "national_turnovers_pre"
        ],
    )

    opponent_first_downs = actual_stat_value(
        opponent_row,
        [
            "first_downs",
        ],
        raw_snapshot[
            "national_first_downs_pre"
        ],
    )

    return pd.Series(
        {
            "team_points":
            team_points,

            "opponent_points":
            opponent_points,

            "yards_per_play_proxy":
            team_ypp,

            "turnovers":
            team_turnovers,

            "first_downs":
            team_first_downs,

            "opponent_yards_per_play":
            opponent_ypp,

            "opponent_turnovers":
            opponent_turnovers,

            "opponent_first_downs":
            opponent_first_downs,

            "opponent_offense_rating_pre":
            opponent_raw_snapshot[
                "offense_rating_pre"
            ],

            "opponent_defense_rating_pre":
            opponent_raw_snapshot[
                "defense_rating_pre"
            ],

            "rating_state_national_points_pre":
            raw_snapshot[
                "national_points_pre"
            ],

            "rating_state_national_ypp_pre":
            raw_snapshot[
                "national_ypp_pre"
            ],

            "rating_state_national_turnovers_pre":
            raw_snapshot[
                "national_turnovers_pre"
            ],

            "rating_state_national_first_downs_pre":
            raw_snapshot[
                "national_first_downs_pre"
            ],
        }
    )


# ============================================================
# PREDICT ONE GAME
# ============================================================

def predict_one_game(
    game_row: pd.Series,
    elo_engine: EloEngine,
    raw_engine: TeamRatingsEngine,
    adjusted_engine: OpponentAdjustmentEngine,
    matchup_engine: MatchupEngine,
    production_model: ProductionProbabilityModel,
    model_version: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    season = safe_int(
        game_row.get(
            "season"
        ),
        CURRENT_SEASON,
    )

    home_team = normalise_name(
        game_row.get(
            "home_team"
        )
    )

    away_team = normalise_name(
        game_row.get(
            "away_team"
        )
    )

    if (
        not home_team
        or
        not away_team
    ):

        raise ValueError(
            "Game has missing home/away team."
        )

    neutral_site = normalise_bool(
        game_row.get(
            "neutral_site"
        )
    )

    canonical_game_id = (
        canonical_schedule_game_id(
            game_row
        )
    )

    # ========================================================
    # ELO PREGAME STATE
    # ========================================================

    home_elo = elo_engine.get_rating(
        home_team,
        game_row.get("home_classification"),
    )

    away_elo = elo_engine.get_rating(
        away_team,
        game_row.get("away_classification"),
    )

    elo_probability = (
        elo_engine.expected_home_win(
            home_rating=home_elo,
            away_rating=away_elo,
            neutral_site=neutral_site,
        )
    )

    elo_logit = probability_logit(
        elo_probability
    )

    # ========================================================
    # RAW PREGAME SNAPSHOTS
    # ========================================================

    home_raw = raw_engine.team_snapshot(
        team=home_team,
        season=season,
    )

    away_raw = raw_engine.team_snapshot(
        team=away_team,
        season=season,
    )

    # ========================================================
    # OPPONENT-ADJUSTED PREGAME SNAPSHOTS
    # ========================================================

    home_adjusted = (
        adjusted_engine.snapshot(
            team=home_team,
            season=season,
        )
    )

    away_adjusted = (
        adjusted_engine.snapshot(
            team=away_team,
            season=season,
        )
    )

    # ========================================================
    # MATCHUP
    # ========================================================

    home_matchup_row = (
        build_matchup_team_row(
            game_row=game_row,
            team_name=home_team,
            home_away="home",
            raw_snapshot=home_raw,
            adjusted_snapshot=home_adjusted,
        )
    )

    away_matchup_row = (
        build_matchup_team_row(
            game_row=game_row,
            team_name=away_team,
            home_away="away",
            raw_snapshot=away_raw,
            adjusted_snapshot=away_adjusted,
        )
    )

    elo_row = pd.Series(
        {
            "home_elo_pre":
            home_elo,

            "away_elo_pre":
            away_elo,

            "home_elo_win_probability":
            elo_probability,
        }
    )

    matchup = (
        matchup_engine
        .build_game_matchup(
            home_row=home_matchup_row,
            away_row=away_matchup_row,
            elo_row=elo_row,
        )
    )

    offensive_matchup_difference = (
        safe_float(
            matchup.get(
                "offensive_matchup_difference"
            ),
            0.0,
        )
    )

        # ========================================================
    # FROZEN PRODUCTION MODEL
    # ========================================================

    model_input = pd.DataFrame(
        [
            {
                "home_elo_win_probability":
                elo_probability,

                "elo_logit":
                elo_logit,

                "offensive_matchup_difference":
                offensive_matchup_difference,
            }
        ]
    )

    home_probability = float(
        production_model
        .predict_home_probability(
            model_input
        )[0]
    )

    home_probability = float(
        np.clip(
            home_probability,
            0.0,
            1.0,
        )
    )

    away_probability = (
        1.0
        -
        home_probability
    )

    predicted_winner = (
        home_team
        if home_probability >= 0.50
        else away_team
    )

    predicted_probability = max(
        home_probability,
        away_probability,
    )

    # ========================================================
    # ACTUAL RESULT
    # ========================================================

    completed = is_completed_game(
        game_row
    )

    home_points = safe_float(
        game_row.get(
            "home_points"
        ),
        np.nan,
    )

    away_points = safe_float(
        game_row.get(
            "away_points"
        ),
        np.nan,
    )

    actual_winner = ""
    prediction_correct = np.nan
    actual_home_result = np.nan

    if completed:

        if home_points > away_points:

            actual_winner = home_team
            actual_home_result = 1.0

        elif away_points > home_points:

            actual_winner = away_team
            actual_home_result = 0.0

        else:

            actual_winner = "TIE"
            actual_home_result = 0.5

        if actual_winner == "TIE":

            prediction_correct = 0.5

        else:

            prediction_correct = float(
                predicted_winner
                ==
                actual_winner
            )

    # ========================================================
    # OUTPUT RECORD
    # ========================================================

    prediction = {
        "model_version":
        model_version,

        "season":
        season,

        "week":
        safe_int(
            game_row.get(
                "week"
            ),
            0,
        ),

        "season_type":
        game_row.get(
            "season_type"
        ),

        "game_id":
        canonical_game_id,

        "cfbd_game_id":
        canonical_game_id,

        "start_date":
        game_row.get(
            "start_date"
        ),

        "home_team":
        home_team,

        "away_team":
        away_team,

        "neutral_site":
        neutral_site,

        "conference_game":
        normalise_bool(
            game_row.get(
                "conference_game"
            )
        ),

        "home_classification":
        normalise_classification(
            game_row.get(
                "home_classification"
            )
        ),

        "away_classification":
        normalise_classification(
            game_row.get(
                "away_classification"
            )
        ),

        # --------------------------------------------
        # WIN PROBABILITY
        # --------------------------------------------

        "home_win_probability":
        home_probability,

        "away_win_probability":
        away_probability,

        "predicted_winner":
        predicted_winner,

        "predicted_winner_probability":
        predicted_probability,

        # --------------------------------------------
        # ELO
        # --------------------------------------------

        "home_elo_pre":
        home_elo,

        "away_elo_pre":
        away_elo,

        "elo_difference":
        home_elo
        -
        away_elo,

        "home_elo_win_probability":
        elo_probability,

        "elo_logit":
        elo_logit,

        # --------------------------------------------
        # RAW RATINGS
        # --------------------------------------------

        "home_raw_offense_rating_pre":
        home_raw[
            "offense_rating_pre"
        ],

        "home_raw_defense_rating_pre":
        home_raw[
            "defense_rating_pre"
        ],

        "away_raw_offense_rating_pre":
        away_raw[
            "offense_rating_pre"
        ],

        "away_raw_defense_rating_pre":
        away_raw[
            "defense_rating_pre"
        ],

        # --------------------------------------------
        # ADJUSTED RATINGS
        # --------------------------------------------

        "home_adjusted_offense_rating_pre":
        home_adjusted[
            "adjusted_offense_rating_pre"
        ],

        "home_adjusted_defense_rating_pre":
        home_adjusted[
            "adjusted_defense_rating_pre"
        ],

        "away_adjusted_offense_rating_pre":
        away_adjusted[
            "adjusted_offense_rating_pre"
        ],

        "away_adjusted_defense_rating_pre":
        away_adjusted[
            "adjusted_defense_rating_pre"
        ],

        "home_adjusted_games_pre":
        home_adjusted[
            "games_pre"
        ],

        "away_adjusted_games_pre":
        away_adjusted[
            "games_pre"
        ],

        # --------------------------------------------
        # MATCHUP
        # --------------------------------------------

        "home_offensive_matchup_index":
        matchup.get(
            "home_offensive_matchup_index"
        ),

        "away_offensive_matchup_index":
        matchup.get(
            "away_offensive_matchup_index"
        ),

        "offensive_matchup_difference":
        offensive_matchup_difference,

        "home_adjusted_overall_strength":
        matchup.get(
            "home_adjusted_overall_strength"
        ),

        "away_adjusted_overall_strength":
        matchup.get(
            "away_adjusted_overall_strength"
        ),

        "adjusted_strength_difference":
        matchup.get(
            "adjusted_strength_difference"
        ),

        "matchup_rating_maturity":
        matchup.get(
            "matchup_rating_maturity"
        ),

        # --------------------------------------------
        # RESULT
        # --------------------------------------------

        "completed":
        completed,

        "home_points":
        (
            home_points
            if completed
            else np.nan
        ),

        "away_points":
        (
            away_points
            if completed
            else np.nan
        ),

        "actual_winner":
        actual_winner,

        "actual_home_result":
        actual_home_result,

        "prediction_correct":
        prediction_correct,
    }

    state_context = {
        "canonical_game_id":
        canonical_game_id,

        "home_team":
        home_team,

        "away_team":
        away_team,

        "season":
        season,

        "neutral_site":
        neutral_site,

        "home_classification":
        normalise_classification(
            game_row.get(
                "home_classification"
            )
        ),

        "away_classification":
        normalise_classification(
            game_row.get(
                "away_classification"
            )
        ),

        "home_points":
        home_points,

        "away_points":
        away_points,

        "home_raw_snapshot":
        home_raw,

        "away_raw_snapshot":
        away_raw,
    }

    return (
        prediction,
        state_context,
    )


# ============================================================
# UPDATE COMPLETED GAME
# ============================================================

def update_completed_game(
    game_row: pd.Series,
    state_context: dict[str, Any],
    team_rows: pd.DataFrame,
    elo_engine: EloEngine,
    raw_engine: TeamRatingsEngine,
    adjusted_engine: OpponentAdjustmentEngine,
) -> None:

    if len(team_rows) != 2:

        raise ValueError(
            "Completed game does not have exactly "
            f"two team-stat rows: "
            f"{state_context['canonical_game_id']}"
        )

    home_team = state_context[
        "home_team"
    ]

    away_team = state_context[
        "away_team"
    ]

    season = state_context[
        "season"
    ]

    home_points = state_context[
        "home_points"
    ]

    away_points = state_context[
        "away_points"
    ]

    home_raw = state_context[
        "home_raw_snapshot"
    ]

    away_raw = state_context[
        "away_raw_snapshot"
    ]

    home_team_row = find_team_row(
        rows=team_rows,
        team_name=home_team,
        home_away="home",
    )

    away_team_row = find_team_row(
        rows=team_rows,
        team_name=away_team,
        home_away="away",
    )

    # ========================================================
    # ACTUAL STATS
    # ========================================================

    home_ypp = actual_stat_value(
        home_team_row,
        [
            "yards_per_play_proxy",
            "yards_per_play",
        ],
        home_raw[
            "national_ypp_pre"
        ],
    )

    away_ypp = actual_stat_value(
        away_team_row,
        [
            "yards_per_play_proxy",
            "yards_per_play",
        ],
        away_raw[
            "national_ypp_pre"
        ],
    )

    home_turnovers = actual_stat_value(
        home_team_row,
        ["turnovers"],
        home_raw[
            "national_turnovers_pre"
        ],
    )

    away_turnovers = actual_stat_value(
        away_team_row,
        ["turnovers"],
        away_raw[
            "national_turnovers_pre"
        ],
    )

    home_first_downs = actual_stat_value(
        home_team_row,
        ["first_downs"],
        home_raw[
            "national_first_downs_pre"
        ],
    )

    away_first_downs = actual_stat_value(
        away_team_row,
        ["first_downs"],
        away_raw[
            "national_first_downs_pre"
        ],
    )

    # ========================================================
    # ADJUSTED GAME PERFORMANCE
    # ========================================================

    home_adjustment_row = (
        build_adjustment_row(
            team_row=home_team_row,
            opponent_row=away_team_row,
            raw_snapshot=home_raw,
            opponent_raw_snapshot=away_raw,
            team_points=home_points,
            opponent_points=away_points,
        )
    )

    away_adjustment_row = (
        build_adjustment_row(
            team_row=away_team_row,
            opponent_row=home_team_row,
            raw_snapshot=away_raw,
            opponent_raw_snapshot=home_raw,
            team_points=away_points,
            opponent_points=home_points,
        )
    )

    home_adjusted_offense_game = (
        adjusted_engine
        .offense_game_rating(
            home_adjustment_row
        )
    )

    home_adjusted_defense_game = (
        adjusted_engine
        .defense_game_rating(
            home_adjustment_row
        )
    )

    away_adjusted_offense_game = (
        adjusted_engine
        .offense_game_rating(
            away_adjustment_row
        )
    )

    away_adjusted_defense_game = (
        adjusted_engine
        .defense_game_rating(
            away_adjustment_row
        )
    )

    # ========================================================
    # ELO UPDATE
    # ========================================================

    elo_engine.update_game(
        home_team=home_team,
        away_team=away_team,
        home_points=home_points,
        away_points=away_points,
        neutral_site=state_context[
            "neutral_site"
        ],
        home_classification=(
            state_context[
                "home_classification"
            ]
        ),
        away_classification=(
            state_context[
                "away_classification"
            ]
        ),
    )

    # ========================================================
    # RAW TEAM UPDATE
    # ========================================================

    raw_engine.update_team(
        team=home_team,
        season=season,
        points=home_points,
        yards_per_play=home_ypp,
        turnovers=home_turnovers,
        first_downs=home_first_downs,
        opponent_points=away_points,
        opponent_yards_per_play=away_ypp,
        opponent_turnovers=away_turnovers,
        opponent_first_downs=away_first_downs,
    )

    raw_engine.update_team(
        team=away_team,
        season=season,
        points=away_points,
        yards_per_play=away_ypp,
        turnovers=away_turnovers,
        first_downs=away_first_downs,
        opponent_points=home_points,
        opponent_yards_per_play=home_ypp,
        opponent_turnovers=home_turnovers,
        opponent_first_downs=home_first_downs,
    )

    # ========================================================
    # NATIONAL BASELINE UPDATE
    # ========================================================

    raw_engine.update_national(
        points=home_points,
        yards_per_play=home_ypp,
        turnovers=home_turnovers,
        first_downs=home_first_downs,
    )

    raw_engine.update_national(
        points=away_points,
        yards_per_play=away_ypp,
        turnovers=away_turnovers,
        first_downs=away_first_downs,
    )

    # ========================================================
    # OPPONENT ADJUSTMENT UPDATE
    # ========================================================

    adjusted_engine.update(
        team=home_team,
        season=season,
        offense_game_rating=(
            home_adjusted_offense_game
        ),
        defense_game_rating=(
            home_adjusted_defense_game
        ),
    )

    adjusted_engine.update(
        team=away_team,
        season=season,
        offense_game_rating=(
            away_adjusted_offense_game
        ),
        defense_game_rating=(
            away_adjusted_defense_game
        ),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_completed_metrics(
    completed: pd.DataFrame,
) -> dict[str, float]:

    if completed.empty:

        return {}

    valid = completed[
        completed[
            "actual_home_result"
        ].notna()
    ].copy()

    if valid.empty:

        return {}

    y = pd.to_numeric(
        valid[
            "actual_home_result"
        ],
        errors="coerce",
    )

    p = pd.to_numeric(
        valid[
            "home_win_probability"
        ],
        errors="coerce",
    )

    mask = (
        y.notna()
        &
        p.notna()
    )

    y = y[
        mask
    ]

    p = p[
        mask
    ]

    if len(y) == 0:

        return {}

    p_clipped = np.clip(
        p.to_numpy(
            dtype=float
        ),
        1e-12,
        1.0 - 1e-12,
    )

    y_values = y.to_numpy(
        dtype=float
    )

    brier = float(
        np.mean(
            (
                p_clipped
                -
                y_values
            )
            ** 2
        )
    )

    log_loss = float(
        -np.mean(
            (
                y_values
                *
                np.log(
                    p_clipped
                )
            )
            +
            (
                1.0
                -
                y_values
            )
            *
            np.log(
                1.0
                -
                p_clipped
            )
        )
    )

    non_ties = valid[
        valid[
            "actual_winner"
        ]
        !=
        "TIE"
    ]

    if len(non_ties):

        accuracy = float(
            non_ties[
                "prediction_correct"
            ]
            .astype(float)
            .mean()
        )

    else:

        accuracy = np.nan

    return {
        "games":
        float(len(valid)),

        "accuracy":
        accuracy,

        "brier":
        brier,

        "log_loss":
        log_loss,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— LIVE FORWARD RUN"
    )

    print(
        "Architecture:"
    )

    print(
        "ELO + opponent-adjusted offensive "
        "matchup -> frozen win probability model"
    )

    print()
    print(
        "Historical model architecture is frozen."
    )

    print(
        "2026 results are used only to update "
        "state AFTER each pregame prediction."
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    section(
        "1. REQUIRED FILES"
    )

    for path in (
        HISTORICAL_GAMES_PATH,
        HISTORICAL_TEAM_GAMES_PATH,
        FORWARD_GAMES_PATH,
        FORWARD_TEAM_GAMES_PATH,
        MODEL_PATH,
    ):

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
        "2. LOAD DATA"
    )

    historical_games = pd.read_csv(
        HISTORICAL_GAMES_PATH,
        low_memory=False,
    )

    historical_team_games = pd.read_csv(
        HISTORICAL_TEAM_GAMES_PATH,
        low_memory=False,
    )

    forward_games = pd.read_csv(
        FORWARD_GAMES_PATH,
        low_memory=False,
    )

    forward_team_games = pd.read_csv(
        FORWARD_TEAM_GAMES_PATH,
        low_memory=False,
    )

    print(
        f"Historical games:           "
        f"{len(historical_games):,}"
    )

    print(
        f"Historical team-games:      "
        f"{len(historical_team_games):,}"
    )

    print(
        f"2026 eligible games:        "
        f"{len(forward_games):,}"
    )

    print(
        f"2026 completed team rows:   "
        f"{len(forward_team_games):,}"
    )

    # ========================================================
    # PREPARE HISTORICAL REPLAY
    # ========================================================

    section(
        "3. REPLAY 2023-2025 MODEL STATE"
    )

    historical_model_games = (
        build_historical_model_universe(
            historical_games
        )
    )

    historical_model_team_games = (
        filter_historical_team_games(
            historical_team_games,
            historical_model_games,
        )
    )

    historical_model_team_games = (
        ensure_opponent_name(
            historical_model_team_games
        )
    )

    print(
        f"Historical model games:     "
        f"{len(historical_model_games):,}"
    )

    print(
        f"Historical team rows:       "
        f"{len(historical_model_team_games):,}"
    )

    # ========================================================
    # INITIALISE ENGINES
    # ========================================================

    elo_engine = EloEngine(
        config=build_frozen_elo_config()
    )

    raw_engine = TeamRatingsEngine(
        config=RatingConfig()
    )

    adjusted_engine = (
        OpponentAdjustmentEngine(
            config=(
                OpponentAdjustmentConfig()
            )
        )
    )

    matchup_engine = MatchupEngine(
        config=MatchupConfig()
    )

    production_model = (
        ProductionProbabilityModel.load(
            MODEL_PATH
        )
    )

    model_version = (
        read_model_version()
    )

    # ========================================================
    # HISTORICAL STATE REPLAY
    # ========================================================

    elo_history = (
        elo_engine.process_games(
            historical_model_games
        )
    )

    raw_history = (
        raw_engine.process_games(
            historical_model_team_games
        )
    )

    adjusted_history = (
        adjusted_engine.process_history(
            raw_history
        )
    )

    print(
        f"ELO replay rows:            "
        f"{len(elo_history):,}"
    )

    print(
        f"Raw rating replay rows:     "
        f"{len(raw_history):,}"
    )

    print(
        f"Adjusted replay rows:       "
        f"{len(adjusted_history):,}"
    )

    print(
        f"End-2025 ELO teams:         "
        f"{len(elo_engine.current_ratings()):,}"
    )

    print(
        f"End-2025 raw teams:         "
        f"{len(raw_engine.current_ratings()):,}"
    )

    print(
        f"End-2025 adjusted teams:    "
        f"{len(adjusted_engine.current_ratings()):,}"
    )

    # ========================================================
    # 2026 PRESEASON TRANSITION
    # ========================================================

    section(
        "4. APPLY 2026 PRESEASON TRANSITION"
    )

    elo_engine.regress_for_new_season()

    print(
        "ELO preseason regression applied once."
    )

    print(
        "Raw ratings will regress to 2026 on each "
        "team's first 2026 snapshot."
    )

    print(
        "Opponent-adjusted ratings will regress "
        "the same way."
    )

    # ========================================================
    # PREPARE 2026 SCHEDULE
    # ========================================================

    forward_games[
        "start_date"
    ] = pd.to_datetime(
        forward_games[
            "start_date"
        ],
        errors="coerce",
        utc=True,
    )

    if forward_games[
        "start_date"
    ].isna().any():

        raise ValueError(
            "2026 forward games contain invalid "
            "start_date values."
        )

    forward_games[
        "_canonical_cfbd_game_id"
    ] = (
        forward_games.apply(
            canonical_schedule_game_id,
            axis=1,
        )
    )

    forward_games = (
        forward_games
        .sort_values(
            [
                "start_date",
                "_canonical_cfbd_game_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if not forward_team_games.empty:

        forward_team_games[
            "start_date"
        ] = pd.to_datetime(
            forward_team_games[
                "start_date"
            ],
            errors="coerce",
            utc=True,
        )

        forward_team_games = (
            ensure_opponent_name(
                forward_team_games
            )
        )

    (
        team_game_lookup,
        team_game_key_column,
    ) = build_team_game_lookup(
        forward_team_games
    )

    completed_schedule_games = int(
        forward_games
        .apply(
            is_completed_game,
            axis=1,
        )
        .sum()
    )

    print()
    print(
        f"2026 schedule games:        "
        f"{len(forward_games):,}"
    )

    print(
        f"Completed schedule games:   "
        f"{completed_schedule_games:,}"
    )

    print(
        f"Upcoming schedule games:    "
        f"{len(forward_games) - completed_schedule_games:,}"
    )

    print()
    print(
        "Forward schedule join key:  "
        "CFBD game ID"
    )

    print(
        "Team-stat source key:       "
        f"{team_game_key_column}"
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    section(
        "5. COMPLETED GAME BOX-SCORE SAFETY CHECK"
    )

    missing_completed_stats: list[
        int
    ] = []

    invalid_completed_stats: list[
        tuple[int, int]
    ] = []

    for _, game_row in (
        forward_games.iterrows()
    ):

        if not is_completed_game(
            game_row
        ):

            continue

        game_id = (
            canonical_schedule_game_id(
                game_row
            )
        )

        rows = team_game_lookup.get(
            game_id
        )

        if rows is None:

            missing_completed_stats.append(
                game_id
            )

            continue

        if len(rows) != 2:

            invalid_completed_stats.append(
                (
                    game_id,
                    len(rows),
                )
            )

    if missing_completed_stats:

        print(
            "WARNING — some completed eligible games have a final "
            "score but no CFBD team box-score rows."
        )

        print(
            "These games will update ELO from the verified final "
            "score, but will NOT update raw or opponent-adjusted "
            "stat ratings."
        )

        print(
            "Missing CFBD game IDs: "
            f"{missing_completed_stats}"
        )

    if invalid_completed_stats:

        raise ValueError(
            "Some completed 2026 games do not "
            "have exactly two team rows.\n"
            f"{invalid_completed_stats}"
        )

    print(
        "PASS — every completed eligible game "
        "matched its team stats using CFBD game ID."
    )

    print(
        "PASS — every completed eligible game "
        "has exactly two team-stat rows."
    )

    # ========================================================
    # CHRONOLOGICAL FORWARD RUN
    # ========================================================

    section(
        "6. GENERATE LEAKAGE-SAFE 2026 PREDICTIONS"
    )

    prediction_rows: list[
        dict[str, Any]
    ] = []

    completed_updates = 0

    timestamp_groups = (
        forward_games.groupby(
            "start_date",
            sort=True,
        )
    )

    for (
        start_date,
        timestamp_games,
    ) in timestamp_groups:

        timestamp_predictions: list[
            tuple[
                pd.Series,
                dict[str, Any],
            ]
        ] = []

        # ----------------------------------------------------
        # PREDICT ALL SAME-KICKOFF GAMES FIRST
        # ----------------------------------------------------

        for _, game_row in (
            timestamp_games.iterrows()
        ):

            (
                prediction,
                state_context,
            ) = predict_one_game(
                game_row=game_row,
                elo_engine=elo_engine,
                raw_engine=raw_engine,
                adjusted_engine=adjusted_engine,
                matchup_engine=matchup_engine,
                production_model=production_model,
                model_version=model_version,
            )

            prediction_rows.append(
                prediction
            )

            timestamp_predictions.append(
                (
                    game_row,
                    state_context,
                )
            )

        # ----------------------------------------------------
        # UPDATE ONLY AFTER ALL SAME-TIME PREDICTIONS
        # ----------------------------------------------------

        for (
            game_row,
            state_context,
        ) in timestamp_predictions:

            if not is_completed_game(
                game_row
            ):

                continue

            game_id = (
                state_context[
                    "canonical_game_id"
                ]
            )

            team_rows = (
                team_game_lookup.get(
                    game_id
                )
            )

            if team_rows is None:

                # CFBD occasionally publishes a verified final score
                # without publishing team box-score statistics.
                #
                # Preserve the result-dependent ELO update, which only
                # requires the final score and game context. Do not
                # fabricate statistical inputs for the raw-rating or
                # opponent-adjustment engines.

                elo_engine.update_game(
                    home_team=state_context[
                        "home_team"
                    ],
                    away_team=state_context[
                        "away_team"
                    ],
                    home_points=state_context[
                        "home_points"
                    ],
                    away_points=state_context[
                        "away_points"
                    ],
                    neutral_site=state_context[
                        "neutral_site"
                    ],
                    home_classification=(
                        state_context[
                            "home_classification"
                        ]
                    ),
                    away_classification=(
                        state_context[
                            "away_classification"
                        ]
                    ),
                )

                print(
                    "WARNING — result-only ELO update for "
                    f"CFBD game {game_id}; raw and "
                    "opponent-adjusted stat updates skipped "
                    "because CFBD supplied no team box score."
                )

                completed_updates += 1

                continue

            update_completed_game(
                game_row=game_row,
                state_context=state_context,
                team_rows=team_rows,
                elo_engine=elo_engine,
                raw_engine=raw_engine,
                adjusted_engine=adjusted_engine,
            )

            completed_updates += 1

    # ========================================================
    # OUTPUT DATAFRAME
    # ========================================================

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions[
        "start_date"
    ] = pd.to_datetime(
        predictions[
            "start_date"
        ],
        errors="coerce",
        utc=True,
    )

    predictions = (
        predictions
        .sort_values(
            [
                "start_date",
                "cfbd_game_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "7. FORWARD RUN VALIDATION"
    )

    probability_sum = (
        predictions[
            "home_win_probability"
        ]
        +
        predictions[
            "away_win_probability"
        ]
    )

    invalid_probability_rows = int(
        (
            ~np.isclose(
                probability_sum,
                1.0,
                atol=1e-12,
            )
        )
        .sum()
    )

    missing_probabilities = int(
        predictions[
            [
                "home_win_probability",
                "away_win_probability",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    duplicate_games = int(
        predictions[
            "cfbd_game_id"
        ]
        .duplicated()
        .sum()
    )

    completed_predictions = (
        predictions[
            predictions[
                "completed"
            ]
            ==
            True
        ]
        .copy()
    )

    upcoming_predictions = (
        predictions[
            predictions[
                "completed"
            ]
            ==
            False
        ]
        .copy()
    )

    print(
        f"Prediction rows:            "
        f"{len(predictions):,}"
    )

    print(
        f"Completed predictions:      "
        f"{len(completed_predictions):,}"
    )

    print(
        f"Upcoming predictions:       "
        f"{len(upcoming_predictions):,}"
    )

    print(
        f"Completed state updates:    "
        f"{completed_updates:,}"
    )

    print(
        f"Missing probabilities:      "
        f"{missing_probabilities:,}"
    )

    print(
        f"Invalid probability sums:   "
        f"{invalid_probability_rows:,}"
    )

    print(
        f"Duplicate CFBD game IDs:    "
        f"{duplicate_games:,}"
    )

    if len(predictions) != len(
        forward_games
    ):

        raise ValueError(
            "Prediction row count does not "
            "match 2026 schedule."
        )

    if missing_probabilities:

        raise ValueError(
            "Some prediction probabilities "
            "are missing."
        )

    if invalid_probability_rows:

        raise ValueError(
            "Home + away probabilities do "
            "not equal 1.0."
        )

    if duplicate_games:

        raise ValueError(
            "Duplicate 2026 prediction rows "
            "were generated."
        )

    if (
        completed_updates
        !=
        completed_schedule_games
    ):

        raise ValueError(
            "Not every completed game updated "
            "the live model state."
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    section(
        "8. SAVE PREDICTIONS"
    )

    predictions.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    completed_predictions.to_csv(
        COMPLETED_OUTPUT_PATH,
        index=False,
    )

    upcoming_predictions.to_csv(
        UPCOMING_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Saved: {COMPLETED_OUTPUT_PATH}"
    )

    print(
        f"Saved: {UPCOMING_OUTPUT_PATH}"
    )

    # ========================================================
    # COMPLETED PERFORMANCE
    # ========================================================

    section(
        "9. 2026 COMPLETED FORWARD PERFORMANCE"
    )

    metrics = (
        calculate_completed_metrics(
            completed_predictions
        )
    )

    if metrics:

        print(
            f"Games:       "
            f"{int(metrics['games'])}"
        )

        print(
            f"Accuracy:    "
            f"{metrics['accuracy']:.2%}"
        )

        print(
            f"Brier:       "
            f"{metrics['brier']:.4f}"
        )

        print(
            f"Log Loss:    "
            f"{metrics['log_loss']:.4f}"
        )

        print()
        print(
            "IMPORTANT: this is a tiny 2026 "
            "forward sample."
        )

        print(
            "These results are diagnostic only "
            "and must NOT be used to retune "
            "the frozen model."
        )

    else:

        print(
            "No completed games available "
            "for scoring."
        )

    # ========================================================
    # COMPLETED DISPLAY
    # ========================================================

    section(
        "10. COMPLETED 2026 V2 PREDICTIONS"
    )

    if completed_predictions.empty:

        print(
            "No completed games."
        )

    else:

        display = (
            completed_predictions[
                [
                    "week",
                    "away_team",
                    "home_team",
                    "away_win_probability",
                    "home_win_probability",
                    "predicted_winner",
                    "actual_winner",
                    "prediction_correct",
                ]
            ]
            .copy()
        )

        display[
            "away_win_probability"
        ] = (
            display[
                "away_win_probability"
            ]
            *
            100.0
        ).round(1)

        display[
            "home_win_probability"
        ] = (
            display[
                "home_win_probability"
            ]
            *
            100.0
        ).round(1)

        print(
            display.to_string(
                index=False
            )
        )

    # ========================================================
    # UPCOMING DISPLAY
    # ========================================================

    section(
        "11. NEXT UPCOMING PREDICTIONS"
    )

    if upcoming_predictions.empty:

        print(
            "No upcoming games."
        )

    else:

        upcoming_display = (
            upcoming_predictions[
                [
                    "start_date",
                    "week",
                    "away_team",
                    "home_team",
                    "away_win_probability",
                    "home_win_probability",
                    "predicted_winner",
                    "predicted_winner_probability",
                ]
            ]
            .head(20)
            .copy()
        )

        upcoming_display[
            "away_win_probability"
        ] = (
            upcoming_display[
                "away_win_probability"
            ]
            *
            100.0
        ).round(1)

        upcoming_display[
            "home_win_probability"
        ] = (
            upcoming_display[
                "home_win_probability"
            ]
            *
            100.0
        ).round(1)

        upcoming_display[
            "predicted_winner_probability"
        ] = (
            upcoming_display[
                "predicted_winner_probability"
            ]
            *
            100.0
        ).round(1)

        print(
            upcoming_display.to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "2026 FORWARD RUN COMPLETE"
    )

    print(
        "✓ Historical 2023-2025 state replayed."
    )

    print(
        "✓ 2026 preseason transition applied."
    )

    print(
        "✓ Schedule and box scores joined by "
        "external CFBD game ID."
    )

    print(
        "✓ Every game predicted before its own result."
    )

    print(
        "✓ Same-kickoff games predicted before "
        "timestamp state updates."
    )

    print(
        "✓ Completed games advanced ELO."
    )

    print(
        "✓ Completed games advanced raw "
        "offence/defence."
    )

    print(
        "✓ Completed games advanced "
        "opponent-adjusted ratings."
    )

    print(
        "✓ Frozen probability model used."
    )

    print()
    print(
        f"Official output:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
