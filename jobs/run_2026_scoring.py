from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
# ENGINE IMPORTS
# ============================================================

from engine.environment import (
    EnvironmentConfig,
    GameEnvironmentEngine,
)

from engine.possessions import (
    PossessionConfig,
    PossessionModel,
)

from engine.scoring import (
    ScoringConfig,
    ScoringModel,
)

from engine.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)


HISTORICAL_TEAM_GAMES_PATH = (
    PROCESSED_DIR
    / "historical_team_games.csv"
)

MATCHUP_HISTORY_PATH = (
    PROCESSED_DIR
    / "matchup_history.csv"
)

WIN_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "2026_forward_predictions.csv"
)

FORWARD_TEAM_GAMES_PATH = (
    PROCESSED_DIR
    / "2026_forward_team_games.csv"
)

FROZEN_SCORE_MODEL_PATH = (
    PROCESSED_DIR
    / "v2_score_model.joblib"
)

CORRECTED_UNCERTAINTY_PATH = (
    PROCESSED_DIR
    / "v2_corrected_score_uncertainty.json"
)


SCORING_OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_scoring_predictions.csv"
)

COMPLETED_SCORING_OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_completed_scoring_predictions.csv"
)

UPCOMING_SCORING_OUTPUT_PATH = (
    PREDICTIONS_DIR
    / "2026_upcoming_scoring_predictions.csv"
)


# ============================================================
# MONTE CARLO
# ============================================================

SIMULATIONS = 3000
RANDOM_SEED = 2026


# ============================================================
# FROZEN SCORE ARCHITECTURE
# ============================================================

EXPECTED_SCORE_MODEL_NAME = (
    "compact_full"
)

EXPECTED_MARGIN_FEATURES = [
    "expected_home_margin",
    "elo_difference",
    "offensive_matchup_difference",
    "adjusted_strength_difference",
]

EXPECTED_TOTAL_FEATURES = [
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
# GENERIC HELPERS
# ============================================================

def safe_float(
    value: Any,
    fallback: float = np.nan,
) -> float:

    try:

        number = float(
            value
        )

        if np.isfinite(
            number
        ):
            return number

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(
        fallback
    )


def safe_int(
    value: Any,
    fallback: int = 0,
) -> int:

    try:

        number = float(
            value
        )

        if np.isfinite(
            number
        ):
            return int(
                number
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return int(
        fallback
    )


def normalise_bool_value(
    value: Any,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(
        value
    ):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    if "start_date" in frame.columns:

        frame["start_date"] = pd.to_datetime(
            frame["start_date"],
            errors="coerce",
            utc=True,
        )

    return frame


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
        f"Saved {len(frame):,} rows -> {path}"
    )


# ============================================================
# HISTORICAL OPPONENT DERIVATION
# ============================================================

def derive_opponent_name(
    team_games: pd.DataFrame,
) -> pd.DataFrame:

    data = team_games.copy()

    if "opponent_name" in data.columns:

        opponent = (
            data[
                "opponent_name"
            ]
            .astype("string")
            .str.strip()
        )

        if opponent.notna().all():

            return data

    required = [
        "game_id",
        "team_name",
        "home_away",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Cannot derive opponent_name. "
            f"Missing columns: {missing}"
        )

    data["game_id"] = pd.to_numeric(
        data["game_id"],
        errors="coerce",
    ).astype("Int64")

    data["team_name"] = (
        data[
            "team_name"
        ]
        .astype("string")
        .str.strip()
    )

    data["home_away"] = (
        data[
            "home_away"
        ]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    lookup: dict[
        tuple[int, str],
        str,
    ] = {}

    for game_id, rows in data.groupby(
        "game_id",
        sort=False,
    ):

        if (
            pd.isna(
                game_id
            )
            or
            len(rows)
            !=
            2
        ):
            continue

        home_rows = rows[
            rows[
                "home_away"
            ]
            ==
            "home"
        ]

        away_rows = rows[
            rows[
                "home_away"
            ]
            ==
            "away"
        ]

        if (
            len(
                home_rows
            )
            !=
            1
            or
            len(
                away_rows
            )
            !=
            1
        ):
            continue

        home_team = str(
            home_rows.iloc[
                0
            ][
                "team_name"
            ]
        ).strip()

        away_team = str(
            away_rows.iloc[
                0
            ][
                "team_name"
            ]
        ).strip()

        gid = int(
            game_id
        )

        lookup[
            (
                gid,
                home_team,
            )
        ] = away_team

        lookup[
            (
                gid,
                away_team,
            )
        ] = home_team

    data[
        "opponent_name"
    ] = [
        lookup.get(
            (
                int(
                    game_id
                )
                if pd.notna(
                    game_id
                )
                else -1,
                str(
                    team
                ).strip(),
            ),
            pd.NA,
        )
        for game_id, team
        in zip(
            data[
                "game_id"
            ],
            data[
                "team_name"
            ],
        )
    ]

    return data


# ============================================================
# FIND EXTERNAL CFBD GAME KEY
# ============================================================

def select_prediction_game_id_column(
    predictions: pd.DataFrame,
    team_games: pd.DataFrame,
) -> str:

    if (
        "cfbd_game_id"
        not in
        team_games.columns
    ):

        raise ValueError(
            "2026_forward_team_games.csv "
            "must contain cfbd_game_id."
        )

    completed_ids = set(
        pd.to_numeric(
            team_games[
                "cfbd_game_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    candidates = [
        "cfbd_game_id",
        "cfbd_id",
        "game_id",
    ]

    best_column = None
    best_overlap = -1

    for column in candidates:

        if column not in predictions.columns:
            continue

        candidate_ids = set(
            pd.to_numeric(
                predictions[
                    column
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        overlap = len(
            candidate_ids
            &
            completed_ids
        )

        print(
            f"Prediction ID candidate "
            f"{column:<18} "
            f"completed overlap={overlap:,}"
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_column = column

    if best_column is None:

        raise RuntimeError(
            "Could not identify prediction "
            "CFBD game ID column."
        )

    print()

    print(
        f"Selected prediction game key: "
        f"{best_column}"
    )

    return best_column


# ============================================================
# COMPLETED TEAM GAME LOOKUP
# ============================================================

def build_team_game_lookup(
    team_games: pd.DataFrame,
) -> dict[
    int,
    tuple[
        pd.Series,
        pd.Series,
    ],
]:

    data = team_games.copy()

    data[
        "cfbd_game_id"
    ] = pd.to_numeric(
        data[
            "cfbd_game_id"
        ],
        errors="coerce",
    ).astype("Int64")

    data[
        "home_away"
    ] = (
        data[
            "home_away"
        ]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    lookup = {}

    for game_id, rows in data.groupby(
        "cfbd_game_id",
        sort=False,
    ):

        if (
            pd.isna(
                game_id
            )
            or
            len(rows)
            !=
            2
        ):
            continue

        home_rows = rows[
            rows[
                "home_away"
            ]
            ==
            "home"
        ]

        away_rows = rows[
            rows[
                "home_away"
            ]
            ==
            "away"
        ]

        if (
            len(
                home_rows
            )
            !=
            1
            or
            len(
                away_rows
            )
            !=
            1
        ):
            continue

        lookup[
            int(
                game_id
            )
        ] = (
            home_rows.iloc[
                0
            ],
            away_rows.iloc[
                0
            ],
        )

    return lookup


# ============================================================
# LOAD FROZEN SCORE MODEL
# ============================================================

def load_score_model() -> dict[
    str,
    Any,
]:

    if not FROZEN_SCORE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Frozen score model not found:\n"
            f"{FROZEN_SCORE_MODEL_PATH}\n\n"
            "Run:\n"
            "python -m jobs.freeze_score_model"
        )

    artifact = joblib.load(
        FROZEN_SCORE_MODEL_PATH
    )

    required_keys = [
        "model_name",
        "margin_features",
        "total_features",
        "margin_model",
        "total_model",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing:

        raise RuntimeError(
            "Frozen score model artifact is "
            f"missing keys: {missing}"
        )

    if (
        artifact[
            "model_name"
        ]
        !=
        EXPECTED_SCORE_MODEL_NAME
    ):

        raise RuntimeError(
            "Frozen score model architecture mismatch.\n"
            f"Expected: {EXPECTED_SCORE_MODEL_NAME}\n"
            f"Found:    {artifact['model_name']}"
        )

    if (
        artifact[
            "margin_features"
        ]
        !=
        EXPECTED_MARGIN_FEATURES
    ):

        raise RuntimeError(
            "Frozen margin feature list mismatch."
        )

    if (
        artifact[
            "total_features"
        ]
        !=
        EXPECTED_TOTAL_FEATURES
    ):

        raise RuntimeError(
            "Frozen total feature list mismatch."
        )

    return artifact


def load_corrected_uncertainty() -> dict[
    str,
    Any,
]:

    if not CORRECTED_UNCERTAINTY_PATH.exists():

        raise FileNotFoundError(
            "Corrected score uncertainty artifact not found:\n"
            f"{CORRECTED_UNCERTAINTY_PATH}\n\n"
            "Run:\n"
            "python -m jobs.validate_corrected_score_uncertainty"
        )

    with CORRECTED_UNCERTAINTY_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        artifact = json.load(
            handle
        )

    if (
        artifact.get(
            "artifact_type"
        )
        !=
        "corrected_score_uncertainty"
    ):

        raise RuntimeError(
            "Corrected uncertainty artifact type mismatch."
        )

    if (
        artifact.get(
            "score_model_name"
        )
        !=
        EXPECTED_SCORE_MODEL_NAME
    ):

        raise RuntimeError(
            "Corrected uncertainty score-model mismatch.\n"
            f"Expected: {EXPECTED_SCORE_MODEL_NAME}\n"
            f"Found:    {artifact.get('score_model_name')}"
        )

    if artifact.get(
        "uses_2026_for_fitting",
        True,
    ):

        raise RuntimeError(
            "Corrected uncertainty artifact reports that "
            "2026 was used for fitting. Production run stopped."
        )

    state = artifact.get(
        "end_2025_state"
    )

    if not isinstance(
        state,
        dict,
    ):

        raise RuntimeError(
            "Corrected uncertainty artifact is missing "
            "end_2025_state."
        )

    required_state_fields = [
        "games_seen",
        "home_score_sd",
        "away_score_sd",
        "score_residual_correlation",
    ]

    missing = [
        field
        for field in required_state_fields
        if field not in state
    ]

    if missing:

        raise RuntimeError(
            "Corrected uncertainty end_2025_state "
            f"is missing fields: {missing}"
        )

    home_sd = safe_float(
        state.get(
            "home_score_sd"
        ),
        np.nan,
    )

    away_sd = safe_float(
        state.get(
            "away_score_sd"
        ),
        np.nan,
    )

    correlation = safe_float(
        state.get(
            "score_residual_correlation"
        ),
        np.nan,
    )

    games_seen = safe_int(
        state.get(
            "games_seen"
        ),
        -1,
    )

    if (
        not np.isfinite(
            home_sd
        )
        or
        not np.isfinite(
            away_sd
        )
        or
        not np.isfinite(
            correlation
        )
        or
        home_sd <= 0.0
        or
        away_sd <= 0.0
        or
        games_seen < 1
    ):

        raise RuntimeError(
            "Corrected uncertainty end-2025 state "
            "contains invalid values."
        )

    oos_rows = safe_int(
        artifact.get(
            "oos_rows"
        ),
        -1,
    )

    if (
        oos_rows > 0
        and
        games_seen != oos_rows
    ):

        raise RuntimeError(
            "Corrected uncertainty artifact is internally inconsistent.\n"
            f"OOS rows:   {oos_rows}\n"
            f"Games seen: {games_seen}"
        )

    return artifact


def seed_corrected_uncertainty(
    engine: MonteCarloEngine,
    artifact: dict[
        str,
        Any,
    ],
) -> None:

    state = artifact[
        "end_2025_state"
    ]

    home_sd = safe_float(
        state[
            "home_score_sd"
        ]
    )

    away_sd = safe_float(
        state[
            "away_score_sd"
        ]
    )

    correlation = safe_float(
        state[
            "score_residual_correlation"
        ]
    )

    home_variance = safe_float(
        state.get(
            "home_variance"
        ),
        home_sd ** 2,
    )

    away_variance = safe_float(
        state.get(
            "away_variance"
        ),
        away_sd ** 2,
    )

    covariance = safe_float(
        state.get(
            "covariance"
        ),
        correlation
        *
        home_sd
        *
        away_sd,
    )

    engine.state.games_seen = safe_int(
        state[
            "games_seen"
        ]
    )

    engine.state.home_variance = (
        home_variance
    )

    engine.state.away_variance = (
        away_variance
    )

    engine.state.covariance = (
        covariance
    )


# ============================================================
# ENVIRONMENT PREDICTION
# ============================================================

def predict_environment(
    engine: GameEnvironmentEngine,
    home_team: str,
    away_team: str,
    season: int,
) -> dict[
    str,
    float,
]:

    home = engine.snapshot(
        home_team,
        season,
    )

    away = engine.snapshot(
        away_team,
        season,
    )

    expected_home_plays = (
        engine.expected_team_plays(
            offense_plays=(
                home[
                    "offense_plays_pre"
                ]
            ),
            opponent_defense_plays_allowed=(
                away[
                    "defense_plays_allowed_pre"
                ]
            ),
        )
    )

    expected_away_plays = (
        engine.expected_team_plays(
            offense_plays=(
                away[
                    "offense_plays_pre"
                ]
            ),
            opponent_defense_plays_allowed=(
                home[
                    "defense_plays_allowed_pre"
                ]
            ),
        )
    )

    expected_total_plays = float(
        np.clip(
            (
                expected_home_plays
                +
                expected_away_plays
            ),
            engine.config.minimum_total_plays,
            engine.config.maximum_total_plays,
        )
    )

    (
        home_possession_share,
        away_possession_share,
    ) = engine.expected_possession_shares(
        home[
            "possession_share_pre"
        ],
        away[
            "possession_share_pre"
        ],
    )

    national_total_plays = (
        2.0
        *
        engine.national.average_team_plays
    )

    if (
        national_total_plays
        <=
        0
    ):

        pace_index = 100.0

    else:

        pace_index = (
            100.0
            *
            expected_total_plays
            /
            national_total_plays
        )

    return {
        "home_environment_games_pre": (
            home[
                "environment_games_pre"
            ]
        ),
        "away_environment_games_pre": (
            away[
                "environment_games_pre"
            ]
        ),
        "expected_home_plays": (
            expected_home_plays
        ),
        "expected_away_plays": (
            expected_away_plays
        ),
        "expected_total_plays": (
            expected_total_plays
        ),
        "expected_home_possession_share": (
            home_possession_share
        ),
        "expected_away_possession_share": (
            away_possession_share
        ),
        "expected_home_possession_minutes": (
            60.0
            *
            home_possession_share
        ),
        "expected_away_possession_minutes": (
            60.0
            *
            away_possession_share
        ),
        "game_pace_index": (
            pace_index
        ),
    }


# ============================================================
# POSSESSION PREDICTION
# ============================================================

def predict_possessions(
    engine: PossessionModel,
    home_team: str,
    away_team: str,
    season: int,
    environment: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:

    home = engine.snapshot(
        home_team,
        season,
    )

    away = engine.snapshot(
        away_team,
        season,
    )

    home_sustain = (
        engine.offensive_sustainability(
            home
        )
    )

    away_sustain = (
        engine.offensive_sustainability(
            away
        )
    )

    home_resistance = (
        engine.defensive_resistance(
            home
        )
    )

    away_resistance = (
        engine.defensive_resistance(
            away
        )
    )

    home_plays_per_drive = (
        engine.expected_plays_per_drive(
            offense_snapshot=home,
            defense_snapshot=away,
        )
    )

    away_plays_per_drive = (
        engine.expected_plays_per_drive(
            offense_snapshot=away,
            defense_snapshot=home,
        )
    )

    raw_home_drives = (
        engine.estimate_drives(
            environment[
                "expected_home_plays"
            ],
            home_plays_per_drive,
        )
    )

    raw_away_drives = (
        engine.estimate_drives(
            environment[
                "expected_away_plays"
            ],
            away_plays_per_drive,
        )
    )

    common_drive_environment = (
        raw_home_drives
        +
        raw_away_drives
    ) / 2.0

    possession_difference = (
        environment[
            "expected_home_possession_share"
        ]
        -
        environment[
            "expected_away_possession_share"
        ]
    )

    drive_adjustment = (
        possession_difference
        *
        engine.config
        .possession_drive_adjustment
    )

    expected_home_drives = float(
        np.clip(
            (
                common_drive_environment
                +
                drive_adjustment
            ),
            engine.config.minimum_estimated_drives,
            engine.config.maximum_estimated_drives,
        )
    )

    expected_away_drives = float(
        np.clip(
            (
                common_drive_environment
                -
                drive_adjustment
            ),
            engine.config.minimum_estimated_drives,
            engine.config.maximum_estimated_drives,
        )
    )

    return {
        "home_possession_games_pre": (
            home[
                "possession_games_pre"
            ]
        ),
        "away_possession_games_pre": (
            away[
                "possession_games_pre"
            ]
        ),
        "home_offensive_sustainability_index": (
            100.0
            *
            home_sustain
        ),
        "away_offensive_sustainability_index": (
            100.0
            *
            away_sustain
        ),
        "home_defensive_resistance_index": (
            100.0
            *
            home_resistance
        ),
        "away_defensive_resistance_index": (
            100.0
            *
            away_resistance
        ),
        "expected_home_plays_per_drive": (
            home_plays_per_drive
        ),
        "expected_away_plays_per_drive": (
            away_plays_per_drive
        ),
        "raw_home_estimated_drives": (
            raw_home_drives
        ),
        "raw_away_estimated_drives": (
            raw_away_drives
        ),
        "common_drive_environment": (
            common_drive_environment
        ),
        "expected_home_drives": (
            expected_home_drives
        ),
        "expected_away_drives": (
            expected_away_drives
        ),
        "expected_total_drives": (
            expected_home_drives
            +
            expected_away_drives
        ),
    }


# ============================================================
# RAW SCORING PREDICTION
# ============================================================

def predict_raw_score(
    engine: ScoringModel,
    home_team: str,
    away_team: str,
    season: int,
    neutral_site: bool,
    home_offensive_matchup_index: float,
    away_offensive_matchup_index: float,
    possession: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:

    home = engine.snapshot(
        home_team,
        season,
    )

    away = engine.snapshot(
        away_team,
        season,
    )

    (
        home_location_multiplier,
        away_location_multiplier,
    ) = engine.location_multipliers(
        neutral_site
    )

    expected_home_ppp = (
        engine.expected_points_per_play(
            offense_snapshot=home,
            defense_snapshot=away,
            offensive_matchup_index=(
                home_offensive_matchup_index
            ),
            location_multiplier=(
                home_location_multiplier
            ),
        )
    )

    expected_away_ppp = (
        engine.expected_points_per_play(
            offense_snapshot=away,
            defense_snapshot=home,
            offensive_matchup_index=(
                away_offensive_matchup_index
            ),
            location_multiplier=(
                away_location_multiplier
            ),
        )
    )

    expected_home_ppd = (
        engine.expected_points_per_drive(
            expected_points_per_play=(
                expected_home_ppp
            ),
            expected_plays_per_drive=(
                possession[
                    "expected_home_plays_per_drive"
                ]
            ),
        )
    )

    expected_away_ppd = (
        engine.expected_points_per_drive(
            expected_points_per_play=(
                expected_away_ppp
            ),
            expected_plays_per_drive=(
                possession[
                    "expected_away_plays_per_drive"
                ]
            ),
        )
    )

    expected_home_points = (
        engine.expected_team_points(
            expected_points_per_drive=(
                expected_home_ppd
            ),
            expected_drives=(
                possession[
                    "expected_home_drives"
                ]
            ),
        )
    )

    expected_away_points = (
        engine.expected_team_points(
            expected_points_per_drive=(
                expected_away_ppd
            ),
            expected_drives=(
                possession[
                    "expected_away_drives"
                ]
            ),
        )
    )

    expected_total_points = (
        expected_home_points
        +
        expected_away_points
    )

    expected_home_margin = (
        expected_home_points
        -
        expected_away_points
    )

    return {
        "home_scoring_games_pre": (
            home[
                "scoring_games_pre"
            ]
        ),
        "away_scoring_games_pre": (
            away[
                "scoring_games_pre"
            ]
        ),

        "home_offense_points_per_play_pre": (
            home[
                "offense_points_per_play_pre"
            ]
        ),
        "away_offense_points_per_play_pre": (
            away[
                "offense_points_per_play_pre"
            ]
        ),

        "home_defense_points_per_play_allowed_pre": (
            home[
                "defense_points_per_play_allowed_pre"
            ]
        ),
        "away_defense_points_per_play_allowed_pre": (
            away[
                "defense_points_per_play_allowed_pre"
            ]
        ),

        "home_scoring_location_multiplier": (
            home_location_multiplier
        ),
        "away_scoring_location_multiplier": (
            away_location_multiplier
        ),

        "home_scoring_matchup_factor": (
            engine.scoring_matchup_factor(
                home_offensive_matchup_index
            )
        ),
        "away_scoring_matchup_factor": (
            engine.scoring_matchup_factor(
                away_offensive_matchup_index
            )
        ),

        "expected_home_points_per_play": (
            expected_home_ppp
        ),
        "expected_away_points_per_play": (
            expected_away_ppp
        ),

        "expected_home_points_per_drive": (
            expected_home_ppd
        ),
        "expected_away_points_per_drive": (
            expected_away_ppd
        ),

        "raw_expected_home_points": (
            expected_home_points
        ),
        "raw_expected_away_points": (
            expected_away_points
        ),
        "raw_expected_total_points": (
            expected_total_points
        ),
        "raw_expected_home_margin": (
            expected_home_margin
        ),
    }


# ============================================================
# APPLY FROZEN SCORE CORRECTION
# ============================================================

def apply_frozen_score_model(
    artifact: dict[
        str,
        Any,
    ],
    game_row: pd.Series,
    environment: dict[
        str,
        float,
    ],
    possession: dict[
        str,
        float,
    ],
    raw_score: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:

    margin_features = {
        "expected_home_margin": (
            raw_score[
                "raw_expected_home_margin"
            ]
        ),
        "elo_difference": (
            safe_float(
                game_row.get(
                    "elo_difference"
                ),
                0.0,
            )
        ),
        "offensive_matchup_difference": (
            safe_float(
                game_row.get(
                    "offensive_matchup_difference"
                ),
                0.0,
            )
        ),
        "adjusted_strength_difference": (
            safe_float(
                game_row.get(
                    "adjusted_strength_difference"
                ),
                0.0,
            )
        ),
    }

    total_features = {
        "expected_total_points": (
            raw_score[
                "raw_expected_total_points"
            ]
        ),
        "expected_total_plays": (
            environment[
                "expected_total_plays"
            ]
        ),
        "expected_total_drives": (
            possession[
                "expected_total_drives"
            ]
        ),
        "game_pace_index": (
            environment[
                "game_pace_index"
            ]
        ),
    }

    margin_frame = pd.DataFrame(
        [
            margin_features
        ]
    )

    total_frame = pd.DataFrame(
        [
            total_features
        ]
    )

    final_margin = float(
        artifact[
            "margin_model"
        ].predict(
            margin_frame[
                artifact[
                    "margin_features"
                ]
            ]
        )[0]
    )

    final_total = float(
        artifact[
            "total_model"
        ].predict(
            total_frame[
                artifact[
                    "total_features"
                ]
            ]
        )[0]
    )

    # --------------------------------------------------------
    # SAFETY BOUNDS
    # --------------------------------------------------------

    final_total = float(
        np.clip(
            final_total,
            12.0,
            100.0,
        )
    )

    final_margin = float(
        np.clip(
            final_margin,
            -60.0,
            60.0,
        )
    )

    # --------------------------------------------------------
    # RECONSTRUCT TEAM SCORES
    # --------------------------------------------------------

    final_home_points = (
        final_total
        +
        final_margin
    ) / 2.0

    final_away_points = (
        final_total
        -
        final_margin
    ) / 2.0

    # --------------------------------------------------------
    # NON-NEGATIVE SCORE SAFETY
    #
    # If extreme margin / total combination creates a
    # negative score, preserve the total as much as possible
    # while flooring the team score at zero.
    # --------------------------------------------------------

    if final_home_points < 0.0:

        final_home_points = 0.0
        final_away_points = (
            final_total
        )

    if final_away_points < 0.0:

        final_away_points = 0.0
        final_home_points = (
            final_total
        )

    final_home_points = float(
        np.clip(
            final_home_points,
            0.0,
            80.0,
        )
    )

    final_away_points = float(
        np.clip(
            final_away_points,
            0.0,
            80.0,
        )
    )

    # Recalculate after safety clipping.

    final_total = (
        final_home_points
        +
        final_away_points
    )

    final_margin = (
        final_home_points
        -
        final_away_points
    )

    return {
        "score_model_name": (
            artifact[
                "model_name"
            ]
        ),

        "final_expected_home_margin": (
            final_margin
        ),

        "final_expected_total_points": (
            final_total
        ),

        "final_expected_home_points": (
            final_home_points
        ),

        "final_expected_away_points": (
            final_away_points
        ),

        # Convenience production names
        "expected_home_margin": (
            final_margin
        ),

        "expected_total_points": (
            final_total
        ),

        "expected_home_points": (
            final_home_points
        ),

        "expected_away_points": (
            final_away_points
        ),
    }


# ============================================================
# MONTE CARLO AROUND FINAL SCORE
# ============================================================

def simulate_final_score(
    engine: MonteCarloEngine,
    expected_home_points: float,
    expected_away_points: float,
    game_id: int,
) -> dict[
    str,
    float,
]:

    uncertainty = (
        engine.current_uncertainty()
    )

    simulation = (
        engine.simulate_game(
            expected_home_points=(
                expected_home_points
            ),
            expected_away_points=(
                expected_away_points
            ),
            game_seed=(
                engine.config.random_seed
                +
                int(
                    game_id
                )
            ),
        )
    )

    result = {
        "simulation_count": (
            engine.config.simulations
        ),

        "historical_uncertainty_games_pre": (
            engine.state.games_seen
        ),

        "home_score_residual_sd_pre": (
            uncertainty[
                "home_score_sd"
            ]
        ),

        "away_score_residual_sd_pre": (
            uncertainty[
                "away_score_sd"
            ]
        ),

        "score_residual_correlation_pre": (
            uncertainty[
                "score_residual_correlation"
            ]
        ),
    }

    result.update(
        simulation
    )

    return result


# ============================================================
# UPDATE COMPLETED GAME
# ============================================================

def update_completed_game(
    *,
    season: int,
    home_row: pd.Series,
    away_row: pd.Series,
    expected_home_points: float,
    expected_away_points: float,
    environment_engine: GameEnvironmentEngine,
    possession_engine: PossessionModel,
    scoring_engine: ScoringModel,
    monte_carlo_engine: MonteCarloEngine,
) -> None:

    home_team = str(
        home_row[
            "team_name"
        ]
    ).strip()

    away_team = str(
        away_row[
            "team_name"
        ]
    ).strip()

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    actual_home_plays = (
        environment_engine
        .calculate_actual_team_plays(
            home_row
        )
    )

    actual_away_plays = (
        environment_engine
        .calculate_actual_team_plays(
            away_row
        )
    )

    actual_home_possession = max(
        safe_float(
            home_row.get(
                "possession_minutes"
            ),
            30.0,
        ),
        0.0,
    )

    actual_away_possession = max(
        safe_float(
            away_row.get(
                "possession_minutes"
            ),
            30.0,
        ),
        0.0,
    )

    actual_home_share = (
        environment_engine
        .calculate_possession_share(
            actual_home_possession,
            actual_away_possession,
        )
    )

    actual_away_share = (
        1.0
        -
        actual_home_share
    )

    environment_engine.update_team(
        team=home_team,
        season=season,
        team_plays=actual_home_plays,
        opponent_plays=actual_away_plays,
        possession_minutes=(
            actual_home_possession
        ),
        possession_share=(
            actual_home_share
        ),
    )

    environment_engine.update_team(
        team=away_team,
        season=season,
        team_plays=actual_away_plays,
        opponent_plays=actual_home_plays,
        possession_minutes=(
            actual_away_possession
        ),
        possession_share=(
            actual_away_share
        ),
    )

    environment_engine.update_national(
        team_plays=(
            actual_home_plays
        ),
        possession_minutes=(
            actual_home_possession
        ),
    )

    environment_engine.update_national(
        team_plays=(
            actual_away_plays
        ),
        possession_minutes=(
            actual_away_possession
        ),
    )

    # ========================================================
    # POSSESSION
    # ========================================================

    (
        _,
        home_first_down_rate,
        home_turnover_rate,
    ) = possession_engine.actual_rates(
        home_row
    )

    (
        _,
        away_first_down_rate,
        away_turnover_rate,
    ) = possession_engine.actual_rates(
        away_row
    )

    possession_engine.update_team(
        team=home_team,
        season=season,
        offense_first_down_rate=(
            home_first_down_rate
        ),
        offense_turnover_rate=(
            home_turnover_rate
        ),
        defense_first_down_rate_allowed=(
            away_first_down_rate
        ),
        defense_takeaway_rate=(
            away_turnover_rate
        ),
    )

    possession_engine.update_team(
        team=away_team,
        season=season,
        offense_first_down_rate=(
            away_first_down_rate
        ),
        offense_turnover_rate=(
            away_turnover_rate
        ),
        defense_first_down_rate_allowed=(
            home_first_down_rate
        ),
        defense_takeaway_rate=(
            home_turnover_rate
        ),
    )

    possession_engine.update_national(
        first_down_rate=(
            home_first_down_rate
        ),
        turnover_rate=(
            home_turnover_rate
        ),
    )

    possession_engine.update_national(
        first_down_rate=(
            away_first_down_rate
        ),
        turnover_rate=(
            away_turnover_rate
        ),
    )

    # ========================================================
    # RAW SCORING STATE
    #
    # IMPORTANT:
    # Team scoring state still updates from actual PPP.
    # The frozen correction is a prediction layer, not a
    # replacement for the chronological scoring-state engine.
    # ========================================================

    (
        _,
        home_actual_ppp,
    ) = scoring_engine.actual_points_per_play(
        home_row
    )

    (
        _,
        away_actual_ppp,
    ) = scoring_engine.actual_points_per_play(
        away_row
    )

    scoring_engine.update_team(
        team=home_team,
        season=season,
        offense_points_per_play=(
            home_actual_ppp
        ),
        defense_points_per_play_allowed=(
            away_actual_ppp
        ),
    )

    scoring_engine.update_team(
        team=away_team,
        season=season,
        offense_points_per_play=(
            away_actual_ppp
        ),
        defense_points_per_play_allowed=(
            home_actual_ppp
        ),
    )

    scoring_engine.update_national(
        home_actual_ppp
    )

    scoring_engine.update_national(
        away_actual_ppp
    )

    # ========================================================
    # MONTE CARLO UNCERTAINTY
    #
    # Going forward, residual uncertainty is measured against
    # the FINAL frozen score expectation because that is now
    # the production score model.
    # ========================================================

    actual_home_points = safe_float(
        home_row.get(
            "team_points"
        ),
        np.nan,
    )

    actual_away_points = safe_float(
        away_row.get(
            "team_points"
        ),
        np.nan,
    )

    if (
        np.isfinite(
            actual_home_points
        )
        and
        np.isfinite(
            actual_away_points
        )
    ):

        monte_carlo_engine.update_uncertainty(
            home_error=(
                actual_home_points
                -
                expected_home_points
            ),
            away_error=(
                actual_away_points
                -
                expected_away_points
            ),
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- FROZEN LIVE SCORE MODEL"
    )

    print(
        "Production architecture:"
    )

    print()

    print(
        "Environment"
    )

    print(
        "  -> possession / drive opportunity"
    )

    print(
        "  -> raw scoring engine"
    )

    print(
        "  -> frozen compact_full correction"
    )

    print(
        "  -> final margin / total / score"
    )

    print(
        "  -> Monte Carlo"
    )

    print()

    print(
        "Official winner probability remains "
        "the separate frozen probability model."
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    section(
        "1. REQUIRED FILES"
    )

    required_files = [
        HISTORICAL_TEAM_GAMES_PATH,
        MATCHUP_HISTORY_PATH,
        WIN_PREDICTIONS_PATH,
        FORWARD_TEAM_GAMES_PATH,
        FROZEN_SCORE_MODEL_PATH,
        CORRECTED_UNCERTAINTY_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        print(
            f"FOUND  {path.name}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "2. LOAD DATA AND FROZEN SCORE MODEL"
    )

    historical_team_games = load_csv(
        HISTORICAL_TEAM_GAMES_PATH
    )

    matchup_history = load_csv(
        MATCHUP_HISTORY_PATH
    )

    predictions = load_csv(
        WIN_PREDICTIONS_PATH
    )

    forward_team_games = load_csv(
        FORWARD_TEAM_GAMES_PATH
    )

    score_artifact = (
        load_score_model()
    )

    corrected_uncertainty_artifact = (
        load_corrected_uncertainty()
    )

    print(
        f"Historical team rows:       "
        f"{len(historical_team_games):,}"
    )

    print(
        f"Historical matchups:        "
        f"{len(matchup_history):,}"
    )

    print(
        f"2026 winner predictions:    "
        f"{len(predictions):,}"
    )

    print(
        f"2026 completed team rows:   "
        f"{len(forward_team_games):,}"
    )

    print()

    print(
        f"Frozen score architecture:  "
        f"{score_artifact['model_name']}"
    )

    print(
        f"Margin features:            "
        f"{len(score_artifact['margin_features'])}"
    )

    print(
        f"Total features:             "
        f"{len(score_artifact['total_features'])}"
    )

    print(
        f"Corrected uncertainty rows: "
        f"{safe_int(corrected_uncertainty_artifact.get('oos_rows'), 0):,}"
    )

    print(
        "2026 used in uncertainty:  NO"
    )

    # ========================================================
    # REPLAY HISTORICAL STATE
    # ========================================================

    section(
        "3. REPLAY SCORING STATE / LOAD CORRECTED UNCERTAINTY"
    )

    historical_team_games = (
        derive_opponent_name(
            historical_team_games
        )
    )

    matchup_history[
        "game_id"
    ] = pd.to_numeric(
        matchup_history[
            "game_id"
        ],
        errors="coerce",
    ).astype("Int64")

    model_game_ids = set(
        matchup_history[
            "game_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    historical_model_team_games = (
        historical_team_games[
            pd.to_numeric(
                historical_team_games[
                    "game_id"
                ],
                errors="coerce",
            )
            .isin(
                model_game_ids
            )
        ]
        .copy()
    )

    environment_engine = (
        GameEnvironmentEngine(
            EnvironmentConfig()
        )
    )

    historical_environment = (
        environment_engine
        .process_history(
            team_history=(
                historical_model_team_games
            ),
            matchup_history=(
                matchup_history
            ),
        )
    )

    possession_engine = (
        PossessionModel(
            PossessionConfig()
        )
    )

    historical_possession = (
        possession_engine
        .process_history(
            team_history=(
                historical_model_team_games
            ),
            environment_history=(
                historical_environment
            ),
        )
    )

    scoring_engine = (
        ScoringModel(
            ScoringConfig()
        )
    )

    historical_scoring = (
        scoring_engine
        .process_history(
            team_history=(
                historical_model_team_games
            ),
            possession_history=(
                historical_possession
            ),
        )
    )

    monte_carlo_engine = (
        MonteCarloEngine(
            MonteCarloConfig(
                simulations=(
                    SIMULATIONS
                ),
                random_seed=(
                    RANDOM_SEED
                ),
                uncertainty_update_rate=(
                    safe_float(
                        corrected_uncertainty_artifact.get(
                            "uncertainty_update_rate"
                        ),
                        0.05,
                    )
                ),
                minimum_score_sd=(
                    safe_float(
                        corrected_uncertainty_artifact.get(
                            "minimum_score_sd"
                        ),
                        8.0,
                    )
                ),
                maximum_score_sd=(
                    safe_float(
                        corrected_uncertainty_artifact.get(
                            "maximum_score_sd"
                        ),
                        22.0,
                    )
                ),
                minimum_residual_correlation=(
                    safe_float(
                        corrected_uncertainty_artifact.get(
                            "minimum_correlation"
                        ),
                        -0.35,
                    )
                ),
                maximum_residual_correlation=(
                    safe_float(
                        corrected_uncertainty_artifact.get(
                            "maximum_correlation"
                        ),
                        0.60,
                    )
                ),
            )
        )
    )

    seed_corrected_uncertainty(
        engine=(
            monte_carlo_engine
        ),
        artifact=(
            corrected_uncertainty_artifact
        ),
    )

    print(
        f"Environment replay rows: "
        f"{len(historical_environment):,}"
    )

    print(
        f"Possession replay rows:  "
        f"{len(historical_possession):,}"
    )

    print(
        f"Scoring replay rows:     "
        f"{len(historical_scoring):,}"
    )

    print(
        "Monte Carlo raw residual replay: SKIPPED"
    )

    print(
        "Corrected OOS uncertainty artifact: LOADED"
    )

    uncertainty = (
        monte_carlo_engine
        .current_uncertainty()
    )

    print()

    print(
        f"Corrected uncertainty games: "
        f"{monte_carlo_engine.state.games_seen:,}"
    )

    print(
        f"End-2025 corrected home SD: "
        f"{uncertainty['home_score_sd']:.4f}"
    )

    print(
        f"End-2025 corrected away SD: "
        f"{uncertainty['away_score_sd']:.4f}"
    )

    print(
        f"Corrected residual corr:    "
        f"{uncertainty['score_residual_correlation']:.4f}"
    )

    # ========================================================
    # PREPARE 2026
    # ========================================================

    section(
        "4. PREPARE 2026 FORWARD RUN"
    )

    prediction_id_column = (
        select_prediction_game_id_column(
            predictions,
            forward_team_games,
        )
    )

    predictions[
        "_cfbd_game_id"
    ] = pd.to_numeric(
        predictions[
            prediction_id_column
        ],
        errors="coerce",
    ).astype("Int64")

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
                "_cfbd_game_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    completed_lookup = (
        build_team_game_lookup(
            forward_team_games
        )
    )

    required_prediction_fields = [
        "home_team",
        "away_team",
        "start_date",
        "elo_difference",
        "home_offensive_matchup_index",
        "away_offensive_matchup_index",
        "offensive_matchup_difference",
        "adjusted_strength_difference",
    ]

    missing_prediction_fields = [
        column
        for column in required_prediction_fields
        if column not in predictions.columns
    ]

    if missing_prediction_fields:

        raise ValueError(
            "Winner prediction file is missing "
            "frozen score model inputs:\n"
            f"{missing_prediction_fields}"
        )

    print(
        f"2026 schedule games:       "
        f"{len(predictions):,}"
    )

    print(
        f"Completed box-score games: "
        f"{len(completed_lookup):,}"
    )

    print()

    print(
        "PASS - all frozen score model "
        "features are present."
    )

    # ========================================================
    # LIVE LOOP
    # ========================================================

    section(
        "5. GENERATE FROZEN 2026 SCORE PREDICTIONS"
    )

    outputs = []

    completed_updates = 0

    grouped = predictions.groupby(
        "start_date",
        sort=False,
        dropna=False,
    )

    for (
        start_date,
        timestamp_games,
    ) in grouped:

        pending_updates = []

        # ====================================================
        # PREDICT EVERY GAME AT TIMESTAMP FIRST
        # ====================================================

        for _, game in timestamp_games.iterrows():

            game_id = safe_int(
                game.get(
                    "_cfbd_game_id"
                ),
                0,
            )

            season = safe_int(
                game.get(
                    "season"
                ),
                2026,
            )

            home_team = str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip()

            away_team = str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip()

            neutral_site = (
                normalise_bool_value(
                    game.get(
                        "neutral_site",
                        False,
                    )
                )
            )

            home_matchup = safe_float(
                game.get(
                    "home_offensive_matchup_index"
                ),
                100.0,
            )

            away_matchup = safe_float(
                game.get(
                    "away_offensive_matchup_index"
                ),
                100.0,
            )

            # --------------------------------------------
            # ENVIRONMENT
            # --------------------------------------------

            environment = (
                predict_environment(
                    engine=(
                        environment_engine
                    ),
                    home_team=(
                        home_team
                    ),
                    away_team=(
                        away_team
                    ),
                    season=(
                        season
                    ),
                )
            )

            # --------------------------------------------
            # POSSESSION
            # --------------------------------------------

            possession = (
                predict_possessions(
                    engine=(
                        possession_engine
                    ),
                    home_team=(
                        home_team
                    ),
                    away_team=(
                        away_team
                    ),
                    season=(
                        season
                    ),
                    environment=(
                        environment
                    ),
                )
            )

            # --------------------------------------------
            # RAW SCORE
            # --------------------------------------------

            raw_score = (
                predict_raw_score(
                    engine=(
                        scoring_engine
                    ),
                    home_team=(
                        home_team
                    ),
                    away_team=(
                        away_team
                    ),
                    season=(
                        season
                    ),
                    neutral_site=(
                        neutral_site
                    ),
                    home_offensive_matchup_index=(
                        home_matchup
                    ),
                    away_offensive_matchup_index=(
                        away_matchup
                    ),
                    possession=(
                        possession
                    ),
                )
            )

            # --------------------------------------------
            # FROZEN CORRECTION
            # --------------------------------------------

            final_score = (
                apply_frozen_score_model(
                    artifact=(
                        score_artifact
                    ),
                    game_row=(
                        game
                    ),
                    environment=(
                        environment
                    ),
                    possession=(
                        possession
                    ),
                    raw_score=(
                        raw_score
                    ),
                )
            )

            # --------------------------------------------
            # MONTE CARLO AROUND FINAL SCORE
            # --------------------------------------------

            simulation = (
                simulate_final_score(
                    engine=(
                        monte_carlo_engine
                    ),
                    expected_home_points=(
                        final_score[
                            "final_expected_home_points"
                        ]
                    ),
                    expected_away_points=(
                        final_score[
                            "final_expected_away_points"
                        ]
                    ),
                    game_id=(
                        game_id
                    ),
                )
            )

            # --------------------------------------------
            # BUILD OUTPUT
            # --------------------------------------------

            output = (
                game.to_dict()
            )

            output.update(
                environment
            )

            output.update(
                possession
            )

            output.update(
                raw_score
            )

            output.update(
                final_score
            )

            output.update(
                simulation
            )

            # --------------------------------------------
            # DISPLAY SCORE
            # --------------------------------------------

            output[
                "projected_home_score"
            ] = int(
                np.rint(
                    final_score[
                        "final_expected_home_points"
                    ]
                )
            )

            output[
                "projected_away_score"
            ] = int(
                np.rint(
                    final_score[
                        "final_expected_away_points"
                    ]
                )
            )

            output[
                "projected_margin"
            ] = (
                final_score[
                    "final_expected_home_margin"
                ]
            )

            output[
                "projected_total"
            ] = (
                final_score[
                    "final_expected_total_points"
                ]
            )

            # --------------------------------------------
            # SCORE MODEL WINNER
            # --------------------------------------------

            if (
                final_score[
                    "final_expected_home_margin"
                ]
                >
                0
            ):

                score_model_winner = (
                    home_team
                )

            elif (
                final_score[
                    "final_expected_home_margin"
                ]
                <
                0
            ):

                score_model_winner = (
                    away_team
                )

            else:

                score_model_winner = (
                    "Tie"
                )

            output[
                "score_model_winner"
            ] = (
                score_model_winner
            )

            official_winner = str(
                game.get(
                    "predicted_winner",
                    "",
                )
            ).strip()

            output[
                "score_vs_official_winner_agreement"
            ] = (
                score_model_winner
                ==
                official_winner
            )

            # --------------------------------------------
            # RAW SCORE MODEL WINNER
            # --------------------------------------------

            if (
                raw_score[
                    "raw_expected_home_margin"
                ]
                >
                0
            ):

                raw_score_winner = (
                    home_team
                )

            elif (
                raw_score[
                    "raw_expected_home_margin"
                ]
                <
                0
            ):

                raw_score_winner = (
                    away_team
                )

            else:

                raw_score_winner = (
                    "Tie"
                )

            output[
                "raw_score_model_winner"
            ] = (
                raw_score_winner
            )

            # --------------------------------------------
            # COMPLETED RESULT
            # --------------------------------------------

            if game_id in completed_lookup:

                (
                    home_row,
                    away_row,
                ) = completed_lookup[
                    game_id
                ]

                actual_home_points = (
                    safe_float(
                        home_row.get(
                            "team_points"
                        ),
                        np.nan,
                    )
                )

                actual_away_points = (
                    safe_float(
                        away_row.get(
                            "team_points"
                        ),
                        np.nan,
                    )
                )

                actual_margin = (
                    actual_home_points
                    -
                    actual_away_points
                )

                actual_total = (
                    actual_home_points
                    +
                    actual_away_points
                )

                output[
                    "actual_home_points"
                ] = (
                    actual_home_points
                )

                output[
                    "actual_away_points"
                ] = (
                    actual_away_points
                )

                output[
                    "actual_home_margin"
                ] = (
                    actual_margin
                )

                output[
                    "actual_total_points"
                ] = (
                    actual_total
                )

                # RAW ERRORS

                output[
                    "raw_home_score_error"
                ] = (
                    actual_home_points
                    -
                    raw_score[
                        "raw_expected_home_points"
                    ]
                )

                output[
                    "raw_away_score_error"
                ] = (
                    actual_away_points
                    -
                    raw_score[
                        "raw_expected_away_points"
                    ]
                )

                output[
                    "raw_margin_error"
                ] = (
                    actual_margin
                    -
                    raw_score[
                        "raw_expected_home_margin"
                    ]
                )

                output[
                    "raw_total_error"
                ] = (
                    actual_total
                    -
                    raw_score[
                        "raw_expected_total_points"
                    ]
                )

                # FINAL ERRORS

                output[
                    "home_score_error"
                ] = (
                    actual_home_points
                    -
                    final_score[
                        "final_expected_home_points"
                    ]
                )

                output[
                    "away_score_error"
                ] = (
                    actual_away_points
                    -
                    final_score[
                        "final_expected_away_points"
                    ]
                )

                output[
                    "margin_error"
                ] = (
                    actual_margin
                    -
                    final_score[
                        "final_expected_home_margin"
                    ]
                )

                output[
                    "total_error"
                ] = (
                    actual_total
                    -
                    final_score[
                        "final_expected_total_points"
                    ]
                )

                pending_updates.append(
                    {
                        "season": season,
                        "home_row": home_row,
                        "away_row": away_row,

                        "expected_home_points": (
                            final_score[
                                "final_expected_home_points"
                            ]
                        ),

                        "expected_away_points": (
                            final_score[
                                "final_expected_away_points"
                            ]
                        ),
                    }
                )

            else:

                for column in [
                    "actual_home_points",
                    "actual_away_points",
                    "actual_home_margin",
                    "actual_total_points",

                    "raw_home_score_error",
                    "raw_away_score_error",
                    "raw_margin_error",
                    "raw_total_error",

                    "home_score_error",
                    "away_score_error",
                    "margin_error",
                    "total_error",
                ]:

                    output[
                        column
                    ] = np.nan

            outputs.append(
                output
            )

        # ====================================================
        # UPDATE ONLY AFTER ALL SAME-TIME PREDICTIONS
        # ====================================================

        for update in pending_updates:

            update_completed_game(
                season=(
                    update[
                        "season"
                    ]
                ),
                home_row=(
                    update[
                        "home_row"
                    ]
                ),
                away_row=(
                    update[
                        "away_row"
                    ]
                ),
                expected_home_points=(
                    update[
                        "expected_home_points"
                    ]
                ),
                expected_away_points=(
                    update[
                        "expected_away_points"
                    ]
                ),
                environment_engine=(
                    environment_engine
                ),
                possession_engine=(
                    possession_engine
                ),
                scoring_engine=(
                    scoring_engine
                ),
                monte_carlo_engine=(
                    monte_carlo_engine
                ),
            )

            completed_updates += 1

    # ========================================================
    # BUILD RESULT
    # ========================================================

    result = pd.DataFrame(
        outputs
    )

    result = (
        result
        .sort_values(
            [
                "start_date",
                "_cfbd_game_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    section(
        "6. PRODUCTION SCORE VALIDATION"
    )

    expected_columns = [
        "final_expected_home_points",
        "final_expected_away_points",
        "final_expected_home_margin",
        "final_expected_total_points",
    ]

    missing_values = int(
        result[
            expected_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    nonfinite_values = int(
        (
            ~np.isfinite(
                result[
                    expected_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )
        )
        .sum()
    )

    completed_mask = (
        result[
            "actual_home_points"
        ]
        .notna()
        &
        result[
            "actual_away_points"
        ]
        .notna()
    )

    completed = (
        result[
            completed_mask
        ]
        .copy()
    )

    upcoming = (
        result[
            ~completed_mask
        ]
        .copy()
    )

    print(
        f"Prediction rows:          "
        f"{len(result):,}"
    )

    print(
        f"Completed score rows:     "
        f"{len(completed):,}"
    )

    print(
        f"Upcoming score rows:      "
        f"{len(upcoming):,}"
    )

    print(
        f"Completed state updates:  "
        f"{completed_updates:,}"
    )

    print(
        f"Missing final values:     "
        f"{missing_values:,}"
    )

    print(
        f"Non-finite final values:  "
        f"{nonfinite_values:,}"
    )

    if (
        len(
            result
        )
        !=
        len(
            predictions
        )
    ):

        raise RuntimeError(
            "Production scoring row count "
            "does not match winner predictions."
        )

    if (
        missing_values
        or
        nonfinite_values
    ):

        raise RuntimeError(
            "Invalid frozen score predictions "
            "detected."
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "7. SAVE PRODUCTION SCORE PREDICTIONS"
    )

    save_csv(
        result,
        SCORING_OUTPUT_PATH,
    )

    save_csv(
        completed,
        COMPLETED_SCORING_OUTPUT_PATH,
    )

    save_csv(
        upcoming,
        UPCOMING_SCORING_OUTPUT_PATH,
    )

    # ========================================================
    # RAW VS FINAL 2026 DIAGNOSTIC
    # ========================================================

    section(
        "8. COMPLETED 2026 RAW VS FROZEN SCORE MODEL"
    )

    if completed.empty:

        print(
            "No completed games available."
        )

    else:

        raw_home_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "raw_home_score_error"
                    ]
                )
            )
        )

        raw_away_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "raw_away_score_error"
                    ]
                )
            )
        )

        raw_margin_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "raw_margin_error"
                    ]
                )
            )
        )

        raw_total_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "raw_total_error"
                    ]
                )
            )
        )

        final_home_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "home_score_error"
                    ]
                )
            )
        )

        final_away_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "away_score_error"
                    ]
                )
            )
        )

        final_margin_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "margin_error"
                    ]
                )
            )
        )

        final_total_mae = float(
            np.mean(
                np.abs(
                    completed[
                        "total_error"
                    ]
                )
            )
        )

        actual_winners = (
            completed.apply(
                lambda row: (
                    row[
                        "home_team"
                    ]
                    if (
                        row[
                            "actual_home_points"
                        ]
                        >
                        row[
                            "actual_away_points"
                        ]
                    )
                    else row[
                        "away_team"
                    ]
                ),
                axis=1,
            )
        )

        raw_winner_accuracy = float(
            (
                completed[
                    "raw_score_model_winner"
                ]
                ==
                actual_winners
            )
            .mean()
        )

        final_winner_accuracy = float(
            (
                completed[
                    "score_model_winner"
                ]
                ==
                actual_winners
            )
            .mean()
        )

        official_winner_accuracy = float(
            (
                completed[
                    "predicted_winner"
                ]
                ==
                actual_winners
            )
            .mean()
        )

        print(
            "RAW SCORE ENGINE"
        )

        print(
            f"Home score MAE:  "
            f"{raw_home_mae:.2f}"
        )

        print(
            f"Away score MAE:  "
            f"{raw_away_mae:.2f}"
        )

        print(
            f"Margin MAE:      "
            f"{raw_margin_mae:.2f}"
        )

        print(
            f"Total MAE:       "
            f"{raw_total_mae:.2f}"
        )

        print(
            f"Winner accuracy: "
            f"{raw_winner_accuracy:.2%}"
        )

        print()

        print(
            "FROZEN COMPACT_FULL SCORE MODEL"
        )

        print(
            f"Home score MAE:  "
            f"{final_home_mae:.2f}"
        )

        print(
            f"Away score MAE:  "
            f"{final_away_mae:.2f}"
        )

        print(
            f"Margin MAE:      "
            f"{final_margin_mae:.2f}"
        )

        print(
            f"Total MAE:       "
            f"{final_total_mae:.2f}"
        )

        print(
            f"Winner accuracy: "
            f"{final_winner_accuracy:.2%}"
        )

        print()

        print(
            "OFFICIAL FROZEN WINNER MODEL"
        )

        print(
            f"Winner accuracy: "
            f"{official_winner_accuracy:.2%}"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            f"{len(completed):,} completed games remain "
            "too small for model retuning."
        )

    # ========================================================
    # COMPLETED TABLE
    # ========================================================

    if not completed.empty:

        section(
            "9. COMPLETED FROZEN SCORE PREDICTIONS"
        )

        columns = [
            "week",
            "away_team",
            "home_team",

            "raw_expected_away_points",
            "raw_expected_home_points",

            "final_expected_away_points",
            "final_expected_home_points",

            "actual_away_points",
            "actual_home_points",

            "final_expected_home_margin",
            "final_expected_total_points",

            "score_model_winner",
            "predicted_winner",
        ]

        columns = [
            column
            for column in columns
            if column in completed.columns
        ]

        print(
            completed[
                columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # UPCOMING
    # ========================================================

    if not upcoming.empty:

        section(
            "10. NEXT UPCOMING PRODUCTION PREDICTIONS"
        )

        preview = (
            upcoming
            .head(
                20
            )
        )

        columns = [
            "start_date",
            "week",
            "away_team",
            "home_team",

            "projected_away_score",
            "projected_home_score",

            "final_expected_home_margin",
            "final_expected_total_points",

            "away_win_probability",
            "home_win_probability",
            "predicted_winner",

            "simulation_away_win_probability",
            "simulation_home_win_probability",

            "simulation_away_score_p10",
            "simulation_away_score_p90",
            "simulation_home_score_p10",
            "simulation_home_score_p90",
        ]

        columns = [
            column
            for column in columns
            if column in preview.columns
        ]

        print(
            preview[
                columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "2026 FROZEN SCORE PRODUCTION RUN COMPLETE"
    )

    print(
        "✓ Historical scoring state replayed."
    )

    print(
        "✓ Corrected OOS Monte Carlo uncertainty loaded."
    )

    print(
        "✓ Frozen compact_full model loaded."
    )

    print(
        "✓ Raw scoring predictions generated."
    )

    print(
        "✓ Frozen margin correction applied."
    )

    print(
        "✓ Frozen total correction applied."
    )

    print(
        "✓ Final team scores reconstructed."
    )

    print(
        "✓ Monte Carlo run around FINAL scores."
    )

    print(
        "✓ Same-kickoff leakage protection preserved."
    )

    print(
        "✓ Completed results applied only after prediction."
    )

    print(
        "✓ Official winner probability unchanged."
    )

    print()

    print(
        "Production score output:"
    )

    print(
        SCORING_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()