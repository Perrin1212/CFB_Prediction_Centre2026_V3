from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_NAME = "CFB Prediction Centre 2026 V2"

CURRENT_SEASON = 2026


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# V2 DATABASE
# ============================================================

V2_DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "cfb_prediction_v2.db"
)

V2_DATABASE_URL = (
    f"sqlite:///{V2_DATABASE_PATH}"
)


# ============================================================
# V1 DATABASE
#
# Default expected structure:
#
# Desktop/
# ├── CFB_Prediction_Centre2026/
# │   └── data/
# │       └── cfb_prediction.db
# │
# └── CFB_Prediction_Centre2026_V2/
#
# You can override this in .env using:
#
# V1_DATABASE_PATH=C:\path\to\database.db
# ============================================================

DEFAULT_V1_DATABASE_PATH = (
    PROJECT_ROOT.parent
    / "CFB_Prediction_Centre2026"
    / "data"
    / "cfb_prediction.db"
)

V1_DATABASE_PATH = Path(
    os.getenv(
        "V1_DATABASE_PATH",
        str(DEFAULT_V1_DATABASE_PATH),
    )
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PREDICTIONS_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)


# ============================================================
# SEASONS
# ============================================================

HISTORICAL_SEASONS = (
    2023,
    2024,
    2025,
)

MODEL_SEASONS = (
    2023,
    2024,
    2025,
    2026,
)


# ============================================================
# MODEL DEFAULTS
#
# IMPORTANT:
# These are initial engineering defaults.
#
# They are NOT final statistically optimised values.
#
# V2 will later use historical walk-forward testing to tune
# parameters such as ELO K-factor, home-field advantage,
# preseason regression, possession variance, etc.
# ============================================================

SIMULATIONS = 10_000

NATIONAL_MEAN_ELO = 1500.0

DEFAULT_HOME_FIELD_ADVANTAGE = 55.0

DEFAULT_K_FACTOR = 22.0

DEFAULT_PRESEASON_REGRESSION = 0.35

MIN_HISTORICAL_GAMES = 1


# ============================================================
# NATIONAL BASELINES
#
# Initial defaults used by the V2 scaffold.
# These will eventually be calculated dynamically from
# historical CFB data rather than hard-coded.
# ============================================================

DEFAULT_POSSESSION_STD = 1.5

NATIONAL_PACE = 70.0

NATIONAL_POINTS_PER_DRIVE = 2.0

NATIONAL_YARDS_PER_PLAY = 5.5

NATIONAL_THIRD_DOWN_PCT = 0.40

NATIONAL_RED_ZONE_TD_PCT = 0.60

NATIONAL_EXPLOSIVE_PLAY_RATE = 0.10

NATIONAL_STARTING_FIELD_POSITION = 28.0


# ============================================================
# BACKWARDS-COMPATIBLE MODEL SETTINGS
#
# Some of the V2 scaffold files created previously use:
#
#     from config.settings import SETTINGS
#
# Therefore we retain this object while we progressively
# replace the placeholder architecture with the real model.
# ============================================================

@dataclass(frozen=True)
class ModelSettings:

    current_season: int = CURRENT_SEASON

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    elo_starting_rating: float = NATIONAL_MEAN_ELO

    elo_home_field_advantage: float = (
        DEFAULT_HOME_FIELD_ADVANTAGE
    )

    elo_k_factor: float = DEFAULT_K_FACTOR

    elo_preseason_regression: float = (
        DEFAULT_PRESEASON_REGRESSION
    )

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    simulations: int = SIMULATIONS

    random_seed: int = 42

    # --------------------------------------------------------
    # Possessions
    # --------------------------------------------------------

    possession_std: float = DEFAULT_POSSESSION_STD

    national_pace: float = NATIONAL_PACE

    # --------------------------------------------------------
    # Offensive / defensive baselines
    # --------------------------------------------------------

    national_points_per_drive: float = (
        NATIONAL_POINTS_PER_DRIVE
    )

    national_yards_per_play: float = (
        NATIONAL_YARDS_PER_PLAY
    )

    national_third_down_pct: float = (
        NATIONAL_THIRD_DOWN_PCT
    )

    national_red_zone_td_pct: float = (
        NATIONAL_RED_ZONE_TD_PCT
    )

    national_explosive_play_rate: float = (
        NATIONAL_EXPLOSIVE_PLAY_RATE
    )

    national_starting_field_position: float = (
        NATIONAL_STARTING_FIELD_POSITION
    )


SETTINGS = ModelSettings()