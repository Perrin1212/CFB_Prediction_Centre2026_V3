from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from config.settings import PROCESSED_DATA_DIR

from engine.possessions import (
    PossessionConfig,
    PossessionModel,
)


# ============================================================
# FILES
# ============================================================

TEAM_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "opponent_adjusted_rating_history.csv"
)

ENVIRONMENT_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "game_environment_history.csv"
)

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    / "teams.csv"
)

POSSESSION_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "possession_drive_history.csv"
)

TEAM_POSSESSION_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "team_possession_state_latest.csv"
)

FBS_POSSESSION_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "fbs_possession_state_latest.csv"
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— POSSESSION / DRIVE OPPORTUNITY MODEL"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    for path in [
        TEAM_HISTORY_FILE,
        ENVIRONMENT_HISTORY_FILE,
        TEAMS_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired file does not exist:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING INPUTS"
    )

    team_history = pd.read_csv(
        TEAM_HISTORY_FILE,
        low_memory=False,
    )

    environment = pd.read_csv(
        ENVIRONMENT_HISTORY_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"Team-game rows loaded:    "
        f"{len(team_history):,}"
    )

    print(
        f"Historical games:         "
        f"{team_history['game_id'].nunique():,}"
    )

    print(
        f"Environment games loaded: "
        f"{len(environment):,}"
    )

    print(
        f"Teams loaded:             "
        f"{len(teams):,}"
    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    section(
        "POSSESSION INPUT VALIDATION"
    )

    required_team_columns = [
        "game_id",
        "season",
        "week",

        "team_name",
        "opponent_name",
        "home_away",

        "rushing_attempts",
        "passing_attempts",

        "first_downs",
        "turnovers",
    ]

    missing = [
        column
        for column in required_team_columns
        if column not in team_history.columns
    ]

    if missing:

        raise RuntimeError(
            "Team history is missing required "
            f"fields: {missing}"
        )

    required_environment = [
        "game_id",

        "expected_home_plays",
        "expected_away_plays",

        "expected_home_possession_share",
        "expected_away_possession_share",
    ]

    missing_environment = [
        column
        for column in required_environment
        if column not in environment.columns
    ]

    if missing_environment:

        raise RuntimeError(
            "Environment history is missing "
            f"fields: {missing_environment}"
        )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    for column in [
        "rushing_attempts",
        "passing_attempts",
        "first_downs",
        "turnovers",
    ]:

        coverage = (
            team_history[
                column
            ]
            .notna()
            .mean()
        )

        print(
            f"{column:<25} "
            f"{coverage:>7.2%}"
        )

    print()

    print(
        f"Team-history games: "
        f"{team_history['game_id'].nunique():,}"
    )

    print(
        f"Environment games:  "
        f"{environment['game_id'].nunique():,}"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    section(
        "POSSESSION MODEL CONFIGURATION"
    )

    config = PossessionConfig(
        update_rate=0.25,
        preseason_regression=0.40,

        starting_first_down_rate=0.30,
        starting_turnover_rate=0.022,

        starting_plays_per_drive=6.20,

        national_prior_team_games=100.0,

        offense_sustain_weight=0.55,
        defense_resistance_weight=0.45,

        first_down_weight=0.70,
        turnover_weight=0.30,

        minimum_plays_per_drive=4.50,
        maximum_plays_per_drive=8.50,

        minimum_estimated_drives=7.0,
        maximum_estimated_drives=16.0,

        sustainability_floor=0.70,
        sustainability_ceiling=1.30,

        possession_drive_adjustment=1.00,
    )

    print(
        f"Update rate:                 "
        f"{config.update_rate:.0%}"
    )

    print(
        f"Preseason regression:        "
        f"{config.preseason_regression:.0%}"
    )

    print(
        f"Starting plays/drive proxy:  "
        f"{config.starting_plays_per_drive:.2f}"
    )

    print(
        f"Drive range:                 "
        f"{config.minimum_estimated_drives:.1f}"
        f" – "
        f"{config.maximum_estimated_drives:.1f}"
    )

    print(
        f"Sustainability components:   "
        f"{config.first_down_weight:.0%} first downs / "
        f"{config.turnover_weight:.0%} turnovers"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  These are ESTIMATED drives."
    )

    print(
        "  The source database does not contain true "
        "historical drive counts."
    )

    print(
        "  No fabricated drive observations are being "
        "written into the model."
    )

    # ========================================================
    # BUILD
    # ========================================================

    section(
        "BUILDING CHRONOLOGICAL DRIVE OPPORTUNITIES"
    )

    model = PossessionModel(
        config=config
    )

    possession_history = (
        model.process_history(
            team_history=team_history,
            environment_history=environment,
        )
    )

    current_states = (
        model.current_states()
    )

    print(
        f"Possession game rows created: "
        f"{len(possession_history):,}"
    )

    print(
        f"Teams with possession state:  "
        f"{len(current_states):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "POSSESSION MODEL VALIDATION"
    )

    expected_games = int(
        environment[
            "game_id"
        ]
        .nunique()
    )

    created_games = int(
        possession_history[
            "game_id"
        ]
        .nunique()
    )

    print(
        f"Expected games: "
        f"{expected_games:,}"
    )

    print(
        f"Created games:  "
        f"{created_games:,}"
    )

    print(
        f"Missing games:  "
        f"{expected_games - created_games:,}"
    )

    if created_games != expected_games:

        raise RuntimeError(
            "Possession model failed to create "
            "all historical games."
        )

    core_columns = [
        "expected_home_plays_per_drive",
        "expected_away_plays_per_drive",

        "expected_home_drives",
        "expected_away_drives",
        "expected_total_drives",

        "home_offensive_sustainability_index",
        "away_offensive_sustainability_index",

        "home_defensive_resistance_index",
        "away_defensive_resistance_index",
    ]

    missing_core = int(
        possession_history[
            core_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing core features: "
        f"{missing_core:,}"
    )

    if missing_core > 0:

        raise RuntimeError(
            "Possession output contains missing "
            "pregame features."
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    section(
        "ESTIMATED DRIVE DISTRIBUTION"
    )

    print(
        "HOME ESTIMATED DRIVES"
    )

    print(
        f"  High:   "
        f"{possession_history['expected_home_drives'].max():.2f}"
    )

    print(
        f"  Median: "
        f"{possession_history['expected_home_drives'].median():.2f}"
    )

    print(
        f"  Low:    "
        f"{possession_history['expected_home_drives'].min():.2f}"
    )

    print()

    print(
        "AWAY ESTIMATED DRIVES"
    )

    print(
        f"  High:   "
        f"{possession_history['expected_away_drives'].max():.2f}"
    )

    print(
        f"  Median: "
        f"{possession_history['expected_away_drives'].median():.2f}"
    )

    print(
        f"  Low:    "
        f"{possession_history['expected_away_drives'].min():.2f}"
    )

    print()

    print(
        "TOTAL ESTIMATED DRIVES"
    )

    print(
        f"  High:   "
        f"{possession_history['expected_total_drives'].max():.2f}"
    )

    print(
        f"  Median: "
        f"{possession_history['expected_total_drives'].median():.2f}"
    )

    print(
        f"  Low:    "
        f"{possession_history['expected_total_drives'].min():.2f}"
    )

    print()

    print(
        "EXPECTED PLAYS PER DRIVE"
    )

    combined_ppd = pd.concat(
        [
            possession_history[
                "expected_home_plays_per_drive"
            ],

            possession_history[
                "expected_away_plays_per_drive"
            ],
        ],
        ignore_index=True,
    )

    print(
        f"  High:   "
        f"{combined_ppd.max():.2f}"
    )

    print(
        f"  Median: "
        f"{combined_ppd.median():.2f}"
    )

    print(
        f"  Low:    "
        f"{combined_ppd.min():.2f}"
    )

    # ========================================================
    # SUSTAINABILITY DIAGNOSTIC
    # ========================================================

    section(
        "SUSTAINABILITY DIAGNOSTIC"
    )

    possession_history[
        "sustainability_matchup_difference"
    ] = (
        (
            possession_history[
                "home_offensive_sustainability_index"
            ]
            -
            possession_history[
                "away_defensive_resistance_index"
            ]
        )
        -
        (
            possession_history[
                "away_offensive_sustainability_index"
            ]
            -
            possession_history[
                "home_defensive_resistance_index"
            ]
        )
    )

    possession_history[
        "actual_home_margin"
    ] = (
        pd.to_numeric(
            possession_history[
                "home_points"
            ],
            errors="coerce",
        )
        -
        pd.to_numeric(
            possession_history[
                "away_points"
            ],
            errors="coerce",
        )
    )

    sustain_margin_corr = (
        possession_history[
            "sustainability_matchup_difference"
        ]
        .corr(
            possession_history[
                "actual_home_margin"
            ]
        )
    )

    home_drive_matchup_corr = (
        possession_history[
            "expected_home_plays_per_drive"
        ]
        .corr(
            pd.to_numeric(
                possession_history[
                    "home_points"
                ],
                errors="coerce",
            )
        )
    )

    away_drive_matchup_corr = (
        possession_history[
            "expected_away_plays_per_drive"
        ]
        .corr(
            pd.to_numeric(
                possession_history[
                    "away_points"
                ],
                errors="coerce",
            )
        )
    )

    print(
        "Correlation diagnostics:"
    )

    print(
        f"  Sustainability difference vs "
        f"margin: "
        f"{sustain_margin_corr:.3f}"
    )

    print(
        f"  Home expected plays/drive vs "
        f"home points: "
        f"{home_drive_matchup_corr:.3f}"
    )

    print(
        f"  Away expected plays/drive vs "
        f"away points: "
        f"{away_drive_matchup_corr:.3f}"
    )

    print()

    print(
        "These are structural diagnostics only."
    )

    print(
        "There are no observed historical drive counts "
        "to score drive-count MAE against."
    )

    # ========================================================
    # FBS CURRENT STATES
    # ========================================================

    section(
        "BUILDING FBS-ONLY POSSESSION STATE"
    )

    if "team_name" not in teams.columns:

        raise RuntimeError(
            "teams.csv is missing team_name."
        )

    if "classification" not in teams.columns:

        raise RuntimeError(
            "teams.csv is missing classification."
        )

    teams["classification"] = (
        teams["classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    fbs_names = set(
        teams.loc[
            teams["classification"]
            ==
            "fbs",
            "team_name",
        ]
        .dropna()
        .astype(str)
    )

    fbs_states = (
        current_states[
            current_states[
                "team"
            ].isin(
                fbs_names
            )
        ]
        .copy()
    )

    print(
        f"FBS teams with state: "
        f"{len(fbs_states):,}"
    )

    # ========================================================
    # TOP SUSTAINING OFFENCES
    # ========================================================

    section(
        "TOP 15 CURRENT FBS FIRST-DOWN RATES"
    )

    top_first_down = (
        fbs_states
        .sort_values(
            "offense_first_down_rate",
            ascending=False,
        )
        .head(15)
    )

    for rank, row in enumerate(
        top_first_down.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.offense_first_down_rate:>7.2%}"
        )

    # ========================================================
    # BEST TURNOVER AVOIDANCE
    # ========================================================

    section(
        "TOP 15 CURRENT FBS TURNOVER-AVOIDANCE RATES"
    )

    best_turnover = (
        fbs_states
        .sort_values(
            "offense_turnover_rate",
            ascending=True,
        )
        .head(15)
    )

    for rank, row in enumerate(
        best_turnover.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.offense_turnover_rate:>7.2%}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING POSSESSION OUTPUT"
    )

    save_csv(
        possession_history,
        POSSESSION_HISTORY_FILE,
    )

    save_csv(
        current_states,
        TEAM_POSSESSION_LATEST_FILE,
    )

    save_csv(
        fbs_states,
        FBS_POSSESSION_LATEST_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "POSSESSION / DRIVE OPPORTUNITY MODEL COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {POSSESSION_HISTORY_FILE}"
    )

    print(
        f"  {TEAM_POSSESSION_LATEST_FILE}"
    )

    print(
        f"  {FBS_POSSESSION_LATEST_FILE}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "  ELO"
    )

    print(
        "   ↓"
    )

    print(
        "  OFFENCE / DEFENCE"
    )

    print(
        "   ↓"
    )

    print(
        "  OPPONENT ADJUSTMENT"
    )

    print(
        "   ↓"
    )

    print(
        "  MATCHUP ENGINE"
    )

    print(
        "   ↓"
    )

    print(
        "  GAME ENVIRONMENT / PACE"
    )

    print(
        "   ↓"
    )

    print(
        "  POSSESSION / DRIVE OPPORTUNITIES"
    )

    print(
        "   ↓"
    )

    print(
        "  SCORING MODEL"
    )

    print()


if __name__ == "__main__":
    main()