from __future__ import annotations

from pathlib import Path
import sys

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

from engine.environment import (
    EnvironmentConfig,
    GameEnvironmentEngine,
)


# ============================================================
# FILES
# ============================================================

TEAM_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "opponent_adjusted_rating_history.csv"
)

MATCHUP_FILE = (
    PROCESSED_DATA_DIR
    / "matchup_history.csv"
)

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    / "teams.csv"
)

ENVIRONMENT_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "game_environment_history.csv"
)

TEAM_ENVIRONMENT_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "team_environment_latest.csv"
)

FBS_ENVIRONMENT_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "fbs_environment_latest.csv"
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
        "— GAME ENVIRONMENT / PACE"
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    for path in [
        TEAM_HISTORY_FILE,
        MATCHUP_FILE,
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

    matchups = pd.read_csv(
        MATCHUP_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"Team-game rows loaded: "
        f"{len(team_history):,}"
    )

    print(
        f"Team-game games:       "
        f"{team_history['game_id'].nunique():,}"
    )

    print(
        f"Matchup rows loaded:   "
        f"{len(matchups):,}"
    )

    print(
        f"Teams loaded:          "
        f"{len(teams):,}"
    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    section(
        "ENVIRONMENT INPUT VALIDATION"
    )

    required = [
        "game_id",
        "season",
        "week",
        "team_name",
        "opponent_name",
        "home_away",
        "rushing_attempts",
        "passing_attempts",
        "possession_minutes",
    ]

    missing = [
        column
        for column in required
        if column not in team_history.columns
    ]

    if missing:

        print()
        print(
            "Available columns:"
        )

        for column in team_history.columns:
            print(
                f"  {column}"
            )

        raise RuntimeError(
            "Opponent-adjusted history is missing "
            f"environment fields: {missing}"
        )

    rows_per_game = (
        team_history
        .groupby(
            "game_id"
        )
        .size()
    )

    complete_games = int(
        (
            rows_per_game
            ==
            2
        ).sum()
    )

    invalid_games = int(
        (
            rows_per_game
            !=
            2
        ).sum()
    )

    print(
        f"Games with two team rows: "
        f"{complete_games:,}"
    )

    print(
        f"Invalid game pairs:       "
        f"{invalid_games:,}"
    )

    if invalid_games > 0:

        raise RuntimeError(
            "Environment engine requires exactly "
            "two team rows per game."
        )

    stat_columns = [
        "rushing_attempts",
        "passing_attempts",
        "possession_minutes",
    ]

    for column in stat_columns:

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

    # ========================================================
    # CONFIG
    # ========================================================

    section(
        "ENVIRONMENT CONFIGURATION"
    )

    config = EnvironmentConfig(
        update_rate=0.25,
        preseason_regression=0.40,

        starting_team_plays=68.0,
        starting_possession_minutes=30.0,
        starting_plays_per_possession_minute=2.27,

        national_prior_team_games=100.0,

        offense_environment_weight=0.50,
        defense_environment_weight=0.50,

        minimum_team_plays=40.0,
        maximum_team_plays=100.0,

        minimum_total_plays=90.0,
        maximum_total_plays=190.0,

        minimum_possession_share=0.30,
        maximum_possession_share=0.70,
    )

    print(
        f"Update rate:             "
        f"{config.update_rate:.0%}"
    )

    print(
        f"Preseason regression:    "
        f"{config.preseason_regression:.0%}"
    )

    print(
        f"Starting team plays:     "
        f"{config.starting_team_plays:.1f}"
    )

    print(
        f"Environment blend:       "
        f"{config.offense_environment_weight:.0%} offence / "
        f"{config.defense_environment_weight:.0%} opposing defence"
    )

    print()

    print(
        "Important:"
    )

    print(
        "  This models PLAY ENVIRONMENT."
    )

    print(
        "  Plays are NOT being treated as possessions."
    )

    print(
        "  True drive/possession estimation comes next."
    )

    # ========================================================
    # BUILD
    # ========================================================

    section(
        "BUILDING CHRONOLOGICAL GAME ENVIRONMENT"
    )

    engine = GameEnvironmentEngine(
        config=config
    )

    environment_history = (
        engine.process_history(
            team_history=team_history,
            matchup_history=matchups,
        )
    )

    current_environment = (
        engine.current_environment()
    )

    print(
        f"Environment game rows created: "
        f"{len(environment_history):,}"
    )

    print(
        f"Teams with environment state:  "
        f"{len(current_environment):,}"
    )

    # ========================================================
    # FBS-ONLY CURRENT OUTPUT
    # ========================================================

    section(
        "BUILDING FBS-ONLY ENVIRONMENT"
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

    fbs_environment = (
        current_environment[
            current_environment[
                "team"
            ].isin(
                fbs_names
            )
        ]
        .copy()
    )

    print(
        f"FBS teams with environment state: "
        f"{len(fbs_environment):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "ENVIRONMENT VALIDATION"
    )

    expected_games = int(
        matchups[
            "game_id"
        ]
        .nunique()
    )

    created_games = int(
        environment_history[
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
            "Environment engine did not produce "
            "all matchup games."
        )

    core_features = [
        "expected_home_plays",
        "expected_away_plays",
        "expected_total_plays",

        "expected_home_possession_share",
        "expected_away_possession_share",

        "game_pace_index",
    ]

    missing_core = int(
        environment_history[
            core_features
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing environment features: "
        f"{missing_core:,}"
    )

    if missing_core > 0:

        raise RuntimeError(
            "Environment output contains missing "
            "pregame features."
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    section(
        "GAME ENVIRONMENT DISTRIBUTION"
    )

    print(
        "EXPECTED HOME PLAYS"
    )

    print(
        f"  High:   "
        f"{environment_history['expected_home_plays'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{environment_history['expected_home_plays'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{environment_history['expected_home_plays'].min():.1f}"
    )

    print()

    print(
        "EXPECTED AWAY PLAYS"
    )

    print(
        f"  High:   "
        f"{environment_history['expected_away_plays'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{environment_history['expected_away_plays'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{environment_history['expected_away_plays'].min():.1f}"
    )

    print()

    print(
        "EXPECTED TOTAL PLAYS"
    )

    print(
        f"  High:   "
        f"{environment_history['expected_total_plays'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{environment_history['expected_total_plays'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{environment_history['expected_total_plays'].min():.1f}"
    )

    print()

    print(
        "GAME PACE INDEX"
    )

    print(
        f"  High:   "
        f"{environment_history['game_pace_index'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{environment_history['game_pace_index'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{environment_history['game_pace_index'].min():.1f}"
    )

    # ========================================================
    # HISTORICAL ACCURACY DIAGNOSTICS
    # ========================================================

    section(
        "PACE DIAGNOSTICS"
    )

    environment_history[
        "home_play_error"
    ] = (
        environment_history[
            "expected_home_plays"
        ]
        -
        environment_history[
            "actual_home_plays"
        ]
    )

    environment_history[
        "away_play_error"
    ] = (
        environment_history[
            "expected_away_plays"
        ]
        -
        environment_history[
            "actual_away_plays"
        ]
    )

    environment_history[
        "total_play_error"
    ] = (
        environment_history[
            "expected_total_plays"
        ]
        -
        environment_history[
            "actual_total_plays"
        ]
    )

    home_mae = (
        environment_history[
            "home_play_error"
        ]
        .abs()
        .mean()
    )

    away_mae = (
        environment_history[
            "away_play_error"
        ]
        .abs()
        .mean()
    )

    total_mae = (
        environment_history[
            "total_play_error"
        ]
        .abs()
        .mean()
    )

    total_correlation = (
        environment_history[
            "expected_total_plays"
        ]
        .corr(
            environment_history[
                "actual_total_plays"
            ]
        )
    )

    possession_correlation = (
        environment_history[
            "expected_home_possession_share"
        ]
        .corr(
            environment_history[
                "actual_home_possession_share"
            ]
        )
    )

    print(
        f"Home plays MAE:             "
        f"{home_mae:.2f}"
    )

    print(
        f"Away plays MAE:             "
        f"{away_mae:.2f}"
    )

    print(
        f"Total plays MAE:            "
        f"{total_mae:.2f}"
    )

    print(
        f"Expected vs actual total "
        f"plays correlation: "
        f"{total_correlation:.3f}"
    )

    print(
        f"Expected vs actual home "
        f"possession-share correlation: "
        f"{possession_correlation:.3f}"
    )

    print()

    print(
        "These are diagnostic statistics only."
    )

    print(
        "No parameters are being optimised here."
    )

    # ========================================================
    # FASTEST CURRENT FBS OFFENCES
    # ========================================================

    section(
        "TOP 15 CURRENT FBS OFFENSIVE PLAY VOLUMES"
    )

    fastest = (
        fbs_environment
        .sort_values(
            "offense_plays",
            ascending=False,
        )
        .head(15)
    )

    for rank, row in enumerate(
        fastest.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.offense_plays:>6.1f} plays"
        )

    # ========================================================
    # POSSESSION TEAMS
    # ========================================================

    section(
        "TOP 15 CURRENT FBS POSSESSION SHARES"
    )

    possession = (
        fbs_environment
        .sort_values(
            "possession_share",
            ascending=False,
        )
        .head(15)
    )

    for rank, row in enumerate(
        possession.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.possession_share:>7.2%}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING GAME ENVIRONMENT OUTPUT"
    )

    save_csv(
        environment_history,
        ENVIRONMENT_HISTORY_FILE,
    )

    save_csv(
        current_environment,
        TEAM_ENVIRONMENT_LATEST_FILE,
    )

    save_csv(
        fbs_environment,
        FBS_ENVIRONMENT_LATEST_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "GAME ENVIRONMENT COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {ENVIRONMENT_HISTORY_FILE}"
    )

    print(
        f"  {TEAM_ENVIRONMENT_LATEST_FILE}"
    )

    print(
        f"  {FBS_ENVIRONMENT_LATEST_FILE}"
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
        "  POSSESSION / DRIVE MODEL"
    )

    print()


if __name__ == "__main__":
    main()