from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import (
    PROCESSED_DATA_DIR,
)

from engine.opponent_adjustment import (
    OpponentAdjustmentConfig,
    OpponentAdjustmentEngine,
)


# ============================================================
# FILES
# ============================================================


RAW_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    /
    "raw_rating_history.csv"
)

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    /
    "teams.csv"
)

ADJUSTED_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    /
    "opponent_adjusted_rating_history.csv"
)

LATEST_FILE = (
    PROCESSED_DATA_DIR
    /
    "team_opponent_adjusted_ratings_latest.csv"
)

FBS_LATEST_FILE = (
    PROCESSED_DATA_DIR
    /
    "fbs_opponent_adjusted_ratings_latest.csv"
)


# ============================================================
# HELPERS
# ============================================================


def section(
    title: str,
) -> None:

    print()

    print(
        "=" * 78
    )

    print(
        title
    )

    print(
        "=" * 78
    )


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
        f"✓ Saved {len(df):,} rows "
        f"→ {path}"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— OPPONENT ADJUSTMENT"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    if not RAW_HISTORY_FILE.exists():

        raise FileNotFoundError(
            "\nRaw rating history does not exist:\n"
            f"{RAW_HISTORY_FILE}\n\n"
            "Run:\n"
            "python -m jobs.build_ratings"
        )

    if not TEAMS_FILE.exists():

        raise FileNotFoundError(
            "\nTeam master does not exist:\n"
            f"{TEAMS_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING RAW RATINGS"
    )

    raw_history = pd.read_csv(
        RAW_HISTORY_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"Raw rating rows loaded: "
        f"{len(raw_history):,}"
    )

    print(
        f"Games represented: "
        f"{raw_history['game_id'].nunique():,}"
    )

    print(
        f"Teams loaded: "
        f"{len(teams):,}"
    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    section(
        "INPUT VALIDATION"
    )

    required = [
        "game_id",
        "season",
        "team_name",
        "opponent_name",

        "team_points",
        "opponent_points",

        "yards_per_play_proxy",
        "turnovers",
        "first_downs",

        "team_offense_rating_pre",
        "team_defense_rating_pre",

        "opponent_offense_rating_pre",
        "opponent_defense_rating_pre",

        "rating_state_national_points_pre",
        "rating_state_national_ypp_pre",
        "rating_state_national_turnovers_pre",
        "rating_state_national_first_downs_pre",
    ]

    missing = [
        column
        for column in required
        if column not in raw_history.columns
    ]

    if missing:

        raise RuntimeError(
            "raw_rating_history.csv is missing "
            f"required fields: {missing}"
        )

    rows_per_game = (
        raw_history
        .groupby(
            "game_id"
        )
        .size()
    )

    valid_games = int(
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
        f"Games with exactly two team rows: "
        f"{valid_games:,}"
    )

    print(
        f"Invalid game pairings: "
        f"{invalid_games:,}"
    )

    if invalid_games > 0:

        raise RuntimeError(
            "Opponent adjustment requires exactly "
            "two rows per game."
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    section(
        "OPPONENT-ADJUSTMENT CONFIGURATION"
    )

    config = OpponentAdjustmentConfig(
        update_rate=0.25,

        preseason_regression=0.40,

        rating_floor=50.0,
        rating_ceiling=150.0,

        component_ratio_floor=0.50,
        component_ratio_ceiling=1.50,

        opponent_factor_floor=0.75,
        opponent_factor_ceiling=1.25,

        offense_points_weight=0.40,
        offense_ypp_weight=0.35,
        offense_turnover_weight=0.15,
        offense_first_down_weight=0.10,

        defense_points_weight=0.40,
        defense_ypp_weight=0.35,
        defense_takeaway_weight=0.15,
        defense_first_down_weight=0.10,
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
        f"Opponent factor range:   "
        f"{config.opponent_factor_floor:.2f}"
        f" – "
        f"{config.opponent_factor_ceiling:.2f}"
    )

    print(
        f"Rating range:            "
        f"{config.rating_floor:.0f}"
        f" – "
        f"{config.rating_ceiling:.0f}"
    )

    print()

    print(
        "No parameter optimisation is being run."
    )

    print(
        "This is a single chronological pass."
    )

    # ========================================================
    # BUILD
    # ========================================================

    section(
        "BUILDING OPPONENT-ADJUSTED RATINGS"
    )

    engine = (
        OpponentAdjustmentEngine(
            config=config
        )
    )

    adjusted_history = (
        engine.process_history(
            raw_history
        )
    )

    latest = (
        engine.current_ratings()
    )

    print(
        f"Adjusted history rows: "
        f"{len(adjusted_history):,}"
    )

    print(
        f"Teams with adjusted ratings: "
        f"{len(latest):,}"
    )

    # ========================================================
    # FBS-ONLY OUTPUT
    # ========================================================

    section(
        "BUILDING FBS-ONLY RATINGS"
    )

    teams[
        "classification"
    ] = (
        teams[
            "classification"
        ]
        .astype(
            "string"
        )
        .str.strip()
        .str.lower()
    )

    fbs_names = set(
        teams[
            teams[
                "classification"
            ]
            ==
            "fbs"
        ][
            "team_name"
        ]
        .dropna()
        .astype(
            str
        )
    )

    fbs_latest = latest[
        latest[
            "team"
        ].isin(
            fbs_names
        )
    ].copy()

    fbs_latest = (
        fbs_latest
        .sort_values(
            "adjusted_overall_rating",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    fbs_latest[
        "rank"
    ] = (
        fbs_latest.index
        +
        1
    )

    print(
        f"FBS teams with adjusted ratings: "
        f"{len(fbs_latest):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "OPPONENT-ADJUSTMENT VALIDATION"
    )

    rating_columns = [
        "team_adjusted_offense_rating_pre",
        "team_adjusted_defense_rating_pre",
        "opponent_adjusted_offense_rating_pre",
        "opponent_adjusted_defense_rating_pre",
    ]

    missing_ratings = (
        adjusted_history[
            rating_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    invalid_offense = (
        (
            adjusted_history[
                "team_adjusted_offense_rating_pre"
            ]
            <
            config.rating_floor
        )
        |
        (
            adjusted_history[
                "team_adjusted_offense_rating_pre"
            ]
            >
            config.rating_ceiling
        )
    ).sum()

    invalid_defense = (
        (
            adjusted_history[
                "team_adjusted_defense_rating_pre"
            ]
            <
            config.rating_floor
        )
        |
        (
            adjusted_history[
                "team_adjusted_defense_rating_pre"
            ]
            >
            config.rating_ceiling
        )
    ).sum()

    print(
        f"Missing adjusted pregame ratings: "
        f"{int(missing_ratings):,}"
    )

    print(
        f"Invalid offence ratings:          "
        f"{int(invalid_offense):,}"
    )

    print(
        f"Invalid defence ratings:          "
        f"{int(invalid_defense):,}"
    )

    print()

    print(
        "Game performance index range:"
    )

    print(
        f"  Offence low:  "
        f"{adjusted_history['offense_game_rating_adjusted'].min():.1f}"
    )

    print(
        f"  Offence high: "
        f"{adjusted_history['offense_game_rating_adjusted'].max():.1f}"
    )

    print(
        f"  Defence low:  "
        f"{adjusted_history['defense_game_rating_adjusted'].min():.1f}"
    )

    print(
        f"  Defence high: "
        f"{adjusted_history['defense_game_rating_adjusted'].max():.1f}"
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    section(
        "CURRENT FBS ADJUSTED DISTRIBUTION"
    )

    print(
        "OFFENCE"
    )

    print(
        f"  High:   "
        f"{fbs_latest['adjusted_offense_rating'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{fbs_latest['adjusted_offense_rating'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{fbs_latest['adjusted_offense_rating'].min():.1f}"
    )

    print()

    print(
        "DEFENCE"
    )

    print(
        f"  High:   "
        f"{fbs_latest['adjusted_defense_rating'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{fbs_latest['adjusted_defense_rating'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{fbs_latest['adjusted_defense_rating'].min():.1f}"
    )

    # ========================================================
    # TOP OFFENCES
    # ========================================================

    section(
        "TOP 15 OPPONENT-ADJUSTED FBS OFFENCES"
    )

    top_offense = (
        fbs_latest
        .sort_values(
            "adjusted_offense_rating",
            ascending=False,
        )
        .head(
            15
        )
    )

    for rank, row in enumerate(
        top_offense.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.adjusted_offense_rating:>6.1f}"
        )

    # ========================================================
    # TOP DEFENCES
    # ========================================================

    section(
        "TOP 15 OPPONENT-ADJUSTED FBS DEFENCES"
    )

    top_defense = (
        fbs_latest
        .sort_values(
            "adjusted_defense_rating",
            ascending=False,
        )
        .head(
            15
        )
    )

    for rank, row in enumerate(
        top_defense.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.adjusted_defense_rating:>6.1f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING OPPONENT-ADJUSTED OUTPUT"
    )

    save_csv(
        adjusted_history,
        ADJUSTED_HISTORY_FILE,
    )

    save_csv(
        latest,
        LATEST_FILE,
    )

    save_csv(
        fbs_latest,
        FBS_LATEST_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "OPPONENT ADJUSTMENT COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {ADJUSTED_HISTORY_FILE}"
    )

    print(
        f"  {LATEST_FILE}"
    )

    print(
        f"  {FBS_LATEST_FILE}"
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
        "  RAW OFFENCE / DEFENCE"
    )

    print(
        "   ↓"
    )

    print(
        "  OPPONENT-ADJUSTED OFFENCE / DEFENCE"
    )

    print(
        "   ↓"
    )

    print(
        "  MATCHUP ENGINE"
    )

    print()


if __name__ == "__main__":
    main()