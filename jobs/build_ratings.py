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

from engine.ratings import (
    RatingConfig,
    TeamRatingsEngine,
)


# ============================================================
# FILES
# ============================================================


TEAM_GAMES_FILE = (
    PROCESSED_DATA_DIR
    /
    "historical_team_games.csv"
)

GAMES_FILE = (
    PROCESSED_DATA_DIR
    /
    "historical_games.csv"
)

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    /
    "teams.csv"
)

RATING_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    /
    "raw_rating_history.csv"
)

LATEST_RATINGS_FILE = (
    PROCESSED_DATA_DIR
    /
    "team_raw_ratings_latest.csv"
)

FBS_LATEST_RATINGS_FILE = (
    PROCESSED_DATA_DIR
    /
    "fbs_raw_ratings_latest.csv"
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


def normalise_bool(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
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


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# COLUMN STANDARDISATION
# ============================================================


def standardise_team_game_columns(
    team_games: pd.DataFrame,
) -> pd.DataFrame:

    data = (
        team_games.copy()
    )

    # ========================================================
    # TEAM NAME
    # ========================================================

    if "team_name" not in data.columns:

        team_column = first_existing_column(
            data,
            [
                "team",
                "school",
                "team_school",
            ],
        )

        if team_column is not None:

            data = data.rename(
                columns={
                    team_column: "team_name"
                }
            )

    # ========================================================
    # TEAM POINTS
    # ========================================================

    if "team_points" not in data.columns:

        points_column = first_existing_column(
            data,
            [
                "points",
                "team_score",
                "score",
            ],
        )

        if points_column is not None:

            data = data.rename(
                columns={
                    points_column: "team_points"
                }
            )

    # ========================================================
    # OPPONENT NAME
    # ========================================================

    if "opponent_name" not in data.columns:

        opponent_column = first_existing_column(
            data,
            [
                "opponent",
                "opponent_team",
                "opponent_school",
                "opp_team",
                "opp_name",
            ],
        )

        if opponent_column is not None:

            data = data.rename(
                columns={
                    opponent_column: "opponent_name"
                }
            )

    # ========================================================
    # OPPONENT POINTS
    # ========================================================

    if "opponent_points" not in data.columns:

        opponent_points_column = first_existing_column(
            data,
            [
                "opp_points",
                "opponent_score",
                "opp_score",
            ],
        )

        if opponent_points_column is not None:

            data = data.rename(
                columns={
                    opponent_points_column: "opponent_points"
                }
            )

    # ========================================================
    # BASIC REQUIRED FIELDS BEFORE DERIVATION
    # ========================================================

    base_required = [
        "game_id",
        "season",
        "team_name",
        "team_points",
        "yards_per_play_proxy",
        "turnovers",
        "first_downs",
    ]

    missing_base = [
        column
        for column in base_required
        if column not in data.columns
    ]

    if missing_base:

        print()

        print(
            "Available historical_team_games.csv columns:"
        )

        print()

        for column in data.columns:

            print(
                f"  {column}"
            )

        raise RuntimeError(
            "historical_team_games.csv is missing "
            f"required columns: {missing_base}"
        )

    # ========================================================
    # DERIVE OPPONENT NAME / POINTS FROM PAIRED GAME ROWS
    # ========================================================

    needs_opponent_name = (
        "opponent_name"
        not in data.columns
    )

    needs_opponent_points = (
        "opponent_points"
        not in data.columns
    )

    if (
        needs_opponent_name
        or
        needs_opponent_points
    ):

        print()

        print(
            "Opponent fields not explicitly present."
        )

        print(
            "Deriving opponent information from paired "
            "team-game rows..."
        )

        pair_counts = (
            data
            .groupby(
                "game_id"
            )
            .size()
        )

        valid_pair_ids = set(
            pair_counts[
                pair_counts
                ==
                2
            ].index
        )

        paired = data[
            data[
                "game_id"
            ].isin(
                valid_pair_ids
            )
        ].copy()

        paired["_pair_order"] = (
            paired
            .groupby(
                "game_id"
            )
            .cumcount()
        )

        opponent_lookup = (
            paired[
                [
                    "game_id",
                    "_pair_order",
                    "team_name",
                    "team_points",
                ]
            ]
            .copy()
        )

        opponent_lookup[
            "_pair_order"
        ] = (
            1
            -
            opponent_lookup[
                "_pair_order"
            ]
        )

        opponent_lookup = (
            opponent_lookup
            .rename(
                columns={
                    "team_name": (
                        "_derived_opponent_name"
                    ),
                    "team_points": (
                        "_derived_opponent_points"
                    ),
                }
            )
        )

        paired = paired.merge(
            opponent_lookup,
            on=[
                "game_id",
                "_pair_order",
            ],
            how="left",
        )

        if needs_opponent_name:

            paired[
                "opponent_name"
            ] = paired[
                "_derived_opponent_name"
            ]

        if needs_opponent_points:

            paired[
                "opponent_points"
            ] = paired[
                "_derived_opponent_points"
            ]

        paired = paired.drop(
            columns=[
                "_pair_order",
                "_derived_opponent_name",
                "_derived_opponent_points",
            ],
            errors="ignore",
        )

        incomplete = data[
            ~data[
                "game_id"
            ].isin(
                valid_pair_ids
            )
        ].copy()

        if needs_opponent_name:

            incomplete[
                "opponent_name"
            ] = pd.NA

        if needs_opponent_points:

            incomplete[
                "opponent_points"
            ] = pd.NA

        data = pd.concat(
            [
                paired,
                incomplete,
            ],
            ignore_index=True,
        )

    # ========================================================
    # FINAL REQUIRED FIELD CHECK
    # ========================================================

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
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise RuntimeError(
            "historical_team_games.csv is missing "
            f"required columns after standardisation: "
            f"{missing}"
        )

    return data


# ============================================================
# MODEL UNIVERSE
# ============================================================


def eligible_game_ids(
    games: pd.DataFrame,
) -> set:

    data = (
        games.copy()
    )

    # Some datasets use id instead of game_id.
    if (
        "game_id"
        not in data.columns
        and
        "id" in data.columns
    ):

        data = data.rename(
            columns={
                "id": "game_id"
            }
        )

    required = [
        "game_id",
        "fbs_vs_fbs",
        "fbs_vs_fcs",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise RuntimeError(
            "historical_games.csv is missing "
            f"required columns: {missing}"
        )

    data[
        "fbs_vs_fbs"
    ] = normalise_bool(
        data[
            "fbs_vs_fbs"
        ]
    )

    data[
        "fbs_vs_fcs"
    ] = normalise_bool(
        data[
            "fbs_vs_fcs"
        ]
    )

    eligible = data[
        data[
            "fbs_vs_fbs"
        ]
        |
        data[
            "fbs_vs_fcs"
        ]
    ]

    return set(
        eligible[
            "game_id"
        ]
        .dropna()
        .tolist()
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— RAW TEAM RATINGS"
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    for path in [
        TEAM_GAMES_FILE,
        GAMES_FILE,
        TEAMS_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{path}\n\n"
                "Run:\n"
                "python -m jobs.build_data"
            )

    # ========================================================
    # LOAD DATA
    # ========================================================

    section(
        "LOADING DATA"
    )

    team_games = pd.read_csv(
        TEAM_GAMES_FILE,
        low_memory=False,
    )

    games = pd.read_csv(
        GAMES_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"Team-game rows loaded: "
        f"{len(team_games):,}"
    )

    print(
        f"Historical games loaded: "
        f"{len(games):,}"
    )

    print(
        f"Teams loaded: "
        f"{len(teams):,}"
    )

    team_games = (
        standardise_team_game_columns(
            team_games
        )
    )

    print()

    print(
        "Standardised team-game columns:"
    )

    print(
        f"  Team:            team_name"
    )

    print(
        f"  Opponent:        opponent_name"
    )

    print(
        f"  Team points:     team_points"
    )

    print(
        f"  Opponent points: opponent_points"
    )

    # ========================================================
    # FILTER TO V2 MODEL UNIVERSE
    # ========================================================

    section(
        "FILTERING MODEL UNIVERSE"
    )

    valid_game_ids = (
        eligible_game_ids(
            games
        )
    )

    before = (
        len(
            team_games
        )
    )

    team_games = team_games[
        team_games[
            "game_id"
        ].isin(
            valid_game_ids
        )
    ].copy()

    after = (
        len(
            team_games
        )
    )

    removed = (
        before
        -
        after
    )

    unique_games = (
        team_games[
            "game_id"
        ]
        .nunique()
    )

    print(
        f"Eligible team-game rows: "
        f"{after:,}"
    )

    print(
        f"Eligible games with box-score data: "
        f"{unique_games:,}"
    )

    print(
        f"Lower-division team-game rows removed: "
        f"{removed:,}"
    )

    if team_games.empty:

        raise RuntimeError(
            "No eligible team-game rows remain."
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    section(
        "STAT COVERAGE"
    )

    core_stats = [
        "team_points",
        "opponent_points",
        "yards_per_play_proxy",
        "turnovers",
        "first_downs",
    ]

    for column in core_stats:

        coverage = (
            team_games[
                column
            ]
            .notna()
            .mean()
        )

        print(
            f"{column:<25} "
            f"{coverage:>7.2%}"
        )

    rows_per_game = (
        team_games
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

    incomplete_games = int(
        (
            rows_per_game
            !=
            2
        ).sum()
    )

    print()

    print(
        f"Games with exactly two stat rows: "
        f"{complete_games:,}"
    )

    print(
        f"Incomplete games: "
        f"{incomplete_games:,}"
    )

    valid_complete_ids = set(
        rows_per_game[
            rows_per_game
            ==
            2
        ].index
    )

    team_games = team_games[
        team_games[
            "game_id"
        ].isin(
            valid_complete_ids
        )
    ].copy()

    # ========================================================
    # FINAL DATA SANITY
    # ========================================================

    section(
        "TEAM-GAME SANITY CHECK"
    )

    missing_team = (
        team_games[
            "team_name"
        ]
        .isna()
        .sum()
    )

    missing_opponent = (
        team_games[
            "opponent_name"
        ]
        .isna()
        .sum()
    )

    missing_team_points = (
        team_games[
            "team_points"
        ]
        .isna()
        .sum()
    )

    missing_opponent_points = (
        team_games[
            "opponent_points"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing team names:       "
        f"{missing_team:,}"
    )

    print(
        f"Missing opponent names:   "
        f"{missing_opponent:,}"
    )

    print(
        f"Missing team points:      "
        f"{missing_team_points:,}"
    )

    print(
        f"Missing opponent points:  "
        f"{missing_opponent_points:,}"
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    section(
        "RAW RATING CONFIGURATION"
    )

    config = RatingConfig(
        update_rate=0.25,

        preseason_regression=0.40,

        ratio_floor=0.50,
        ratio_ceiling=1.50,

        rating_floor=50.0,
        rating_ceiling=150.0,

        offense_points_weight=0.40,
        offense_ypp_weight=0.35,
        offense_turnover_weight=0.15,
        offense_first_down_weight=0.10,

        defense_points_weight=0.40,
        defense_ypp_weight=0.35,
        defense_takeaway_weight=0.15,
        defense_first_down_weight=0.10,

        starting_points=26.5,
        starting_yards_per_play=5.50,
        starting_turnovers=1.50,
        starting_first_downs=20.0,

        national_prior_games=50.0,
    )

    print(
        f"Update rate:           "
        f"{config.update_rate:.0%}"
    )

    print(
        f"Preseason regression:  "
        f"{config.preseason_regression:.0%}"
    )

    print()

    print(
        "Offence weights:"
    )

    print(
        f"  Points:       "
        f"{config.offense_points_weight:.0%}"
    )

    print(
        f"  Yards/play:   "
        f"{config.offense_ypp_weight:.0%}"
    )

    print(
        f"  Turnovers:    "
        f"{config.offense_turnover_weight:.0%}"
    )

    print(
        f"  First downs:  "
        f"{config.offense_first_down_weight:.0%}"
    )

    print()

    print(
        "Defence weights:"
    )

    print(
        f"  Points:       "
        f"{config.defense_points_weight:.0%}"
    )

    print(
        f"  Yards/play:   "
        f"{config.defense_ypp_weight:.0%}"
    )

    print(
        f"  Takeaways:    "
        f"{config.defense_takeaway_weight:.0%}"
    )

    print(
        f"  First downs:  "
        f"{config.defense_first_down_weight:.0%}"
    )

    # ========================================================
    # BUILD RATINGS
    # ========================================================

    section(
        "BUILDING CHRONOLOGICAL RAW RATINGS"
    )

    engine = TeamRatingsEngine(
        config=config
    )

    rating_history = (
        engine.process_games(
            team_games
        )
    )

    latest = (
        engine.current_ratings()
    )

    print(
        f"Pregame rating rows created: "
        f"{len(rating_history):,}"
    )

    print(
        f"Teams with current ratings: "
        f"{len(latest):,}"
    )

    # ========================================================
    # FBS-ONLY CURRENT RATINGS
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
            "overall_raw_rating",
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
        f"Official FBS teams rated: "
        f"{len(fbs_latest):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "RATING VALIDATION"
    )

    required_output_columns = [
        "team_offense_rating_pre",
        "team_defense_rating_pre",
        "opponent_offense_rating_pre",
        "opponent_defense_rating_pre",
    ]

    missing_output = [
        column
        for column in required_output_columns
        if column not in rating_history.columns
    ]

    if missing_output:

        raise RuntimeError(
            "Rating history is missing expected "
            f"columns: {missing_output}"
        )

    missing_ratings = (
        rating_history[
            required_output_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    invalid_offense = (
        (
            rating_history[
                "team_offense_rating_pre"
            ]
            <
            config.rating_floor
        )
        |
        (
            rating_history[
                "team_offense_rating_pre"
            ]
            >
            config.rating_ceiling
        )
    ).sum()

    invalid_defense = (
        (
            rating_history[
                "team_defense_rating_pre"
            ]
            <
            config.rating_floor
        )
        |
        (
            rating_history[
                "team_defense_rating_pre"
            ]
            >
            config.rating_ceiling
        )
    ).sum()

    print(
        f"Missing pregame ratings: "
        f"{int(missing_ratings):,}"
    )

    print(
        f"Invalid offence ratings: "
        f"{int(invalid_offense):,}"
    )

    print(
        f"Invalid defence ratings: "
        f"{int(invalid_defense):,}"
    )

    # ========================================================
    # RATING DISTRIBUTIONS
    # ========================================================

    section(
        "CURRENT FBS RATING DISTRIBUTION"
    )

    if not fbs_latest.empty:

        print(
            "OFFENCE"
        )

        print(
            f"  High:   "
            f"{fbs_latest['offense_rating'].max():.1f}"
        )

        print(
            f"  Median: "
            f"{fbs_latest['offense_rating'].median():.1f}"
        )

        print(
            f"  Low:    "
            f"{fbs_latest['offense_rating'].min():.1f}"
        )

        print()

        print(
            "DEFENCE"
        )

        print(
            f"  High:   "
            f"{fbs_latest['defense_rating'].max():.1f}"
        )

        print(
            f"  Median: "
            f"{fbs_latest['defense_rating'].median():.1f}"
        )

        print(
            f"  Low:    "
            f"{fbs_latest['defense_rating'].min():.1f}"
        )

    # ========================================================
    # TOP OFFENCES
    # ========================================================

    section(
        "TOP 15 RAW FBS OFFENCES"
    )

    top_offenses = (
        fbs_latest
        .sort_values(
            "offense_rating",
            ascending=False,
        )
        .head(
            15
        )
    )

    for index, row in enumerate(
        top_offenses.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{index:>2}. "
            f"{row.team:<28} "
            f"{row.offense_rating:>6.1f}"
        )

    # ========================================================
    # TOP DEFENCES
    # ========================================================

    section(
        "TOP 15 RAW FBS DEFENCES"
    )

    top_defenses = (
        fbs_latest
        .sort_values(
            "defense_rating",
            ascending=False,
        )
        .head(
            15
        )
    )

    for index, row in enumerate(
        top_defenses.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{index:>2}. "
            f"{row.team:<28} "
            f"{row.defense_rating:>6.1f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING RAW RATING OUTPUT"
    )

    save_csv(
        rating_history,
        RATING_HISTORY_FILE,
    )

    save_csv(
        latest,
        LATEST_RATINGS_FILE,
    )

    save_csv(
        fbs_latest,
        FBS_LATEST_RATINGS_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "RAW TEAM RATINGS COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {RATING_HISTORY_FILE}"
    )

    print(
        f"  {LATEST_RATINGS_FILE}"
    )

    print(
        f"  {FBS_LATEST_RATINGS_FILE}"
    )

    print()

    print(
        "Important:"
    )

    print(
        "  ✓ Ratings are chronological"
    )

    print(
        "  ✓ Current game cannot affect its "
        "own pregame rating"
    )

    print(
        "  ✓ FBS-vs-FBS included"
    )

    print(
        "  ✓ FBS-vs-FCS included"
    )

    print(
        "  ✓ Lower-division-only games excluded"
    )

    print(
        "  ✓ Offence and defence remain separate"
    )

    print(
        "  ✓ No opponent adjustment applied yet"
    )

    print()

    print(
        "Next layer:"
    )

    print(
        "  RAW RATINGS"
    )

    print(
        "       ↓"
    )

    print(
        "  OPPONENT ADJUSTMENT"
    )

    print()


if __name__ == "__main__":
    main()