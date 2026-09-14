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
# FILES
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SOURCE_AUDIT_FILE = (
    PROCESSED_DIR
    / "2026_source_audit.csv"
)

SOURCE_TEAM_STATS_FILE = (
    PROCESSED_DIR
    / "2026_team_game_stats.csv"
)

HISTORICAL_GAMES_FILE = (
    PROCESSED_DIR
    / "historical_games.csv"
)

HISTORICAL_TEAM_GAMES_FILE = (
    PROCESSED_DIR
    / "historical_team_games.csv"
)


OUTPUT_GAMES_FILE = (
    PROCESSED_DIR
    / "2026_forward_games.csv"
)

OUTPUT_TEAM_GAMES_FILE = (
    PROCESSED_DIR
    / "2026_forward_team_games.csv"
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


def normalise_bool(
    value,
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
        str(
            value
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def numeric(
    value,
) -> float | None:

    try:

        if pd.isna(
            value
        ):
            return None

        return float(
            value
        )

    except (
        ValueError,
        TypeError,
    ):

        return None


def integer(
    value,
) -> int | None:

    number = numeric(
        value
    )

    if number is None:
        return None

    return int(
        number
    )


def possession_to_minutes(
    value,
) -> float | None:

    if value is None:
        return None

    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    if ":" in text:

        pieces = text.split(
            ":"
        )

        if len(
            pieces
        ) == 2:

            try:

                minutes = float(
                    pieces[0]
                )

                seconds = float(
                    pieces[1]
                )

                return (
                    minutes
                    +
                    seconds
                    /
                    60.0
                )

            except ValueError:
                return None

    try:

        return float(
            text
        )

    except ValueError:
        return None


def put_if_exists(
    target: dict,
    columns: set[str],
    aliases: list[str],
    value,
) -> None:

    for alias in aliases:

        if alias in columns:

            target[
                alias
            ] = value


def make_blank_row(
    columns: list[str],
) -> dict:

    return {
        column: np.nan
        for column in columns
    }


# ============================================================
# GAME ADAPTER
# ============================================================

def build_game_rows(
    audit: pd.DataFrame,
    historical_template: pd.DataFrame,
) -> pd.DataFrame:

    template_columns = list(
        historical_template.columns
    )

    column_set = set(
        template_columns
    )

    output_rows = []

    for source in audit.itertuples(
        index=False
    ):

        row = make_blank_row(
            template_columns
        )

        db_id = integer(
            getattr(
                source,
                "id",
                None,
            )
        )

        cfbd_id = integer(
            getattr(
                source,
                "cfbd_id",
                None,
            )
        )

        put_if_exists(
            row,
            column_set,
            [
                "game_db_id",
                "db_game_id",
                "internal_game_id",
            ],
            db_id,
        )

        put_if_exists(
            row,
            column_set,
            [
                "game_id",
                "cfbd_game_id",
                "cfbd_id",
            ],
            cfbd_id,
        )

        put_if_exists(
            row,
            column_set,
            [
                "season",
                "year",
            ],
            2026,
        )

        put_if_exists(
            row,
            column_set,
            [
                "week",
            ],
            integer(
                getattr(
                    source,
                    "week",
                    None,
                )
            ),
        )

        put_if_exists(
            row,
            column_set,
            [
                "season_type",
            ],
            getattr(
                source,
                "season_type",
                "regular",
            ),
        )

        start_date = getattr(
            source,
            "start_date",
            None,
        )

        put_if_exists(
            row,
            column_set,
            [
                "start_date",
                "start_datetime",
                "game_date",
            ],
            start_date,
        )

        home_team = getattr(
            source,
            "home_team",
            None,
        )

        away_team = getattr(
            source,
            "away_team",
            None,
        )

        put_if_exists(
            row,
            column_set,
            [
                "home_team",
            ],
            home_team,
        )

        put_if_exists(
            row,
            column_set,
            [
                "away_team",
            ],
            away_team,
        )

        home_points = numeric(
            getattr(
                source,
                "home_points",
                None,
            )
        )

        away_points = numeric(
            getattr(
                source,
                "away_points",
                None,
            )
        )

        put_if_exists(
            row,
            column_set,
            [
                "home_points",
                "home_score",
            ],
            home_points,
        )

        put_if_exists(
            row,
            column_set,
            [
                "away_points",
                "away_score",
            ],
            away_points,
        )

        neutral = normalise_bool(
            getattr(
                source,
                "neutral_site_bool",
                False,
            )
        )

        completed = normalise_bool(
            getattr(
                source,
                "usable_completed_result",
                False,
            )
        )

        put_if_exists(
            row,
            column_set,
            [
                "neutral_site",
            ],
            neutral,
        )

        put_if_exists(
            row,
            column_set,
            [
                "completed",
            ],
            completed,
        )

        home_classification = getattr(
            source,
            "home_classification",
            None,
        )

        away_classification = getattr(
            source,
            "away_classification",
            None,
        )

        put_if_exists(
            row,
            column_set,
            [
                "home_classification",
            ],
            home_classification,
        )

        put_if_exists(
            row,
            column_set,
            [
                "away_classification",
            ],
            away_classification,
        )

        game_type = str(
            getattr(
                source,
                "game_type",
                "",
            )
        )

        fbs_vs_fbs = (
            game_type
            ==
            "FBS vs FBS"
        )

        fbs_vs_fcs = (
            game_type
            ==
            "FBS vs FCS"
        )

        put_if_exists(
            row,
            column_set,
            [
                "fbs_vs_fbs",
            ],
            fbs_vs_fbs,
        )

        put_if_exists(
            row,
            column_set,
            [
                "fbs_vs_fcs",
            ],
            fbs_vs_fcs,
        )

        put_if_exists(
            row,
            column_set,
            [
                "model_eligible",
            ],
            True,
        )

        if (
            completed
            and
            home_points is not None
            and
            away_points is not None
        ):

            margin = (
                home_points
                -
                away_points
            )

            total = (
                home_points
                +
                away_points
            )

            if margin > 0:

                home_result = 1.0

            elif margin < 0:

                home_result = 0.0

            else:

                home_result = 0.5

            put_if_exists(
                row,
                column_set,
                [
                    "home_margin",
                    "actual_home_margin",
                ],
                margin,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "total_points",
                    "actual_total_points",
                ],
                total,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "actual_home_result",
                    "home_result",
                ],
                home_result,
            )

        output_rows.append(
            row
        )

    return pd.DataFrame(
        output_rows,
        columns=template_columns,
    )


# ============================================================
# TEAM-GAME ADAPTER
# ============================================================

def build_team_game_rows(
    stats: pd.DataFrame,
    audit: pd.DataFrame,
    historical_template: pd.DataFrame,
) -> pd.DataFrame:

    template_columns = list(
        historical_template.columns
    )

    column_set = set(
        template_columns
    )

    game_lookup = (
        audit
        .set_index(
            "cfbd_id",
            drop=False,
        )
    )

    output_rows = []

    for (
        cfbd_game_id,
        game_stats,
    ) in stats.groupby(
        "game_id",
        sort=False,
    ):

        if cfbd_game_id not in game_lookup.index:

            raise RuntimeError(
                "2026 team stats contain a game that "
                "does not exist in 2026_source_audit.csv:\n"
                f"{cfbd_game_id}"
            )

        game = game_lookup.loc[
            cfbd_game_id
        ]

        if isinstance(
            game,
            pd.DataFrame,
        ):

            game = game.iloc[
                0
            ]

        if len(
            game_stats
        ) != 2:

            raise RuntimeError(
                f"Game {cfbd_game_id} does not have "
                "exactly two team-stat rows."
            )

        game_stats = (
            game_stats
            .reset_index(
                drop=True
            )
        )

        for index in range(
            2
        ):

            team = game_stats.iloc[
                index
            ]

            opponent = game_stats.iloc[
                1 - index
            ]

            row = make_blank_row(
                template_columns
            )

            game_db_id = integer(
                team.get(
                    "game_db_id"
                )
            )

            cfbd_id = integer(
                team.get(
                    "game_id"
                )
            )

            put_if_exists(
                row,
                column_set,
                [
                    "game_id",
                    "game_db_id",
                    "internal_game_id",
                ],
                game_db_id,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "cfbd_game_id",
                    "cfbd_id",
                ],
                cfbd_id,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "team_id",
                    "cfbd_team_id",
                ],
                integer(
                    team.get(
                        "team_id"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "season",
                    "year",
                ],
                2026,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "week",
                ],
                integer(
                    team.get(
                        "week"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "season_type",
                ],
                team.get(
                    "season_type"
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "start_date",
                    "game_date",
                ],
                team.get(
                    "start_date"
                ),
            )

            team_name = team.get(
                "team_name"
            )

            opponent_name = opponent.get(
                "team_name"
            )

            put_if_exists(
                row,
                column_set,
                [
                    "team",
                    "team_name",
                    "school",
                ],
                team_name,
            )

            # This is only written when the historical schema
            # already contains such a field.
            #
            # The current historical 88-column schema does not,
            # so downstream engines derive opponent from the
            # paired team row exactly as they do historically.
            put_if_exists(
                row,
                column_set,
                [
                    "opponent",
                    "opponent_name",
                ],
                opponent_name,
            )

            home_away = str(
                team.get(
                    "home_away",
                    "",
                )
            ).lower()

            put_if_exists(
                row,
                column_set,
                [
                    "home_away",
                ],
                home_away,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "home_team",
                ],
                game.get(
                    "home_team"
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "away_team",
                ],
                game.get(
                    "away_team"
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "neutral_site",
                ],
                normalise_bool(
                    game.get(
                        "neutral_site_bool"
                    )
                ),
            )

            team_points = numeric(
                team.get(
                    "points"
                )
            )

            opponent_points = numeric(
                opponent.get(
                    "points"
                )
            )

            put_if_exists(
                row,
                column_set,
                [
                    "points",
                    "team_points",
                ],
                team_points,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_points",
                    "points_allowed",
                ],
                opponent_points,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "rushing_attempts",
                ],
                numeric(
                    team.get(
                        "rushing_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "rushing_yards",
                ],
                numeric(
                    team.get(
                        "rushing_yards"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "rushing_tds",
                ],
                numeric(
                    team.get(
                        "rushing_tds"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "passing_completions",
                ],
                numeric(
                    team.get(
                        "passing_completions"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "passing_attempts",
                ],
                numeric(
                    team.get(
                        "passing_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "net_passing_yards",
                    "passing_yards",
                ],
                numeric(
                    team.get(
                        "net_passing_yards"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "passing_tds",
                ],
                numeric(
                    team.get(
                        "passing_tds"
                    )
                ),
            )

            total_yards = numeric(
                team.get(
                    "total_yards"
                )
            )

            total_plays = numeric(
                team.get(
                    "total_plays_proxy"
                )
            )

            yards_per_play = numeric(
                team.get(
                    "yards_per_play_proxy"
                )
            )

            put_if_exists(
                row,
                column_set,
                [
                    "total_yards",
                ],
                total_yards,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "total_plays",
                    "total_plays_proxy",
                    "plays",
                ],
                total_plays,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "yards_per_play",
                    "yards_per_play_proxy",
                ],
                yards_per_play,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "first_downs",
                ],
                numeric(
                    team.get(
                        "first_downs"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "third_down_made",
                ],
                numeric(
                    team.get(
                        "third_down_made"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "third_down_attempts",
                ],
                numeric(
                    team.get(
                        "third_down_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "fourth_down_made",
                ],
                numeric(
                    team.get(
                        "fourth_down_made"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "fourth_down_attempts",
                ],
                numeric(
                    team.get(
                        "fourth_down_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "turnovers",
                ],
                numeric(
                    team.get(
                        "turnovers"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "interceptions",
                ],
                numeric(
                    team.get(
                        "interceptions"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "fumbles_lost",
                ],
                numeric(
                    team.get(
                        "fumbles_lost"
                    )
                ),
            )

            possession_minutes = (
                possession_to_minutes(
                    team.get(
                        "possession_time"
                    )
                )
            )

            put_if_exists(
                row,
                column_set,
                [
                    "possession_minutes",
                ],
                possession_minutes,
            )

            put_if_exists(
                row,
                column_set,
                [
                    "possession_time",
                ],
                team.get(
                    "possession_time"
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_total_yards",
                ],
                numeric(
                    opponent.get(
                        "total_yards"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_first_downs",
                ],
                numeric(
                    opponent.get(
                        "first_downs"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_turnovers",
                ],
                numeric(
                    opponent.get(
                        "turnovers"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_rushing_attempts",
                ],
                numeric(
                    opponent.get(
                        "rushing_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_passing_attempts",
                ],
                numeric(
                    opponent.get(
                        "passing_attempts"
                    )
                ),
            )

            put_if_exists(
                row,
                column_set,
                [
                    "opponent_yards_per_play_proxy",
                    "opponent_yards_per_play",
                ],
                numeric(
                    opponent.get(
                        "yards_per_play_proxy"
                    )
                ),
            )

            if (
                team_points is not None
                and
                opponent_points is not None
            ):

                point_margin = (
                    team_points
                    -
                    opponent_points
                )

                put_if_exists(
                    row,
                    column_set,
                    [
                        "point_margin",
                        "team_margin",
                    ],
                    point_margin,
                )

                put_if_exists(
                    row,
                    column_set,
                    [
                        "team_win",
                        "win",
                    ],
                    (
                        1.0
                        if point_margin > 0
                        else
                        0.0
                        if point_margin < 0
                        else
                        0.5
                    ),
                )

            output_rows.append(
                row
            )

    return pd.DataFrame(
        output_rows,
        columns=template_columns,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_forward_data(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:

    section(
        "FORWARD DATA VALIDATION"
    )

    print(
        f"2026 eligible schedule games: "
        f"{len(games):,}"
    )

    print(
        f"2026 completed team-game rows: "
        f"{len(team_games):,}"
    )

    # ========================================================
    # COMPLETED GAME COUNT
    # ========================================================

    completed_games = 0

    if "completed" in games.columns:

        completed_games = int(
            games[
                "completed"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
            .sum()
        )

    else:

        score_columns = [
            column
            for column in [
                "home_points",
                "away_points",
            ]
            if column in games.columns
        ]

        if len(
            score_columns
        ) == 2:

            completed_games = int(
                games[
                    score_columns
                ]
                .notna()
                .all(
                    axis=1
                )
                .sum()
            )

    print(
        f"Completed eligible schedule games: "
        f"{completed_games:,}"
    )

    # ========================================================
    # IDENTIFY TEAM-GAME KEY
    # ========================================================

    game_id_column = None

    for candidate in [
        "game_id",
        "game_db_id",
        "internal_game_id",
    ]:

        if candidate in team_games.columns:

            if team_games[
                candidate
            ].notna().any():

                game_id_column = candidate
                break

    if game_id_column is None:

        raise RuntimeError(
            "Could not locate team-game identifier "
            "in adapted 2026 team-game data."
        )

    pair_counts = (
        team_games
        .dropna(
            subset=[
                game_id_column
            ]
        )
        .groupby(
            game_id_column
        )
        .size()
    )

    exactly_two = int(
        (
            pair_counts
            ==
            2
        ).sum()
    )

    invalid_pairs = int(
        (
            pair_counts
            !=
            2
        ).sum()
    )

    print(
        f"Games with exactly two team rows: "
        f"{exactly_two:,}"
    )

    print(
        f"Games with invalid team-row count: "
        f"{invalid_pairs:,}"
    )

    # ========================================================
    # TEAM NAME COLUMN
    # ========================================================

    team_name_column = None

    for candidate in [
        "team_name",
        "team",
        "school",
    ]:

        if (
            candidate
            in team_games.columns
            and
            team_games[
                candidate
            ].notna().any()
        ):

            team_name_column = candidate
            break

    if team_name_column is None:

        raise RuntimeError(
            "No usable team-name column exists."
        )

    # ========================================================
    # OPPONENT VALIDATION
    #
    # Historical schema does not necessarily contain an
    # opponent-name column.
    #
    # When absent, opponent is derived from the second team
    # row belonging to the same game.
    # ========================================================

    opponent_column = None

    for candidate in [
        "opponent_name",
        "opponent",
    ]:

        if (
            candidate
            in team_games.columns
            and
            team_games[
                candidate
            ].notna().any()
        ):

            opponent_column = candidate
            break

    if opponent_column is not None:

        opponent_coverage = (
            team_games[
                opponent_column
            ]
            .notna()
            .mean()
        )

        opponent_ready = (
            opponent_coverage
            ==
            1.0
        )

        opponent_description = (
            f"{opponent_coverage:.2%} "
            f"({opponent_column})"
        )

    else:

        # ----------------------------------------------------
        # DERIVE / VERIFY OPPONENT FROM PAIRED ROW
        # ----------------------------------------------------

        pair_ready = True

        derived_rows = 0

        for (
            _,
            group,
        ) in team_games.groupby(
            game_id_column
        ):

            if len(
                group
            ) != 2:

                pair_ready = False
                continue

            names = (
                group[
                    team_name_column
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

            if (
                len(names)
                !=
                2
            ):

                pair_ready = False
                continue

            if names[0] == names[1]:

                pair_ready = False
                continue

            derived_rows += 2

        opponent_ready = (
            pair_ready
            and
            derived_rows
            ==
            len(team_games)
        )

        opponent_description = (
            "100.00% "
            "(derived from paired team row)"
            if opponent_ready
            else
            "FAILED paired-row derivation"
        )

    # ========================================================
    # STATE FIELDS
    # ========================================================

    section(
        "STATE-ENGINE FIELD COVERAGE"
    )

    field_groups = {
        "team": [
            "team_name",
            "team",
        ],

        "team points": [
            "team_points",
            "points",
        ],

        "opponent points": [
            "opponent_points",
            "points_allowed",
        ],

        "rushing attempts": [
            "rushing_attempts",
        ],

        "rushing yards": [
            "rushing_yards",
        ],

        "passing attempts": [
            "passing_attempts",
        ],

        "passing yards": [
            "net_passing_yards",
            "passing_yards",
        ],

        "total yards": [
            "total_yards",
        ],

        "first downs": [
            "first_downs",
        ],

        "turnovers": [
            "turnovers",
        ],

        "possession minutes": [
            "possession_minutes",
        ],

        "yards/play": [
            "yards_per_play_proxy",
            "yards_per_play",
        ],
    }

    all_ready = True

    for label, candidates in (
        field_groups.items()
    ):

        available = [
            column
            for column in candidates
            if (
                column
                in team_games.columns
                and
                team_games[
                    column
                ].notna().any()
            )
        ]

        if not available:

            print(
                f"{label:<24} "
                "MISSING COLUMN"
            )

            all_ready = False
            continue

        best_column = max(
            available,
            key=lambda column: (
                team_games[
                    column
                ]
                .notna()
                .sum()
            ),
        )

        coverage = (
            team_games[
                best_column
            ]
            .notna()
            .mean()
        )

        print(
            f"{label:<24} "
            f"{coverage:>7.2%} "
            f"({best_column})"
        )

        if coverage < 1.0:

            all_ready = False

    print(
        f"{'opponent':<24} "
        f"{opponent_description}"
    )

    if not opponent_ready:

        all_ready = False

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "FORWARD DATA READINESS"
    )

    expected_team_rows = completed_games * 2

    if (
        len(team_games)
        ==
        expected_team_rows
        and
        exactly_two
        ==
        completed_games
        and
        invalid_pairs
        ==
        0
        and
        all_ready
    ):

        print(
            "✓ 2026 forward data is READY."
        )

        print()

        print(
            "The live 2026 rows match the historical "
            "engine schema."
        )

        print()

        print(
            "Opponent identity is safely derived from the "
            "paired team row, exactly as in the historical "
            "ratings workflow."
        )

        print()

        print(
            "Next:"
        )

        print(
            "  Initialise frozen engines through 2025"
        )

        print(
            "       ↓"
        )

        print(
            "  Process 2026 games chronologically"
        )

        print(
            "       ↓"
        )

        print(
            "  Create PREGAME features"
        )

        print(
            "       ↓"
        )

        print(
            "  Run frozen CFB-V2-2026.1 probability model"
        )

        print(
            "       ↓"
        )

        print(
            "  Apply results only AFTER predictions"
        )

    else:

        print(
            "⚠ Forward data needs review before "
            "state engines are advanced."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— BUILD 2026 FORWARD DATA"
    )

    required_files = [
        SOURCE_AUDIT_FILE,
        SOURCE_TEAM_STATS_FILE,
        HISTORICAL_GAMES_FILE,
        HISTORICAL_TEAM_GAMES_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing:\n{path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING SOURCE DATA"
    )

    audit = pd.read_csv(
        SOURCE_AUDIT_FILE,
        low_memory=False,
    )

    stats = pd.read_csv(
        SOURCE_TEAM_STATS_FILE,
        low_memory=False,
    )

    historical_games = pd.read_csv(
        HISTORICAL_GAMES_FILE,
        low_memory=False,
        nrows=10,
    )

    historical_team_games = (
        pd.read_csv(
            HISTORICAL_TEAM_GAMES_FILE,
            low_memory=False,
            nrows=10,
        )
    )

    print(
        f"2026 audit rows:            "
        f"{len(audit):,}"
    )

    print(
        f"2026 fetched team rows:     "
        f"{len(stats):,}"
    )

    print(
        f"Historical game columns:    "
        f"{len(historical_games.columns):,}"
    )

    print(
        f"Historical team columns:    "
        f"{len(historical_team_games.columns):,}"
    )

    # ========================================================
    # MODEL UNIVERSE
    # ========================================================

    section(
        "FILTERING 2026 MODEL UNIVERSE"
    )

    if (
        "model_eligible"
        not in audit.columns
    ):

        raise RuntimeError(
            "2026_source_audit.csv is missing "
            "model_eligible."
        )

    model_flag = (
        audit[
            "model_eligible"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    eligible_audit = (
        audit[
            model_flag
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Eligible 2026 games: "
        f"{len(eligible_audit):,}"
    )

    # ========================================================
    # GAME DATA
    # ========================================================

    section(
        "ADAPTING 2026 GAME SCHEDULE"
    )

    forward_games = (
        build_game_rows(
            audit=eligible_audit,
            historical_template=historical_games,
        )
    )

    print(
        f"Forward game rows created: "
        f"{len(forward_games):,}"
    )

    print(
        f"Forward game columns:      "
        f"{len(forward_games.columns):,}"
    )

    # ========================================================
    # TEAM DATA
    # ========================================================

    section(
        "ADAPTING 2026 COMPLETED TEAM STATS"
    )

    forward_team_games = (
        build_team_game_rows(
            stats=stats,
            audit=eligible_audit,
            historical_template=(
                historical_team_games
            ),
        )
    )

    print(
        f"Forward team-game rows created: "
        f"{len(forward_team_games):,}"
    )

    print(
        f"Forward team-game columns:      "
        f"{len(forward_team_games.columns):,}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_forward_data(
        games=forward_games,
        team_games=forward_team_games,
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING FORWARD DATA"
    )

    forward_games.to_csv(
        OUTPUT_GAMES_FILE,
        index=False,
    )

    forward_team_games.to_csv(
        OUTPUT_TEAM_GAMES_FILE,
        index=False,
    )

    print(
        f"✓ Saved {len(forward_games):,} games → "
        f"{OUTPUT_GAMES_FILE}"
    )

    print(
        f"✓ Saved {len(forward_team_games):,} team rows → "
        f"{OUTPUT_TEAM_GAMES_FILE}"
    )

    section(
        "2026 FORWARD DATA BUILD COMPLETE"
    )


if __name__ == "__main__":

    main()