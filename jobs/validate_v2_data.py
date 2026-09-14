from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

GAMES_FILE = (
    PROCESSED_DIR
    / "historical_games.csv"
)

TEAM_GAMES_FILE = (
    PROCESSED_DIR
    / "historical_team_games.csv"
)

TEAMS_FILE = (
    PROCESSED_DIR
    / "teams.csv"
)


def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def pct(
    numerator: int | float,
    denominator: int | float,
) -> float:

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100
    )


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 — DATA AUDIT"
    )

    # ========================================================
    # LOAD
    # ========================================================

    games = pd.read_csv(
        GAMES_FILE
    )

    team_games = pd.read_csv(
        TEAM_GAMES_FILE
    )

    teams = pd.read_csv(
        TEAMS_FILE
    )

    print(
        f"Games:      {len(games):,}"
    )

    print(
        f"Team-games: {len(team_games):,}"
    )

    print(
        f"Teams:      {len(teams):,}"
    )

    # ========================================================
    # GAME COUNTS BY SEASON
    # ========================================================

    section(
        "GAME COUNTS BY SEASON"
    )

    game_counts = (
        games
        .groupby("season")
        .size()
    )

    for season, count in game_counts.items():

        print(
            f"{int(season)}: {count:,} games"
        )

    # ========================================================
    # TEAM GAME COUNTS
    # ========================================================

    section(
        "TEAM-GAME STAT ROWS BY SEASON"
    )

    team_game_counts = (
        team_games
        .groupby("season")
        .size()
    )

    for season, count in team_game_counts.items():

        print(
            f"{int(season)}: {count:,} team-game rows"
        )

    # ========================================================
    # EXPECTED STAT COVERAGE
    # ========================================================

    section(
        "STAT COVERAGE"
    )

    expected_team_rows = (
        len(games)
        * 2
    )

    actual_team_rows = len(
        team_games
    )

    coverage = pct(
        actual_team_rows,
        expected_team_rows,
    )

    print(
        f"Expected team-game rows if every game had stats: "
        f"{expected_team_rows:,}"
    )

    print(
        f"Actual team-game rows: "
        f"{actual_team_rows:,}"
    )

    print(
        f"Overall detailed-stat coverage: "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # STAT COVERAGE BY SEASON
    # ========================================================

    section(
        "STAT COVERAGE BY SEASON"
    )

    seasons = sorted(
        games["season"]
        .dropna()
        .astype(int)
        .unique()
    )

    for season in seasons:

        season_games = games[
            games["season"] == season
        ]

        season_team_games = team_games[
            team_games["season"] == season
        ]

        expected = (
            len(season_games)
            * 2
        )

        actual = len(
            season_team_games
        )

        print(
            f"{season}: "
            f"{actual:,}/{expected:,} "
            f"team stat rows "
            f"({pct(actual, expected):.2f}%)"
        )

    # ========================================================
    # MATCH TEAM-GAME STATS BACK TO GAMES
    # ========================================================

    section(
        "NUMBER OF STAT ROWS PER GAME"
    )

    if "game_id" not in team_games.columns:

        raise RuntimeError(
            "historical_team_games.csv does not contain game_id"
        )

    stat_rows_per_game = (
        team_games
        .groupby("game_id")
        .size()
    )

    zero_stats = 0

    one_stat = int(
        (stat_rows_per_game == 1).sum()
    )

    two_stats = int(
        (stat_rows_per_game == 2).sum()
    )

    more_than_two = int(
        (stat_rows_per_game > 2).sum()
    )

    games_with_any_stats = (
        stat_rows_per_game.index.nunique()
    )

    zero_stats = (
        len(games)
        -
        games_with_any_stats
    )

    print(
        f"Games with zero team-stat rows: "
        f"{zero_stats:,}"
    )

    print(
        f"Games with one team-stat row: "
        f"{one_stat:,}"
    )

    print(
        f"Games with two team-stat rows: "
        f"{two_stats:,}"
    )

    print(
        f"Games with >2 team-stat rows: "
        f"{more_than_two:,}"
    )

    # ========================================================
    # FBS COVERAGE
    # ========================================================

    section(
        "FBS GAME COVERAGE"
    )

    if "fbs_involved" in games.columns:

        fbs_games = games[
            games["fbs_involved"].fillna(False)
        ]

        print(
            f"Games involving at least one FBS team: "
            f"{len(fbs_games):,}"
        )

    if "fbs_vs_fbs" in games.columns:

        fbs_vs_fbs = games[
            games["fbs_vs_fbs"].fillna(False)
        ]

        print(
            f"FBS vs FBS games: "
            f"{len(fbs_vs_fbs):,}"
        )

    if "fbs_vs_fcs" in games.columns:

        fbs_vs_fcs = games[
            games["fbs_vs_fcs"].fillna(False)
        ]

        print(
            f"FBS vs FCS games: "
            f"{len(fbs_vs_fcs):,}"
        )

    # ========================================================
    # TEAM CLASSIFICATION
    # ========================================================

    section(
        "TEAM CLASSIFICATIONS"
    )

    if "classification" in teams.columns:

        classifications = (
            teams["classification"]
            .fillna("missing")
            .value_counts(
                dropna=False
            )
        )

        print(
            classifications.to_string()
        )

    # ========================================================
    # SCORE VALIDATION
    # ========================================================

    section(
        "SCORE VALIDATION"
    )

    if (
        "home_score_matches_stats"
        in games.columns
    ):

        home_mismatches = (
            ~games[
                "home_score_matches_stats"
            ].fillna(True)
        ).sum()

        print(
            f"Home score mismatches: "
            f"{int(home_mismatches):,}"
        )

    if (
        "away_score_matches_stats"
        in games.columns
    ):

        away_mismatches = (
            ~games[
                "away_score_matches_stats"
            ].fillna(True)
        ).sum()

        print(
            f"Away score mismatches: "
            f"{int(away_mismatches):,}"
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    section(
        "CORE TEAM-GAME STAT COMPLETENESS"
    )

    important_columns = [
        "team_points",
        "opponent_points",
        "rushing_attempts",
        "rushing_yards",
        "passing_attempts",
        "net_passing_yards",
        "total_yards",
        "first_downs",
        "third_down_made",
        "third_down_attempts",
        "turnovers",
        "possession_minutes",
        "yards_per_play_proxy",
    ]

    for column in important_columns:

        if column not in team_games.columns:
            continue

        missing = int(
            team_games[column]
            .isna()
            .sum()
        )

        total = len(
            team_games
        )

        complete = (
            total
            -
            missing
        )

        print(
            f"{column:<28}"
            f"{complete:>7,}/{total:<7,} "
            f"{pct(complete, total):>6.2f}% complete"
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    section(
        "DUPLICATE CHECKS"
    )

    if "cfbd_id" in games.columns:

        duplicate_games = int(
            games["cfbd_id"]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate CFBD game IDs: "
            f"{duplicate_games:,}"
        )

    if "team_game_id" in team_games.columns:

        duplicate_team_games = int(
            team_games[
                "team_game_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate team-game IDs: "
            f"{duplicate_team_games:,}"
        )

    # ========================================================
    # HOME / AWAY CHECK
    # ========================================================

    section(
        "HOME / AWAY STAT ROWS"
    )

    if "home_away" in team_games.columns:

        orientation = (
            team_games[
                "home_away"
            ]
            .fillna("missing")
            .value_counts()
        )

        print(
            orientation.to_string()
        )

    # ========================================================
    # POSSIBLE BAD GAME ROWS
    # ========================================================

    section(
        "RESULT SANITY CHECK"
    )

    negative_scores = (
        (
            games["home_points"] < 0
        )
        |
        (
            games["away_points"] < 0
        )
    ).sum()

    print(
        f"Games with negative scores: "
        f"{int(negative_scores):,}"
    )

    missing_scores = (
        games[
            [
                "home_points",
                "away_points",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Games with missing scores: "
        f"{int(missing_scores):,}"
    )

    ties = (
        games[
            "home_points"
        ]
        ==
        games[
            "away_points"
        ]
    ).sum()

    print(
        f"Games ending level in source data: "
        f"{int(ties):,}"
    )

    # ========================================================
    # BASIC SCORING DISTRIBUTION
    # ========================================================

    section(
        "SCORING BASELINES"
    )

    print(
        f"Average home points: "
        f"{games['home_points'].mean():.2f}"
    )

    print(
        f"Average away points: "
        f"{games['away_points'].mean():.2f}"
    )

    print(
        f"Average total points: "
        f"{games['total_points'].mean():.2f}"
    )

    print(
        f"Average home margin: "
        f"{games['home_margin'].mean():+.2f}"
    )

    home_win_rate = (
        games[
            "home_win"
        ].mean()
        * 100
    )

    print(
        f"Home win rate: "
        f"{home_win_rate:.2f}%"
    )

    # ========================================================
    # ELO READINESS
    # ========================================================

    section(
        "ELO READINESS"
    )

    elo_required = [
        "season",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "neutral_site",
    ]

    missing_elo_columns = [
        column
        for column in elo_required
        if column not in games.columns
    ]

    if missing_elo_columns:

        print(
            "✗ ELO dataset is missing:"
        )

        for column in missing_elo_columns:

            print(
                f"  - {column}"
            )

    else:

        elo_missing_rows = (
            games[
                elo_required
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        print(
            f"Rows available: "
            f"{len(games):,}"
        )

        print(
            f"Rows with missing required ELO data: "
            f"{int(elo_missing_rows):,}"
        )

        if elo_missing_rows == 0:

            print()
            print(
                "✓ GAME RESULTS DATASET IS READY "
                "FOR CHRONOLOGICAL ELO"
            )

        else:

            print()
            print(
                "⚠ Some rows need cleaning before ELO."
            )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "AUDIT COMPLETE"
    )

    print(
        "No files were changed."
    )

    print(
        "This job only analysed the V2 processed datasets."
    )

    print()


if __name__ == "__main__":

    main()