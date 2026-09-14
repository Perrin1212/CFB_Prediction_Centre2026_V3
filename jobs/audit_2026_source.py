from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

DEFAULT_V1_DB = (
    PROJECT_ROOT.parent
    / "CFB_Prediction_Centre2026"
    / "data"
    / "cfb_prediction.db"
)

V1_DB_PATH = Path(
    os.getenv(
        "CFB_V1_DB_PATH",
        str(DEFAULT_V1_DB),
    )
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "2026_source_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def normalise_classification(value) -> str:
    if pd.isna(value):
        return "missing"

    text = str(value).strip().lower()

    aliases = {
        "fbs": "fbs",
        "fcs": "fcs",

        "i-a": "fbs",
        "ia": "fbs",
        "division i-a": "fbs",

        "i-aa": "fcs",
        "iaa": "fcs",
        "division i-aa": "fcs",

        "ii": "ii",
        "division ii": "ii",

        "iii": "iii",
        "division iii": "iii",
    }

    return aliases.get(text, text)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


def safe_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"

    return f"{numerator / denominator:.2%}"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— 2026 SOURCE DATA AUDIT"
    )

    # ========================================================
    # DATABASE CHECK
    # ========================================================

    print(f"V1 database:")
    print(f"  {V1_DB_PATH}")

    if not V1_DB_PATH.exists():
        raise FileNotFoundError(
            "\nV1 database was not found:\n"
            f"{V1_DB_PATH}\n\n"
            "Expected the original V1 project to remain beside "
            "the V2 project."
        )

    # ========================================================
    # CONNECT READ-ONLY
    # ========================================================

    uri = (
        f"file:{V1_DB_PATH.as_posix()}"
        "?mode=ro"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
    )

    try:

        # ====================================================
        # TABLE CHECK
        # ====================================================

        section("DATABASE TABLES")

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            connection,
        )

        available_tables = set(
            tables["name"].astype(str)
        )

        required_tables = {
            "teams",
            "games",
            "game_team_stats",
        }

        missing_tables = (
            required_tables
            -
            available_tables
        )

        print(
            "Required tables found: "
            f"{len(required_tables) - len(missing_tables)}"
            f"/{len(required_tables)}"
        )

        if missing_tables:
            raise RuntimeError(
                "Missing required V1 tables: "
                f"{sorted(missing_tables)}"
            )

        print("✓ teams")
        print("✓ games")
        print("✓ game_team_stats")

        # ====================================================
        # LOAD TEAMS
        # ====================================================

        section("LOADING TEAM CLASSIFICATIONS")

        teams = pd.read_sql_query(
            """
            SELECT
                id,
                cfbd_id,
                school,
                abbreviation,
                conference,
                classification
            FROM teams
            """,
            connection,
        )

        teams["classification_normalised"] = (
            teams["classification"]
            .apply(normalise_classification)
        )

        print(f"Teams loaded: {len(teams):,}")

        print()
        print("Classification counts:")

        print(
            teams[
                "classification_normalised"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        # ====================================================
        # LOAD 2026 GAMES
        # ====================================================

        section("LOADING 2026 GAMES")

        games = pd.read_sql_query(
            """
            SELECT
                id,
                cfbd_id,
                season,
                week,
                season_type,
                start_date,
                home_team,
                away_team,
                home_points,
                away_points,
                completed,
                neutral_site,
                conference_game,
                venue,
                attendance
            FROM games
            WHERE season = 2026
            ORDER BY
                start_date,
                id
            """,
            connection,
        )

        print(
            f"2026 games loaded: "
            f"{len(games):,}"
        )

        if games.empty:
            raise RuntimeError(
                "No 2026 games currently exist in the V1 database."
            )

        # ====================================================
        # CLEAN GAME FIELDS
        # ====================================================

        games["completed_bool"] = bool_series(
            games["completed"]
        )

        games["neutral_site_bool"] = bool_series(
            games["neutral_site"]
        )

        games["home_points"] = pd.to_numeric(
            games["home_points"],
            errors="coerce",
        )

        games["away_points"] = pd.to_numeric(
            games["away_points"],
            errors="coerce",
        )

        games["week"] = pd.to_numeric(
            games["week"],
            errors="coerce",
        )

        games["start_date_parsed"] = pd.to_datetime(
            games["start_date"],
            errors="coerce",
            utc=True,
        )

        # ====================================================
        # JOIN CLASSIFICATIONS
        # ====================================================

        team_classification = (
            teams[
                [
                    "school",
                    "classification_normalised",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "school",
                ]
            )
        )

        home_lookup = (
            team_classification
            .rename(
                columns={
                    "school": "home_team",
                    "classification_normalised":
                        "home_classification",
                }
            )
        )

        away_lookup = (
            team_classification
            .rename(
                columns={
                    "school": "away_team",
                    "classification_normalised":
                        "away_classification",
                }
            )
        )

        games = (
            games
            .merge(
                home_lookup,
                on="home_team",
                how="left",
            )
            .merge(
                away_lookup,
                on="away_team",
                how="left",
            )
        )

        games[
            "home_classification"
        ] = (
            games[
                "home_classification"
            ]
            .fillna("missing")
        )

        games[
            "away_classification"
        ] = (
            games[
                "away_classification"
            ]
            .fillna("missing")
        )

        # ====================================================
        # MODEL UNIVERSE
        # ====================================================

        home_fbs = (
            games["home_classification"]
            ==
            "fbs"
        )

        away_fbs = (
            games["away_classification"]
            ==
            "fbs"
        )

        home_fcs = (
            games["home_classification"]
            ==
            "fcs"
        )

        away_fcs = (
            games["away_classification"]
            ==
            "fcs"
        )

        games["fbs_vs_fbs"] = (
            home_fbs
            &
            away_fbs
        )

        games["fbs_vs_fcs"] = (
            (
                home_fbs
                &
                away_fcs
            )
            |
            (
                home_fcs
                &
                away_fbs
            )
        )

        games["model_eligible"] = (
            games["fbs_vs_fbs"]
            |
            games["fbs_vs_fcs"]
        )

        games["game_type"] = np.select(
            [
                games["fbs_vs_fbs"],
                games["fbs_vs_fcs"],
            ],
            [
                "FBS vs FBS",
                "FBS vs FCS",
            ],
            default="Excluded",
        )

        # ====================================================
        # LOAD 2026 TEAM STATS
        # ====================================================

        section("LOADING 2026 BOX-SCORE COVERAGE")

        stats = pd.read_sql_query(
            """
            SELECT
                game_id,
                team_id,
                home_away,
                points,
                rushing_attempts,
                rushing_yards,
                passing_attempts,
                net_passing_yards,
                total_yards,
                first_downs,
                third_down_attempts,
                turnovers,
                possession_time
            FROM game_team_stats
            WHERE season = 2026
            """,
            connection,
        )

        print(
            f"2026 team-stat rows: "
            f"{len(stats):,}"
        )

        # ====================================================
        # STAT COUNTS BY GAME
        # ====================================================

        if not stats.empty:

            stats_per_game = (
                stats
                .groupby(
                    "game_id"
                )
                .size()
                .rename(
                    "team_stat_rows"
                )
                .reset_index()
            )

            games = games.merge(
                stats_per_game,
                left_on="id",
                right_on="game_id",
                how="left",
            )

            games[
                "team_stat_rows"
            ] = (
                games[
                    "team_stat_rows"
                ]
                .fillna(0)
                .astype(int)
            )

        else:

            games[
                "team_stat_rows"
            ] = 0

        games[
            "complete_box_score"
        ] = (
            games[
                "team_stat_rows"
            ]
            ==
            2
        )

        # ====================================================
        # COMPLETION CHECK
        # ====================================================

        games[
            "has_final_score"
        ] = (
            games[
                "home_points"
            ].notna()
            &
            games[
                "away_points"
            ].notna()
        )

        games[
            "usable_completed_result"
        ] = (
            games[
                "completed_bool"
            ]
            &
            games[
                "has_final_score"
            ]
        )

        games[
            "usable_completed_model_game"
        ] = (
            games[
                "model_eligible"
            ]
            &
            games[
                "usable_completed_result"
            ]
        )

        games[
            "usable_state_update_game"
        ] = (
            games[
                "usable_completed_model_game"
            ]
            &
            games[
                "complete_box_score"
            ]
        )

        games[
            "future_or_uncompleted_model_game"
        ] = (
            games[
                "model_eligible"
            ]
            &
            ~games[
                "usable_completed_result"
            ]
        )

        # ====================================================
        # OVERALL SUMMARY
        # ====================================================

        section("2026 GAME UNIVERSE")

        total_games = len(games)

        fbs_vs_fbs_count = int(
            games[
                "fbs_vs_fbs"
            ].sum()
        )

        fbs_vs_fcs_count = int(
            games[
                "fbs_vs_fcs"
            ].sum()
        )

        eligible_count = int(
            games[
                "model_eligible"
            ].sum()
        )

        excluded_count = (
            total_games
            -
            eligible_count
        )

        print(
            f"All 2026 games:             "
            f"{total_games:,}"
        )

        print(
            f"FBS vs FBS:                 "
            f"{fbs_vs_fbs_count:,}"
        )

        print(
            f"FBS vs FCS:                 "
            f"{fbs_vs_fcs_count:,}"
        )

        print(
            f"Total model-eligible games: "
            f"{eligible_count:,}"
        )

        print(
            f"Excluded games:             "
            f"{excluded_count:,}"
        )

        # ====================================================
        # RESULTS
        # ====================================================

        section("2026 RESULT AVAILABILITY")

        completed_all = int(
            games[
                "usable_completed_result"
            ].sum()
        )

        completed_model = int(
            games[
                "usable_completed_model_game"
            ].sum()
        )

        upcoming_model = int(
            games[
                "future_or_uncompleted_model_game"
            ].sum()
        )

        print(
            f"Games with completed result:       "
            f"{completed_all:,}"
        )

        print(
            f"Completed model-eligible games:    "
            f"{completed_model:,}"
        )

        print(
            f"Upcoming/uncompleted model games:  "
            f"{upcoming_model:,}"
        )

        # ====================================================
        # BOX SCORE COVERAGE
        # ====================================================

        section("2026 MODEL STATE-UPDATE COVERAGE")

        model_games = (
            games[
                games[
                    "model_eligible"
                ]
            ]
            .copy()
        )

        completed_model_games = (
            model_games[
                model_games[
                    "usable_completed_result"
                ]
            ]
            .copy()
        )

        complete_box = int(
            completed_model_games[
                "complete_box_score"
            ].sum()
        )

        incomplete_box = (
            len(
                completed_model_games
            )
            -
            complete_box
        )

        print(
            f"Completed eligible games:     "
            f"{len(completed_model_games):,}"
        )

        print(
            f"With exactly 2 stat rows:     "
            f"{complete_box:,}"
        )

        print(
            f"Without complete box scores:  "
            f"{incomplete_box:,}"
        )

        print(
            f"State-update coverage:        "
            f"{safe_percent(complete_box, len(completed_model_games))}"
        )

        # ====================================================
        # STAT COMPLETENESS
        # ====================================================

        if not stats.empty:

            section("2026 TEAM-STAT COMPLETENESS")

            eligible_game_ids = set(
                games.loc[
                    games[
                        "model_eligible"
                    ],
                    "id",
                ]
                .astype(int)
                .tolist()
            )

            eligible_stats = (
                stats[
                    stats[
                        "game_id"
                    ]
                    .isin(
                        eligible_game_ids
                    )
                ]
                .copy()
            )

            columns = [
                "points",
                "rushing_attempts",
                "rushing_yards",
                "passing_attempts",
                "net_passing_yards",
                "total_yards",
                "first_downs",
                "third_down_attempts",
                "turnovers",
                "possession_time",
            ]

            for column in columns:

                if column not in eligible_stats.columns:
                    continue

                coverage = (
                    eligible_stats[
                        column
                    ]
                    .notna()
                    .mean()
                )

                print(
                    f"{column:<28} "
                    f"{coverage:>7.2%}"
                )

        # ====================================================
        # WEEK SUMMARY
        # ====================================================

        section("2026 MODEL GAMES BY WEEK")

        week_summary = (
            model_games
            .groupby(
                "week",
                dropna=False,
            )
            .agg(
                games=(
                    "id",
                    "size",
                ),

                completed=(
                    "usable_completed_result",
                    "sum",
                ),

                complete_box_scores=(
                    "complete_box_score",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "week"
            )
        )

        if week_summary.empty:

            print(
                "No model-eligible games found."
            )

        else:

            for row in week_summary.itertuples(
                index=False
            ):

                week_text = (
                    "NA"
                    if pd.isna(row.week)
                    else str(
                        int(row.week)
                    )
                )

                print(
                    f"Week {week_text:<3} | "
                    f"{int(row.games):>3} games | "
                    f"{int(row.completed):>3} completed | "
                    f"{int(row.complete_box_scores):>3} box scores"
                )

        # ====================================================
        # CLASSIFICATION ANOMALIES
        # ====================================================

        section("CLASSIFICATION / UNIVERSE ANOMALIES")

        missing_classification = (
            games[
                (
                    games[
                        "home_classification"
                    ]
                    ==
                    "missing"
                )
                |
                (
                    games[
                        "away_classification"
                    ]
                    ==
                    "missing"
                )
            ]
        )

        print(
            f"Games with missing team classification: "
            f"{len(missing_classification):,}"
        )

        fbs_involved_but_excluded = (
            games[
                (
                    (
                        games[
                            "home_classification"
                        ]
                        ==
                        "fbs"
                    )
                    |
                    (
                        games[
                            "away_classification"
                        ]
                        ==
                        "fbs"
                    )
                )
                &
                ~games[
                    "model_eligible"
                ]
            ]
        )

        print(
            f"FBS-involved games excluded by universe rule: "
            f"{len(fbs_involved_but_excluded):,}"
        )

        if not fbs_involved_but_excluded.empty:

            print()
            print(
                fbs_involved_but_excluded[
                    [
                        "week",
                        "away_team",
                        "away_classification",
                        "home_team",
                        "home_classification",
                    ]
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

        # ====================================================
        # READY / NOT READY
        # ====================================================

        section("FORWARD PIPELINE READINESS")

        problems = []

        if eligible_count == 0:
            problems.append(
                "No model-eligible 2026 games."
            )

        if completed_model > 0 and complete_box == 0:
            problems.append(
                "Completed 2026 model games exist but none "
                "have complete two-team box scores."
            )

        if len(
            fbs_involved_but_excluded
        ) > 0:
            problems.append(
                "Some FBS-involved games have non-FCS/missing "
                "opponent classifications and require review."
            )

        if not problems:

            print(
                "✓ 2026 source data is structurally ready "
                "for the forward pipeline."
            )

            print()
            print(
                "We can now:"
            )

            print(
                "  1. initialise states from 2023-2025"
            )

            print(
                "  2. process completed 2026 games chronologically"
            )

            print(
                "  3. make each prediction BEFORE applying "
                "that game's result"
            )

            print(
                "  4. update states only after completed games"
            )

            print(
                "  5. produce predictions for upcoming games"
            )

        else:

            print(
                "2026 source data needs review before the "
                "forward pipeline:"
            )

            for problem in problems:
                print(
                    f"  - {problem}"
                )

        # ====================================================
        # SAVE
        # ====================================================

        section("SAVING AUDIT")

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_columns = [
            "id",
            "cfbd_id",
            "season",
            "week",
            "season_type",
            "start_date",

            "away_team",
            "away_classification",

            "home_team",
            "home_classification",

            "home_points",
            "away_points",

            "completed_bool",
            "neutral_site_bool",

            "game_type",
            "model_eligible",

            "team_stat_rows",
            "complete_box_score",

            "usable_completed_result",
            "usable_completed_model_game",
            "usable_state_update_game",
            "future_or_uncompleted_model_game",

            "venue",
        ]

        games[
            output_columns
        ].to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            f"✓ Saved {len(games):,} rows → "
            f"{OUTPUT_FILE}"
        )

        section("2026 SOURCE AUDIT COMPLETE")

        print(
            "No model state or prediction was changed."
        )

        print()
        print(
            "Next:"
        )

        print(
            "  Build the frozen chronological 2026 "
            "forward-prediction pipeline."
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()