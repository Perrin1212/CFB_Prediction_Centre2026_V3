from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd


class CFBDataLoader:
    """
    Read the existing V1 CFB SQLite database and expose
    clean pandas DataFrames for the V2 modelling pipeline.

    IMPORTANT
    ---------
    V2 reads V1 in read-only mode.

    It does not modify the V1 database.

    Database relationships in V1:

        games.id
            ↓
        game_team_stats.game_id

        teams.id
            ↓
        game_team_stats.team_id

    teams.cfbd_id is the external CFBD team identifier.
    """

    REQUIRED_TABLES = {
        "teams",
        "games",
        "game_team_stats",
    }

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:

        self.database_path = Path(database_path)

    # ========================================================
    # DATABASE CONNECTION
    # ========================================================

    def _validate_database(self) -> None:

        if not self.database_path.exists():

            raise FileNotFoundError(
                "\nV1 database not found:\n"
                f"{self.database_path}\n\n"
                "Check config/settings.py or set "
                "V1_DATABASE_PATH in your .env file."
            )

    def _connect(self) -> sqlite3.Connection:

        self._validate_database()

        uri = (
            f"file:{self.database_path.resolve()}"
            "?mode=ro"
        )

        return sqlite3.connect(
            uri,
            uri=True,
        )

    # ========================================================
    # DATABASE DISCOVERY
    # ========================================================

    def get_tables(self) -> list[str]:

        with self._connect() as connection:

            query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """

            df = pd.read_sql_query(
                query,
                connection,
            )

        return df["name"].tolist()

    def get_table_columns(
        self,
        table_name: str,
    ) -> list[str]:

        with self._connect() as connection:

            rows = connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

        return [
            row[1]
            for row in rows
        ]

    def validate_schema(self) -> None:

        tables = set(
            self.get_tables()
        )

        missing = (
            self.REQUIRED_TABLES
            - tables
        )

        if missing:

            raise RuntimeError(
                "V1 database is missing required tables: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # GENERIC TABLE READER
    # ========================================================

    def read_table(
        self,
        table_name: str,
    ) -> pd.DataFrame:

        self.validate_schema()

        if table_name not in self.REQUIRED_TABLES:

            raise ValueError(
                f"Unsupported table: {table_name}"
            )

        with self._connect() as connection:

            return pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                connection,
            )

    # ========================================================
    # TEAMS
    # ========================================================

    def load_teams(self) -> pd.DataFrame:
        """
        Return clean team master data.

        team_db_id:
            Internal V1 SQLite ID.

        cfbd_team_id:
            CollegeFootballData external ID.
        """

        df = self.read_table(
            "teams"
        )

        columns = [
            "id",
            "cfbd_id",
            "school",
            "abbreviation",
            "mascot",
            "conference",
            "classification",
            "color",
            "logo_url",
        ]

        columns = [
            column
            for column in columns
            if column in df.columns
        ]

        df = df[
            columns
        ].copy()

        df = df.rename(
            columns={
                "id": "team_db_id",
                "cfbd_id": "cfbd_team_id",
                "school": "team_name",
            }
        )

        if "team_db_id" in df.columns:

            df["team_db_id"] = pd.to_numeric(
                df["team_db_id"],
                errors="coerce",
            )

        if "cfbd_team_id" in df.columns:

            df["cfbd_team_id"] = pd.to_numeric(
                df["cfbd_team_id"],
                errors="coerce",
            )

        if "classification" in df.columns:

            df["classification"] = (
                df["classification"]
                .astype("string")
                .str.lower()
                .str.strip()
            )

        df = df.drop_duplicates(
            subset=["team_db_id"]
        )

        return df.reset_index(
            drop=True
        )

    # ========================================================
    # GAMES
    # ========================================================

    def load_games(
        self,
        seasons: Iterable[int] | None = None,
        completed_only: bool = False,
    ) -> pd.DataFrame:

        df = self.read_table(
            "games"
        )

        # ----------------------------------------------------
        # Numeric columns
        # ----------------------------------------------------

        numeric_columns = [
            "id",
            "cfbd_id",
            "season",
            "week",
            "home_points",
            "away_points",
            "attendance",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        if "start_date" in df.columns:

            df["start_date"] = pd.to_datetime(
                df["start_date"],
                errors="coerce",
                utc=True,
            )

        # ----------------------------------------------------
        # Boolean values
        # ----------------------------------------------------

        boolean_columns = [
            "completed",
            "neutral_site",
            "conference_game",
        ]

        for column in boolean_columns:

            if column in df.columns:

                df[column] = (
                    pd.to_numeric(
                        df[column],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                    .astype(bool)
                )

        # ----------------------------------------------------
        # Season filter
        # ----------------------------------------------------

        if seasons is not None:

            season_values = list(
                seasons
            )

            df = df[
                df["season"].isin(
                    season_values
                )
            ].copy()

        # ----------------------------------------------------
        # Completed only
        # ----------------------------------------------------

        if completed_only:

            mask = (
                df["home_points"].notna()
                &
                df["away_points"].notna()
            )

            if "completed" in df.columns:

                mask = (
                    mask
                    &
                    df["completed"]
                )

            df = df[
                mask
            ].copy()

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        sort_columns = [
            column
            for column in [
                "season",
                "week",
                "start_date",
                "cfbd_id",
            ]
            if column in df.columns
        ]

        df = df.sort_values(
            sort_columns,
            kind="stable",
        )

        return df.reset_index(
            drop=True
        )

    # ========================================================
    # GAME TEAM STATS
    # ========================================================

    def load_game_team_stats(
        self,
        seasons: Iterable[int] | None = None,
    ) -> pd.DataFrame:

        df = self.read_table(
            "game_team_stats"
        )

        numeric_columns = [
            "id",
            "game_id",
            "season",
            "week",
            "team_id",
            "points",
            "rushing_tds",
            "rushing_attempts",
            "rushing_yards",
            "yards_per_rush",
            "passing_tds",
            "passing_completions",
            "passing_attempts",
            "net_passing_yards",
            "yards_per_pass",
            "total_yards",
            "first_downs",
            "third_down_made",
            "third_down_attempts",
            "fourth_down_made",
            "fourth_down_attempts",
            "turnovers",
            "interceptions",
            "passes_intercepted",
            "fumbles",
            "fumbles_lost",
            "fumbles_recovered",
            "punt_returns",
            "punt_return_yards",
            "punt_return_tds",
            "kick_returns",
            "kick_return_yards",
            "kick_return_tds",
            "kicking_points",
            "interception_yards",
            "interception_tds",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        if seasons is not None:

            season_values = list(
                seasons
            )

            df = df[
                df["season"].isin(
                    season_values
                )
            ].copy()

        if "home_away" in df.columns:

            df["home_away"] = (
                df["home_away"]
                .astype("string")
                .str.lower()
                .str.strip()
            )

        # ----------------------------------------------------
        # Parse possession time
        # ----------------------------------------------------

        if "possession_time" in df.columns:

            df["possession_minutes"] = (
                df["possession_time"]
                .apply(
                    self._parse_time_to_minutes
                )
            )

        # ----------------------------------------------------
        # Parse penalty string
        #
        # V1 stores examples like:
        #
        # 5-45
        #
        # where:
        #   penalties = 5
        #   penalty yards = 45
        # ----------------------------------------------------

        if "penalties_yards" in df.columns:

            parsed = (
                df["penalties_yards"]
                .apply(
                    self._parse_penalties
                )
            )

            df["penalties"] = (
                parsed.apply(
                    lambda value: value[0]
                )
            )

            df["penalty_yards"] = (
                parsed.apply(
                    lambda value: value[1]
                )
            )

        sort_columns = [
            column
            for column in [
                "season",
                "week",
                "game_id",
                "team_id",
            ]
            if column in df.columns
        ]

        df = df.sort_values(
            sort_columns,
            kind="stable",
        )

        return df.reset_index(
            drop=True
        )

    # ========================================================
    # CANONICAL GAME DATASET
    # ========================================================

    def build_game_dataset(
        self,
        seasons: Iterable[int] | None = None,
        completed_only: bool = True,
    ) -> pd.DataFrame:
        """
        Build one-row-per-game historical data.

        Official score fields from the games table are retained
        as:

            home_points
            away_points

        Statistics from game_team_stats are deliberately named:

            home_stat_...
            away_stat_...

        This prevents collisions with official game columns.
        """

        games = self.load_games(
            seasons=seasons,
            completed_only=completed_only,
        )

        teams = self.load_teams()

        stats = self.load_game_team_stats(
            seasons=seasons,
        )

        if games.empty:

            return pd.DataFrame()

        # ----------------------------------------------------
        # Attach team names to stats using INTERNAL DB IDs.
        #
        # game_team_stats.team_id → teams.id
        # ----------------------------------------------------

        team_lookup = teams[
            [
                "team_db_id",
                "team_name",
                "cfbd_team_id",
                "conference",
                "classification",
            ]
        ].copy()

        stats = stats.merge(
            team_lookup,
            left_on="team_id",
            right_on="team_db_id",
            how="left",
            validate="many_to_one",
        )

        # ----------------------------------------------------
        # Statistics to pivot
        #
        # We can safely include stats.points now because it will
        # become home_stat_points instead of home_points.
        # ----------------------------------------------------

        stat_columns = [
            "points",
            "rushing_tds",
            "rushing_attempts",
            "rushing_yards",
            "yards_per_rush",
            "passing_tds",
            "passing_completions",
            "passing_attempts",
            "net_passing_yards",
            "yards_per_pass",
            "total_yards",
            "first_downs",
            "third_down_made",
            "third_down_attempts",
            "fourth_down_made",
            "fourth_down_attempts",
            "turnovers",
            "interceptions",
            "passes_intercepted",
            "fumbles",
            "fumbles_lost",
            "fumbles_recovered",
            "possession_minutes",
            "penalties",
            "penalty_yards",
            "punt_returns",
            "punt_return_yards",
            "punt_return_tds",
            "kick_returns",
            "kick_return_yards",
            "kick_return_tds",
            "kicking_points",
            "interception_yards",
            "interception_tds",
        ]

        stat_columns = [
            column
            for column in stat_columns
            if column in stats.columns
        ]

        # ----------------------------------------------------
        # Pivot home / away
        # ----------------------------------------------------

        if stat_columns:

            pivot = stats.pivot_table(
                index="game_id",
                columns="home_away",
                values=stat_columns,
                aggfunc="first",
            )

            pivot.columns = [
                f"{side}_stat_{stat}"
                for stat, side in pivot.columns
            ]

            pivot = pivot.reset_index()

            games = games.merge(
                pivot,
                left_on="id",
                right_on="game_id",
                how="left",
                validate="one_to_one",
            )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        games["home_win"] = (
            games["home_points"]
            >
            games["away_points"]
        ).astype(int)

        games["away_win"] = (
            games["away_points"]
            >
            games["home_points"]
        ).astype(int)

        games["tie"] = (
            games["home_points"]
            ==
            games["away_points"]
        ).astype(int)

        games["home_margin"] = (
            games["home_points"]
            -
            games["away_points"]
        )

        games["total_points"] = (
            games["home_points"]
            +
            games["away_points"]
        )

        # ----------------------------------------------------
        # Team classification lookup
        # ----------------------------------------------------

        team_name_lookup = (
            teams[
                [
                    "team_name",
                    "classification",
                    "conference",
                    "cfbd_team_id",
                    "team_db_id",
                ]
            ]
            .drop_duplicates(
                subset=["team_name"]
            )
        )

        home_lookup = (
            team_name_lookup.rename(
                columns={
                    "team_name": "home_team",
                    "classification":
                        "home_classification",
                    "conference":
                        "home_conference",
                    "cfbd_team_id":
                        "home_cfbd_team_id",
                    "team_db_id":
                        "home_team_db_id",
                }
            )
        )

        away_lookup = (
            team_name_lookup.rename(
                columns={
                    "team_name": "away_team",
                    "classification":
                        "away_classification",
                    "conference":
                        "away_conference",
                    "cfbd_team_id":
                        "away_cfbd_team_id",
                    "team_db_id":
                        "away_team_db_id",
                }
            )
        )

        games = games.merge(
            home_lookup,
            on="home_team",
            how="left",
            validate="many_to_one",
        )

        games = games.merge(
            away_lookup,
            on="away_team",
            how="left",
            validate="many_to_one",
        )

        games["home_is_fbs"] = (
            games["home_classification"]
            .astype("string")
            .str.lower()
            .eq("fbs")
        )

        games["away_is_fbs"] = (
            games["away_classification"]
            .astype("string")
            .str.lower()
            .eq("fbs")
        )

        games["home_is_fcs"] = (
            games["home_classification"]
            .astype("string")
            .str.lower()
            .eq("fcs")
        )

        games["away_is_fcs"] = (
            games["away_classification"]
            .astype("string")
            .str.lower()
            .eq("fcs")
        )

        games["fbs_involved"] = (
            games["home_is_fbs"]
            |
            games["away_is_fbs"]
        )

        games["fbs_vs_fbs"] = (
            games["home_is_fbs"]
            &
            games["away_is_fbs"]
        )

        games["fbs_vs_fcs"] = (
            (
                games["home_is_fbs"]
                &
                games["away_is_fcs"]
            )
            |
            (
                games["home_is_fcs"]
                &
                games["away_is_fbs"]
            )
        )

        # ----------------------------------------------------
        # Check score consistency
        #
        # The official games score and stats score should agree.
        # We don't overwrite either.
        # ----------------------------------------------------

        if "home_stat_points" in games.columns:

            games["home_score_matches_stats"] = (
                games["home_stat_points"].isna()
                |
                (
                    games["home_points"]
                    ==
                    games["home_stat_points"]
                )
            )

        if "away_stat_points" in games.columns:

            games["away_score_matches_stats"] = (
                games["away_stat_points"].isna()
                |
                (
                    games["away_points"]
                    ==
                    games["away_stat_points"]
                )
            )

        # ----------------------------------------------------
        # Chronological ordering
        # ----------------------------------------------------

        sort_columns = [
            column
            for column in [
                "season",
                "week",
                "start_date",
                "cfbd_id",
            ]
            if column in games.columns
        ]

        games = games.sort_values(
            sort_columns,
            kind="stable",
        )

        return games.reset_index(
            drop=True
        )

    # ========================================================
    # TEAM-GAME LONG DATASET
    # ========================================================

    def build_team_game_dataset(
        self,
        seasons: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        """
        Build the core long-format historical dataset.

        One row represents one team's performance in one game.

        This will eventually feed:

            ELO
            offensive ratings
            defensive ratings
            special teams ratings
            opponent adjustment
            pace model
            possession model
            drive model
        """

        games = self.load_games(
            seasons=seasons,
            completed_only=True,
        )

        stats = self.load_game_team_stats(
            seasons=seasons,
        )

        teams = self.load_teams()

        if games.empty or stats.empty:

            return pd.DataFrame()

        # ----------------------------------------------------
        # Team mapping
        #
        # IMPORTANT:
        # stats.team_id is teams.id
        # ----------------------------------------------------

        team_lookup = teams[
            [
                "team_db_id",
                "cfbd_team_id",
                "team_name",
                "conference",
                "classification",
            ]
        ].copy()

        stats = stats.merge(
            team_lookup,
            left_on="team_id",
            right_on="team_db_id",
            how="left",
            validate="many_to_one",
        )

        # ----------------------------------------------------
        # Game information
        # ----------------------------------------------------

        game_columns = [
            "id",
            "cfbd_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "completed",
            "neutral_site",
            "conference_game",
            "venue",
        ]

        game_columns = [
            column
            for column in game_columns
            if column in games.columns
        ]

        game_info = games[
            game_columns
        ].copy()

        game_info = game_info.rename(
            columns={
                "id": "game_db_id",
                "cfbd_id": "cfbd_game_id",
            }
        )

        # stats already has season/week, so avoid duplicate
        # _x / _y versions from merge.
        merge_game_columns = [
            column
            for column in game_info.columns
            if column not in {
                "season",
                "week",
            }
        ]

        stats = stats.merge(
            game_info[
                merge_game_columns
            ],
            left_on="game_id",
            right_on="game_db_id",
            how="inner",
            validate="many_to_one",
        )

        # ----------------------------------------------------
        # Home / away orientation
        # ----------------------------------------------------

        stats["is_home"] = (
            stats["home_away"]
            .astype("string")
            .str.lower()
            .eq("home")
        )

        stats["is_away"] = (
            stats["home_away"]
            .astype("string")
            .str.lower()
            .eq("away")
        )

        stats["opponent_team"] = (
            stats["away_team"]
            .where(
                stats["is_home"],
                stats["home_team"],
            )
        )

        stats["team_points"] = (
            stats["points"]
        )

        stats["opponent_points"] = (
            stats["away_points"]
            .where(
                stats["is_home"],
                stats["home_points"],
            )
        )

        stats["point_margin"] = (
            stats["team_points"]
            -
            stats["opponent_points"]
        )

        stats["win"] = (
            stats["point_margin"] > 0
        ).astype(int)

        stats["loss"] = (
            stats["point_margin"] < 0
        ).astype(int)

        stats["tie"] = (
            stats["point_margin"] == 0
        ).astype(int)

        # ----------------------------------------------------
        # Opponent identity
        # ----------------------------------------------------

        opponent_lookup = teams[
            [
                "team_db_id",
                "cfbd_team_id",
                "team_name",
                "conference",
                "classification",
            ]
        ].rename(
            columns={
                "team_db_id":
                    "opponent_team_db_id",
                "cfbd_team_id":
                    "opponent_cfbd_team_id",
                "team_name":
                    "opponent_team",
                "conference":
                    "opponent_conference",
                "classification":
                    "opponent_classification",
            }
        )

        stats = stats.merge(
            opponent_lookup,
            on="opponent_team",
            how="left",
            validate="many_to_one",
        )

        # ----------------------------------------------------
        # Classification flags
        # ----------------------------------------------------

        stats["is_fbs"] = (
            stats["classification"]
            .astype("string")
            .str.lower()
            .eq("fbs")
        )

        stats["is_fcs"] = (
            stats["classification"]
            .astype("string")
            .str.lower()
            .eq("fcs")
        )

        stats["opponent_is_fbs"] = (
            stats["opponent_classification"]
            .astype("string")
            .str.lower()
            .eq("fbs")
        )

        stats["opponent_is_fcs"] = (
            stats["opponent_classification"]
            .astype("string")
            .str.lower()
            .eq("fcs")
        )

        # ----------------------------------------------------
        # Rate statistics
        # ----------------------------------------------------

        stats["third_down_pct"] = (
            self._safe_rate(
                stats["third_down_made"],
                stats["third_down_attempts"],
            )
        )

        stats["fourth_down_pct"] = (
            self._safe_rate(
                stats["fourth_down_made"],
                stats["fourth_down_attempts"],
            )
        )

        stats["completion_pct"] = (
            self._safe_rate(
                stats["passing_completions"],
                stats["passing_attempts"],
            )
        )

        # ----------------------------------------------------
        # Play-volume proxy
        #
        # NOTE:
        # This is NOT a perfect NCAA play count because sacks
        # may be represented differently in source stats.
        #
        # We preserve it explicitly as a proxy rather than
        # pretending it is exact.
        # ----------------------------------------------------

        stats["offensive_play_proxy"] = (
            stats["rushing_attempts"].fillna(0)
            +
            stats["passing_attempts"].fillna(0)
        )

        stats["yards_per_play_proxy"] = (
            stats["total_yards"]
            /
            stats[
                "offensive_play_proxy"
            ].replace(
                0,
                pd.NA,
            )
        )

        stats["points_per_play_proxy"] = (
            stats["team_points"]
            /
            stats[
                "offensive_play_proxy"
            ].replace(
                0,
                pd.NA,
            )
        )

        stats["turnover_rate_per_play"] = (
            stats["turnovers"]
            /
            stats[
                "offensive_play_proxy"
            ].replace(
                0,
                pd.NA,
            )
        )

        stats["passing_play_rate_proxy"] = (
            stats["passing_attempts"]
            /
            stats[
                "offensive_play_proxy"
            ].replace(
                0,
                pd.NA,
            )
        )

        stats["rushing_play_rate_proxy"] = (
            stats["rushing_attempts"]
            /
            stats[
                "offensive_play_proxy"
            ].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------
        # Possession
        # ----------------------------------------------------

        stats["possession_seconds"] = (
            stats["possession_minutes"]
            * 60
        )

        # ----------------------------------------------------
        # Team-game identifier
        # ----------------------------------------------------

        stats["team_game_id"] = (
            stats["game_id"]
            .astype("Int64")
            .astype(str)
            +
            "_"
            +
            stats["team_db_id"]
            .astype("Int64")
            .astype(str)
        )

        # ----------------------------------------------------
        # Chronological order
        # ----------------------------------------------------

        sort_columns = [
            column
            for column in [
                "season",
                "week",
                "start_date",
                "game_id",
                "team_db_id",
            ]
            if column in stats.columns
        ]

        stats = stats.sort_values(
            sort_columns,
            kind="stable",
        )

        return stats.reset_index(
            drop=True
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _parse_time_to_minutes(
        value,
    ) -> float | None:

        if pd.isna(value):

            return None

        text = str(
            value
        ).strip()

        if not text:

            return None

        try:

            pieces = text.split(
                ":"
            )

            if len(pieces) != 2:

                return None

            minutes = float(
                pieces[0]
            )

            seconds = float(
                pieces[1]
            )

            return (
                minutes
                +
                seconds / 60.0
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    @staticmethod
    def _parse_penalties(
        value,
    ) -> tuple[
        float | None,
        float | None,
    ]:

        if pd.isna(value):

            return (
                None,
                None,
            )

        text = str(
            value
        ).strip()

        if not text:

            return (
                None,
                None,
            )

        try:

            pieces = text.split(
                "-"
            )

            if len(pieces) != 2:

                return (
                    None,
                    None,
                )

            penalties = float(
                pieces[0]
            )

            yards = float(
                pieces[1]
            )

            return (
                penalties,
                yards,
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                None,
                None,
            )

    @staticmethod
    def _safe_rate(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:

        denominator = denominator.replace(
            0,
            pd.NA,
        )

        return (
            numerator
            /
            denominator
        )