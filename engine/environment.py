from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Chronological game-environment / pace configuration.

    Important:
        This engine models PLAY ENVIRONMENT.

        It does NOT pretend that plays == possessions.

    True possessions/drives will be estimated in the next
    modelling layer.
    """

    update_rate: float = 0.25
    preseason_regression: float = 0.40

    # National priors
    starting_team_plays: float = 68.0
    starting_possession_minutes: float = 30.0
    starting_plays_per_possession_minute: float = 2.27

    national_prior_team_games: float = 100.0

    # Environment blending.
    #
    # This is deliberately neutral rather than optimised:
    # 50% offensive historical play volume
    # 50% opposing defensive play volume allowed
    offense_environment_weight: float = 0.50
    defense_environment_weight: float = 0.50

    # Safety bounds
    minimum_team_plays: float = 40.0
    maximum_team_plays: float = 100.0

    minimum_total_plays: float = 90.0
    maximum_total_plays: float = 190.0

    minimum_possession_share: float = 0.30
    maximum_possession_share: float = 0.70


# ============================================================
# TEAM STATE
# ============================================================


@dataclass
class TeamEnvironmentState:
    """
    Chronological pace/environment state for one team.
    """

    team: str

    games_played: int = 0

    offense_plays: float = 68.0

    defense_plays_allowed: float = 68.0

    possession_minutes: float = 30.0

    possession_share: float = 0.50

    plays_per_possession_minute: float = 2.27

    last_season: int | None = None


# ============================================================
# NATIONAL STATE
# ============================================================


@dataclass
class NationalEnvironmentState:
    """
    Running national environment.

    Uses a prior so Week 1 does not begin from a tiny sample.
    """

    team_games_seen: float = 100.0

    total_team_plays: float = 6800.0

    total_possession_minutes: float = 3000.0

    total_plays_per_possession_minute: float = 227.0

    @property
    def average_team_plays(self) -> float:

        if self.team_games_seen <= 0:
            return 68.0

        return (
            self.total_team_plays
            /
            self.team_games_seen
        )

    @property
    def average_possession_minutes(self) -> float:

        if self.team_games_seen <= 0:
            return 30.0

        return (
            self.total_possession_minutes
            /
            self.team_games_seen
        )

    @property
    def average_plays_per_possession_minute(
        self,
    ) -> float:

        if self.team_games_seen <= 0:
            return 2.27

        return (
            self.total_plays_per_possession_minute
            /
            self.team_games_seen
        )


# ============================================================
# ENGINE
# ============================================================


class GameEnvironmentEngine:
    """
    Chronological pace / play-environment engine.

    For every game:

        1. Snapshot both teams BEFORE the game.
        2. Estimate expected play volume/environment.
        3. Save pregame features.
        4. Only then update both teams using actual stats.

    Therefore the current game cannot leak into its own
    pregame environment.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else EnvironmentConfig()
        )

        self.states: dict[
            str,
            TeamEnvironmentState,
        ] = {}

        prior_games = (
            self.config.national_prior_team_games
        )

        self.national = (
            NationalEnvironmentState(
                team_games_seen=prior_games,

                total_team_plays=(
                    prior_games
                    *
                    self.config.starting_team_plays
                ),

                total_possession_minutes=(
                    prior_games
                    *
                    self.config.starting_possession_minutes
                ),

                total_plays_per_possession_minute=(
                    prior_games
                    *
                    self.config.starting_plays_per_possession_minute
                ),
            )
        )

    # ========================================================
    # SAFE CONVERSIONS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float,
    ) -> float:

        try:

            number = float(value)

            if not np.isfinite(number):
                return float(fallback)

            return number

        except (
            TypeError,
            ValueError,
        ):

            return float(fallback)

    @staticmethod
    def _safe_int(
        value: Any,
        fallback: int = 0,
    ) -> int:

        try:

            number = float(value)

            if not np.isfinite(number):
                return int(fallback)

            return int(number)

        except (
            TypeError,
            ValueError,
        ):

            return int(fallback)

    @staticmethod
    def _normalise_team(
        value: Any,
    ) -> str:

        if pd.isna(value):
            return ""

        return str(value).strip()

    @staticmethod
    def _ewma(
        old: float,
        new: float,
        rate: float,
    ) -> float:

        return (
            (1.0 - rate) * old
            +
            rate * new
        )

    # ========================================================
    # STATE
    # ========================================================

    def _get_state(
        self,
        team: str,
    ) -> TeamEnvironmentState:

        if team not in self.states:

            self.states[
                team
            ] = TeamEnvironmentState(
                team=team,

                offense_plays=(
                    self.config.starting_team_plays
                ),

                defense_plays_allowed=(
                    self.config.starting_team_plays
                ),

                possession_minutes=(
                    self.config.starting_possession_minutes
                ),

                possession_share=0.50,

                plays_per_possession_minute=(
                    self.config
                    .starting_plays_per_possession_minute
                ),
            )

        return self.states[
            team
        ]

    # ========================================================
    # SEASON REGRESSION
    # ========================================================

    def _apply_season_regression(
        self,
        state: TeamEnvironmentState,
        season: int,
    ) -> None:

        if state.last_season is None:

            state.last_season = season
            return

        if season <= state.last_season:
            return

        regression = (
            self.config.preseason_regression
        )

        keep = (
            1.0
            -
            regression
        )

        national_plays = (
            self.national.average_team_plays
        )

        national_possession = (
            self.national.average_possession_minutes
        )

        national_tempo = (
            self.national
            .average_plays_per_possession_minute
        )

        state.offense_plays = (
            keep
            *
            state.offense_plays
            +
            regression
            *
            national_plays
        )

        state.defense_plays_allowed = (
            keep
            *
            state.defense_plays_allowed
            +
            regression
            *
            national_plays
        )

        state.possession_minutes = (
            keep
            *
            state.possession_minutes
            +
            regression
            *
            national_possession
        )

        state.possession_share = (
            keep
            *
            state.possession_share
            +
            regression
            *
            0.50
        )

        state.plays_per_possession_minute = (
            keep
            *
            state.plays_per_possession_minute
            +
            regression
            *
            national_tempo
        )

        state.last_season = season

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
        team: str,
        season: int,
    ) -> dict[str, Any]:

        state = self._get_state(
            team
        )

        self._apply_season_regression(
            state,
            season,
        )

        return {
            "team": team,

            "environment_games_pre": (
                state.games_played
            ),

            "offense_plays_pre": (
                state.offense_plays
            ),

            "defense_plays_allowed_pre": (
                state.defense_plays_allowed
            ),

            "possession_minutes_pre": (
                state.possession_minutes
            ),

            "possession_share_pre": (
                state.possession_share
            ),

            "plays_per_possession_minute_pre": (
                state.plays_per_possession_minute
            ),

            "national_team_plays_pre": (
                self.national.average_team_plays
            ),

            "national_possession_minutes_pre": (
                self.national.average_possession_minutes
            ),

            "national_plays_per_possession_minute_pre": (
                self.national
                .average_plays_per_possession_minute
            ),
        }

    # ========================================================
    # ACTUAL GAME STATS
    # ========================================================

    def calculate_actual_team_plays(
        self,
        row: pd.Series,
    ) -> float:

        rushing_attempts = max(
            self._safe_float(
                row.get(
                    "rushing_attempts"
                ),
                0.0,
            ),
            0.0,
        )

        passing_attempts = max(
            self._safe_float(
                row.get(
                    "passing_attempts"
                ),
                0.0,
            ),
            0.0,
        )

        plays = (
            rushing_attempts
            +
            passing_attempts
        )

        return float(
            np.clip(
                plays,
                self.config.minimum_team_plays,
                self.config.maximum_team_plays,
            )
        )

    def calculate_possession_share(
        self,
        team_minutes: float,
        opponent_minutes: float,
    ) -> float:

        total = (
            team_minutes
            +
            opponent_minutes
        )

        if total <= 0:
            return 0.50

        share = (
            team_minutes
            /
            total
        )

        return float(
            np.clip(
                share,
                self.config.minimum_possession_share,
                self.config.maximum_possession_share,
            )
        )

    # ========================================================
    # EXPECTED GAME ENVIRONMENT
    # ========================================================

    def expected_team_plays(
        self,
        offense_plays: float,
        opponent_defense_plays_allowed: float,
    ) -> float:

        offense_weight = (
            self.config.offense_environment_weight
        )

        defense_weight = (
            self.config.defense_environment_weight
        )

        denominator = (
            offense_weight
            +
            defense_weight
        )

        if denominator <= 0:
            denominator = 1.0

        expected = (
            offense_weight
            *
            offense_plays
            +
            defense_weight
            *
            opponent_defense_plays_allowed
        ) / denominator

        return float(
            np.clip(
                expected,
                self.config.minimum_team_plays,
                self.config.maximum_team_plays,
            )
        )

    def expected_possession_shares(
        self,
        home_share: float,
        away_share: float,
    ) -> tuple[float, float]:

        home = max(
            self._safe_float(
                home_share,
                0.50,
            ),
            0.01,
        )

        away = max(
            self._safe_float(
                away_share,
                0.50,
            ),
            0.01,
        )

        total = (
            home
            +
            away
        )

        if total <= 0:
            return 0.50, 0.50

        home_expected = (
            home
            /
            total
        )

        home_expected = float(
            np.clip(
                home_expected,
                self.config.minimum_possession_share,
                self.config.maximum_possession_share,
            )
        )

        away_expected = (
            1.0
            -
            home_expected
        )

        return (
            home_expected,
            away_expected,
        )

    # ========================================================
    # UPDATE TEAM
    # ========================================================

    def update_team(
        self,
        team: str,
        season: int,
        team_plays: float,
        opponent_plays: float,
        possession_minutes: float,
        possession_share: float,
    ) -> None:

        state = self._get_state(
            team
        )

        self._apply_season_regression(
            state,
            season,
        )

        rate = (
            self.config.update_rate
        )

        possession_minutes = max(
            possession_minutes,
            1.0,
        )

        tempo = (
            team_plays
            /
            possession_minutes
        )

        state.offense_plays = (
            self._ewma(
                state.offense_plays,
                team_plays,
                rate,
            )
        )

        state.defense_plays_allowed = (
            self._ewma(
                state.defense_plays_allowed,
                opponent_plays,
                rate,
            )
        )

        state.possession_minutes = (
            self._ewma(
                state.possession_minutes,
                possession_minutes,
                rate,
            )
        )

        state.possession_share = (
            self._ewma(
                state.possession_share,
                possession_share,
                rate,
            )
        )

        state.plays_per_possession_minute = (
            self._ewma(
                state.plays_per_possession_minute,
                tempo,
                rate,
            )
        )

        state.games_played += 1
        state.last_season = season

    # ========================================================
    # UPDATE NATIONAL
    # ========================================================

    def update_national(
        self,
        team_plays: float,
        possession_minutes: float,
    ) -> None:

        possession_minutes = max(
            possession_minutes,
            1.0,
        )

        tempo = (
            team_plays
            /
            possession_minutes
        )

        self.national.team_games_seen += 1.0

        self.national.total_team_plays += (
            team_plays
        )

        self.national.total_possession_minutes += (
            possession_minutes
        )

        self.national.total_plays_per_possession_minute += (
            tempo
        )

    # ========================================================
    # PROCESS HISTORY
    # ========================================================

    def process_history(
        self,
        team_history: pd.DataFrame,
        matchup_history: pd.DataFrame,
    ) -> pd.DataFrame:

        required_team_columns = [
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

        missing_team = [
            column
            for column in required_team_columns
            if column not in team_history.columns
        ]

        if missing_team:

            raise ValueError(
                "Team history is missing required "
                f"environment columns: {missing_team}"
            )

        if "game_id" not in matchup_history.columns:

            raise ValueError(
                "Matchup history is missing game_id."
            )

        data = (
            team_history.copy()
        )

        matchups = (
            matchup_history.copy()
        )

        data["game_id"] = pd.to_numeric(
            data["game_id"],
            errors="coerce",
        ).astype("Int64")

        matchups["game_id"] = pd.to_numeric(
            matchups["game_id"],
            errors="coerce",
        ).astype("Int64")

        data["home_away"] = (
            data["home_away"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # ----------------------------------------------------
        # CHRONOLOGICAL ORDER
        # ----------------------------------------------------

        if "start_date" in data.columns:

            data["start_date"] = pd.to_datetime(
                data["start_date"],
                errors="coerce",
                utc=True,
            )

            sort_columns = [
                "season",
                "start_date",
                "game_id",
            ]

        else:

            sort_columns = [
                "season",
                "week",
                "game_id",
            ]

        data = (
            data
            .sort_values(
                sort_columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        matchup_lookup = (
            matchups
            .set_index(
                "game_id",
                drop=False,
            )
        )

        outputs: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # GAME LOOP
        # ====================================================

        for game_id, rows in data.groupby(
            "game_id",
            sort=False,
        ):

            if len(rows) != 2:
                continue

            home_rows = rows[
                rows["home_away"]
                ==
                "home"
            ]

            away_rows = rows[
                rows["home_away"]
                ==
                "away"
            ]

            if (
                len(home_rows) != 1
                or
                len(away_rows) != 1
            ):
                continue

            if game_id not in matchup_lookup.index:
                continue

            home_row = home_rows.iloc[0]
            away_row = away_rows.iloc[0]

            matchup_row = (
                matchup_lookup.loc[
                    game_id
                ]
            )

            # Defensive guard in case pandas returns a frame.
            if isinstance(
                matchup_row,
                pd.DataFrame,
            ):
                matchup_row = (
                    matchup_row.iloc[0]
                )

            season = self._safe_int(
                home_row.get(
                    "season"
                ),
                0,
            )

            home_team = self._normalise_team(
                home_row.get(
                    "team_name"
                )
            )

            away_team = self._normalise_team(
                away_row.get(
                    "team_name"
                )
            )

            # ------------------------------------------------
            # BOTH SNAPSHOTS BEFORE UPDATES
            # ------------------------------------------------

            home_snapshot = self.snapshot(
                home_team,
                season,
            )

            away_snapshot = self.snapshot(
                away_team,
                season,
            )

            # ------------------------------------------------
            # EXPECTED PLAY ENVIRONMENT
            # ------------------------------------------------

            expected_home_plays = (
                self.expected_team_plays(
                    offense_plays=(
                        home_snapshot[
                            "offense_plays_pre"
                        ]
                    ),
                    opponent_defense_plays_allowed=(
                        away_snapshot[
                            "defense_plays_allowed_pre"
                        ]
                    ),
                )
            )

            expected_away_plays = (
                self.expected_team_plays(
                    offense_plays=(
                        away_snapshot[
                            "offense_plays_pre"
                        ]
                    ),
                    opponent_defense_plays_allowed=(
                        home_snapshot[
                            "defense_plays_allowed_pre"
                        ]
                    ),
                )
            )

            expected_total_plays = (
                expected_home_plays
                +
                expected_away_plays
            )

            expected_total_plays = float(
                np.clip(
                    expected_total_plays,
                    self.config.minimum_total_plays,
                    self.config.maximum_total_plays,
                )
            )

            national_expected_total = (
                2.0
                *
                self.national.average_team_plays
            )

            if national_expected_total <= 0:
                pace_index = 100.0
            else:
                pace_index = (
                    100.0
                    *
                    expected_total_plays
                    /
                    national_expected_total
                )

            (
                expected_home_possession_share,
                expected_away_possession_share,
            ) = self.expected_possession_shares(
                home_snapshot[
                    "possession_share_pre"
                ],
                away_snapshot[
                    "possession_share_pre"
                ],
            )

            expected_home_possession_minutes = (
                60.0
                *
                expected_home_possession_share
            )

            expected_away_possession_minutes = (
                60.0
                *
                expected_away_possession_share
            )

            # ------------------------------------------------
            # OUTPUT = MATCHUP ROW + ENVIRONMENT
            # ------------------------------------------------

            output = (
                matchup_row.to_dict()
            )

            output.update(
                {
                    # ----------------------------------------
                    # HOME PACE STATE
                    # ----------------------------------------

                    "home_environment_games_pre": (
                        home_snapshot[
                            "environment_games_pre"
                        ]
                    ),

                    "home_offense_plays_pre": (
                        home_snapshot[
                            "offense_plays_pre"
                        ]
                    ),

                    "home_defense_plays_allowed_pre": (
                        home_snapshot[
                            "defense_plays_allowed_pre"
                        ]
                    ),

                    "home_possession_minutes_pre": (
                        home_snapshot[
                            "possession_minutes_pre"
                        ]
                    ),

                    "home_possession_share_pre": (
                        home_snapshot[
                            "possession_share_pre"
                        ]
                    ),

                    "home_plays_per_possession_minute_pre": (
                        home_snapshot[
                            "plays_per_possession_minute_pre"
                        ]
                    ),

                    # ----------------------------------------
                    # AWAY PACE STATE
                    # ----------------------------------------

                    "away_environment_games_pre": (
                        away_snapshot[
                            "environment_games_pre"
                        ]
                    ),

                    "away_offense_plays_pre": (
                        away_snapshot[
                            "offense_plays_pre"
                        ]
                    ),

                    "away_defense_plays_allowed_pre": (
                        away_snapshot[
                            "defense_plays_allowed_pre"
                        ]
                    ),

                    "away_possession_minutes_pre": (
                        away_snapshot[
                            "possession_minutes_pre"
                        ]
                    ),

                    "away_possession_share_pre": (
                        away_snapshot[
                            "possession_share_pre"
                        ]
                    ),

                    "away_plays_per_possession_minute_pre": (
                        away_snapshot[
                            "plays_per_possession_minute_pre"
                        ]
                    ),

                    # ----------------------------------------
                    # NATIONAL BASELINE
                    # ----------------------------------------

                    "national_team_plays_pre": (
                        self.national.average_team_plays
                    ),

                    "national_possession_minutes_pre": (
                        self.national.average_possession_minutes
                    ),

                    "national_plays_per_possession_minute_pre": (
                        self.national
                        .average_plays_per_possession_minute
                    ),

                    # ----------------------------------------
                    # EXPECTED GAME ENVIRONMENT
                    # ----------------------------------------

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
                        expected_home_possession_share
                    ),

                    "expected_away_possession_share": (
                        expected_away_possession_share
                    ),

                    "expected_home_possession_minutes": (
                        expected_home_possession_minutes
                    ),

                    "expected_away_possession_minutes": (
                        expected_away_possession_minutes
                    ),

                    "game_pace_index": (
                        pace_index
                    ),
                }
            )

            # ------------------------------------------------
            # ACTUAL STATS FOR VALIDATION / UPDATE
            #
            # These are post-game fields and must never be
            # used as same-game pregame features.
            # ------------------------------------------------

            actual_home_plays = (
                self.calculate_actual_team_plays(
                    home_row
                )
            )

            actual_away_plays = (
                self.calculate_actual_team_plays(
                    away_row
                )
            )

            actual_home_possession_minutes = max(
                self._safe_float(
                    home_row.get(
                        "possession_minutes"
                    ),
                    30.0,
                ),
                0.0,
            )

            actual_away_possession_minutes = max(
                self._safe_float(
                    away_row.get(
                        "possession_minutes"
                    ),
                    30.0,
                ),
                0.0,
            )

            actual_home_possession_share = (
                self.calculate_possession_share(
                    actual_home_possession_minutes,
                    actual_away_possession_minutes,
                )
            )

            actual_away_possession_share = (
                1.0
                -
                actual_home_possession_share
            )

            actual_total_plays = (
                actual_home_plays
                +
                actual_away_plays
            )

            output.update(
                {
                    "actual_home_plays": (
                        actual_home_plays
                    ),

                    "actual_away_plays": (
                        actual_away_plays
                    ),

                    "actual_total_plays": (
                        actual_total_plays
                    ),

                    "actual_home_possession_minutes": (
                        actual_home_possession_minutes
                    ),

                    "actual_away_possession_minutes": (
                        actual_away_possession_minutes
                    ),

                    "actual_home_possession_share": (
                        actual_home_possession_share
                    ),

                    "actual_away_possession_share": (
                        actual_away_possession_share
                    ),
                }
            )

            outputs.append(
                output
            )

            # ------------------------------------------------
            # UPDATE ONLY AFTER PRE-GAME OUTPUT CREATED
            # ------------------------------------------------

            self.update_team(
                team=home_team,
                season=season,
                team_plays=actual_home_plays,
                opponent_plays=actual_away_plays,
                possession_minutes=(
                    actual_home_possession_minutes
                ),
                possession_share=(
                    actual_home_possession_share
                ),
            )

            self.update_team(
                team=away_team,
                season=season,
                team_plays=actual_away_plays,
                opponent_plays=actual_home_plays,
                possession_minutes=(
                    actual_away_possession_minutes
                ),
                possession_share=(
                    actual_away_possession_share
                ),
            )

            self.update_national(
                team_plays=actual_home_plays,
                possession_minutes=(
                    actual_home_possession_minutes
                ),
            )

            self.update_national(
                team_plays=actual_away_plays,
                possession_minutes=(
                    actual_away_possession_minutes
                ),
            )

        result = pd.DataFrame(
            outputs
        )

        if not result.empty:

            if "start_date" in result.columns:

                result["start_date"] = pd.to_datetime(
                    result["start_date"],
                    errors="coerce",
                    utc=True,
                )

                final_sort = [
                    "season",
                    "start_date",
                    "game_id",
                ]

            else:

                final_sort = [
                    "season",
                    "week",
                    "game_id",
                ]

            result = (
                result
                .sort_values(
                    final_sort,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        return result

    # ========================================================
    # CURRENT STATES
    # ========================================================

    def current_environment(
        self,
    ) -> pd.DataFrame:

        rows: list[
            dict[str, Any]
        ] = []

        for team, state in self.states.items():

            rows.append(
                {
                    "team": team,

                    "games_played": (
                        state.games_played
                    ),

                    "offense_plays": (
                        state.offense_plays
                    ),

                    "defense_plays_allowed": (
                        state.defense_plays_allowed
                    ),

                    "possession_minutes": (
                        state.possession_minutes
                    ),

                    "possession_share": (
                        state.possession_share
                    ),

                    "plays_per_possession_minute": (
                        state.plays_per_possession_minute
                    ),

                    "last_season": (
                        state.last_season
                    ),
                }
            )

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                "offense_plays",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )