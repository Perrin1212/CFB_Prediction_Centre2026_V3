from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class OpponentAdjustmentConfig:
    """
    Configuration for chronological opponent-adjusted ratings.

    Scale:

        100 = national average
        >100 = above average
        <100 = below average

    This layer takes raw game performance and modifies its
    value according to the quality of the opponent faced.
    """

    update_rate: float = 0.25

    preseason_regression: float = 0.40

    rating_floor: float = 50.0
    rating_ceiling: float = 150.0

    component_ratio_floor: float = 0.50
    component_ratio_ceiling: float = 1.50

    opponent_factor_floor: float = 0.75
    opponent_factor_ceiling: float = 1.25

    # --------------------------------------------------------
    # OFFENCE
    # --------------------------------------------------------

    offense_points_weight: float = 0.40
    offense_ypp_weight: float = 0.35
    offense_turnover_weight: float = 0.15
    offense_first_down_weight: float = 0.10

    # --------------------------------------------------------
    # DEFENCE
    # --------------------------------------------------------

    defense_points_weight: float = 0.40
    defense_ypp_weight: float = 0.35
    defense_takeaway_weight: float = 0.15
    defense_first_down_weight: float = 0.10


# ============================================================
# TEAM STATE
# ============================================================


@dataclass
class AdjustedTeamState:
    """
    Current chronological opponent-adjusted team strength.
    """

    team: str

    games_played: int = 0

    offense_rating: float = 100.0
    defense_rating: float = 100.0

    last_season: int | None = None


# ============================================================
# ENGINE
# ============================================================


class OpponentAdjustmentEngine:
    """
    Builds chronological opponent-adjusted offence and defence.

    Architecture:

        raw pregame strength
              ↓
        actual game performance
              ↓
        quality of opponent faced
              ↓
        opponent-adjusted game performance
              ↓
        update adjusted team state

    The current game never affects its own pregame rating.
    """

    def __init__(
        self,
        config: OpponentAdjustmentConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else OpponentAdjustmentConfig()
        )

        self.states: dict[
            str,
            AdjustedTeamState,
        ] = {}

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float,
    ) -> float:

        try:

            numeric = float(value)

            if not np.isfinite(numeric):
                return float(fallback)

            return numeric

        except (
            TypeError,
            ValueError,
        ):

            return float(fallback)

    @staticmethod
    def _normalise_team(
        value: Any,
    ) -> str:

        if pd.isna(value):
            return ""

        return str(value).strip()

    def _clip_component(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.component_ratio_floor,
                self.config.component_ratio_ceiling,
            )
        )

    def _clip_opponent_factor(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.opponent_factor_floor,
                self.config.opponent_factor_ceiling,
            )
        )

    def _clip_rating(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.rating_floor,
                self.config.rating_ceiling,
            )
        )

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
    # TEAM STATE
    # ========================================================

    def _get_state(
        self,
        team: str,
    ) -> AdjustedTeamState:

        if team not in self.states:

            self.states[
                team
            ] = AdjustedTeamState(
                team=team
            )

        return self.states[
            team
        ]

    # ========================================================
    # SEASON REGRESSION
    # ========================================================

    def _apply_season_regression(
        self,
        state: AdjustedTeamState,
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

        state.offense_rating = (
            keep
            *
            state.offense_rating
            +
            regression
            *
            100.0
        )

        state.defense_rating = (
            keep
            *
            state.defense_rating
            +
            regression
            *
            100.0
        )

        state.last_season = season

    # ========================================================
    # PREGAME SNAPSHOT
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

            "games_pre": (
                state.games_played
            ),

            "adjusted_offense_rating_pre": (
                state.offense_rating
            ),

            "adjusted_defense_rating_pre": (
                state.defense_rating
            ),
        }

    # ========================================================
    # OFFENCE GAME PERFORMANCE
    # ========================================================

    def offense_game_rating(
        self,
        row: pd.Series,
    ) -> float:
        """
        Rate one team's offensive performance after accounting
        for the opponent's PRE-GAME raw defence strength.
        """

        national_points = max(
            self._safe_float(
                row.get(
                    "rating_state_national_points_pre"
                ),
                26.5,
            ),
            0.01,
        )

        national_ypp = max(
            self._safe_float(
                row.get(
                    "rating_state_national_ypp_pre"
                ),
                5.5,
            ),
            0.01,
        )

        national_turnovers = max(
            self._safe_float(
                row.get(
                    "rating_state_national_turnovers_pre"
                ),
                1.5,
            ),
            0.01,
        )

        national_first_downs = max(
            self._safe_float(
                row.get(
                    "rating_state_national_first_downs_pre"
                ),
                20.0,
            ),
            0.01,
        )

        team_points = max(
            self._safe_float(
                row.get(
                    "team_points"
                ),
                national_points,
            ),
            0.0,
        )

        team_ypp = max(
            self._safe_float(
                row.get(
                    "yards_per_play_proxy"
                ),
                national_ypp,
            ),
            0.1,
        )

        team_turnovers = max(
            self._safe_float(
                row.get(
                    "turnovers"
                ),
                national_turnovers,
            ),
            0.20,
        )

        team_first_downs = max(
            self._safe_float(
                row.get(
                    "first_downs"
                ),
                national_first_downs,
            ),
            0.1,
        )

        opponent_defense = (
            self._safe_float(
                row.get(
                    "opponent_defense_rating_pre"
                ),
                100.0,
            )
        )

        opponent_factor = (
            self._clip_opponent_factor(
                opponent_defense
                /
                100.0
            )
        )

        # ----------------------------------------------------
        # COMPONENT RATIOS
        # ----------------------------------------------------

        points_ratio = self._clip_component(
            (
                team_points
                /
                national_points
            )
            *
            opponent_factor
        )

        ypp_ratio = self._clip_component(
            (
                team_ypp
                /
                national_ypp
            )
            *
            opponent_factor
        )

        # Fewer turnovers is better.
        turnover_ratio = self._clip_component(
            (
                national_turnovers
                /
                team_turnovers
            )
            *
            opponent_factor
        )

        first_down_ratio = self._clip_component(
            (
                team_first_downs
                /
                national_first_downs
            )
            *
            opponent_factor
        )

        composite = (
            self.config.offense_points_weight
            *
            points_ratio
            +
            self.config.offense_ypp_weight
            *
            ypp_ratio
            +
            self.config.offense_turnover_weight
            *
            turnover_ratio
            +
            self.config.offense_first_down_weight
            *
            first_down_ratio
        )

        return self._clip_rating(
            composite
            *
            100.0
        )

    # ========================================================
    # DEFENCE GAME PERFORMANCE
    # ========================================================

    def defense_game_rating(
        self,
        row: pd.Series,
    ) -> float:
        """
        Rate one team's defensive performance after accounting
        for the opponent's PRE-GAME raw offence strength.
        """

        national_points = max(
            self._safe_float(
                row.get(
                    "rating_state_national_points_pre"
                ),
                26.5,
            ),
            0.01,
        )

        national_ypp = max(
            self._safe_float(
                row.get(
                    "rating_state_national_ypp_pre"
                ),
                5.5,
            ),
            0.01,
        )

        national_turnovers = max(
            self._safe_float(
                row.get(
                    "rating_state_national_turnovers_pre"
                ),
                1.5,
            ),
            0.01,
        )

        national_first_downs = max(
            self._safe_float(
                row.get(
                    "rating_state_national_first_downs_pre"
                ),
                20.0,
            ),
            0.01,
        )

        opponent_points = max(
            self._safe_float(
                row.get(
                    "opponent_points"
                ),
                national_points,
            ),
            1.0,
        )

        # ----------------------------------------------------
        # The opposing row contains the opponent's own
        # yards/play, turnovers and first downs.
        #
        # These fields are added before this function is called.
        # ----------------------------------------------------

        opponent_ypp = max(
            self._safe_float(
                row.get(
                    "opponent_yards_per_play"
                ),
                national_ypp,
            ),
            0.1,
        )

        opponent_turnovers = max(
            self._safe_float(
                row.get(
                    "opponent_turnovers"
                ),
                national_turnovers,
            ),
            0.20,
        )

        opponent_first_downs = max(
            self._safe_float(
                row.get(
                    "opponent_first_downs"
                ),
                national_first_downs,
            ),
            0.1,
        )

        opponent_offense = (
            self._safe_float(
                row.get(
                    "opponent_offense_rating_pre"
                ),
                100.0,
            )
        )

        opponent_factor = (
            self._clip_opponent_factor(
                opponent_offense
                /
                100.0
            )
        )

        # ----------------------------------------------------
        # COMPONENT RATIOS
        # ----------------------------------------------------

        # Fewer points allowed is better.
        points_ratio = self._clip_component(
            (
                national_points
                /
                opponent_points
            )
            *
            opponent_factor
        )

        # Lower opponent YPP is better.
        ypp_ratio = self._clip_component(
            (
                national_ypp
                /
                opponent_ypp
            )
            *
            opponent_factor
        )

        # More takeaways is better.
        takeaway_ratio = self._clip_component(
            (
                opponent_turnovers
                /
                national_turnovers
            )
            *
            opponent_factor
        )

        # Fewer first downs allowed is better.
        first_down_ratio = self._clip_component(
            (
                national_first_downs
                /
                opponent_first_downs
            )
            *
            opponent_factor
        )

        composite = (
            self.config.defense_points_weight
            *
            points_ratio
            +
            self.config.defense_ypp_weight
            *
            ypp_ratio
            +
            self.config.defense_takeaway_weight
            *
            takeaway_ratio
            +
            self.config.defense_first_down_weight
            *
            first_down_ratio
        )

        return self._clip_rating(
            composite
            *
            100.0
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        team: str,
        season: int,
        offense_game_rating: float,
        defense_game_rating: float,
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

        state.offense_rating = (
            self._ewma(
                state.offense_rating,
                offense_game_rating,
                rate,
            )
        )

        state.defense_rating = (
            self._ewma(
                state.defense_rating,
                defense_game_rating,
                rate,
            )
        )

        state.offense_rating = (
            self._clip_rating(
                state.offense_rating
            )
        )

        state.defense_rating = (
            self._clip_rating(
                state.defense_rating
            )
        )

        state.games_played += 1
        state.last_season = season

    # ========================================================
    # ADD OPPONENT ACTUAL STATS
    # ========================================================

    @staticmethod
    def add_opponent_actual_stats(
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        result = (
            data.copy()
        )

        pair = result[
            [
                "game_id",
                "team_name",
                "yards_per_play_proxy",
                "turnovers",
                "first_downs",
            ]
        ].copy()

        pair = pair.rename(
            columns={
                "team_name": (
                    "opponent_name_lookup"
                ),
                "yards_per_play_proxy": (
                    "opponent_yards_per_play"
                ),
                "turnovers": (
                    "opponent_turnovers"
                ),
                "first_downs": (
                    "opponent_first_downs"
                ),
            }
        )

        result = result.merge(
            pair,
            left_on=[
                "game_id",
                "opponent_name",
            ],
            right_on=[
                "game_id",
                "opponent_name_lookup",
            ],
            how="left",
        )

        result = result.drop(
            columns=[
                "opponent_name_lookup",
            ],
            errors="ignore",
        )

        return result

    # ========================================================
    # PROCESS HISTORY
    # ========================================================

    def process_history(
        self,
        rating_history: pd.DataFrame,
    ) -> pd.DataFrame:

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
        ]

        missing = [
            column
            for column in required
            if column not in rating_history.columns
        ]

        if missing:

            raise ValueError(
                "Raw rating history is missing required "
                f"columns: {missing}"
            )

        data = (
            rating_history.copy()
        )

        data = self.add_opponent_actual_stats(
            data
        )

        if "start_date" in data.columns:

            data[
                "start_date"
            ] = pd.to_datetime(
                data[
                    "start_date"
                ],
                errors="coerce",
                utc=True,
            )

            sort_columns = [
                "season",
                "start_date",
                "game_id",
                "team_name",
            ]

        else:

            sort_columns = [
                "season",
                "week",
                "game_id",
                "team_name",
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

        outputs: list[
            dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # GAME-BY-GAME
        # ----------------------------------------------------

        for game_id, game_rows in data.groupby(
            "game_id",
            sort=False,
        ):

            if len(game_rows) != 2:
                continue

            game_rows = (
                game_rows.copy()
            )

            season = int(
                game_rows.iloc[0][
                    "season"
                ]
            )

            snapshots: dict[
                str,
                dict[str, Any],
            ] = {}

            # ------------------------------------------------
            # CREATE BOTH PRE-GAME SNAPSHOTS FIRST
            # ------------------------------------------------

            for _, row in game_rows.iterrows():

                team = (
                    self._normalise_team(
                        row[
                            "team_name"
                        ]
                    )
                )

                snapshots[
                    team
                ] = self.snapshot(
                    team=team,
                    season=season,
                )

            update_rows: list[
                tuple[
                    str,
                    float,
                    float,
                ]
            ] = []

            # ------------------------------------------------
            # CALCULATE GAME PERFORMANCE
            # ------------------------------------------------

            for _, row in game_rows.iterrows():

                team = (
                    self._normalise_team(
                        row[
                            "team_name"
                        ]
                    )
                )

                opponent = (
                    self._normalise_team(
                        row[
                            "opponent_name"
                        ]
                    )
                )

                snapshot = snapshots[
                    team
                ]

                opponent_snapshot = (
                    snapshots[
                        opponent
                    ]
                )

                offense_game_rating = (
                    self.offense_game_rating(
                        row
                    )
                )

                defense_game_rating = (
                    self.defense_game_rating(
                        row
                    )
                )

                output = (
                    row.to_dict()
                )

                # --------------------------------------------
                # ADJUSTED PRE-GAME RATINGS
                # --------------------------------------------

                output[
                    "team_adjusted_offense_rating_pre"
                ] = (
                    snapshot[
                        "adjusted_offense_rating_pre"
                    ]
                )

                output[
                    "team_adjusted_defense_rating_pre"
                ] = (
                    snapshot[
                        "adjusted_defense_rating_pre"
                    ]
                )

                output[
                    "opponent_adjusted_offense_rating_pre"
                ] = (
                    opponent_snapshot[
                        "adjusted_offense_rating_pre"
                    ]
                )

                output[
                    "opponent_adjusted_defense_rating_pre"
                ] = (
                    opponent_snapshot[
                        "adjusted_defense_rating_pre"
                    ]
                )

                output[
                    "adjusted_games_pre"
                ] = (
                    snapshot[
                        "games_pre"
                    ]
                )

                # --------------------------------------------
                # POST-GAME PERFORMANCE INDEX
                #
                # Useful for diagnostics/training, but it must
                # never be used as a pregame feature for this
                # same game.
                # --------------------------------------------

                output[
                    "offense_game_rating_adjusted"
                ] = (
                    offense_game_rating
                )

                output[
                    "defense_game_rating_adjusted"
                ] = (
                    defense_game_rating
                )

                output[
                    "opponent_raw_defense_factor"
                ] = (
                    self._clip_opponent_factor(
                        self._safe_float(
                            row[
                                "opponent_defense_rating_pre"
                            ],
                            100.0,
                        )
                        /
                        100.0
                    )
                )

                output[
                    "opponent_raw_offense_factor"
                ] = (
                    self._clip_opponent_factor(
                        self._safe_float(
                            row[
                                "opponent_offense_rating_pre"
                            ],
                            100.0,
                        )
                        /
                        100.0
                    )
                )

                outputs.append(
                    output
                )

                update_rows.append(
                    (
                        team,
                        offense_game_rating,
                        defense_game_rating,
                    )
                )

            # ------------------------------------------------
            # UPDATE ONLY AFTER BOTH PREGAME ROWS CREATED
            # ------------------------------------------------

            for (
                team,
                offense_game_rating,
                defense_game_rating,
            ) in update_rows:

                self.update(
                    team=team,
                    season=season,
                    offense_game_rating=(
                        offense_game_rating
                    ),
                    defense_game_rating=(
                        defense_game_rating
                    ),
                )

        output_df = (
            pd.DataFrame(
                outputs
            )
        )

        if not output_df.empty:

            output_df = (
                output_df
                .sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        return output_df

    # ========================================================
    # CURRENT RATINGS
    # ========================================================

    def current_ratings(
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

                    "adjusted_offense_rating": (
                        state.offense_rating
                    ),

                    "adjusted_defense_rating": (
                        state.defense_rating
                    ),

                    "adjusted_overall_rating": (
                        (
                            state.offense_rating
                            +
                            state.defense_rating
                        )
                        /
                        2.0
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
                [
                    "adjusted_overall_rating",
                    "adjusted_offense_rating",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )