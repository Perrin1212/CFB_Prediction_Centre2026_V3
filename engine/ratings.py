from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class RatingConfig:
    """
    Configuration for the raw chronological offence and defence
    rating engine.

    Rating interpretation:

        100 = approximately national average
        >100 = above average
        <100 = below average

    These ratings are deliberately NOT opponent-adjusted.

    Opponent adjustment happens later in the V2 pipeline.
    """

    # How quickly a team's underlying performance estimate
    # responds to a new game.
    update_rate: float = 0.25

    # At the start of a new season, regress this proportion
    # of the previous season's state back toward the national
    # baseline.
    preseason_regression: float = 0.40

    # Minimum / maximum individual component ratios.
    ratio_floor: float = 0.50
    ratio_ceiling: float = 1.50

    # Final rating range.
    rating_floor: float = 50.0
    rating_ceiling: float = 150.0

    # --------------------------------------------------------
    # Offensive composite weights
    # --------------------------------------------------------

    offense_points_weight: float = 0.40
    offense_ypp_weight: float = 0.35
    offense_turnover_weight: float = 0.15
    offense_first_down_weight: float = 0.10

    # --------------------------------------------------------
    # Defensive composite weights
    # --------------------------------------------------------

    defense_points_weight: float = 0.40
    defense_ypp_weight: float = 0.35
    defense_takeaway_weight: float = 0.15
    defense_first_down_weight: float = 0.10

    # --------------------------------------------------------
    # Starting national reference values
    #
    # These are only startup anchors.
    #
    # Once historical games start being processed, the engine
    # maintains its own chronological national baselines.
    # --------------------------------------------------------

    starting_points: float = 26.5
    starting_yards_per_play: float = 5.50
    starting_turnovers: float = 1.50
    starting_first_downs: float = 20.0

    # Amount of prior pseudo-observations represented by the
    # startup national baseline.
    national_prior_games: float = 50.0


# ============================================================
# TEAM STATE
# ============================================================


@dataclass
class TeamRatingState:
    """
    Current underlying raw football state for one team.

    Offence fields represent what this team's offence tends
    to produce.

    Defence fields represent what opponents tend to produce
    against this team's defence.
    """

    team: str

    games_played: int = 0

    # Offence
    offense_points: float = 26.5
    offense_ypp: float = 5.50
    offense_turnovers: float = 1.50
    offense_first_downs: float = 20.0

    # Defence / opponent output allowed
    defense_points_allowed: float = 26.5
    defense_ypp_allowed: float = 5.50
    defense_takeaways: float = 1.50
    defense_first_downs_allowed: float = 20.0

    last_season: int | None = None


# ============================================================
# NATIONAL STATE
# ============================================================


@dataclass
class NationalState:
    """
    Chronological national football baseline.

    This state is updated only AFTER a game has been rated,
    ensuring the current game's result cannot leak into its
    own pregame features.
    """

    observations: float

    points_sum: float
    ypp_sum: float
    turnovers_sum: float
    first_downs_sum: float

    @property
    def points(self) -> float:

        if self.observations <= 0:
            return 26.5

        return (
            self.points_sum
            /
            self.observations
        )

    @property
    def ypp(self) -> float:

        if self.observations <= 0:
            return 5.50

        return (
            self.ypp_sum
            /
            self.observations
        )

    @property
    def turnovers(self) -> float:

        if self.observations <= 0:
            return 1.50

        return (
            self.turnovers_sum
            /
            self.observations
        )

    @property
    def first_downs(self) -> float:

        if self.observations <= 0:
            return 20.0

        return (
            self.first_downs_sum
            /
            self.observations
        )


# ============================================================
# RATINGS ENGINE
# ============================================================


class TeamRatingsEngine:
    """
    Leakage-safe chronological offence / defence ratings.

    Important architectural rule:

        Pregame rating
            ↓
        game happens
            ↓
        state updates

    The game being predicted therefore never contributes to
    its own pregame rating.

    This is the RAW ratings layer.

    Opponent strength is deliberately ignored here and is
    handled later in opponent_adjustment.py.
    """

    def __init__(
        self,
        config: RatingConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RatingConfig()
        )

        self.states: dict[
            str,
            TeamRatingState,
        ] = {}

        prior = float(
            self.config.national_prior_games
        )

        self.national = NationalState(
            observations=prior,

            points_sum=(
                prior
                *
                self.config.starting_points
            ),

            ypp_sum=(
                prior
                *
                self.config.starting_yards_per_play
            ),

            turnovers_sum=(
                prior
                *
                self.config.starting_turnovers
            ),

            first_downs_sum=(
                prior
                *
                self.config.starting_first_downs
            ),
        )

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
    def _normalise_name(
        value: Any,
    ) -> str:

        if pd.isna(value):
            return ""

        return str(value).strip()

    def _clip_ratio(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.ratio_floor,
                self.config.ratio_ceiling,
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

    def _new_team_state(
        self,
        team: str,
    ) -> TeamRatingState:

        return TeamRatingState(
            team=team,

            offense_points=(
                self.national.points
            ),

            offense_ypp=(
                self.national.ypp
            ),

            offense_turnovers=(
                self.national.turnovers
            ),

            offense_first_downs=(
                self.national.first_downs
            ),

            defense_points_allowed=(
                self.national.points
            ),

            defense_ypp_allowed=(
                self.national.ypp
            ),

            defense_takeaways=(
                self.national.turnovers
            ),

            defense_first_downs_allowed=(
                self.national.first_downs
            ),
        )

    def _get_state(
        self,
        team: str,
    ) -> TeamRatingState:

        if team not in self.states:

            self.states[
                team
            ] = self._new_team_state(
                team
            )

        return self.states[
            team
        ]

    # ========================================================
    # SEASON REGRESSION
    # ========================================================

    def _apply_season_regression(
        self,
        state: TeamRatingState,
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

        # ----------------------------------------------------
        # Offence regresses toward current national baseline
        # ----------------------------------------------------

        state.offense_points = (
            keep * state.offense_points
            +
            regression * self.national.points
        )

        state.offense_ypp = (
            keep * state.offense_ypp
            +
            regression * self.national.ypp
        )

        state.offense_turnovers = (
            keep * state.offense_turnovers
            +
            regression * self.national.turnovers
        )

        state.offense_first_downs = (
            keep * state.offense_first_downs
            +
            regression * self.national.first_downs
        )

        # ----------------------------------------------------
        # Defence regresses toward current national baseline
        # ----------------------------------------------------

        state.defense_points_allowed = (
            keep
            *
            state.defense_points_allowed
            +
            regression
            *
            self.national.points
        )

        state.defense_ypp_allowed = (
            keep
            *
            state.defense_ypp_allowed
            +
            regression
            *
            self.national.ypp
        )

        state.defense_takeaways = (
            keep
            *
            state.defense_takeaways
            +
            regression
            *
            self.national.turnovers
        )

        state.defense_first_downs_allowed = (
            keep
            *
            state.defense_first_downs_allowed
            +
            regression
            *
            self.national.first_downs
        )

        state.last_season = season

    # ========================================================
    # OFFENCE RATING
    # ========================================================

    def offense_rating(
        self,
        state: TeamRatingState,
    ) -> float:

        national_points = max(
            self.national.points,
            0.01,
        )

        national_ypp = max(
            self.national.ypp,
            0.01,
        )

        national_turnovers = max(
            self.national.turnovers,
            0.01,
        )

        national_first_downs = max(
            self.national.first_downs,
            0.01,
        )

        # Higher scoring is good.
        points_ratio = self._clip_ratio(
            state.offense_points
            /
            national_points
        )

        # Higher YPP is good.
        ypp_ratio = self._clip_ratio(
            state.offense_ypp
            /
            national_ypp
        )

        # Lower turnovers is good.
        turnovers = max(
            state.offense_turnovers,
            0.20,
        )

        turnover_ratio = self._clip_ratio(
            national_turnovers
            /
            turnovers
        )

        # More first downs is good.
        first_down_ratio = self._clip_ratio(
            state.offense_first_downs
            /
            national_first_downs
        )

        weighted_ratio = (
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
            weighted_ratio
            *
            100.0
        )

    # ========================================================
    # DEFENCE RATING
    # ========================================================

    def defense_rating(
        self,
        state: TeamRatingState,
    ) -> float:

        national_points = max(
            self.national.points,
            0.01,
        )

        national_ypp = max(
            self.national.ypp,
            0.01,
        )

        national_turnovers = max(
            self.national.turnovers,
            0.01,
        )

        national_first_downs = max(
            self.national.first_downs,
            0.01,
        )

        points_allowed = max(
            state.defense_points_allowed,
            1.0,
        )

        ypp_allowed = max(
            state.defense_ypp_allowed,
            1.0,
        )

        first_downs_allowed = max(
            state.defense_first_downs_allowed,
            1.0,
        )

        # Lower points allowed = better defence.
        points_ratio = self._clip_ratio(
            national_points
            /
            points_allowed
        )

        # Lower YPP allowed = better defence.
        ypp_ratio = self._clip_ratio(
            national_ypp
            /
            ypp_allowed
        )

        # More takeaways = better defence.
        takeaway_ratio = self._clip_ratio(
            state.defense_takeaways
            /
            national_turnovers
        )

        # Fewer first downs allowed = better defence.
        first_down_ratio = self._clip_ratio(
            national_first_downs
            /
            first_downs_allowed
        )

        weighted_ratio = (
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
            weighted_ratio
            *
            100.0
        )

    # ========================================================
    # PREGAME SNAPSHOT
    # ========================================================

    def team_snapshot(
        self,
        team: str,
        season: int,
    ) -> dict[str, float | int | str]:

        state = self._get_state(
            team
        )

        self._apply_season_regression(
            state,
            season,
        )

        return {
            "team": team,
            "games_pre": state.games_played,

            "offense_rating_pre": (
                self.offense_rating(
                    state
                )
            ),

            "defense_rating_pre": (
                self.defense_rating(
                    state
                )
            ),

            "offense_points_pre": (
                state.offense_points
            ),

            "offense_ypp_pre": (
                state.offense_ypp
            ),

            "offense_turnovers_pre": (
                state.offense_turnovers
            ),

            "offense_first_downs_pre": (
                state.offense_first_downs
            ),

            "defense_points_allowed_pre": (
                state.defense_points_allowed
            ),

            "defense_ypp_allowed_pre": (
                state.defense_ypp_allowed
            ),

            "defense_takeaways_pre": (
                state.defense_takeaways
            ),

            "defense_first_downs_allowed_pre": (
                state.defense_first_downs_allowed
            ),

            "national_points_pre": (
                self.national.points
            ),

            "national_ypp_pre": (
                self.national.ypp
            ),

            "national_turnovers_pre": (
                self.national.turnovers
            ),

            "national_first_downs_pre": (
                self.national.first_downs
            ),
        }

    # ========================================================
    # UPDATE TEAM STATE
    # ========================================================

    def update_team(
        self,
        team: str,
        season: int,
        points: float,
        yards_per_play: float,
        turnovers: float,
        first_downs: float,
        opponent_points: float,
        opponent_yards_per_play: float,
        opponent_turnovers: float,
        opponent_first_downs: float,
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

        # ----------------------------------------------------
        # OFFENCE
        # ----------------------------------------------------

        state.offense_points = self._ewma(
            state.offense_points,
            points,
            rate,
        )

        state.offense_ypp = self._ewma(
            state.offense_ypp,
            yards_per_play,
            rate,
        )

        state.offense_turnovers = self._ewma(
            state.offense_turnovers,
            turnovers,
            rate,
        )

        state.offense_first_downs = self._ewma(
            state.offense_first_downs,
            first_downs,
            rate,
        )

        # ----------------------------------------------------
        # DEFENCE
        #
        # Opponent offensive production becomes this team's
        # defensive output allowed.
        # ----------------------------------------------------

        state.defense_points_allowed = self._ewma(
            state.defense_points_allowed,
            opponent_points,
            rate,
        )

        state.defense_ypp_allowed = self._ewma(
            state.defense_ypp_allowed,
            opponent_yards_per_play,
            rate,
        )

        state.defense_takeaways = self._ewma(
            state.defense_takeaways,
            opponent_turnovers,
            rate,
        )

        state.defense_first_downs_allowed = self._ewma(
            state.defense_first_downs_allowed,
            opponent_first_downs,
            rate,
        )

        state.games_played += 1
        state.last_season = season

    # ========================================================
    # NATIONAL UPDATE
    # ========================================================

    def update_national(
        self,
        points: float,
        yards_per_play: float,
        turnovers: float,
        first_downs: float,
    ) -> None:

        self.national.observations += 1.0

        self.national.points_sum += points
        self.national.ypp_sum += yards_per_play
        self.national.turnovers_sum += turnovers
        self.national.first_downs_sum += first_downs

    # ========================================================
    # PROCESS HISTORICAL GAMES
    # ========================================================

    def process_games(
        self,
        team_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process one row per team per game.

        Expected columns:

            game_id
            season
            week
            start_date
            team_name
            opponent_name
            team_points
            opponent_points
            yards_per_play_proxy
            turnovers
            first_downs

        Two rows should exist for each game.

        Pregame snapshots are created for BOTH teams before
        either team's state is updated.
        """

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
            if column not in team_games.columns
        ]

        if missing:

            raise ValueError(
                "Team ratings input is missing required "
                f"columns: {missing}"
            )

        data = team_games.copy()

        # ----------------------------------------------------
        # Clean basic fields
        # ----------------------------------------------------

        data["team_name"] = (
            data["team_name"]
            .map(
                self._normalise_name
            )
        )

        data["opponent_name"] = (
            data["opponent_name"]
            .map(
                self._normalise_name
            )
        )

        numeric_columns = [
            "season",
            "week",
            "team_points",
            "opponent_points",
            "yards_per_play_proxy",
            "turnovers",
            "first_downs",
        ]

        for column in numeric_columns:

            if column in data.columns:

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Chronological sort
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
        # Process game-by-game.
        #
        # This guarantees neither side is updated before the
        # other side receives its pregame snapshot.
        # ----------------------------------------------------

        for game_id, game_rows in data.groupby(
            "game_id",
            sort=False,
        ):

            game_rows = game_rows.copy()

            if len(game_rows) != 2:
                continue

            first = game_rows.iloc[0]
            second = game_rows.iloc[1]

            season = int(
                first["season"]
            )

            team_a = str(
                first["team_name"]
            )

            team_b = str(
                second["team_name"]
            )

            if (
                not team_a
                or
                not team_b
            ):
                continue

            # ------------------------------------------------
            # PREGAME SNAPSHOTS
            # ------------------------------------------------

            snapshots: dict[
                str,
                dict[str, Any],
            ] = {}

            for row in (
                first,
                second,
            ):

                team = str(
                    row["team_name"]
                )

                snapshots[
                    team
                ] = self.team_snapshot(
                    team=team,
                    season=season,
                )

            # ------------------------------------------------
            # OUTPUT PREGAME ROWS
            # ------------------------------------------------

            for _, row in game_rows.iterrows():

                team = str(
                    row["team_name"]
                )

                opponent = str(
                    row["opponent_name"]
                )

                snapshot = snapshots[
                    team
                ]

                opponent_snapshot = snapshots.get(
                    opponent
                )

                if opponent_snapshot is None:

                    opponent_snapshot = self.team_snapshot(
                        team=opponent,
                        season=season,
                    )

                output = row.to_dict()

                # Team ratings
                output[
                    "team_offense_rating_pre"
                ] = snapshot[
                    "offense_rating_pre"
                ]

                output[
                    "team_defense_rating_pre"
                ] = snapshot[
                    "defense_rating_pre"
                ]

                # Opponent ratings
                output[
                    "opponent_offense_rating_pre"
                ] = opponent_snapshot[
                    "offense_rating_pre"
                ]

                output[
                    "opponent_defense_rating_pre"
                ] = opponent_snapshot[
                    "defense_rating_pre"
                ]

                # Detailed team state
                for key, value in snapshot.items():

                    if key in {
                        "team",
                        "offense_rating_pre",
                        "defense_rating_pre",
                    }:
                        continue

                    output[
                        f"rating_state_{key}"
                    ] = value

                outputs.append(
                    output
                )

            # ------------------------------------------------
            # READ ACTUAL GAME PERFORMANCE
            # ------------------------------------------------

            a_points = self._safe_float(
                first["team_points"],
                self.national.points,
            )

            b_points = self._safe_float(
                second["team_points"],
                self.national.points,
            )

            a_ypp = self._safe_float(
                first["yards_per_play_proxy"],
                self.national.ypp,
            )

            b_ypp = self._safe_float(
                second["yards_per_play_proxy"],
                self.national.ypp,
            )

            a_turnovers = self._safe_float(
                first["turnovers"],
                self.national.turnovers,
            )

            b_turnovers = self._safe_float(
                second["turnovers"],
                self.national.turnovers,
            )

            a_first_downs = self._safe_float(
                first["first_downs"],
                self.national.first_downs,
            )

            b_first_downs = self._safe_float(
                second["first_downs"],
                self.national.first_downs,
            )

            # ------------------------------------------------
            # UPDATE BOTH TEAM STATES
            # ------------------------------------------------

            self.update_team(
                team=team_a,
                season=season,

                points=a_points,
                yards_per_play=a_ypp,
                turnovers=a_turnovers,
                first_downs=a_first_downs,

                opponent_points=b_points,
                opponent_yards_per_play=b_ypp,
                opponent_turnovers=b_turnovers,
                opponent_first_downs=b_first_downs,
            )

            self.update_team(
                team=team_b,
                season=season,

                points=b_points,
                yards_per_play=b_ypp,
                turnovers=b_turnovers,
                first_downs=b_first_downs,

                opponent_points=a_points,
                opponent_yards_per_play=a_ypp,
                opponent_turnovers=a_turnovers,
                opponent_first_downs=a_first_downs,
            )

            # ------------------------------------------------
            # NATIONAL BASELINE UPDATE
            #
            # Done AFTER pregame snapshots.
            # ------------------------------------------------

            self.update_national(
                points=a_points,
                yards_per_play=a_ypp,
                turnovers=a_turnovers,
                first_downs=a_first_downs,
            )

            self.update_national(
                points=b_points,
                yards_per_play=b_ypp,
                turnovers=b_turnovers,
                first_downs=b_first_downs,
            )

        result = pd.DataFrame(
            outputs
        )

        if not result.empty:

            result = (
                result
                .sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        return result

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

                    "offense_rating": (
                        self.offense_rating(
                            state
                        )
                    ),

                    "defense_rating": (
                        self.defense_rating(
                            state
                        )
                    ),

                    "offense_points": (
                        state.offense_points
                    ),

                    "offense_ypp": (
                        state.offense_ypp
                    ),

                    "offense_turnovers": (
                        state.offense_turnovers
                    ),

                    "offense_first_downs": (
                        state.offense_first_downs
                    ),

                    "defense_points_allowed": (
                        state.defense_points_allowed
                    ),

                    "defense_ypp_allowed": (
                        state.defense_ypp_allowed
                    ),

                    "defense_takeaways": (
                        state.defense_takeaways
                    ),

                    "defense_first_downs_allowed": (
                        state.defense_first_downs_allowed
                    ),

                    "overall_raw_rating": (
                        (
                            self.offense_rating(
                                state
                            )
                            +
                            self.defense_rating(
                                state
                            )
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
                    "overall_raw_rating",
                    "offense_rating",
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