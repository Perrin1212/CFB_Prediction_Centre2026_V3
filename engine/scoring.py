from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ScoringConfig:
    """
    Chronological scoring-efficiency model.

    Historical truth used by this layer:

        points scored
        offensive plays
        points allowed
        opponent offensive plays

    We deliberately learn POINTS PER PLAY rather than points
    per drive because true historical drive counts are not
    currently available.

    Expected points per estimated drive is then derived from:

        expected points/play
            ×
        expected plays/drive
    """

    update_rate: float = 0.25
    preseason_regression: float = 0.40

    # Approximate national starting environment.
    starting_points_per_play: float = 0.39

    national_prior_team_games: float = 100.0

    # Scoring-efficiency matchup blend.
    offense_scoring_weight: float = 0.55
    defense_scoring_weight: float = 0.45

    # How strongly the opponent-adjusted football matchup
    # rating modifies scoring efficiency.
    matchup_factor_weight: float = 0.35

    # Modest explicit home scoring context.
    #
    # This is not a win-probability home-field advantage.
    # It only modifies expected scoring environment.
    home_scoring_multiplier: float = 1.025
    away_scoring_multiplier: float = 0.985

    # Neutral games receive neither.
    neutral_scoring_multiplier: float = 1.0

    # Safety bounds
    minimum_points_per_play: float = 0.10
    maximum_points_per_play: float = 0.85

    minimum_points_per_drive: float = 0.50
    maximum_points_per_drive: float = 5.50

    minimum_expected_points: float = 6.0
    maximum_expected_points: float = 65.0

    matchup_factor_floor: float = 0.75
    matchup_factor_ceiling: float = 1.25


# ============================================================
# TEAM STATE
# ============================================================


@dataclass
class TeamScoringState:

    team: str

    games_played: int = 0

    offense_points_per_play: float = 0.39
    defense_points_per_play_allowed: float = 0.39

    last_season: int | None = None


# ============================================================
# NATIONAL STATE
# ============================================================


@dataclass
class NationalScoringState:

    team_games_seen: float = 100.0
    total_points_per_play: float = 39.0

    @property
    def points_per_play(self) -> float:

        if self.team_games_seen <= 0:
            return 0.39

        return (
            self.total_points_per_play
            /
            self.team_games_seen
        )


# ============================================================
# ENGINE
# ============================================================


class ScoringModel:
    """
    Chronological scoring-efficiency model.

    For each game:

        team offensive scoring efficiency
                    +
        opponent defensive scoring resistance
                    +
        opponent-adjusted matchup quality
                    +
        game environment / expected plays
                    +
        estimated drive structure
                    ↓
        expected points/play
                    ↓
        expected points/estimated drive
                    ↓
        expected points

    Actual game scoring is used only AFTER pregame features
    have been produced.
    """

    def __init__(
        self,
        config: ScoringConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ScoringConfig()
        )

        self.states: dict[
            str,
            TeamScoringState,
        ] = {}

        prior = (
            self.config.national_prior_team_games
        )

        self.national = NationalScoringState(
            team_games_seen=prior,
            total_points_per_play=(
                prior
                *
                self.config.starting_points_per_play
            ),
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float,
    ) -> float:

        try:

            value = float(value)

            if not np.isfinite(value):
                return float(fallback)

            return value

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

            value = float(value)

            if not np.isfinite(value):
                return int(fallback)

            return int(value)

        except (
            TypeError,
            ValueError,
        ):

            return int(fallback)

    @staticmethod
    def _team_name(
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
            old
            *
            (1.0 - rate)
            +
            new
            *
            rate
        )

    # ========================================================
    # TEAM STATE
    # ========================================================

    def _get_state(
        self,
        team: str,
    ) -> TeamScoringState:

        if team not in self.states:

            self.states[
                team
            ] = TeamScoringState(
                team=team,

                offense_points_per_play=(
                    self.config.starting_points_per_play
                ),

                defense_points_per_play_allowed=(
                    self.config.starting_points_per_play
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
        state: TeamScoringState,
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

        national_ppp = (
            self.national.points_per_play
        )

        state.offense_points_per_play = (
            keep
            *
            state.offense_points_per_play
            +
            regression
            *
            national_ppp
        )

        state.defense_points_per_play_allowed = (
            keep
            *
            state.defense_points_per_play_allowed
            +
            regression
            *
            national_ppp
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

            "scoring_games_pre": (
                state.games_played
            ),

            "offense_points_per_play_pre": (
                state.offense_points_per_play
            ),

            "defense_points_per_play_allowed_pre": (
                state.defense_points_per_play_allowed
            ),

            "national_points_per_play_pre": (
                self.national.points_per_play
            ),
        }

    # ========================================================
    # MATCHUP FACTOR
    # ========================================================

    def scoring_matchup_factor(
        self,
        offensive_matchup_index: float,
    ) -> float:
        """
        Translate matchup index into a modest scoring modifier.

        Example with weight 0.35:

            matchup 120

        raw advantage = +20%

        scoring modifier ≈ +7%

        This prevents the matchup rating from overwhelming the
        directly observed scoring-efficiency history.
        """

        matchup = (
            self._safe_float(
                offensive_matchup_index,
                100.0,
            )
            /
            100.0
        )

        factor = (
            1.0
            +
            self.config.matchup_factor_weight
            *
            (
                matchup
                -
                1.0
            )
        )

        return float(
            np.clip(
                factor,
                self.config.matchup_factor_floor,
                self.config.matchup_factor_ceiling,
            )
        )

    # ========================================================
    # EXPECTED POINTS PER PLAY
    # ========================================================

    def expected_points_per_play(
        self,
        offense_snapshot: dict[str, Any],
        defense_snapshot: dict[str, Any],
        offensive_matchup_index: float,
        location_multiplier: float = 1.0,
    ) -> float:

        offense_ppp = max(
            self._safe_float(
                offense_snapshot[
                    "offense_points_per_play_pre"
                ],
                self.config.starting_points_per_play,
            ),
            0.001,
        )

        defense_ppp_allowed = max(
            self._safe_float(
                defense_snapshot[
                    "defense_points_per_play_allowed_pre"
                ],
                self.config.starting_points_per_play,
            ),
            0.001,
        )

        offense_weight = (
            self.config.offense_scoring_weight
        )

        defense_weight = (
            self.config.defense_scoring_weight
        )

        denominator = (
            offense_weight
            +
            defense_weight
        )

        if denominator <= 0:
            denominator = 1.0

        base_ppp = (
            offense_weight
            *
            offense_ppp
            +
            defense_weight
            *
            defense_ppp_allowed
        ) / denominator

        matchup_factor = (
            self.scoring_matchup_factor(
                offensive_matchup_index
            )
        )

        expected = (
            base_ppp
            *
            matchup_factor
            *
            location_multiplier
        )

        return float(
            np.clip(
                expected,
                self.config.minimum_points_per_play,
                self.config.maximum_points_per_play,
            )
        )

    # ========================================================
    # EXPECTED POINTS / ESTIMATED DRIVE
    # ========================================================

    def expected_points_per_drive(
        self,
        expected_points_per_play: float,
        expected_plays_per_drive: float,
    ) -> float:

        points_per_play = (
            self._safe_float(
                expected_points_per_play,
                self.config.starting_points_per_play,
            )
        )

        plays_per_drive = (
            self._safe_float(
                expected_plays_per_drive,
                6.20,
            )
        )

        expected = (
            points_per_play
            *
            plays_per_drive
        )

        return float(
            np.clip(
                expected,
                self.config.minimum_points_per_drive,
                self.config.maximum_points_per_drive,
            )
        )

    # ========================================================
    # EXPECTED POINTS
    # ========================================================

    def expected_team_points(
        self,
        expected_points_per_drive: float,
        expected_drives: float,
    ) -> float:

        ppd = (
            self._safe_float(
                expected_points_per_drive,
                2.4,
            )
        )

        drives = (
            self._safe_float(
                expected_drives,
                10.5,
            )
        )

        expected = (
            ppd
            *
            drives
        )

        return float(
            np.clip(
                expected,
                self.config.minimum_expected_points,
                self.config.maximum_expected_points,
            )
        )

    # ========================================================
    # LOCATION CONTEXT
    # ========================================================

    def location_multipliers(
        self,
        neutral_site: Any,
    ) -> tuple[
        float,
        float,
    ]:

        neutral = False

        if isinstance(
            neutral_site,
            bool,
        ):

            neutral = neutral_site

        elif isinstance(
            neutral_site,
            (int, float),
        ):

            neutral = bool(
                neutral_site
            )

        elif isinstance(
            neutral_site,
            str,
        ):

            neutral = (
                neutral_site
                .strip()
                .lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                }
            )

        if neutral:

            return (
                self.config.neutral_scoring_multiplier,
                self.config.neutral_scoring_multiplier,
            )

        return (
            self.config.home_scoring_multiplier,
            self.config.away_scoring_multiplier,
        )

    # ========================================================
    # ACTUAL SCORING RATE
    # ========================================================

    def actual_points_per_play(
        self,
        row: pd.Series,
    ) -> tuple[
        float,
        float,
    ]:

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

        plays = max(
            rushing_attempts
            +
            passing_attempts,
            1.0,
        )

        points = max(
            self._safe_float(
                row.get(
                    "team_points"
                ),
                0.0,
            ),
            0.0,
        )

        points_per_play = (
            points
            /
            plays
        )

        return (
            plays,
            points_per_play,
        )

    # ========================================================
    # UPDATE TEAM
    # ========================================================

    def update_team(
        self,
        team: str,
        season: int,
        offense_points_per_play: float,
        defense_points_per_play_allowed: float,
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

        state.offense_points_per_play = (
            self._ewma(
                state.offense_points_per_play,
                offense_points_per_play,
                rate,
            )
        )

        state.defense_points_per_play_allowed = (
            self._ewma(
                state.defense_points_per_play_allowed,
                defense_points_per_play_allowed,
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
        points_per_play: float,
    ) -> None:

        self.national.team_games_seen += 1.0

        self.national.total_points_per_play += (
            points_per_play
        )

    # ========================================================
    # PROCESS HISTORY
    # ========================================================

    def process_history(
        self,
        team_history: pd.DataFrame,
        possession_history: pd.DataFrame,
    ) -> pd.DataFrame:

        required_team_columns = [
            "game_id",
            "season",
            "week",

            "team_name",
            "home_away",

            "team_points",

            "rushing_attempts",
            "passing_attempts",
        ]

        missing_team = [
            column
            for column in required_team_columns
            if column not in team_history.columns
        ]

        if missing_team:

            raise ValueError(
                "Team history is missing scoring "
                f"fields: {missing_team}"
            )

        required_possession_columns = [
            "game_id",

            "home_offensive_matchup_index",
            "away_offensive_matchup_index",

            "expected_home_plays",
            "expected_away_plays",

            "expected_home_plays_per_drive",
            "expected_away_plays_per_drive",

            "expected_home_drives",
            "expected_away_drives",
        ]

        missing_possession = [
            column
            for column in required_possession_columns
            if column not in possession_history.columns
        ]

        if missing_possession:

            raise ValueError(
                "Possession history is missing scoring "
                f"inputs: {missing_possession}"
            )

        data = (
            team_history.copy()
        )

        possession = (
            possession_history.copy()
        )

        data["game_id"] = pd.to_numeric(
            data["game_id"],
            errors="coerce",
        ).astype("Int64")

        possession["game_id"] = pd.to_numeric(
            possession["game_id"],
            errors="coerce",
        ).astype("Int64")

        data["home_away"] = (
            data["home_away"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # ----------------------------------------------------
        # CHRONOLOGICAL SORT
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

        possession_lookup = (
            possession
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

            if game_id not in possession_lookup.index:
                continue

            home_row = (
                home_rows.iloc[0]
            )

            away_row = (
                away_rows.iloc[0]
            )

            possession_row = (
                possession_lookup.loc[
                    game_id
                ]
            )

            if isinstance(
                possession_row,
                pd.DataFrame,
            ):

                possession_row = (
                    possession_row.iloc[0]
                )

            season = self._safe_int(
                home_row.get(
                    "season"
                ),
                0,
            )

            home_team = self._team_name(
                home_row.get(
                    "team_name"
                )
            )

            away_team = self._team_name(
                away_row.get(
                    "team_name"
                )
            )

            # ------------------------------------------------
            # PRE-GAME SCORING SNAPSHOTS
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
            # LOCATION
            # ------------------------------------------------

            (
                home_location_multiplier,
                away_location_multiplier,
            ) = self.location_multipliers(
                possession_row.get(
                    "neutral_site",
                    False,
                )
            )

            # ------------------------------------------------
            # EXPECTED PPP
            # ------------------------------------------------

            home_expected_ppp = (
                self.expected_points_per_play(
                    offense_snapshot=home_snapshot,
                    defense_snapshot=away_snapshot,

                    offensive_matchup_index=(
                        possession_row.get(
                            "home_offensive_matchup_index",
                            100.0,
                        )
                    ),

                    location_multiplier=(
                        home_location_multiplier
                    ),
                )
            )

            away_expected_ppp = (
                self.expected_points_per_play(
                    offense_snapshot=away_snapshot,
                    defense_snapshot=home_snapshot,

                    offensive_matchup_index=(
                        possession_row.get(
                            "away_offensive_matchup_index",
                            100.0,
                        )
                    ),

                    location_multiplier=(
                        away_location_multiplier
                    ),
                )
            )

            # ------------------------------------------------
            # EXPECTED POINTS PER ESTIMATED DRIVE
            # ------------------------------------------------

            home_expected_ppd = (
                self.expected_points_per_drive(
                    expected_points_per_play=(
                        home_expected_ppp
                    ),

                    expected_plays_per_drive=(
                        possession_row.get(
                            "expected_home_plays_per_drive",
                            6.20,
                        )
                    ),
                )
            )

            away_expected_ppd = (
                self.expected_points_per_drive(
                    expected_points_per_play=(
                        away_expected_ppp
                    ),

                    expected_plays_per_drive=(
                        possession_row.get(
                            "expected_away_plays_per_drive",
                            6.20,
                        )
                    ),
                )
            )

            # ------------------------------------------------
            # EXPECTED POINTS
            # ------------------------------------------------

            expected_home_points = (
                self.expected_team_points(
                    expected_points_per_drive=(
                        home_expected_ppd
                    ),

                    expected_drives=(
                        possession_row.get(
                            "expected_home_drives",
                            10.5,
                        )
                    ),
                )
            )

            expected_away_points = (
                self.expected_team_points(
                    expected_points_per_drive=(
                        away_expected_ppd
                    ),

                    expected_drives=(
                        possession_row.get(
                            "expected_away_drives",
                            10.5,
                        )
                    ),
                )
            )

            expected_total_points = (
                expected_home_points
                +
                expected_away_points
            )

            expected_home_margin = (
                expected_home_points
                -
                expected_away_points
            )

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            output = (
                possession_row.to_dict()
            )

            output.update(
                {
                    # ----------------------------------------
                    # SCORING STATES
                    # ----------------------------------------

                    "home_scoring_games_pre": (
                        home_snapshot[
                            "scoring_games_pre"
                        ]
                    ),

                    "away_scoring_games_pre": (
                        away_snapshot[
                            "scoring_games_pre"
                        ]
                    ),

                    "home_offense_points_per_play_pre": (
                        home_snapshot[
                            "offense_points_per_play_pre"
                        ]
                    ),

                    "away_offense_points_per_play_pre": (
                        away_snapshot[
                            "offense_points_per_play_pre"
                        ]
                    ),

                    "home_defense_points_per_play_allowed_pre": (
                        home_snapshot[
                            "defense_points_per_play_allowed_pre"
                        ]
                    ),

                    "away_defense_points_per_play_allowed_pre": (
                        away_snapshot[
                            "defense_points_per_play_allowed_pre"
                        ]
                    ),

                    "national_points_per_play_pre": (
                        self.national.points_per_play
                    ),

                    # ----------------------------------------
                    # CONTEXT
                    # ----------------------------------------

                    "home_scoring_location_multiplier": (
                        home_location_multiplier
                    ),

                    "away_scoring_location_multiplier": (
                        away_location_multiplier
                    ),

                    "home_scoring_matchup_factor": (
                        self.scoring_matchup_factor(
                            possession_row.get(
                                "home_offensive_matchup_index",
                                100.0,
                            )
                        )
                    ),

                    "away_scoring_matchup_factor": (
                        self.scoring_matchup_factor(
                            possession_row.get(
                                "away_offensive_matchup_index",
                                100.0,
                            )
                        )
                    ),

                    # ----------------------------------------
                    # SCORING EFFICIENCY
                    # ----------------------------------------

                    "expected_home_points_per_play": (
                        home_expected_ppp
                    ),

                    "expected_away_points_per_play": (
                        away_expected_ppp
                    ),

                    "expected_home_points_per_drive": (
                        home_expected_ppd
                    ),

                    "expected_away_points_per_drive": (
                        away_expected_ppd
                    ),

                    # ----------------------------------------
                    # SCORE EXPECTATION
                    # ----------------------------------------

                    "expected_home_points": (
                        expected_home_points
                    ),

                    "expected_away_points": (
                        expected_away_points
                    ),

                    "expected_total_points": (
                        expected_total_points
                    ),

                    "expected_home_margin": (
                        expected_home_margin
                    ),
                }
            )

            outputs.append(
                output
            )

            # ------------------------------------------------
            # ACTUAL GAME SCORING
            #
            # Only calculated after pregame output is created.
            # ------------------------------------------------

            (
                home_actual_plays,
                home_actual_ppp,
            ) = self.actual_points_per_play(
                home_row
            )

            (
                away_actual_plays,
                away_actual_ppp,
            ) = self.actual_points_per_play(
                away_row
            )

            # ------------------------------------------------
            # TEAM UPDATES
            # ------------------------------------------------

            self.update_team(
                team=home_team,
                season=season,

                offense_points_per_play=(
                    home_actual_ppp
                ),

                defense_points_per_play_allowed=(
                    away_actual_ppp
                ),
            )

            self.update_team(
                team=away_team,
                season=season,

                offense_points_per_play=(
                    away_actual_ppp
                ),

                defense_points_per_play_allowed=(
                    home_actual_ppp
                ),
            )

            # ------------------------------------------------
            # NATIONAL UPDATE
            # ------------------------------------------------

            self.update_national(
                home_actual_ppp
            )

            self.update_national(
                away_actual_ppp
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

    def current_states(
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

                    "offense_points_per_play": (
                        state.offense_points_per_play
                    ),

                    "defense_points_per_play_allowed": (
                        state.defense_points_per_play_allowed
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
                "offense_points_per_play",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )