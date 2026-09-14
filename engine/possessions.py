from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class PossessionConfig:
    """
    Chronological possession / drive-opportunity model.

    IMPORTANT
    ---------
    We do not currently possess true historical drive counts.

    Therefore:

        estimated_drives

    is a latent football opportunity estimate derived from:

        expected play volume
        historical first-down rate
        turnover rate
        opponent resistance
        expected possession share

    It must NOT be interpreted as observed drive data.
    """

    update_rate: float = 0.25
    preseason_regression: float = 0.40

    # National priors
    starting_first_down_rate: float = 0.30
    starting_turnover_rate: float = 0.022

    # Typical offensive plays per possession/drive.
    starting_plays_per_drive: float = 6.20

    national_prior_team_games: float = 100.0

    # Sustainability blend
    offense_sustain_weight: float = 0.55
    defense_resistance_weight: float = 0.45

    # Component weights within sustainability
    first_down_weight: float = 0.70
    turnover_weight: float = 0.30

    # Bounds
    minimum_plays_per_drive: float = 4.50
    maximum_plays_per_drive: float = 8.50

    minimum_estimated_drives: float = 7.0
    maximum_estimated_drives: float = 16.0

    sustainability_floor: float = 0.70
    sustainability_ceiling: float = 1.30

    # Possession share should only make a modest difference to
    # drive counts. Football possessions alternate heavily.
    possession_drive_adjustment: float = 1.00


# ============================================================
# TEAM STATE
# ============================================================


@dataclass
class TeamPossessionState:

    team: str

    games_played: int = 0

    offense_first_down_rate: float = 0.30
    offense_turnover_rate: float = 0.022

    defense_first_down_rate_allowed: float = 0.30
    defense_takeaway_rate: float = 0.022

    last_season: int | None = None


# ============================================================
# NATIONAL STATE
# ============================================================


@dataclass
class NationalPossessionState:

    team_games_seen: float = 100.0

    total_first_down_rate: float = 30.0
    total_turnover_rate: float = 2.2

    @property
    def first_down_rate(self) -> float:

        if self.team_games_seen <= 0:
            return 0.30

        return (
            self.total_first_down_rate
            /
            self.team_games_seen
        )

    @property
    def turnover_rate(self) -> float:

        if self.team_games_seen <= 0:
            return 0.022

        return (
            self.total_turnover_rate
            /
            self.team_games_seen
        )


# ============================================================
# ENGINE
# ============================================================


class PossessionModel:
    """
    Chronological latent possession / drive model.

    Pipeline:

        expected plays
             ↓
        offensive sustainability
             +
        opposing defensive resistance
             ↓
        expected plays per drive
             ↓
        estimated offensive drives

    No score prediction occurs here.
    """

    def __init__(
        self,
        config: PossessionConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else PossessionConfig()
        )

        self.states: dict[
            str,
            TeamPossessionState,
        ] = {}

        prior = (
            self.config.national_prior_team_games
        )

        self.national = NationalPossessionState(
            team_games_seen=prior,

            total_first_down_rate=(
                prior
                *
                self.config.starting_first_down_rate
            ),

            total_turnover_rate=(
                prior
                *
                self.config.starting_turnover_rate
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

    def _clip_sustainability(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.sustainability_floor,
                self.config.sustainability_ceiling,
            )
        )

    # ========================================================
    # TEAM STATE
    # ========================================================

    def _get_state(
        self,
        team: str,
    ) -> TeamPossessionState:

        if team not in self.states:

            self.states[
                team
            ] = TeamPossessionState(
                team=team,

                offense_first_down_rate=(
                    self.config.starting_first_down_rate
                ),

                offense_turnover_rate=(
                    self.config.starting_turnover_rate
                ),

                defense_first_down_rate_allowed=(
                    self.config.starting_first_down_rate
                ),

                defense_takeaway_rate=(
                    self.config.starting_turnover_rate
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
        state: TeamPossessionState,
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

        national_fd = (
            self.national.first_down_rate
        )

        national_to = (
            self.national.turnover_rate
        )

        state.offense_first_down_rate = (
            keep
            *
            state.offense_first_down_rate
            +
            regression
            *
            national_fd
        )

        state.offense_turnover_rate = (
            keep
            *
            state.offense_turnover_rate
            +
            regression
            *
            national_to
        )

        state.defense_first_down_rate_allowed = (
            keep
            *
            state.defense_first_down_rate_allowed
            +
            regression
            *
            national_fd
        )

        state.defense_takeaway_rate = (
            keep
            *
            state.defense_takeaway_rate
            +
            regression
            *
            national_to
        )

        state.last_season = season

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
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

            "possession_games_pre": (
                state.games_played
            ),

            "offense_first_down_rate_pre": (
                state.offense_first_down_rate
            ),

            "offense_turnover_rate_pre": (
                state.offense_turnover_rate
            ),

            "defense_first_down_rate_allowed_pre": (
                state.defense_first_down_rate_allowed
            ),

            "defense_takeaway_rate_pre": (
                state.defense_takeaway_rate
            ),

            "national_first_down_rate_pre": (
                self.national.first_down_rate
            ),

            "national_turnover_rate_pre": (
                self.national.turnover_rate
            ),
        }

    # ========================================================
    # SUSTAINABILITY RATINGS
    # ========================================================

    def offensive_sustainability(
        self,
        snapshot: dict[str, Any],
    ) -> float:

        national_fd = max(
            self._safe_float(
                snapshot[
                    "national_first_down_rate_pre"
                ],
                0.30,
            ),
            0.001,
        )

        national_to = max(
            self._safe_float(
                snapshot[
                    "national_turnover_rate_pre"
                ],
                0.022,
            ),
            0.001,
        )

        first_down_rate = max(
            self._safe_float(
                snapshot[
                    "offense_first_down_rate_pre"
                ],
                national_fd,
            ),
            0.001,
        )

        turnover_rate = max(
            self._safe_float(
                snapshot[
                    "offense_turnover_rate_pre"
                ],
                national_to,
            ),
            0.001,
        )

        first_down_component = (
            first_down_rate
            /
            national_fd
        )

        # Lower turnovers improve drive sustainability.
        turnover_component = (
            national_to
            /
            turnover_rate
        )

        sustainability = (
            self.config.first_down_weight
            *
            first_down_component
            +
            self.config.turnover_weight
            *
            turnover_component
        )

        return self._clip_sustainability(
            sustainability
        )

    def defensive_resistance(
        self,
        snapshot: dict[str, Any],
    ) -> float:
        """
        >1 means the defence tends to end opposing drives
        sooner than average.

        <1 means opponents sustain drives relatively easily.
        """

        national_fd = max(
            self._safe_float(
                snapshot[
                    "national_first_down_rate_pre"
                ],
                0.30,
            ),
            0.001,
        )

        national_to = max(
            self._safe_float(
                snapshot[
                    "national_turnover_rate_pre"
                ],
                0.022,
            ),
            0.001,
        )

        fd_allowed = max(
            self._safe_float(
                snapshot[
                    "defense_first_down_rate_allowed_pre"
                ],
                national_fd,
            ),
            0.001,
        )

        takeaway_rate = max(
            self._safe_float(
                snapshot[
                    "defense_takeaway_rate_pre"
                ],
                national_to,
            ),
            0.001,
        )

        first_down_component = (
            national_fd
            /
            fd_allowed
        )

        takeaway_component = (
            takeaway_rate
            /
            national_to
        )

        resistance = (
            self.config.first_down_weight
            *
            first_down_component
            +
            self.config.turnover_weight
            *
            takeaway_component
        )

        return self._clip_sustainability(
            resistance
        )

    # ========================================================
    # EXPECTED DRIVE LENGTH
    # ========================================================

    def expected_plays_per_drive(
        self,
        offense_snapshot: dict[str, Any],
        defense_snapshot: dict[str, Any],
    ) -> float:
        """
        Estimate effective plays per offensive possession.

        Strong drive sustainability increases plays/drive.

        Strong opposing defensive resistance decreases
        plays/drive.
        """

        offensive_sustain = (
            self.offensive_sustainability(
                offense_snapshot
            )
        )

        defensive_resistance = (
            self.defensive_resistance(
                defense_snapshot
            )
        )

        offense_weight = (
            self.config.offense_sustain_weight
        )

        defense_weight = (
            self.config.defense_resistance_weight
        )

        denominator = (
            offense_weight
            +
            defense_weight
        )

        if denominator <= 0:
            denominator = 1.0

        # Defence appears inversely:
        # strong defence reduces drive length.
        matchup_factor = (
            offense_weight
            *
            offensive_sustain
            +
            defense_weight
            *
            (
                1.0
                /
                max(
                    defensive_resistance,
                    0.01,
                )
            )
        ) / denominator

        plays_per_drive = (
            self.config.starting_plays_per_drive
            *
            matchup_factor
        )

        return float(
            np.clip(
                plays_per_drive,
                self.config.minimum_plays_per_drive,
                self.config.maximum_plays_per_drive,
            )
        )

    # ========================================================
    # DRIVE COUNT
    # ========================================================

    def estimate_drives(
        self,
        expected_plays: float,
        plays_per_drive: float,
    ) -> float:

        expected_plays = max(
            self._safe_float(
                expected_plays,
                68.0,
            ),
            1.0,
        )

        plays_per_drive = max(
            self._safe_float(
                plays_per_drive,
                self.config.starting_plays_per_drive,
            ),
            1.0,
        )

        drives = (
            expected_plays
            /
            plays_per_drive
        )

        return float(
            np.clip(
                drives,
                self.config.minimum_estimated_drives,
                self.config.maximum_estimated_drives,
            )
        )

    # ========================================================
    # ACTUAL RATE CALCULATION
    # ========================================================

    def actual_rates(
        self,
        row: pd.Series,
    ) -> tuple[
        float,
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

        first_downs = max(
            self._safe_float(
                row.get(
                    "first_downs"
                ),
                0.0,
            ),
            0.0,
        )

        turnovers = max(
            self._safe_float(
                row.get(
                    "turnovers"
                ),
                0.0,
            ),
            0.0,
        )

        first_down_rate = (
            first_downs
            /
            plays
        )

        turnover_rate = (
            turnovers
            /
            plays
        )

        return (
            plays,
            first_down_rate,
            turnover_rate,
        )

    # ========================================================
    # TEAM UPDATE
    # ========================================================

    def update_team(
        self,
        team: str,
        season: int,
        offense_first_down_rate: float,
        offense_turnover_rate: float,
        defense_first_down_rate_allowed: float,
        defense_takeaway_rate: float,
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

        state.offense_first_down_rate = (
            self._ewma(
                state.offense_first_down_rate,
                offense_first_down_rate,
                rate,
            )
        )

        state.offense_turnover_rate = (
            self._ewma(
                state.offense_turnover_rate,
                offense_turnover_rate,
                rate,
            )
        )

        state.defense_first_down_rate_allowed = (
            self._ewma(
                state.defense_first_down_rate_allowed,
                defense_first_down_rate_allowed,
                rate,
            )
        )

        state.defense_takeaway_rate = (
            self._ewma(
                state.defense_takeaway_rate,
                defense_takeaway_rate,
                rate,
            )
        )

        state.games_played += 1
        state.last_season = season

    # ========================================================
    # NATIONAL UPDATE
    # ========================================================

    def update_national(
        self,
        first_down_rate: float,
        turnover_rate: float,
    ) -> None:

        self.national.team_games_seen += 1.0

        self.national.total_first_down_rate += (
            first_down_rate
        )

        self.national.total_turnover_rate += (
            turnover_rate
        )

    # ========================================================
    # PROCESS HISTORY
    # ========================================================

    def process_history(
        self,
        team_history: pd.DataFrame,
        environment_history: pd.DataFrame,
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

            "first_downs",
            "turnovers",
        ]

        missing = [
            column
            for column in required_team_columns
            if column not in team_history.columns
        ]

        if missing:

            raise ValueError(
                "Team history is missing possession "
                f"fields: {missing}"
            )

        required_environment = [
            "game_id",
            "expected_home_plays",
            "expected_away_plays",

            "expected_home_possession_share",
            "expected_away_possession_share",
        ]

        missing_environment = [
            column
            for column in required_environment
            if column not in environment_history.columns
        ]

        if missing_environment:

            raise ValueError(
                "Environment history is missing "
                f"fields: {missing_environment}"
            )

        data = (
            team_history.copy()
        )

        environment = (
            environment_history.copy()
        )

        data["game_id"] = pd.to_numeric(
            data["game_id"],
            errors="coerce",
        ).astype("Int64")

        environment["game_id"] = pd.to_numeric(
            environment["game_id"],
            errors="coerce",
        ).astype("Int64")

        data["home_away"] = (
            data["home_away"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # ----------------------------------------------------
        # SORT
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

        environment_lookup = (
            environment
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

            if game_id not in environment_lookup.index:
                continue

            home_row = home_rows.iloc[0]
            away_row = away_rows.iloc[0]

            environment_row = (
                environment_lookup.loc[
                    game_id
                ]
            )

            if isinstance(
                environment_row,
                pd.DataFrame,
            ):
                environment_row = (
                    environment_row.iloc[0]
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
            # PRE-GAME SNAPSHOTS
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
            # SUSTAINABILITY
            # ------------------------------------------------

            home_sustainability = (
                self.offensive_sustainability(
                    home_snapshot
                )
            )

            away_sustainability = (
                self.offensive_sustainability(
                    away_snapshot
                )
            )

            home_defensive_resistance = (
                self.defensive_resistance(
                    home_snapshot
                )
            )

            away_defensive_resistance = (
                self.defensive_resistance(
                    away_snapshot
                )
            )

            # ------------------------------------------------
            # EXPECTED DRIVE LENGTH
            # ------------------------------------------------

            home_plays_per_drive = (
                self.expected_plays_per_drive(
                    offense_snapshot=home_snapshot,
                    defense_snapshot=away_snapshot,
                )
            )

            away_plays_per_drive = (
                self.expected_plays_per_drive(
                    offense_snapshot=away_snapshot,
                    defense_snapshot=home_snapshot,
                )
            )

            expected_home_plays = (
                self._safe_float(
                    environment_row.get(
                        "expected_home_plays"
                    ),
                    68.0,
                )
            )

            expected_away_plays = (
                self._safe_float(
                    environment_row.get(
                        "expected_away_plays"
                    ),
                    68.0,
                )
            )

            raw_home_drives = (
                self.estimate_drives(
                    expected_home_plays,
                    home_plays_per_drive,
                )
            )

            raw_away_drives = (
                self.estimate_drives(
                    expected_away_plays,
                    away_plays_per_drive,
                )
            )

            # ------------------------------------------------
            # COMMON GAME DRIVE ENVIRONMENT
            #
            # Possession totals should remain relatively close
            # because teams generally alternate possession.
            # ------------------------------------------------

            common_drive_environment = (
                raw_home_drives
                +
                raw_away_drives
            ) / 2.0

            home_possession_share = (
                self._safe_float(
                    environment_row.get(
                        "expected_home_possession_share"
                    ),
                    0.50,
                )
            )

            away_possession_share = (
                self._safe_float(
                    environment_row.get(
                        "expected_away_possession_share"
                    ),
                    0.50,
                )
            )

            possession_difference = (
                home_possession_share
                -
                away_possession_share
            )

            drive_adjustment = (
                possession_difference
                *
                self.config.possession_drive_adjustment
            )

            expected_home_drives = float(
                np.clip(
                    common_drive_environment
                    +
                    drive_adjustment,
                    self.config.minimum_estimated_drives,
                    self.config.maximum_estimated_drives,
                )
            )

            expected_away_drives = float(
                np.clip(
                    common_drive_environment
                    -
                    drive_adjustment,
                    self.config.minimum_estimated_drives,
                    self.config.maximum_estimated_drives,
                )
            )

            expected_total_drives = (
                expected_home_drives
                +
                expected_away_drives
            )

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            output = (
                environment_row.to_dict()
            )

            output.update(
                {
                    "home_possession_games_pre": (
                        home_snapshot[
                            "possession_games_pre"
                        ]
                    ),

                    "away_possession_games_pre": (
                        away_snapshot[
                            "possession_games_pre"
                        ]
                    ),

                    # ----------------------------------------
                    # HOME SUSTAINABILITY
                    # ----------------------------------------

                    "home_offense_first_down_rate_pre": (
                        home_snapshot[
                            "offense_first_down_rate_pre"
                        ]
                    ),

                    "home_offense_turnover_rate_pre": (
                        home_snapshot[
                            "offense_turnover_rate_pre"
                        ]
                    ),

                    "home_defense_first_down_rate_allowed_pre": (
                        home_snapshot[
                            "defense_first_down_rate_allowed_pre"
                        ]
                    ),

                    "home_defense_takeaway_rate_pre": (
                        home_snapshot[
                            "defense_takeaway_rate_pre"
                        ]
                    ),

                    # ----------------------------------------
                    # AWAY SUSTAINABILITY
                    # ----------------------------------------

                    "away_offense_first_down_rate_pre": (
                        away_snapshot[
                            "offense_first_down_rate_pre"
                        ]
                    ),

                    "away_offense_turnover_rate_pre": (
                        away_snapshot[
                            "offense_turnover_rate_pre"
                        ]
                    ),

                    "away_defense_first_down_rate_allowed_pre": (
                        away_snapshot[
                            "defense_first_down_rate_allowed_pre"
                        ]
                    ),

                    "away_defense_takeaway_rate_pre": (
                        away_snapshot[
                            "defense_takeaway_rate_pre"
                        ]
                    ),

                    # ----------------------------------------
                    # NATIONAL
                    # ----------------------------------------

                    "national_first_down_rate_pre": (
                        self.national.first_down_rate
                    ),

                    "national_turnover_rate_pre": (
                        self.national.turnover_rate
                    ),

                    # ----------------------------------------
                    # MATCHUP SUSTAINABILITY
                    # ----------------------------------------

                    "home_offensive_sustainability_index": (
                        100.0
                        *
                        home_sustainability
                    ),

                    "away_offensive_sustainability_index": (
                        100.0
                        *
                        away_sustainability
                    ),

                    "home_defensive_resistance_index": (
                        100.0
                        *
                        home_defensive_resistance
                    ),

                    "away_defensive_resistance_index": (
                        100.0
                        *
                        away_defensive_resistance
                    ),

                    # ----------------------------------------
                    # LATENT DRIVE MODEL
                    # ----------------------------------------

                    "expected_home_plays_per_drive": (
                        home_plays_per_drive
                    ),

                    "expected_away_plays_per_drive": (
                        away_plays_per_drive
                    ),

                    "raw_home_estimated_drives": (
                        raw_home_drives
                    ),

                    "raw_away_estimated_drives": (
                        raw_away_drives
                    ),

                    "common_drive_environment": (
                        common_drive_environment
                    ),

                    "expected_home_drives": (
                        expected_home_drives
                    ),

                    "expected_away_drives": (
                        expected_away_drives
                    ),

                    "expected_total_drives": (
                        expected_total_drives
                    ),
                }
            )

            outputs.append(
                output
            )

            # ------------------------------------------------
            # ACTUAL SAME-GAME RATES
            #
            # Used only AFTER the pregame output has been
            # created.
            # ------------------------------------------------

            (
                home_actual_plays,
                home_fd_rate,
                home_to_rate,
            ) = self.actual_rates(
                home_row
            )

            (
                away_actual_plays,
                away_fd_rate,
                away_to_rate,
            ) = self.actual_rates(
                away_row
            )

            # ------------------------------------------------
            # UPDATE TEAMS
            # ------------------------------------------------

            self.update_team(
                team=home_team,
                season=season,

                offense_first_down_rate=(
                    home_fd_rate
                ),

                offense_turnover_rate=(
                    home_to_rate
                ),

                defense_first_down_rate_allowed=(
                    away_fd_rate
                ),

                defense_takeaway_rate=(
                    away_to_rate
                ),
            )

            self.update_team(
                team=away_team,
                season=season,

                offense_first_down_rate=(
                    away_fd_rate
                ),

                offense_turnover_rate=(
                    away_to_rate
                ),

                defense_first_down_rate_allowed=(
                    home_fd_rate
                ),

                defense_takeaway_rate=(
                    home_to_rate
                ),
            )

            # ------------------------------------------------
            # UPDATE NATIONAL
            # ------------------------------------------------

            self.update_national(
                home_fd_rate,
                home_to_rate,
            )

            self.update_national(
                away_fd_rate,
                away_to_rate,
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
    # CURRENT TEAM STATES
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

                    "offense_first_down_rate": (
                        state.offense_first_down_rate
                    ),

                    "offense_turnover_rate": (
                        state.offense_turnover_rate
                    ),

                    "defense_first_down_rate_allowed": (
                        state.defense_first_down_rate_allowed
                    ),

                    "defense_takeaway_rate": (
                        state.defense_takeaway_rate
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
                "offense_first_down_rate",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )