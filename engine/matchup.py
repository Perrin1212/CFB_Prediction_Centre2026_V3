from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class MatchupConfig:
    """
    Configuration for the game-level matchup engine.

    Ratings entering this layer use:

        100 = national average
        >100 = stronger
        <100 = weaker

    No win probability or score prediction is produced here.

    This layer only converts team-strength ratings into
    game-specific matchup features.
    """

    matchup_index_floor: float = 60.0
    matchup_index_ceiling: float = 140.0

    strength_floor: float = 50.0
    strength_ceiling: float = 150.0

    # Number of historical games required before we consider
    # a team's adjusted rating broadly established.
    established_games: int = 6


# ============================================================
# ENGINE
# ============================================================


class MatchupEngine:
    """
    Convert chronological pregame team ratings into
    game-specific matchup features.

    Core concept:

        Good offence vs weak defence
            -> favourable offensive matchup

        Good offence vs good defence
            -> closer to neutral

        Weak offence vs good defence
            -> unfavourable offensive matchup

    The engine does NOT create win probabilities.
    """

    def __init__(
        self,
        config: MatchupConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else MatchupConfig()
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float = 100.0,
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
    def _safe_int(
        value: Any,
        fallback: int = 0,
    ) -> int:

        try:

            numeric = float(value)

            if not np.isfinite(numeric):
                return int(fallback)

            return int(numeric)

        except (
            TypeError,
            ValueError,
        ):

            return int(fallback)

    def _clip_matchup(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.matchup_index_floor,
                self.config.matchup_index_ceiling,
            )
        )

    def _clip_strength(
        self,
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                self.config.strength_floor,
                self.config.strength_ceiling,
            )
        )

    # ========================================================
    # OFFENSIVE MATCHUP INDEX
    # ========================================================

    def offensive_matchup_index(
        self,
        offense_rating: float,
        opponent_defense_rating: float,
    ) -> float:
        """
        Convert offence and opposing defence into a
        game-specific offensive matchup index.

        Examples:

            offence 120 vs defence 100
                -> 120

            offence 120 vs defence 120
                -> 100

            offence 90 vs defence 120
                -> 75

        Because defensive ratings are defined so HIGHER is
        better, the defensive rating belongs in the
        denominator.
        """

        offense = max(
            self._safe_float(
                offense_rating,
                100.0,
            ),
            1.0,
        )

        defense = max(
            self._safe_float(
                opponent_defense_rating,
                100.0,
            ),
            1.0,
        )

        matchup = (
            100.0
            *
            offense
            /
            defense
        )

        return self._clip_matchup(
            matchup
        )

    # ========================================================
    # TEAM BALANCE
    # ========================================================

    def overall_strength(
        self,
        offense_rating: float,
        defense_rating: float,
    ) -> float:
        """
        Simple balanced strength representation.

        This is NOT a replacement for ELO.

        It merely provides a useful summary of the separate
        opponent-adjusted offence and defence ratings.
        """

        offense = self._safe_float(
            offense_rating,
            100.0,
        )

        defense = self._safe_float(
            defense_rating,
            100.0,
        )

        # Geometric mean avoids one extreme unit completely
        # dominating the overall strength measure.
        strength = np.sqrt(
            max(
                offense,
                1.0,
            )
            *
            max(
                defense,
                1.0,
            )
        )

        return self._clip_strength(
            float(
                strength
            )
        )

    # ========================================================
    # EXPERIENCE / STABILITY
    # ========================================================

    def rating_maturity(
        self,
        games_pre: int,
    ) -> float:
        """
        0–1 measure describing how established the current
        season/team rating is.

        This does NOT modify ratings.

        It gives later layers a way to understand that a team
        with 1 observed game has more uncertainty than a team
        with 8+ observations.
        """

        games = max(
            self._safe_int(
                games_pre,
                0,
            ),
            0,
        )

        established = max(
            int(
                self.config.established_games
            ),
            1,
        )

        return float(
            np.clip(
                games
                /
                established,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # BUILD ONE GAME
    # ========================================================

    def compare(self, home_team, away_team):
        """Compatibility object API retained for the original interactive predictor.

        The production historical pipeline uses build_game_matchup/dataframe states;
        this adapter prevents the earlier lightweight predictor interface from
        silently breaking.
        """
        from .types import MatchupResult
        he=float(getattr(home_team, "elo", 1500.0)); ae=float(getattr(away_team, "elo", 1500.0))
        ho=float(getattr(home_team, "offensive_rating", 1.0)); ao=float(getattr(away_team, "offensive_rating", 1.0))
        hd=float(getattr(home_team, "defensive_rating", 1.0)); ad=float(getattr(away_team, "defensive_rating", 1.0))
        hs=float(getattr(home_team, "special_teams_rating", 0.0)); ass=float(getattr(away_team, "special_teams_rating", 0.0))
        hoe=ho-ad; aoe=ao-hd; hde=hd-ao; ade=ad-ho; hse=hs-ass; ase=ass-hs
        hov=(he-ae)/400.0 + hoe + .5*hde + .15*hse; aov=-hov
        return MatchupResult(home_team=home_team.team_name, away_team=away_team.team_name, home_elo=he, away_elo=ae,
            home_offensive_edge=hoe, away_offensive_edge=aoe, home_defensive_edge=hde, away_defensive_edge=ade,
            home_special_teams_edge=hse, away_special_teams_edge=ase, home_overall_edge=hov, away_overall_edge=aov,
            home_matchup_score=100+10*hov, away_matchup_score=100+10*aov, advantages={}, risks={})

    def build_game_matchup(
        self,
        home_row: pd.Series,
        away_row: pd.Series,
        elo_row: pd.Series | None = None,
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # TEAM IDENTITIES
        # ----------------------------------------------------

        game_id = home_row.get(
            "game_id"
        )

        season = self._safe_int(
            home_row.get(
                "season"
            ),
            0,
        )

        week = self._safe_int(
            home_row.get(
                "week"
            ),
            0,
        )

        home_team = str(
            home_row.get(
                "team_name",
                "",
            )
        ).strip()

        away_team = str(
            away_row.get(
                "team_name",
                "",
            )
        ).strip()

        # ----------------------------------------------------
        # ADJUSTED RATINGS
        # ----------------------------------------------------

        home_offense = self._safe_float(
            home_row.get(
                "team_adjusted_offense_rating_pre"
            ),
            100.0,
        )

        home_defense = self._safe_float(
            home_row.get(
                "team_adjusted_defense_rating_pre"
            ),
            100.0,
        )

        away_offense = self._safe_float(
            away_row.get(
                "team_adjusted_offense_rating_pre"
            ),
            100.0,
        )

        away_defense = self._safe_float(
            away_row.get(
                "team_adjusted_defense_rating_pre"
            ),
            100.0,
        )

        # ----------------------------------------------------
        # RAW RATINGS
        # ----------------------------------------------------

        home_raw_offense = self._safe_float(
            home_row.get(
                "team_offense_rating_pre"
            ),
            100.0,
        )

        home_raw_defense = self._safe_float(
            home_row.get(
                "team_defense_rating_pre"
            ),
            100.0,
        )

        away_raw_offense = self._safe_float(
            away_row.get(
                "team_offense_rating_pre"
            ),
            100.0,
        )

        away_raw_defense = self._safe_float(
            away_row.get(
                "team_defense_rating_pre"
            ),
            100.0,
        )

        # ----------------------------------------------------
        # GAME-SPECIFIC OFFENSIVE MATCHUPS
        # ----------------------------------------------------

        home_offensive_matchup = (
            self.offensive_matchup_index(
                offense_rating=home_offense,
                opponent_defense_rating=away_defense,
            )
        )

        away_offensive_matchup = (
            self.offensive_matchup_index(
                offense_rating=away_offense,
                opponent_defense_rating=home_defense,
            )
        )

        # ----------------------------------------------------
        # UNIT EDGES
        #
        # Positive number favours the named team.
        # ----------------------------------------------------

        home_offense_edge = (
            home_offense
            -
            away_defense
        )

        away_offense_edge = (
            away_offense
            -
            home_defense
        )

        home_defense_edge = (
            home_defense
            -
            away_offense
        )

        away_defense_edge = (
            away_defense
            -
            home_offense
        )

        # ----------------------------------------------------
        # BALANCED TEAM STRENGTH
        # ----------------------------------------------------

        home_overall = (
            self.overall_strength(
                offense_rating=home_offense,
                defense_rating=home_defense,
            )
        )

        away_overall = (
            self.overall_strength(
                offense_rating=away_offense,
                defense_rating=away_defense,
            )
        )

        adjusted_strength_difference = (
            home_overall
            -
            away_overall
        )

        offensive_matchup_difference = (
            home_offensive_matchup
            -
            away_offensive_matchup
        )

        # ----------------------------------------------------
        # RATING MATURITY
        # ----------------------------------------------------

        home_games_pre = self._safe_int(
            home_row.get(
                "adjusted_games_pre"
            ),
            0,
        )

        away_games_pre = self._safe_int(
            away_row.get(
                "adjusted_games_pre"
            ),
            0,
        )

        home_maturity = (
            self.rating_maturity(
                home_games_pre
            )
        )

        away_maturity = (
            self.rating_maturity(
                away_games_pre
            )
        )

        matchup_maturity = (
            (
                home_maturity
                +
                away_maturity
            )
            /
            2.0
        )

        # ----------------------------------------------------
        # GAME INFORMATION
        # ----------------------------------------------------

        output: dict[
            str,
            Any,
        ] = {
            "game_id": game_id,
            "season": season,
            "week": week,

            "start_date": home_row.get(
                "start_date"
            ),

            "neutral_site": home_row.get(
                "neutral_site"
            ),

            "conference_game": home_row.get(
                "conference_game"
            ),

            "home_team": home_team,
            "away_team": away_team,

            # Actual result retained only for historical
            # validation. These must NEVER be used as pregame
            # prediction features.
            "home_points": home_row.get(
                "team_points"
            ),

            "away_points": away_row.get(
                "team_points"
            ),

            # --------------------------------------------
            # RAW PRE-GAME RATINGS
            # --------------------------------------------

            "home_raw_offense_rating_pre": (
                home_raw_offense
            ),

            "home_raw_defense_rating_pre": (
                home_raw_defense
            ),

            "away_raw_offense_rating_pre": (
                away_raw_offense
            ),

            "away_raw_defense_rating_pre": (
                away_raw_defense
            ),

            # --------------------------------------------
            # OPPONENT-ADJUSTED PRE-GAME RATINGS
            # --------------------------------------------

            "home_adjusted_offense_rating_pre": (
                home_offense
            ),

            "home_adjusted_defense_rating_pre": (
                home_defense
            ),

            "away_adjusted_offense_rating_pre": (
                away_offense
            ),

            "away_adjusted_defense_rating_pre": (
                away_defense
            ),

            # --------------------------------------------
            # MATCHUP INDICES
            # --------------------------------------------

            "home_offensive_matchup_index": (
                home_offensive_matchup
            ),

            "away_offensive_matchup_index": (
                away_offensive_matchup
            ),

            "offensive_matchup_difference": (
                offensive_matchup_difference
            ),

            # --------------------------------------------
            # UNIT EDGES
            # --------------------------------------------

            "home_offense_edge": (
                home_offense_edge
            ),

            "away_offense_edge": (
                away_offense_edge
            ),

            "home_defense_edge": (
                home_defense_edge
            ),

            "away_defense_edge": (
                away_defense_edge
            ),

            # --------------------------------------------
            # BALANCED STRENGTH
            # --------------------------------------------

            "home_adjusted_overall_strength": (
                home_overall
            ),

            "away_adjusted_overall_strength": (
                away_overall
            ),

            "adjusted_strength_difference": (
                adjusted_strength_difference
            ),

            # --------------------------------------------
            # MATURITY
            # --------------------------------------------

            "home_adjusted_games_pre": (
                home_games_pre
            ),

            "away_adjusted_games_pre": (
                away_games_pre
            ),

            "home_rating_maturity": (
                home_maturity
            ),

            "away_rating_maturity": (
                away_maturity
            ),

            "matchup_rating_maturity": (
                matchup_maturity
            ),
        }

        # ====================================================
        # ELO
        # ====================================================

        if elo_row is not None:

            home_elo = self._safe_float(
                elo_row.get(
                    "home_elo_pre"
                ),
                1500.0,
            )

            away_elo = self._safe_float(
                elo_row.get(
                    "away_elo_pre"
                ),
                1500.0,
            )

            home_elo_probability = (
                self._safe_float(
                    elo_row.get(
                        "home_elo_win_probability"
                    ),
                    0.50,
                )
            )

            output[
                "home_elo_pre"
            ] = home_elo

            output[
                "away_elo_pre"
            ] = away_elo

            output[
                "elo_difference"
            ] = (
                home_elo
                -
                away_elo
            )

            output[
                "home_elo_win_probability"
            ] = (
                home_elo_probability
            )

            output[
                "away_elo_win_probability"
            ] = (
                1.0
                -
                home_elo_probability
            )

        return output

    # ========================================================
    # BUILD HISTORY
    # ========================================================

    def build_history(
        self,
        adjusted_history: pd.DataFrame,
        elo_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:

        required = [
            "game_id",
            "team_name",
            "team_points",

            "team_offense_rating_pre",
            "team_defense_rating_pre",

            "team_adjusted_offense_rating_pre",
            "team_adjusted_defense_rating_pre",

            "adjusted_games_pre",
        ]

        missing = [
            column
            for column in required
            if column not in adjusted_history.columns
        ]

        if missing:

            raise ValueError(
                "Opponent-adjusted history is missing "
                f"required columns: {missing}"
            )

        data = (
            adjusted_history.copy()
        )

        # ====================================================
        # HOME / AWAY IDENTIFICATION
        # ====================================================

        if "home_away" not in data.columns:

            raise ValueError(
                "Opponent-adjusted history does not contain "
                "'home_away'. The matchup engine requires "
                "home/away orientation."
            )

        data[
            "home_away"
        ] = (
            data[
                "home_away"
            ]
            .astype(
                "string"
            )
            .str.strip()
            .str.lower()
        )

        # ====================================================
        # ELO LOOKUP
        # ====================================================

        elo_lookup: dict[
            Any,
            pd.Series,
        ] = {}

        if elo_history is not None:

            elo_data = (
                elo_history.copy()
            )

            if "game_id" not in elo_data.columns:

                raise ValueError(
                    "ELO history is missing game_id."
                )

            for _, row in elo_data.iterrows():

                elo_lookup[
                    row[
                        "game_id"
                    ]
                ] = row

        # ====================================================
        # CHRONOLOGICAL ORDER
        # ====================================================

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

        outputs: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # ONE OUTPUT ROW PER GAME
        # ====================================================

        for game_id, rows in data.groupby(
            "game_id",
            sort=False,
        ):

            if len(rows) != 2:
                continue

            home_rows = rows[
                rows[
                    "home_away"
                ]
                ==
                "home"
            ]

            away_rows = rows[
                rows[
                    "home_away"
                ]
                ==
                "away"
            ]

            if (
                len(home_rows) != 1
                or
                len(away_rows) != 1
            ):

                continue

            home_row = (
                home_rows.iloc[0]
            )

            away_row = (
                away_rows.iloc[0]
            )

            elo_row = (
                elo_lookup.get(
                    game_id
                )
            )

            output = (
                self.build_game_matchup(
                    home_row=home_row,
                    away_row=away_row,
                    elo_row=elo_row,
                )
            )

            outputs.append(
                output
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