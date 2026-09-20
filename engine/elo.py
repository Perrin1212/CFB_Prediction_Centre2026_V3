from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================


@dataclass(frozen=True)
class EloConfig:
    """
    Configuration for the College Football ELO engine.

    These are initial defaults only.

    Later we will optimise them using historical walk-forward
    testing rather than treating them as final values.
    """

    starting_rating: float = 1500.0

    # Cross-division holdout testing on the project's historical universe
    # supports a materially lower prior for a previously unseen FCS team.
    # Established FCS teams can still earn their way above this value.
    fcs_starting_rating: float = 1100.0

    home_field_advantage: float = 55.0

    k_factor: float = 22.0

    preseason_regression: float = 0.35

    mov_multiplier_enabled: bool = True

    # --------------------------------------------------------
    # FBS / FCS weighting
    #
    # Routine FBS wins over FCS teams carry limited value.
    #
    # FCS upsets over FBS teams still carry meaningful value
    # because they are much more informative.
    # --------------------------------------------------------

    fbs_over_fcs_weight: float = 0.25

    fcs_over_fbs_weight: float = 0.70

    fbs_fcs_mov_cap: float = 1.35

    minimum_rating: float = 900.0

    maximum_rating: float = 2200.0


# ============================================================
# ELO ENGINE
# ============================================================


class EloEngine:
    """
    Chronological College Football ELO engine.

    Core rules:

        - Games processed chronologically.
        - Pregame ELO frozen before every result.
        - No future leakage.
        - Neutral games receive no HFA.
        - Ties count as 0.5.
        - MOV has logarithmic influence.
        - Ratings regress toward the national mean each season.
        - FBS vs FCS games receive classification-aware weight.
    """

    def __init__(
        self,
        config: EloConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else EloConfig()
        )

        self.ratings: Dict[str, float] = {}
        self.classifications: Dict[str, str] = {}

    # ========================================================
    # BASIC RATING ACCESS
    # ========================================================

    def reset(self) -> None:

        self.ratings = {}
        self.classifications = {}

    @staticmethod
    def _normalise_classification(
        classification: str | None,
    ) -> str:
        value = str(classification or "").strip().casefold()
        return value if value in {"fbs", "fcs"} else ""

    def _prior_for_classification(
        self,
        classification: str | None,
    ) -> float:
        return (
            self.config.fcs_starting_rating
            if self._normalise_classification(classification) == "fcs"
            else self.config.starting_rating
        )

    def get_rating(
        self,
        team: str,
        classification: str | None = None,
    ) -> float:

        team = str(team).strip()
        normalised_class = self._normalise_classification(classification)

        if normalised_class:
            self.classifications[team] = normalised_class

        if team not in self.ratings:

            self.ratings[
                team
            ] = self._prior_for_classification(normalised_class)

        return self.ratings[
            team
        ]

    def set_rating(
        self,
        team: str,
        rating: float,
    ) -> None:

        bounded_rating = max(
            self.config.minimum_rating,
            min(
                self.config.maximum_rating,
                float(rating),
            ),
        )

        self.ratings[
            team
        ] = bounded_rating

    # ========================================================
    # EXPECTED RESULT
    # ========================================================

    @staticmethod
    def expected_score(
        rating_a: float,
        rating_b: float,
    ) -> float:

        return 1.0 / (
            1.0
            +
            10.0
            ** (
                (
                    rating_b
                    -
                    rating_a
                )
                /
                400.0
            )
        )

    def expected_home_win(
        self,
        home_rating: float,
        away_rating: float,
        neutral_site: bool = False,
    ) -> float:

        hfa = (
            0.0
            if neutral_site
            else self.config.home_field_advantage
        )

        adjusted_home_rating = (
            home_rating
            +
            hfa
        )

        return self.expected_score(
            adjusted_home_rating,
            away_rating,
        )

    # ========================================================
    # MARGIN OF VICTORY
    # ========================================================

    def margin_multiplier(
        self,
        point_margin: float,
        rating_difference: float,
    ) -> float:
        """
        Logarithmic MOV adjustment.

        Blowouts matter, but with diminishing returns.
        """

        if not self.config.mov_multiplier_enabled:

            return 1.0

        margin = abs(
            float(point_margin)
        )

        if margin <= 0:

            return 1.0

        margin_component = log(
            margin + 1.0
        )

        rating_adjustment = (
            2.2
            /
            (
                (
                    abs(
                        float(
                            rating_difference
                        )
                    )
                    * 0.001
                )
                +
                2.2
            )
        )

        multiplier = (
            margin_component
            *
            rating_adjustment
        )

        return max(
            1.0,
            multiplier,
        )

    # ========================================================
    # CLASSIFICATION WEIGHTING
    # ========================================================

    def game_weight(
        self,
        home_classification: str | None,
        away_classification: str | None,
        home_points: float,
        away_points: float,
    ) -> tuple[float, str]:
        """
        Determine how strongly the game should affect ELO.

        FBS vs FBS:
            full weight.

        FBS beats FCS:
            reduced weight.

        FCS beats FBS:
            meaningful but still slightly dampened weight.

        Ties in FBS/FCS games:
            use the lower routine-cross-division weight.
        """

        home_class = (
            str(
                home_classification
            )
            .strip()
            .lower()
            if home_classification is not None
            else ""
        )

        away_class = (
            str(
                away_classification
            )
            .strip()
            .lower()
            if away_classification is not None
            else ""
        )

        # ----------------------------------------------------
        # FBS vs FBS
        # ----------------------------------------------------

        if (
            home_class == "fbs"
            and
            away_class == "fbs"
        ):

            return (
                1.0,
                "fbs_vs_fbs",
            )

        # ----------------------------------------------------
        # FBS vs FCS
        # ----------------------------------------------------

        is_cross_division = (
            {
                home_class,
                away_class,
            }
            ==
            {
                "fbs",
                "fcs",
            }
        )

        if not is_cross_division:

            return (
                1.0,
                "other",
            )

        # Tie
        if home_points == away_points:

            return (
                self.config.fbs_over_fcs_weight,
                "fbs_fcs_tie",
            )

        home_won = (
            home_points
            >
            away_points
        )

        winning_classification = (
            home_class
            if home_won
            else away_class
        )

        if winning_classification == "fbs":

            return (
                self.config.fbs_over_fcs_weight,
                "fbs_over_fcs",
            )

        return (
            self.config.fcs_over_fbs_weight,
            "fcs_over_fbs",
        )

    # ========================================================
    # SEASON REGRESSION
    # ========================================================

    def regress_for_new_season(
        self,
    ) -> None:

        regression = (
            self.config.preseason_regression
        )

        for team in list(
            self.ratings.keys()
        ):

            old_rating = (
                self.ratings[
                    team
                ]
            )

            mean = self._prior_for_classification(
                self.classifications.get(team)
            )

            new_rating = (
                old_rating
                +
                (
                    mean
                    -
                    old_rating
                )
                *
                regression
            )

            self.set_rating(
                team,
                new_rating,
            )

    # ========================================================
    # SINGLE GAME UPDATE
    # ========================================================

    def update_game(
        self,
        home_team: str,
        away_team: str,
        home_points: float,
        away_points: float,
        neutral_site: bool = False,
        home_classification: str | None = None,
        away_classification: str | None = None,
    ) -> dict:

        home_pre = self.get_rating(
            home_team,
            home_classification,
        )

        away_pre = self.get_rating(
            away_team,
            away_classification,
        )

        expected_home = (
            self.expected_home_win(
                home_rating=home_pre,
                away_rating=away_pre,
                neutral_site=neutral_site,
            )
        )

        expected_away = (
            1.0
            -
            expected_home
        )

        # ----------------------------------------------------
        # Actual result
        # ----------------------------------------------------

        if home_points > away_points:

            actual_home = 1.0

            actual_away = 0.0

            winner = home_team

        elif away_points > home_points:

            actual_home = 0.0

            actual_away = 1.0

            winner = away_team

        else:

            actual_home = 0.5

            actual_away = 0.5

            winner = "TIE"

        point_margin = (
            float(home_points)
            -
            float(away_points)
        )

        # ----------------------------------------------------
        # Effective rating difference
        # ----------------------------------------------------

        hfa = (
            0.0
            if neutral_site
            else self.config.home_field_advantage
        )

        effective_rating_difference = (
            (
                home_pre
                +
                hfa
            )
            -
            away_pre
        )

        mov_multiplier = (
            self.margin_multiplier(
                point_margin=point_margin,
                rating_difference=(
                    effective_rating_difference
                ),
            )
        )

        # ----------------------------------------------------
        # Game classification weight
        # ----------------------------------------------------

        classification_weight, game_type = (
            self.game_weight(
                home_classification=(
                    home_classification
                ),
                away_classification=(
                    away_classification
                ),
                home_points=(
                    home_points
                ),
                away_points=(
                    away_points
                ),
            )
        )

        # ----------------------------------------------------
        # Cap MOV for FBS/FCS games
        #
        # Prevent 60-point FBS wins from becoming major ELO
        # inflation events.
        # ----------------------------------------------------

        if game_type in {
            "fbs_over_fcs",
            "fcs_over_fbs",
            "fbs_fcs_tie",
        }:

            mov_multiplier = min(
                mov_multiplier,
                self.config.fbs_fcs_mov_cap,
            )

        # ----------------------------------------------------
        # Effective K
        # ----------------------------------------------------

        effective_k = (
            self.config.k_factor
            *
            classification_weight
        )

        rating_change = (
            effective_k
            *
            mov_multiplier
            *
            (
                actual_home
                -
                expected_home
            )
        )

        home_post = (
            home_pre
            +
            rating_change
        )

        away_post = (
            away_pre
            -
            rating_change
        )

        self.set_rating(
            home_team,
            home_post,
        )

        self.set_rating(
            away_team,
            away_post,
        )

        return {
            "home_team": home_team,
            "away_team": away_team,

            "home_points": home_points,
            "away_points": away_points,

            "winner": winner,

            "neutral_site": bool(
                neutral_site
            ),

            "home_classification": (
                home_classification
            ),

            "away_classification": (
                away_classification
            ),

            "game_type": (
                game_type
            ),

            "game_weight": (
                classification_weight
            ),

            "home_elo_pre": (
                home_pre
            ),

            "away_elo_pre": (
                away_pre
            ),

            "home_elo_advantage_pre": (
                home_pre
                -
                away_pre
            ),

            "effective_home_elo_advantage": (
                effective_rating_difference
            ),

            "home_elo_win_probability": (
                expected_home
            ),

            "away_elo_win_probability": (
                expected_away
            ),

            "actual_home_result": (
                actual_home
            ),

            "actual_away_result": (
                actual_away
            ),

            "point_margin": (
                point_margin
            ),

            "mov_multiplier": (
                mov_multiplier
            ),

            "effective_k": (
                effective_k
            ),

            "elo_change": (
                rating_change
            ),

            "home_elo_post": (
                self.get_rating(
                    home_team
                )
            ),

            "away_elo_post": (
                self.get_rating(
                    away_team
                )
            ),
        }

    # ========================================================
    # HISTORICAL PROCESSING
    # ========================================================

    def process_games(
        self,
        games: pd.DataFrame,
    ) -> pd.DataFrame:

        required_columns = [
            "season",
            "week",
            "start_date",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "neutral_site",
            "home_classification",
            "away_classification",
        ]

        missing = [
            column
            for column in required_columns
            if column not in games.columns
        ]

        if missing:

            raise ValueError(
                "Games dataset is missing required "
                f"columns: {missing}"
            )

        data = games.copy()

        data["start_date"] = (
            pd.to_datetime(
                data["start_date"],
                errors="coerce",
                utc=True,
            )
        )

        sort_columns = [
            "season",
            "week",
            "start_date",
        ]

        if "cfbd_id" in data.columns:

            sort_columns.append(
                "cfbd_id"
            )

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

        history: list[dict] = []

        current_season: int | None = None

        for row in data.itertuples(
            index=False
        ):

            season = int(
                row.season
            )

            # ------------------------------------------------
            # Season transition
            # ------------------------------------------------

            if current_season is None:

                current_season = (
                    season
                )

            elif season != current_season:

                self.regress_for_new_season()

                current_season = (
                    season
                )

            # ------------------------------------------------
            # Process game
            # ------------------------------------------------

            result = (
                self.update_game(
                    home_team=str(
                        row.home_team
                    ),
                    away_team=str(
                        row.away_team
                    ),
                    home_points=float(
                        row.home_points
                    ),
                    away_points=float(
                        row.away_points
                    ),
                    neutral_site=bool(
                        row.neutral_site
                    ),
                    home_classification=str(
                        row.home_classification
                    ),
                    away_classification=str(
                        row.away_classification
                    ),
                )
            )

            record = {
                "season": season,

                "week": row.week,

                "start_date": (
                    row.start_date
                ),

                **result,
            }

            if hasattr(
                row,
                "id",
            ):

                record[
                    "game_db_id"
                ] = row.id

            if hasattr(
                row,
                "cfbd_id",
            ):

                record[
                    "cfbd_game_id"
                ] = row.cfbd_id

            history.append(
                record
            )

        return pd.DataFrame(
            history
        )

    # ========================================================
    # CURRENT RATINGS
    # ========================================================

    def current_ratings(
        self,
    ) -> pd.DataFrame:

        rows = [
            {
                "team": team,
                "elo": rating,
            }
            for team, rating
            in self.ratings.items()
        ]

        if not rows:

            return pd.DataFrame(
                columns=[
                    "rank",
                    "team",
                    "elo",
                ]
            )

        df = pd.DataFrame(
            rows
        )

        df = (
            df
            .sort_values(
                "elo",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        df["rank"] = (
            df.index
            +
            1
        )

        return df[
            [
                "rank",
                "team",
                "elo",
            ]
        ]
