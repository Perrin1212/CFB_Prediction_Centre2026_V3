from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Monte Carlo score-distribution configuration.

    Development:
        2,000–5,000 simulations/game

    Final official predictions can later use:
        25,000–50,000 simulations/game

    Historical uncertainty is updated chronologically using
    only scoring residuals from PRIOR games.
    """

    simulations: int = 3000
    random_seed: int = 2026

    # EWMA rate for updating historical residual variance /
    # covariance.
    uncertainty_update_rate: float = 0.05

    # Starting residual assumptions before enough history
    # exists.
    starting_home_score_sd: float = 14.0
    starting_away_score_sd: float = 14.0
    starting_residual_correlation: float = 0.10

    # Safety ranges
    minimum_score_sd: float = 8.0
    maximum_score_sd: float = 22.0

    minimum_residual_correlation: float = -0.35
    maximum_residual_correlation: float = 0.60

    minimum_score: float = 0.0
    maximum_score: float = 100.0


# ============================================================
# UNCERTAINTY STATE
# ============================================================


@dataclass
class ResidualUncertaintyState:
    """
    Chronological scoring-error uncertainty state.

    Variance/covariance describes errors remaining AFTER the
    deterministic scoring model.

    Therefore it captures a mixture of:

        unexpected pace
        unexpected drive efficiency
        turnover swings
        explosive plays
        red-zone variance
        game script
        other residual football randomness

    without pretending we have observed drive-level labels.
    """

    games_seen: int = 0

    home_variance: float = 14.0 ** 2
    away_variance: float = 14.0 ** 2

    covariance: float = (
        0.10
        *
        14.0
        *
        14.0
    )


# ============================================================
# ENGINE
# ============================================================


class MonteCarloEngine:

    def __init__(
        self,
        config: MonteCarloConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else MonteCarloConfig()
        )

        self.state = ResidualUncertaintyState(
            home_variance=(
                self.config.starting_home_score_sd
                ** 2
            ),

            away_variance=(
                self.config.starting_away_score_sd
                ** 2
            ),

            covariance=(
                self.config.starting_residual_correlation
                *
                self.config.starting_home_score_sd
                *
                self.config.starting_away_score_sd
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

    # ========================================================
    # CURRENT UNCERTAINTY
    # ========================================================

    def current_uncertainty(
        self,
    ) -> dict[str, float]:

        home_sd = float(
            np.sqrt(
                max(
                    self.state.home_variance,
                    0.0,
                )
            )
        )

        away_sd = float(
            np.sqrt(
                max(
                    self.state.away_variance,
                    0.0,
                )
            )
        )

        home_sd = float(
            np.clip(
                home_sd,
                self.config.minimum_score_sd,
                self.config.maximum_score_sd,
            )
        )

        away_sd = float(
            np.clip(
                away_sd,
                self.config.minimum_score_sd,
                self.config.maximum_score_sd,
            )
        )

        denominator = (
            home_sd
            *
            away_sd
        )

        if denominator <= 0:

            correlation = 0.0

        else:

            correlation = (
                self.state.covariance
                /
                denominator
            )

        correlation = float(
            np.clip(
                correlation,
                self.config.minimum_residual_correlation,
                self.config.maximum_residual_correlation,
            )
        )

        return {
            "home_score_sd": home_sd,
            "away_score_sd": away_sd,
            "score_residual_correlation": correlation,
        }

    # ========================================================
    # COVARIANCE MATRIX
    # ========================================================

    def covariance_matrix(
        self,
    ) -> np.ndarray:

        uncertainty = (
            self.current_uncertainty()
        )

        home_sd = (
            uncertainty[
                "home_score_sd"
            ]
        )

        away_sd = (
            uncertainty[
                "away_score_sd"
            ]
        )

        correlation = (
            uncertainty[
                "score_residual_correlation"
            ]
        )

        covariance = (
            correlation
            *
            home_sd
            *
            away_sd
        )

        matrix = np.array(
            [
                [
                    home_sd ** 2,
                    covariance,
                ],
                [
                    covariance,
                    away_sd ** 2,
                ],
            ],
            dtype=float,
        )

        return matrix

    # ========================================================
    # SIMULATE ONE GAME
    # ========================================================

    def simulate_game(
        self,
        expected_home_points: float,
        expected_away_points: float,
        game_seed: int,
    ) -> dict[str, float]:

        home_mean = self._safe_float(
            expected_home_points,
            27.0,
        )

        away_mean = self._safe_float(
            expected_away_points,
            24.0,
        )

        covariance = (
            self.covariance_matrix()
        )

        rng = np.random.default_rng(
            game_seed
        )

        residuals = (
            rng.multivariate_normal(
                mean=[
                    0.0,
                    0.0,
                ],

                cov=covariance,

                size=(
                    self.config.simulations
                ),
            )
        )

        home_scores = (
            home_mean
            +
            residuals[
                :,
                0
            ]
        )

        away_scores = (
            away_mean
            +
            residuals[
                :,
                1
            ]
        )

        # Football scores cannot be negative.
        home_scores = np.clip(
            home_scores,
            self.config.minimum_score,
            self.config.maximum_score,
        )

        away_scores = np.clip(
            away_scores,
            self.config.minimum_score,
            self.config.maximum_score,
        )

        # Convert continuous scoring draw to a scoreboard-like
        # integer output.
        home_scores = np.rint(
            home_scores
        )

        away_scores = np.rint(
            away_scores
        )

        margins = (
            home_scores
            -
            away_scores
        )

        totals = (
            home_scores
            +
            away_scores
        )

        home_wins = (
            margins
            >
            0
        )

        away_wins = (
            margins
            <
            0
        )

        ties = (
            margins
            ==
            0
        )

        # NCAA games eventually resolve ties, but the score
        # simulator may land on an equal regulation-style score.
        #
        # Split simulation ties 50/50 for win probability.
        home_win_probability = (
            home_wins.mean()
            +
            0.5
            *
            ties.mean()
        )

        away_win_probability = (
            away_wins.mean()
            +
            0.5
            *
            ties.mean()
        )

        return {
            # ------------------------------------------------
            # WIN PROBABILITY
            # ------------------------------------------------

            "simulation_home_win_probability": (
                float(
                    home_win_probability
                )
            ),

            "simulation_away_win_probability": (
                float(
                    away_win_probability
                )
            ),

            "simulation_tie_rate": (
                float(
                    ties.mean()
                )
            ),

            # ------------------------------------------------
            # MEAN SCORES
            # ------------------------------------------------

            "simulation_home_score_mean": (
                float(
                    home_scores.mean()
                )
            ),

            "simulation_away_score_mean": (
                float(
                    away_scores.mean()
                )
            ),

            "simulation_total_mean": (
                float(
                    totals.mean()
                )
            ),

            "simulation_margin_mean": (
                float(
                    margins.mean()
                )
            ),

            # ------------------------------------------------
            # SCORE MEDIANS
            # ------------------------------------------------

            "simulation_home_score_median": (
                float(
                    np.median(
                        home_scores
                    )
                )
            ),

            "simulation_away_score_median": (
                float(
                    np.median(
                        away_scores
                    )
                )
            ),

            # ------------------------------------------------
            # SCORE INTERVALS
            # ------------------------------------------------

            "simulation_home_score_p10": (
                float(
                    np.percentile(
                        home_scores,
                        10,
                    )
                )
            ),

            "simulation_home_score_p90": (
                float(
                    np.percentile(
                        home_scores,
                        90,
                    )
                )
            ),

            "simulation_away_score_p10": (
                float(
                    np.percentile(
                        away_scores,
                        10,
                    )
                )
            ),

            "simulation_away_score_p90": (
                float(
                    np.percentile(
                        away_scores,
                        90,
                    )
                )
            ),

            # ------------------------------------------------
            # MARGIN DISTRIBUTION
            # ------------------------------------------------

            "simulation_margin_sd": (
                float(
                    margins.std(
                        ddof=0
                    )
                )
            ),

            "simulation_margin_p05": (
                float(
                    np.percentile(
                        margins,
                        5,
                    )
                )
            ),

            "simulation_margin_p10": (
                float(
                    np.percentile(
                        margins,
                        10,
                    )
                )
            ),

            "simulation_margin_p25": (
                float(
                    np.percentile(
                        margins,
                        25,
                    )
                )
            ),

            "simulation_margin_p50": (
                float(
                    np.percentile(
                        margins,
                        50,
                    )
                )
            ),

            "simulation_margin_p75": (
                float(
                    np.percentile(
                        margins,
                        75,
                    )
                )
            ),

            "simulation_margin_p90": (
                float(
                    np.percentile(
                        margins,
                        90,
                    )
                )
            ),

            "simulation_margin_p95": (
                float(
                    np.percentile(
                        margins,
                        95,
                    )
                )
            ),

            # ------------------------------------------------
            # TOTAL DISTRIBUTION
            # ------------------------------------------------

            "simulation_total_sd": (
                float(
                    totals.std(
                        ddof=0
                    )
                )
            ),

            "simulation_total_p10": (
                float(
                    np.percentile(
                        totals,
                        10,
                    )
                )
            ),

            "simulation_total_p25": (
                float(
                    np.percentile(
                        totals,
                        25,
                    )
                )
            ),

            "simulation_total_p50": (
                float(
                    np.percentile(
                        totals,
                        50,
                    )
                )
            ),

            "simulation_total_p75": (
                float(
                    np.percentile(
                        totals,
                        75,
                    )
                )
            ),

            "simulation_total_p90": (
                float(
                    np.percentile(
                        totals,
                        90,
                    )
                )
            ),
        }

    # ========================================================
    # UPDATE UNCERTAINTY
    # ========================================================

    def update_uncertainty(
        self,
        home_error: float,
        away_error: float,
    ) -> None:

        """
        Update scoring residual variance/covariance.

        IMPORTANT:
        Called only AFTER that game's simulations have been
        generated.
        """

        home_error = self._safe_float(
            home_error,
            0.0,
        )

        away_error = self._safe_float(
            away_error,
            0.0,
        )

        rate = (
            self.config.uncertainty_update_rate
        )

        keep = (
            1.0
            -
            rate
        )

        observed_home_variance = (
            home_error
            ** 2
        )

        observed_away_variance = (
            away_error
            ** 2
        )

        observed_covariance = (
            home_error
            *
            away_error
        )

        self.state.home_variance = (
            keep
            *
            self.state.home_variance
            +
            rate
            *
            observed_home_variance
        )

        self.state.away_variance = (
            keep
            *
            self.state.away_variance
            +
            rate
            *
            observed_away_variance
        )

        self.state.covariance = (
            keep
            *
            self.state.covariance
            +
            rate
            *
            observed_covariance
        )

        self.state.games_seen += 1

    # ========================================================
    # PROCESS HISTORICAL GAMES
    # ========================================================

    def process_history(
        self,
        scoring_history: pd.DataFrame,
    ) -> pd.DataFrame:

        required = [
            "game_id",
            "season",
            "week",

            "home_team",
            "away_team",

            "home_points",
            "away_points",

            "expected_home_points",
            "expected_away_points",

            "expected_total_points",
            "expected_home_margin",
        ]

        missing = [
            column
            for column in required
            if column not in scoring_history.columns
        ]

        if missing:

            raise ValueError(
                "Scoring history is missing Monte Carlo "
                f"fields: {missing}"
            )

        data = (
            scoring_history.copy()
        )

        data["game_id"] = pd.to_numeric(
            data["game_id"],
            errors="coerce",
        ).astype("Int64")

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

        outputs: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # CHRONOLOGICAL LOOP
        # ====================================================

        for row in data.itertuples(
            index=False
        ):

            row_dict = row._asdict()

            game_id = self._safe_int(
                row_dict.get(
                    "game_id"
                ),
                0,
            )

            # ------------------------------------------------
            # PRE-GAME UNCERTAINTY SNAPSHOT
            # ------------------------------------------------

            uncertainty = (
                self.current_uncertainty()
            )

            # ------------------------------------------------
            # DETERMINISTIC GAME-SPECIFIC SEED
            # ------------------------------------------------

            game_seed = (
                self.config.random_seed
                +
                game_id
            )

            simulation = self.simulate_game(
                expected_home_points=(
                    row_dict[
                        "expected_home_points"
                    ]
                ),

                expected_away_points=(
                    row_dict[
                        "expected_away_points"
                    ]
                ),

                game_seed=game_seed,
            )

            output = dict(
                row_dict
            )

            output.update(
                {
                    "simulation_count": (
                        self.config.simulations
                    ),

                    "historical_uncertainty_games_pre": (
                        self.state.games_seen
                    ),

                    "home_score_residual_sd_pre": (
                        uncertainty[
                            "home_score_sd"
                        ]
                    ),

                    "away_score_residual_sd_pre": (
                        uncertainty[
                            "away_score_sd"
                        ]
                    ),

                    "score_residual_correlation_pre": (
                        uncertainty[
                            "score_residual_correlation"
                        ]
                    ),
                }
            )

            output.update(
                simulation
            )

            outputs.append(
                output
            )

            # ------------------------------------------------
            # OBSERVED RESULT
            # ------------------------------------------------

            actual_home = self._safe_float(
                row_dict.get(
                    "home_points"
                ),
                np.nan,
            )

            actual_away = self._safe_float(
                row_dict.get(
                    "away_points"
                ),
                np.nan,
            )

            expected_home = self._safe_float(
                row_dict.get(
                    "expected_home_points"
                ),
                np.nan,
            )

            expected_away = self._safe_float(
                row_dict.get(
                    "expected_away_points"
                ),
                np.nan,
            )

            # Only update after simulation.
            if (
                np.isfinite(
                    actual_home
                )
                and
                np.isfinite(
                    actual_away
                )
                and
                np.isfinite(
                    expected_home
                )
                and
                np.isfinite(
                    expected_away
                )
            ):

                home_error = (
                    actual_home
                    -
                    expected_home
                )

                away_error = (
                    actual_away
                    -
                    expected_away
                )

                self.update_uncertainty(
                    home_error=home_error,
                    away_error=away_error,
                )

        result = pd.DataFrame(
            outputs
        )

        return result