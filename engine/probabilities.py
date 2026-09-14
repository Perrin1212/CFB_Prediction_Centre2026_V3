from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ProbabilityConfig:
    """
    Probability-engine configuration.

    Development design:

        2023 -> train candidate architectures
        2024 -> select architecture
        2025 -> forward holdout

    Important modelling principle:

    Directional features answer:

        "Which team is stronger?"

    Uncertainty features answer:

        "How sure are we?"

    Rating maturity belongs to the second category and is
    therefore deliberately excluded from this directional
    logistic probability model.
    """

    train_season: int = 2023
    validation_season: int = 2024
    holdout_season: int = 2025

    logistic_c: float = 1.0
    max_iter: int = 2000

    probability_floor: float = 0.001
    probability_ceiling: float = 0.999


# ============================================================
# MODEL BUNDLE
# ============================================================


@dataclass
class ProbabilityModelBundle:

    name: str

    features: list[str]

    scaler: StandardScaler

    model: LogisticRegression


# ============================================================
# ENGINE
# ============================================================


class ProbabilityEngine:

    def __init__(
        self,
        config: ProbabilityConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ProbabilityConfig()
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _safe_logit(
        probability: pd.Series,
    ) -> pd.Series:

        p = pd.to_numeric(
            probability,
            errors="coerce",
        )

        p = p.clip(
            1e-5,
            1.0 - 1e-5,
        )

        return np.log(
            p
            /
            (
                1.0
                -
                p
            )
        )

    @staticmethod
    def _binary_log_loss(
        actual: pd.Series,
        probability: pd.Series,
    ) -> float:

        y = (
            pd.to_numeric(
                actual,
                errors="coerce",
            )
            .astype(float)
            .to_numpy()
        )

        p = (
            pd.to_numeric(
                probability,
                errors="coerce",
            )
            .astype(float)
            .clip(
                1e-6,
                1.0 - 1e-6,
            )
            .to_numpy()
        )

        return float(
            -np.mean(
                y
                *
                np.log(
                    p
                )
                +
                (
                    1.0
                    -
                    y
                )
                *
                np.log(
                    1.0
                    -
                    p
                )
            )
        )

    @staticmethod
    def _brier(
        actual: pd.Series,
        probability: pd.Series,
    ) -> float:

        y = (
            pd.to_numeric(
                actual,
                errors="coerce",
            )
            .astype(float)
            .to_numpy()
        )

        p = (
            pd.to_numeric(
                probability,
                errors="coerce",
            )
            .astype(float)
            .to_numpy()
        )

        return float(
            np.mean(
                (
                    p
                    -
                    y
                )
                ** 2
            )
        )

    @staticmethod
    def _accuracy(
        actual: pd.Series,
        probability: pd.Series,
    ) -> float:

        y = (
            pd.to_numeric(
                actual,
                errors="coerce",
            )
            .astype(int)
            .to_numpy()
        )

        prediction = (
            pd.to_numeric(
                probability,
                errors="coerce",
            )
            .astype(float)
            .to_numpy()
            >
            0.50
        ).astype(int)

        return float(
            np.mean(
                prediction
                ==
                y
            )
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_dataset(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        data = history.copy()

        required = [
            "season",

            "home_points",
            "away_points",

            "home_elo_win_probability",

            "adjusted_strength_difference",
            "offensive_matchup_difference",

            "expected_home_margin",

            "simulation_home_win_probability",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Probability history is missing required "
                f"fields: {missing}"
            )

        # ----------------------------------------------------
        # REMOVE HISTORICAL TIES
        # ----------------------------------------------------

        data["home_points"] = pd.to_numeric(
            data["home_points"],
            errors="coerce",
        )

        data["away_points"] = pd.to_numeric(
            data["away_points"],
            errors="coerce",
        )

        data = data[
            data["home_points"]
            !=
            data["away_points"]
        ].copy()

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        data["actual_home_win"] = (
            data["home_points"]
            >
            data["away_points"]
        ).astype(int)

        # ----------------------------------------------------
        # PROBABILITY -> LOG ODDS
        # ----------------------------------------------------

        data["elo_logit"] = (
            self._safe_logit(
                data[
                    "home_elo_win_probability"
                ]
            )
        )

        data["simulation_logit"] = (
            self._safe_logit(
                data[
                    "simulation_home_win_probability"
                ]
            )
        )

        # ----------------------------------------------------
        # NUMERIC CLEANUP
        # ----------------------------------------------------

        numeric_columns = [
            "season",

            "actual_home_win",

            "elo_logit",

            "adjusted_strength_difference",
            "offensive_matchup_difference",

            "expected_home_margin",

            "simulation_logit",
        ]

        for column in numeric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = (
            data
            .dropna(
                subset=numeric_columns
            )
            .reset_index(
                drop=True
            )
        )

        return data

    # ========================================================
    # CANDIDATE ARCHITECTURES
    # ========================================================

    @staticmethod
    def candidate_architectures(
    ) -> dict[str, list[str]]:
        """
        Compact directional architectures.

        Deliberately excluded:
            matchup_rating_maturity

        because maturity describes uncertainty rather than
        team-directional strength.

        ELO probability and ELO rating difference are also not
        both included because they are near-deterministic
        transformations of the same underlying rating signal.
        """

        return {
            # -----------------------------------------------
            # ELO BASELINE
            # -----------------------------------------------

            "elo_only": [
                "elo_logit",
            ],

            # -----------------------------------------------
            # ELO + OVERALL ADJUSTED STRENGTH
            # -----------------------------------------------

            "elo_strength": [
                "elo_logit",
                "adjusted_strength_difference",
            ],

            # -----------------------------------------------
            # ELO + MATCHUP ADVANTAGE
            # -----------------------------------------------

            "elo_matchup": [
                "elo_logit",
                "offensive_matchup_difference",
            ],

            # -----------------------------------------------
            # ELO + BOTH RATING DIFFERENCES
            # -----------------------------------------------

            "elo_strength_matchup": [
                "elo_logit",
                "adjusted_strength_difference",
                "offensive_matchup_difference",
            ],

            # -----------------------------------------------
            # ELO + EXPECTED SCORE
            # -----------------------------------------------

            "elo_scoring": [
                "elo_logit",
                "expected_home_margin",
            ],

            # -----------------------------------------------
            # ELO + MONTE CARLO
            # -----------------------------------------------

            "elo_simulation": [
                "elo_logit",
                "simulation_logit",
            ],

            # -----------------------------------------------
            # COMPACT FULL MODEL
            #
            # This intentionally remains small.
            # -----------------------------------------------

            "compact_full": [
                "elo_logit",

                "adjusted_strength_difference",

                "expected_home_margin",

                "simulation_logit",
            ],
        }

    # ========================================================
    # FIT MODEL
    # ========================================================

    def fit_model(
        self,
        data: pd.DataFrame,
        model_name: str,
        features: list[str],
    ) -> ProbabilityModelBundle:

        X = (
            data[
                features
            ]
            .astype(float)
            .to_numpy()
        )

        y = (
            data[
                "actual_home_win"
            ]
            .astype(int)
            .to_numpy()
        )

        scaler = StandardScaler()

        X_scaled = (
            scaler.fit_transform(
                X
            )
        )

        model = LogisticRegression(
            C=self.config.logistic_c,

            max_iter=(
                self.config.max_iter
            ),

            solver="lbfgs",

            random_state=2026,
        )

        model.fit(
            X_scaled,
            y,
        )

        return ProbabilityModelBundle(
            name=model_name,
            features=list(
                features
            ),
            scaler=scaler,
            model=model,
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        bundle: ProbabilityModelBundle,
        data: pd.DataFrame,
    ) -> np.ndarray:

        X = (
            data[
                bundle.features
            ]
            .astype(float)
            .to_numpy()
        )

        X_scaled = (
            bundle.scaler.transform(
                X
            )
        )

        probability = (
            bundle.model.predict_proba(
                X_scaled
            )[
                :,
                1
            ]
        )

        probability = np.clip(
            probability,

            self.config.probability_floor,

            self.config.probability_ceiling,
        )

        return probability

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate_probabilities(
        self,
        data: pd.DataFrame,
        probability: np.ndarray,
    ) -> dict[str, float]:

        probability_series = pd.Series(
            probability,
            index=data.index,
        )

        actual = (
            data[
                "actual_home_win"
            ]
        )

        return {
            "games": int(
                len(data)
            ),

            "accuracy": (
                self._accuracy(
                    actual,
                    probability_series,
                )
            ),

            "brier": (
                self._brier(
                    actual,
                    probability_series,
                )
            ),

            "log_loss": (
                self._binary_log_loss(
                    actual,
                    probability_series,
                )
            ),

            "mean_probability": float(
                probability_series.mean()
            ),

            "minimum_probability": float(
                probability_series.min()
            ),

            "maximum_probability": float(
                probability_series.max()
            ),
        }

    # ========================================================
    # CALIBRATION TABLE
    # ========================================================

    def calibration_table(
        self,
        actual: pd.Series,
        probability: pd.Series,
        bins: int = 10,
    ) -> pd.DataFrame:

        frame = pd.DataFrame(
            {
                "actual": (
                    pd.to_numeric(
                        actual,
                        errors="coerce",
                    )
                ),

                "probability": (
                    pd.to_numeric(
                        probability,
                        errors="coerce",
                    )
                ),
            }
        ).dropna()

        frame["bin"] = pd.cut(
            frame["probability"],

            bins=np.linspace(
                0.0,
                1.0,
                bins + 1,
            ),

            include_lowest=True,

            right=True,
        )

        grouped = (
            frame
            .groupby(
                "bin",
                observed=False,
            )
            .agg(
                games=(
                    "actual",
                    "size",
                ),

                average_probability=(
                    "probability",
                    "mean",
                ),

                actual_home_win_rate=(
                    "actual",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped["calibration_error"] = (
            grouped[
                "actual_home_win_rate"
            ]
            -
            grouped[
                "average_probability"
            ]
        )

        return grouped