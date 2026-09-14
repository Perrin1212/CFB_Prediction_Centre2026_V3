from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Calibration design:

        2024 OOS predictions
            early 60% -> fit learned calibrators
            late 40%  -> choose architecture

        2025
            untouched forward evaluation

    Maturity is NOT allowed to act as a directional predictor.

    For maturity-shrink candidates:

        adjusted_logit
            =
        raw_logit
            *
        [1 - alpha * (1 - maturity)]

    Therefore lower maturity can only move probabilities
    toward 50%, never make them more extreme.
    """

    development_season: int = 2024
    holdout_season: int = 2025

    calibration_train_fraction: float = 0.60

    logistic_c: float = 1.0
    max_iter: int = 2000

    probability_floor: float = 0.001
    probability_ceiling: float = 0.999


@dataclass
class CalibrationModelBundle:

    name: str
    model: LogisticRegression | None = None
    shrink_alpha: float = 0.0


class CalibrationEngine:

    def __init__(
        self,
        config: CalibrationConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else CalibrationConfig()
        )

    # ========================================================
    # MATH
    # ========================================================

    @staticmethod
    def safe_logit(
        probability,
    ) -> np.ndarray:

        p = np.asarray(
            probability,
            dtype=float,
        )

        p = np.clip(
            p,
            1e-6,
            1.0 - 1e-6,
        )

        return np.log(
            p / (1.0 - p)
        )

    @staticmethod
    def sigmoid(
        value,
    ) -> np.ndarray:

        x = np.asarray(
            value,
            dtype=float,
        )

        return (
            1.0
            /
            (
                1.0
                +
                np.exp(-x)
            )
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_dataset(
        self,
        probability_history: pd.DataFrame,
    ) -> pd.DataFrame:

        data = probability_history.copy()

        required = [
            "season",
            "actual_home_win",
            "v2_raw_home_win_probability",
            "matchup_rating_maturity",
            "probability_is_out_of_sample",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Probability history is missing calibration "
                f"fields: {missing}"
            )

        for column in [
            "season",
            "actual_home_win",
            "v2_raw_home_win_probability",
            "matchup_rating_maturity",
        ]:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        # ----------------------------------------------------
        # NORMALISE OOS FLAG
        # ----------------------------------------------------

        flag = (
            data[
                "probability_is_out_of_sample"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data[
            "probability_is_out_of_sample"
        ] = flag.isin(
            [
                "true",
                "1",
                "yes",
            ]
        )

        data = data[
            data[
                "probability_is_out_of_sample"
            ]
        ].copy()

        data = data.dropna(
            subset=[
                "season",
                "actual_home_win",
                "v2_raw_home_win_probability",
                "matchup_rating_maturity",
            ]
        )

        # ----------------------------------------------------
        # MATURITY SAFETY
        # ----------------------------------------------------

        data[
            "matchup_rating_maturity"
        ] = (
            data[
                "matchup_rating_maturity"
            ]
            .clip(
                0.0,
                1.0,
            )
        )

        data[
            "raw_v2_logit"
        ] = self.safe_logit(
            data[
                "v2_raw_home_win_probability"
            ]
        )

        # ----------------------------------------------------
        # CHRONOLOGY
        # ----------------------------------------------------

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

        elif "week" in data.columns:

            sort_columns = [
                "season",
                "week",
                "game_id",
            ]

        else:

            sort_columns = [
                "season",
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

        return data

    # ========================================================
    # CANDIDATES
    # ========================================================

    @staticmethod
    def candidate_architectures(
    ) -> dict[str, float | None]:

        return {
            "identity": None,
            "platt": None,

            "maturity_shrink_25": 0.25,
            "maturity_shrink_50": 0.50,
            "maturity_shrink_75": 0.75,
            "maturity_shrink_100": 1.00,
        }

    # ========================================================
    # FIT
    # ========================================================

    def fit_model(
        self,
        data: pd.DataFrame,
        model_name: str,
        shrink_alpha: float | None,
    ) -> CalibrationModelBundle:

        if model_name == "platt":

            X = (
                data[
                    [
                        "raw_v2_logit"
                    ]
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

            model = LogisticRegression(
                C=self.config.logistic_c,
                max_iter=self.config.max_iter,
                solver="lbfgs",
                random_state=2026,
            )

            model.fit(
                X,
                y,
            )

            return CalibrationModelBundle(
                name=model_name,
                model=model,
                shrink_alpha=0.0,
            )

        return CalibrationModelBundle(
            name=model_name,
            model=None,
            shrink_alpha=(
                0.0
                if shrink_alpha is None
                else float(
                    shrink_alpha
                )
            ),
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        bundle: CalibrationModelBundle,
        data: pd.DataFrame,
    ) -> np.ndarray:

        raw_probability = (
            data[
                "v2_raw_home_win_probability"
            ]
            .astype(float)
            .to_numpy()
        )

        if bundle.name == "identity":

            probability = raw_probability

        elif bundle.name == "platt":

            if bundle.model is None:

                raise RuntimeError(
                    "Platt calibrator has no fitted model."
                )

            X = (
                data[
                    [
                        "raw_v2_logit"
                    ]
                ]
                .astype(float)
                .to_numpy()
            )

            probability = (
                bundle.model
                .predict_proba(
                    X
                )[
                    :,
                    1
                ]
            )

        elif bundle.name.startswith(
            "maturity_shrink_"
        ):

            raw_logit = (
                data[
                    "raw_v2_logit"
                ]
                .astype(float)
                .to_numpy()
            )

            maturity = (
                data[
                    "matchup_rating_maturity"
                ]
                .astype(float)
                .clip(
                    0.0,
                    1.0,
                )
                .to_numpy()
            )

            # ------------------------------------------------
            # GUARANTEED CONFIDENCE SHRINKAGE
            # ------------------------------------------------

            shrink_factor = (
                1.0
                -
                bundle.shrink_alpha
                *
                (
                    1.0
                    -
                    maturity
                )
            )

            shrink_factor = np.clip(
                shrink_factor,
                0.0,
                1.0,
            )

            adjusted_logit = (
                raw_logit
                *
                shrink_factor
            )

            probability = self.sigmoid(
                adjusted_logit
            )

        else:

            raise ValueError(
                f"Unknown calibrator: {bundle.name}"
            )

        return np.clip(
            probability,
            self.config.probability_floor,
            self.config.probability_ceiling,
        )

    # ========================================================
    # METRICS
    # ========================================================

    @staticmethod
    def accuracy(
        actual,
        probability,
    ) -> float:

        y = np.asarray(
            actual,
            dtype=int,
        )

        p = np.asarray(
            probability,
            dtype=float,
        )

        return float(
            np.mean(
                (
                    p > 0.50
                ).astype(int)
                ==
                y
            )
        )

    @staticmethod
    def brier(
        actual,
        probability,
    ) -> float:

        y = np.asarray(
            actual,
            dtype=float,
        )

        p = np.asarray(
            probability,
            dtype=float,
        )

        return float(
            np.mean(
                (
                    p - y
                )
                ** 2
            )
        )

    @staticmethod
    def binary_log_loss(
        actual,
        probability,
    ) -> float:

        y = np.asarray(
            actual,
            dtype=float,
        )

        p = np.asarray(
            probability,
            dtype=float,
        )

        p = np.clip(
            p,
            1e-6,
            1.0 - 1e-6,
        )

        return float(
            -np.mean(
                y
                *
                np.log(p)
                +
                (
                    1.0 - y
                )
                *
                np.log(
                    1.0 - p
                )
            )
        )

    def evaluate(
        self,
        data: pd.DataFrame,
        probability: np.ndarray,
    ) -> dict[str, float]:

        actual = (
            data[
                "actual_home_win"
            ]
            .astype(int)
            .to_numpy()
        )

        return {
            "games": int(
                len(data)
            ),

            "accuracy": self.accuracy(
                actual,
                probability,
            ),

            "brier": self.brier(
                actual,
                probability,
            ),

            "log_loss": self.binary_log_loss(
                actual,
                probability,
            ),

            "minimum_probability": float(
                np.min(
                    probability
                )
            ),

            "median_probability": float(
                np.median(
                    probability
                )
            ),

            "maximum_probability": float(
                np.max(
                    probability
                )
            ),
        }

    # ========================================================
    # CALIBRATION TABLE
    # ========================================================

    def calibration_table(
        self,
        data: pd.DataFrame,
        probability: np.ndarray,
        bins: int = 10,
    ) -> pd.DataFrame:

        frame = pd.DataFrame(
            {
                "actual": (
                    data[
                        "actual_home_win"
                    ]
                    .astype(float)
                    .to_numpy()
                ),

                "probability": probability,
            }
        )

        frame["bin"] = pd.cut(
            frame[
                "probability"
            ],

            bins=np.linspace(
                0.0,
                1.0,
                bins + 1,
            ),

            include_lowest=True,
        )

        result = (
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

        result[
            "calibration_error"
        ] = (
            result[
                "actual_home_win_rate"
            ]
            -
            result[
                "average_probability"
            ]
        )

        return result