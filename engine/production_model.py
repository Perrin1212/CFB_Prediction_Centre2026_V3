from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PRODUCTION MODEL
# ============================================================


@dataclass
class ProductionProbabilityModel:
    """
    Frozen CFB Prediction Centre V2 probability model.

    Final directional architecture:

        ELO log-odds
            +
        offensive matchup difference
            ↓
        logistic probability model

    No post-model probability calibration is currently applied.

    The fitted sklearn scaler/model are stored inside this
    object through the model bundle dictionary.
    """

    architecture_name: str

    features: list[str]

    scaler: Any

    model: Any

    probability_floor: float = 0.001
    probability_ceiling: float = 0.999

    # ========================================================
    # MATH
    # ========================================================

    @staticmethod
    def safe_logit(
        probability: pd.Series | np.ndarray,
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
            p
            /
            (
                1.0
                -
                p
            )
        )

    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def prepare_features(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        data = frame.copy()

        required = [
            "home_elo_win_probability",
            "offensive_matchup_difference",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Production prediction input is missing "
                f"required fields: {missing}"
            )

        data[
            "home_elo_win_probability"
        ] = pd.to_numeric(
            data[
                "home_elo_win_probability"
            ],
            errors="coerce",
        )

        data[
            "offensive_matchup_difference"
        ] = pd.to_numeric(
            data[
                "offensive_matchup_difference"
            ],
            errors="coerce",
        )

        data[
            "elo_logit"
        ] = self.safe_logit(
            data[
                "home_elo_win_probability"
            ]
        )

        missing_values = (
            data[
                self.features
            ]
            .isna()
            .any(
                axis=1
            )
        )

        if missing_values.any():

            bad_rows = int(
                missing_values.sum()
            )

            raise ValueError(
                f"{bad_rows:,} production rows contain "
                "missing model features."
            )

        return data

    # ========================================================
    # PREDICT
    # ========================================================

    def predict_home_probability(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:

        data = self.prepare_features(
            frame
        )

        X = (
            data[
                self.features
            ]
            .astype(float)
            .to_numpy()
        )

        X_scaled = (
            self.scaler.transform(
                X
            )
        )

        probability = (
            self.model.predict_proba(
                X_scaled
            )[
                :,
                1
            ]
        )

        return np.clip(
            probability,
            self.probability_floor,
            self.probability_ceiling,
        )

    # ========================================================
    # FULL PREDICTION OUTPUT
    # ========================================================

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        output = frame.copy()

        home_probability = (
            self.predict_home_probability(
                output
            )
        )

        output[
            "v2_home_win_probability"
        ] = home_probability

        output[
            "v2_away_win_probability"
        ] = (
            1.0
            -
            output[
                "v2_home_win_probability"
            ]
        )

        output[
            "v2_predicted_home"
        ] = (
            output[
                "v2_home_win_probability"
            ]
            >
            0.50
        )

        output[
            "v2_predicted_team"
        ] = np.where(
            output[
                "v2_predicted_home"
            ],

            output.get(
                "home_team",
                "HOME",
            ),

            output.get(
                "away_team",
                "AWAY",
            ),
        )

        output[
            "v2_prediction_probability"
        ] = np.maximum(
            output[
                "v2_home_win_probability"
            ],

            output[
                "v2_away_win_probability"
            ],
        )

        output[
            "v2_probability_architecture"
        ] = (
            self.architecture_name
        )

        output[
            "v2_calibration_method"
        ] = "identity"

        return output

    # ========================================================
    # SAVE / LOAD
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        destination = Path(
            path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ProductionProbabilityModel":

        source = Path(
            path
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Production model not found:\n{source}"
            )

        model = joblib.load(
            source
        )

        if not isinstance(
            model,
            cls,
        ):

            raise TypeError(
                "Loaded object is not a "
                "ProductionProbabilityModel."
            )

        return model