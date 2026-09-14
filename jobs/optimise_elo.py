from __future__ import annotations

from dataclasses import asdict
from itertools import product
from math import log
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import (
    NATIONAL_MEAN_ELO,
    PROCESSED_DATA_DIR,
)

from engine.elo import (
    EloConfig,
    EloEngine,
)


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "historical_games.csv"
)

RESULTS_FILE = (
    PROCESSED_DATA_DIR
    / "elo_optimisation_results.csv"
)

BEST_CONFIG_FILE = (
    PROCESSED_DATA_DIR
    / "elo_best_config.csv"
)


# ============================================================
# SPLIT RULES
# ============================================================

WARMUP_SEASON = 2023
TUNING_SEASON = 2024
HOLDOUT_SEASON = 2025


# ============================================================
# PARAMETER GRID
# ============================================================

HOME_FIELD_ADVANTAGES = [
    35.0,
    45.0,
    55.0,
    65.0,
]

K_FACTORS = [
    14.0,
    18.0,
    22.0,
    26.0,
]

PRESEASON_REGRESSIONS = [
    0.25,
    0.35,
    0.45,
    0.55,
]

FBS_OVER_FCS_WEIGHTS = [
    0.10,
    0.20,
    0.30,
    0.40,
]

FCS_OVER_FBS_WEIGHTS = [
    0.50,
    0.70,
    0.90,
    1.00,
]

FBS_FCS_MOV_CAPS = [
    1.00,
    1.20,
    1.35,
    1.50,
]


# ============================================================
# HELPERS
# ============================================================


def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def normalise_bool(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def binary_log_loss(
    y_true: pd.Series,
    y_prob: pd.Series,
) -> float:
    """
    Binary log loss.

    Ties are excluded before this function is called.
    """

    eps = 1e-15

    probs = (
        y_prob
        .astype(float)
        .clip(
            lower=eps,
            upper=1.0 - eps,
        )
    )

    truth = (
        y_true
        .astype(float)
    )

    losses = -(
        truth
        * probs.map(log)
        +
        (1.0 - truth)
        * (1.0 - probs).map(log)
    )

    return float(
        losses.mean()
    )


def evaluate_season(
    history: pd.DataFrame,
    season: int,
) -> dict:
    """
    Evaluate one season.

    Accuracy and log loss exclude ties.
    Brier retains ties as 0.5 because ELO itself models
    expected score rather than only binary winner.
    """

    df = history[
        history["season"] == season
    ].copy()

    if df.empty:

        raise ValueError(
            f"No rows found for season {season}"
        )

    non_ties = df[
        df["actual_home_result"] != 0.5
    ].copy()

    non_ties["predicted_home_win"] = (
        non_ties[
            "home_elo_win_probability"
        ]
        >= 0.50
    ).astype(int)

    non_ties["actual_home_win"] = (
        non_ties[
            "actual_home_result"
        ]
        ==
        1.0
    ).astype(int)

    accuracy = float(
        (
            non_ties["predicted_home_win"]
            ==
            non_ties["actual_home_win"]
        ).mean()
    )

    brier = float(
        (
            (
                df[
                    "home_elo_win_probability"
                ]
                -
                df[
                    "actual_home_result"
                ]
            )
            ** 2
        ).mean()
    )

    log_loss = binary_log_loss(
        y_true=non_ties[
            "actual_home_win"
        ],
        y_prob=non_ties[
            "home_elo_win_probability"
        ],
    )

    return {
        "games": len(df),
        "non_tie_games": len(non_ties),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
    }


def build_model_universe(
    games: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "fbs_vs_fbs",
        "fbs_vs_fcs",
        "home_classification",
        "away_classification",
    ]

    missing = [
        column
        for column in required
        if column not in games.columns
    ]

    if missing:

        raise RuntimeError(
            "historical_games.csv is missing required "
            f"columns: {missing}"
        )

    data = games.copy()

    data["fbs_vs_fbs"] = (
        normalise_bool(
            data["fbs_vs_fbs"]
        )
    )

    data["fbs_vs_fcs"] = (
        normalise_bool(
            data["fbs_vs_fcs"]
        )
    )

    data = data[
        data["fbs_vs_fbs"]
        |
        data["fbs_vs_fcs"]
    ].copy()

    return data


def run_config(
    games: pd.DataFrame,
    config: EloConfig,
) -> tuple[
    pd.DataFrame,
    dict,
    dict,
]:

    engine = EloEngine(
        config=config
    )

    history = (
        engine.process_games(
            games
        )
    )

    tune = evaluate_season(
        history,
        TUNING_SEASON,
    )

    holdout = evaluate_season(
        history,
        HOLDOUT_SEASON,
    )

    return (
        history,
        tune,
        holdout,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 — ELO OPTIMISATION"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nHistorical game dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run:\n"
            "python -m jobs.build_data"
        )

    games = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    games = build_model_universe(
        games
    )

    games = games[
        games["season"].isin(
            [
                WARMUP_SEASON,
                TUNING_SEASON,
                HOLDOUT_SEASON,
            ]
        )
    ].copy()

    section(
        "DATA SPLIT"
    )

    print(
        f"{WARMUP_SEASON}: warm-up / rating establishment"
    )

    print(
        f"{TUNING_SEASON}: tuning season"
    )

    print(
        f"{HOLDOUT_SEASON}: untouched holdout"
    )

    print()

    for season in [
        WARMUP_SEASON,
        TUNING_SEASON,
        HOLDOUT_SEASON,
    ]:

        count = (
            games[
                games["season"]
                ==
                season
            ]
            .shape[0]
        )

        print(
            f"{season}: {count:,} eligible games"
        )

    # ========================================================
    # CURRENT BASELINE
    # ========================================================

    section(
        "CURRENT DEFAULT BASELINE"
    )

    baseline_config = EloConfig(
        starting_rating=NATIONAL_MEAN_ELO,
        home_field_advantage=55.0,
        k_factor=22.0,
        preseason_regression=0.35,
        mov_multiplier_enabled=True,
        fbs_over_fcs_weight=0.25,
        fcs_over_fbs_weight=0.70,
        fbs_fcs_mov_cap=1.35,
    )

    (
        _,
        baseline_tune,
        baseline_holdout,
    ) = run_config(
        games,
        baseline_config,
    )

    print(
        f"{TUNING_SEASON} tuning:"
    )

    print(
        f"  Accuracy: {baseline_tune['accuracy']:.2%}"
    )

    print(
        f"  Brier:    {baseline_tune['brier']:.4f}"
    )

    print(
        f"  LogLoss:  {baseline_tune['log_loss']:.4f}"
    )

    print()

    print(
        f"{HOLDOUT_SEASON} holdout:"
    )

    print(
        f"  Accuracy: {baseline_holdout['accuracy']:.2%}"
    )

    print(
        f"  Brier:    {baseline_holdout['brier']:.4f}"
    )

    print(
        f"  LogLoss:  {baseline_holdout['log_loss']:.4f}"
    )

    # ========================================================
    # GRID SIZE
    # ========================================================

    combinations = list(
        product(
            HOME_FIELD_ADVANTAGES,
            K_FACTORS,
            PRESEASON_REGRESSIONS,
            FBS_OVER_FCS_WEIGHTS,
            FCS_OVER_FBS_WEIGHTS,
            FBS_FCS_MOV_CAPS,
        )
    )

    section(
        "PARAMETER SEARCH"
    )

    print(
        f"Configurations to test: "
        f"{len(combinations):,}"
    )

    print()

    print(
        "Selection rule:"
    )

    print(
        f"  1. Best {TUNING_SEASON} Brier score"
    )

    print(
        f"  2. Then best {TUNING_SEASON} LogLoss"
    )

    print(
        f"  3. Then best {TUNING_SEASON} accuracy"
    )

    print()

    print(
        f"{HOLDOUT_SEASON} is NOT used to choose parameters."
    )

    # ========================================================
    # SEARCH
    # ========================================================

    results: list[dict] = []

    total = len(
        combinations
    )

    for index, values in enumerate(
        combinations,
        start=1,
    ):

        (
            hfa,
            k_factor,
            regression,
            fbs_win_weight,
            fcs_upset_weight,
            mov_cap,
        ) = values

        config = EloConfig(
            starting_rating=(
                NATIONAL_MEAN_ELO
            ),
            home_field_advantage=hfa,
            k_factor=k_factor,
            preseason_regression=regression,
            mov_multiplier_enabled=True,
            fbs_over_fcs_weight=(
                fbs_win_weight
            ),
            fcs_over_fbs_weight=(
                fcs_upset_weight
            ),
            fbs_fcs_mov_cap=(
                mov_cap
            ),
        )

        (
            _,
            tune,
            holdout,
        ) = run_config(
            games,
            config,
        )

        results.append(
            {
                "home_field_advantage": (
                    hfa
                ),
                "k_factor": (
                    k_factor
                ),
                "preseason_regression": (
                    regression
                ),
                "fbs_over_fcs_weight": (
                    fbs_win_weight
                ),
                "fcs_over_fbs_weight": (
                    fcs_upset_weight
                ),
                "fbs_fcs_mov_cap": (
                    mov_cap
                ),

                "tuning_accuracy": (
                    tune[
                        "accuracy"
                    ]
                ),
                "tuning_brier": (
                    tune[
                        "brier"
                    ]
                ),
                "tuning_log_loss": (
                    tune[
                        "log_loss"
                    ]
                ),

                "holdout_accuracy": (
                    holdout[
                        "accuracy"
                    ]
                ),
                "holdout_brier": (
                    holdout[
                        "brier"
                    ]
                ),
                "holdout_log_loss": (
                    holdout[
                        "log_loss"
                    ]
                ),
            }
        )

        if (
            index % 250 == 0
            or
            index == total
        ):

            print(
                f"Completed "
                f"{index:,}/{total:,}"
            )

    # ========================================================
    # RANK
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "tuning_brier",
                "tuning_log_loss",
                "tuning_accuracy",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "tuning_rank"
    ] = (
        results_df.index
        +
        1
    )

    # ========================================================
    # BEST CONFIG
    # ========================================================

    best = (
        results_df
        .iloc[0]
    )

    best_config = EloConfig(
        starting_rating=(
            NATIONAL_MEAN_ELO
        ),
        home_field_advantage=float(
            best[
                "home_field_advantage"
            ]
        ),
        k_factor=float(
            best[
                "k_factor"
            ]
        ),
        preseason_regression=float(
            best[
                "preseason_regression"
            ]
        ),
        mov_multiplier_enabled=True,
        fbs_over_fcs_weight=float(
            best[
                "fbs_over_fcs_weight"
            ]
        ),
        fcs_over_fbs_weight=float(
            best[
                "fcs_over_fbs_weight"
            ]
        ),
        fbs_fcs_mov_cap=float(
            best[
                "fbs_fcs_mov_cap"
            ]
        ),
    )

    section(
        "BEST TUNING CONFIGURATION"
    )

    print(
        f"Home-field advantage: "
        f"{best_config.home_field_advantage:.1f}"
    )

    print(
        f"K factor:             "
        f"{best_config.k_factor:.1f}"
    )

    print(
        f"Preseason regression: "
        f"{best_config.preseason_regression:.0%}"
    )

    print(
        f"FBS-over-FCS weight:   "
        f"{best_config.fbs_over_fcs_weight:.2f}"
    )

    print(
        f"FCS-upset weight:      "
        f"{best_config.fcs_over_fbs_weight:.2f}"
    )

    print(
        f"FBS/FCS MOV cap:       "
        f"{best_config.fbs_fcs_mov_cap:.2f}"
    )

    print()

    print(
        f"{TUNING_SEASON}:"
    )

    print(
        f"  Accuracy: "
        f"{best['tuning_accuracy']:.2%}"
    )

    print(
        f"  Brier:    "
        f"{best['tuning_brier']:.4f}"
    )

    print(
        f"  LogLoss:  "
        f"{best['tuning_log_loss']:.4f}"
    )

    # ========================================================
    # HOLDOUT REVEAL
    # ========================================================

    section(
        f"{HOLDOUT_SEASON} HOLDOUT RESULT"
    )

    print(
        f"Accuracy: "
        f"{best['holdout_accuracy']:.2%}"
    )

    print(
        f"Brier:    "
        f"{best['holdout_brier']:.4f}"
    )

    print(
        f"LogLoss:  "
        f"{best['holdout_log_loss']:.4f}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    section(
        "DEFAULT VS OPTIMISED — HOLDOUT"
    )

    accuracy_change = (
        float(
            best[
                "holdout_accuracy"
            ]
        )
        -
        baseline_holdout[
            "accuracy"
        ]
    )

    brier_change = (
        float(
            best[
                "holdout_brier"
            ]
        )
        -
        baseline_holdout[
            "brier"
        ]
    )

    logloss_change = (
        float(
            best[
                "holdout_log_loss"
            ]
        )
        -
        baseline_holdout[
            "log_loss"
        ]
    )

    print(
        "Accuracy:"
    )

    print(
        f"  Default:   "
        f"{baseline_holdout['accuracy']:.2%}"
    )

    print(
        f"  Optimised: "
        f"{best['holdout_accuracy']:.2%}"
    )

    print(
        f"  Change:    "
        f"{accuracy_change:+.2%}"
    )

    print()

    print(
        "Brier:"
    )

    print(
        f"  Default:   "
        f"{baseline_holdout['brier']:.4f}"
    )

    print(
        f"  Optimised: "
        f"{best['holdout_brier']:.4f}"
    )

    print(
        f"  Change:    "
        f"{brier_change:+.4f}"
    )

    print()

    print(
        "LogLoss:"
    )

    print(
        f"  Default:   "
        f"{baseline_holdout['log_loss']:.4f}"
    )

    print(
        f"  Optimised: "
        f"{best['holdout_log_loss']:.4f}"
    )

    print(
        f"  Change:    "
        f"{logloss_change:+.4f}"
    )

    # ========================================================
    # TOP CONFIGURATIONS
    # ========================================================

    section(
        "TOP 10 TUNING CONFIGURATIONS"
    )

    display_columns = [
        "tuning_rank",
        "home_field_advantage",
        "k_factor",
        "preseason_regression",
        "fbs_over_fcs_weight",
        "fcs_over_fbs_weight",
        "fbs_fcs_mov_cap",
        "tuning_accuracy",
        "tuning_brier",
        "tuning_log_loss",
        "holdout_accuracy",
        "holdout_brier",
        "holdout_log_loss",
    ]

    print(
        results_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING OPTIMISATION OUTPUT"
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        f"✓ Saved "
        f"{len(results_df):,} configurations "
        f"→ {RESULTS_FILE}"
    )

    best_config_df = pd.DataFrame(
        [
            asdict(
                best_config
            )
        ]
    )

    best_config_df.to_csv(
        BEST_CONFIG_FILE,
        index=False,
    )

    print(
        f"✓ Saved best config "
        f"→ {BEST_CONFIG_FILE}"
    )

        # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "ELO OPTIMISATION COMPLETE"
    )

    print(
        f"The winning parameters were chosen "
        f"using {TUNING_SEASON} only."
    )

    print(
        f"{HOLDOUT_SEASON} was used only after selection "
        f"to test generalisation."
    )

    print()


if __name__ == "__main__":
    main()