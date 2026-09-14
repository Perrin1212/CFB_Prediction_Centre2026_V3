from __future__ import annotations

from dataclasses import asdict
from itertools import product
from math import log
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    / "elo_refinement_results.csv"
)

BEST_CONFIG_FILE = (
    PROCESSED_DATA_DIR
    / "elo_refined_best_config.csv"
)


# ============================================================
# SEASON SPLIT
# ============================================================

WARMUP_SEASON = 2023
TUNING_SEASON = 2024
HOLDOUT_SEASON = 2025


# ============================================================
# REFINED PARAMETER GRID
# ============================================================

HOME_FIELD_ADVANTAGES = [
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
]

K_FACTORS = [
    22.0,
    24.0,
    26.0,
    28.0,
    30.0,
]

PRESEASON_REGRESSIONS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
]

FBS_OVER_FCS_WEIGHTS = [
    0.30,
    0.40,
    0.50,
    0.60,
]

FCS_OVER_FBS_WEIGHTS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

FBS_FCS_MOV_CAPS = [
    1.35,
    1.50,
    1.65,
    1.80,
]


# ============================================================
# HELPERS
# ============================================================


def section(title: str) -> None:

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

    non_ties["actual_home_win"] = (
        non_ties["actual_home_result"]
        ==
        1.0
    ).astype(int)

    non_ties["predicted_home_win"] = (
        non_ties[
            "home_elo_win_probability"
        ]
        >=
        0.50
    ).astype(int)

    accuracy = float(
        (
            non_ties[
                "predicted_home_win"
            ]
            ==
            non_ties[
                "actual_home_win"
            ]
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
        "fbs_vs_fbs",
        "fbs_vs_fcs",
    ]

    missing = [
        column
        for column in required_columns
        if column not in games.columns
    ]

    if missing:

        raise RuntimeError(
            "historical_games.csv is missing "
            f"required columns: {missing}"
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

    tuning = evaluate_season(
        history,
        TUNING_SEASON,
    )

    holdout = evaluate_season(
        history,
        HOLDOUT_SEASON,
    )

    return (
        history,
        tuning,
        holdout,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 — ELO REFINEMENT"
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

    # ========================================================
    # DATA SPLIT
    # ========================================================

    section(
        "DATA SPLIT"
    )

    print(
        f"{WARMUP_SEASON}: warm-up / rating establishment"
    )

    print(
        f"{TUNING_SEASON}: optimisation target"
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
            f"{season}: {count:,} games"
        )

    # ========================================================
    # ORIGINAL DEFAULT
    # ========================================================

    section(
        "ORIGINAL DEFAULT CONFIG"
    )

    original_config = EloConfig(
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
        original_tuning,
        original_holdout,
    ) = run_config(
        games,
        original_config,
    )

    print(
        f"{TUNING_SEASON}: "
        f"{original_tuning['accuracy']:.2%} accuracy | "
        f"{original_tuning['brier']:.4f} Brier | "
        f"{original_tuning['log_loss']:.4f} LogLoss"
    )

    print(
        f"{HOLDOUT_SEASON}: "
        f"{original_holdout['accuracy']:.2%} accuracy | "
        f"{original_holdout['brier']:.4f} Brier | "
        f"{original_holdout['log_loss']:.4f} LogLoss"
    )

    # ========================================================
    # FIRST-PASS WINNER
    # ========================================================

    section(
        "FIRST-PASS OPTIMISED CONFIG"
    )

    first_pass_config = EloConfig(
        starting_rating=NATIONAL_MEAN_ELO,
        home_field_advantage=65.0,
        k_factor=26.0,
        preseason_regression=0.45,
        mov_multiplier_enabled=True,
        fbs_over_fcs_weight=0.40,
        fcs_over_fbs_weight=0.50,
        fbs_fcs_mov_cap=1.50,
    )

    (
        _,
        first_tuning,
        first_holdout,
    ) = run_config(
        games,
        first_pass_config,
    )

    print(
        f"{TUNING_SEASON}: "
        f"{first_tuning['accuracy']:.2%} accuracy | "
        f"{first_tuning['brier']:.4f} Brier | "
        f"{first_tuning['log_loss']:.4f} LogLoss"
    )

    print(
        f"{HOLDOUT_SEASON}: "
        f"{first_holdout['accuracy']:.2%} accuracy | "
        f"{first_holdout['brier']:.4f} Brier | "
        f"{first_holdout['log_loss']:.4f} LogLoss"
    )

    # ========================================================
    # GRID
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

    total = len(
        combinations
    )

    section(
        "REFINEMENT SEARCH"
    )

    print(
        f"Configurations to test: {total:,}"
    )

    print()

    print(
        "Selection uses 2024 only."
    )

    print(
        "Ranking order:"
    )

    print(
        "  1. Lowest Brier score"
    )

    print(
        "  2. Lowest LogLoss"
    )

    print(
        "  3. Highest winner accuracy"
    )

    print()

    print(
        "2025 remains hidden from parameter selection."
    )

    # ========================================================
    # SEARCH
    # ========================================================

    results: list[dict] = []

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
            starting_rating=NATIONAL_MEAN_ELO,
            home_field_advantage=hfa,
            k_factor=k_factor,
            preseason_regression=regression,
            mov_multiplier_enabled=True,
            fbs_over_fcs_weight=fbs_win_weight,
            fcs_over_fbs_weight=fcs_upset_weight,
            fbs_fcs_mov_cap=mov_cap,
        )

        (
            _,
            tuning,
            holdout,
        ) = run_config(
            games,
            config,
        )

        results.append(
            {
                "home_field_advantage": hfa,
                "k_factor": k_factor,
                "preseason_regression": regression,
                "fbs_over_fcs_weight": fbs_win_weight,
                "fcs_over_fbs_weight": fcs_upset_weight,
                "fbs_fcs_mov_cap": mov_cap,

                "tuning_accuracy": (
                    tuning["accuracy"]
                ),

                "tuning_brier": (
                    tuning["brier"]
                ),

                "tuning_log_loss": (
                    tuning["log_loss"]
                ),

                "holdout_accuracy": (
                    holdout["accuracy"]
                ),

                "holdout_brier": (
                    holdout["brier"]
                ),

                "holdout_log_loss": (
                    holdout["log_loss"]
                ),
            }
        )

        if (
            index % 500 == 0
            or
            index == total
        ):

            print(
                f"Completed "
                f"{index:,}/{total:,}"
            )

    # ========================================================
    # RANK RESULTS
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

    best = (
        results_df
        .iloc[0]
    )

    best_config = EloConfig(
        starting_rating=NATIONAL_MEAN_ELO,

        home_field_advantage=float(
            best["home_field_advantage"]
        ),

        k_factor=float(
            best["k_factor"]
        ),

        preseason_regression=float(
            best["preseason_regression"]
        ),

        mov_multiplier_enabled=True,

        fbs_over_fcs_weight=float(
            best["fbs_over_fcs_weight"]
        ),

        fcs_over_fbs_weight=float(
            best["fcs_over_fbs_weight"]
        ),

        fbs_fcs_mov_cap=float(
            best["fbs_fcs_mov_cap"]
        ),
    )

    # ========================================================
    # BEST CONFIG
    # ========================================================

    section(
        "BEST REFINED CONFIGURATION"
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
        f"{HOLDOUT_SEASON} HOLDOUT REVEAL"
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
    # THREE-WAY COMPARISON
    # ========================================================

    section(
        "DEFAULT VS FIRST PASS VS REFINED"
    )

    print(
        "2025 HOLDOUT"
    )

    print()

    print(
        "DEFAULT"
    )

    print(
        f"  Accuracy: "
        f"{original_holdout['accuracy']:.2%}"
    )

    print(
        f"  Brier:    "
        f"{original_holdout['brier']:.4f}"
    )

    print(
        f"  LogLoss:  "
        f"{original_holdout['log_loss']:.4f}"
    )

    print()

    print(
        "FIRST PASS"
    )

    print(
        f"  Accuracy: "
        f"{first_holdout['accuracy']:.2%}"
    )

    print(
        f"  Brier:    "
        f"{first_holdout['brier']:.4f}"
    )

    print(
        f"  LogLoss:  "
        f"{first_holdout['log_loss']:.4f}"
    )

    print()

    print(
        "REFINED"
    )

    print(
        f"  Accuracy: "
        f"{best['holdout_accuracy']:.2%}"
    )

    print(
        f"  Brier:    "
        f"{best['holdout_brier']:.4f}"
    )

    print(
        f"  LogLoss:  "
        f"{best['holdout_log_loss']:.4f}"
    )

    # ========================================================
    # CHECK WHETHER BEST IS STILL ON GRID EDGE
    # ========================================================

    section(
        "BOUNDARY CHECK"
    )

    boundary_flags = []

    if (
        best_config.home_field_advantage
        ==
        max(HOME_FIELD_ADVANTAGES)
    ):

        boundary_flags.append(
            "Home-field advantage hit upper boundary"
        )

    if (
        best_config.home_field_advantage
        ==
        min(HOME_FIELD_ADVANTAGES)
    ):

        boundary_flags.append(
            "Home-field advantage hit lower boundary"
        )

    if (
        best_config.k_factor
        ==
        max(K_FACTORS)
    ):

        boundary_flags.append(
            "K factor hit upper boundary"
        )

    if (
        best_config.k_factor
        ==
        min(K_FACTORS)
    ):

        boundary_flags.append(
            "K factor hit lower boundary"
        )

    if (
        best_config.preseason_regression
        ==
        max(PRESEASON_REGRESSIONS)
    ):

        boundary_flags.append(
            "Preseason regression hit upper boundary"
        )

    if (
        best_config.preseason_regression
        ==
        min(PRESEASON_REGRESSIONS)
    ):

        boundary_flags.append(
            "Preseason regression hit lower boundary"
        )

    if (
        best_config.fbs_over_fcs_weight
        ==
        max(FBS_OVER_FCS_WEIGHTS)
    ):

        boundary_flags.append(
            "FBS-over-FCS weight hit upper boundary"
        )

    if (
        best_config.fbs_over_fcs_weight
        ==
        min(FBS_OVER_FCS_WEIGHTS)
    ):

        boundary_flags.append(
            "FBS-over-FCS weight hit lower boundary"
        )

    if (
        best_config.fcs_over_fbs_weight
        ==
        max(FCS_OVER_FBS_WEIGHTS)
    ):

        boundary_flags.append(
            "FCS-upset weight hit upper boundary"
        )

    if (
        best_config.fcs_over_fbs_weight
        ==
        min(FCS_OVER_FBS_WEIGHTS)
    ):

        boundary_flags.append(
            "FCS-upset weight hit lower boundary"
        )

    if (
        best_config.fbs_fcs_mov_cap
        ==
        max(FBS_FCS_MOV_CAPS)
    ):

        boundary_flags.append(
            "FBS/FCS MOV cap hit upper boundary"
        )

    if (
        best_config.fbs_fcs_mov_cap
        ==
        min(FBS_FCS_MOV_CAPS)
    ):

        boundary_flags.append(
            "FBS/FCS MOV cap hit lower boundary"
        )

    if boundary_flags:

        print(
            "Some parameters still sit on "
            "the edge of the search space:"
        )

        print()

        for flag in boundary_flags:

            print(
                f"  ⚠ {flag}"
            )

    else:

        print(
            "✓ Best configuration is inside "
            "the refined parameter space."
        )

        print(
            "This is a good signal that the "
            "search region is broad enough."
        )

    # ========================================================
    # TOP 15
    # ========================================================

    section(
        "TOP 15 REFINED CONFIGURATIONS"
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
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING REFINEMENT OUTPUT"
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        f"✓ Saved {len(results_df):,} configurations "
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
        f"✓ Saved best refined config "
        f"→ {BEST_CONFIG_FILE}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "ELO REFINEMENT COMPLETE"
    )

    print(
        f"The refined parameters were selected "
        f"using {TUNING_SEASON} only."
    )

    print(
        f"{HOLDOUT_SEASON} was revealed only "
        f"after parameter selection."
    )

    print()


if __name__ == "__main__":
    main()