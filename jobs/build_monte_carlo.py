from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from config.settings import PROCESSED_DATA_DIR

from engine.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
)


# ============================================================
# FILES
# ============================================================

SCORING_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "scoring_history.csv"
)

SIMULATION_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "simulation_history.csv"
)


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


def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"✓ Saved {len(df):,} rows → {path}"
    )


def binary_log_loss(
    actual: pd.Series,
    probability: pd.Series,
) -> float:

    p = np.clip(
        probability.astype(float),
        1e-6,
        1.0 - 1e-6,
    )

    y = actual.astype(float)

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— MONTE CARLO SIMULATION"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    if not SCORING_HISTORY_FILE.exists():

        raise FileNotFoundError(
            "\nRequired file does not exist:\n"
            f"{SCORING_HISTORY_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING SCORING HISTORY"
    )

    scoring = pd.read_csv(
        SCORING_HISTORY_FILE,
        low_memory=False,
    )

    print(
        f"Scoring game rows: "
        f"{len(scoring):,}"
    )

    print(
        f"Unique games:      "
        f"{scoring['game_id'].nunique():,}"
    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    section(
        "SIMULATION INPUT VALIDATION"
    )

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

        "home_elo_win_probability",
    ]

    missing = [
        column
        for column in required
        if column not in scoring.columns
    ]

    if missing:

        raise RuntimeError(
            "Scoring history is missing Monte Carlo "
            f"inputs: {missing}"
        )

    core_missing = int(
        scoring[
            required
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing required values: "
        f"{core_missing:,}"
    )

    if core_missing > 0:

        raise RuntimeError(
            "Monte Carlo input contains missing values."
        )

    # ========================================================
    # CONFIG
    # ========================================================

    section(
        "MONTE CARLO CONFIGURATION"
    )

    config = MonteCarloConfig(
        simulations=3000,
        random_seed=2026,

        uncertainty_update_rate=0.05,

        starting_home_score_sd=14.0,
        starting_away_score_sd=14.0,

        starting_residual_correlation=0.10,

        minimum_score_sd=8.0,
        maximum_score_sd=22.0,

        minimum_residual_correlation=-0.35,
        maximum_residual_correlation=0.60,

        minimum_score=0.0,
        maximum_score=100.0,
    )

    print(
        f"Simulations per game:       "
        f"{config.simulations:,}"
    )

    print(
        f"Random seed:                "
        f"{config.random_seed}"
    )

    print(
        f"Uncertainty update rate:    "
        f"{config.uncertainty_update_rate:.0%}"
    )

    print(
        f"Starting home score SD:     "
        f"{config.starting_home_score_sd:.1f}"
    )

    print(
        f"Starting away score SD:     "
        f"{config.starting_away_score_sd:.1f}"
    )

    print(
        f"Starting residual corr:     "
        f"{config.starting_residual_correlation:.2f}"
    )

    print()

    print(
        "Historical residual uncertainty is updated "
        "chronologically."
    )

    print(
        "The current game's result never affects its own "
        "simulation."
    )

    print()

    print(
        "This first Monte Carlo layer produces the SCORE "
        "DISTRIBUTION."
    )

    print(
        "It is NOT yet the final calibrated win-probability "
        "engine."
    )

    # ========================================================
    # SIMULATE
    # ========================================================

    section(
        "RUNNING HISTORICAL MONTE CARLO"
    )

    engine = MonteCarloEngine(
        config=config
    )

    simulation = (
        engine.process_history(
            scoring_history=scoring
        )
    )

    print(
        f"Simulation rows created: "
        f"{len(simulation):,}"
    )

    print(
        f"Total simulated game outcomes: "
        f"{len(simulation) * config.simulations:,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "SIMULATION VALIDATION"
    )

    expected_games = int(
        scoring[
            "game_id"
        ]
        .nunique()
    )

    created_games = int(
        simulation[
            "game_id"
        ]
        .nunique()
    )

    print(
        f"Expected games: "
        f"{expected_games:,}"
    )

    print(
        f"Created games:  "
        f"{created_games:,}"
    )

    print(
        f"Missing games:  "
        f"{expected_games - created_games:,}"
    )

    core_outputs = [
        "simulation_home_win_probability",
        "simulation_away_win_probability",

        "simulation_home_score_mean",
        "simulation_away_score_mean",

        "simulation_margin_mean",
        "simulation_margin_sd",

        "simulation_total_mean",
        "simulation_total_sd",
    ]

    missing_outputs = int(
        simulation[
            core_outputs
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing simulation outputs: "
        f"{missing_outputs:,}"
    )

    probability_sum_error = (
        (
            simulation[
                "simulation_home_win_probability"
            ]
            +
            simulation[
                "simulation_away_win_probability"
            ]
            -
            1.0
        )
        .abs()
        .max()
    )

    print(
        f"Maximum probability-sum error: "
        f"{probability_sum_error:.8f}"
    )

    if created_games != expected_games:

        raise RuntimeError(
            "Monte Carlo did not simulate all games."
        )

    if missing_outputs > 0:

        raise RuntimeError(
            "Simulation contains missing outputs."
        )

    # ========================================================
    # UNCERTAINTY DISTRIBUTION
    # ========================================================

    section(
        "LEARNED HISTORICAL UNCERTAINTY"
    )

    print(
        "HOME SCORE RESIDUAL SD"
    )

    print(
        f"  High:   "
        f"{simulation['home_score_residual_sd_pre'].max():.2f}"
    )

    print(
        f"  Median: "
        f"{simulation['home_score_residual_sd_pre'].median():.2f}"
    )

    print(
        f"  Low:    "
        f"{simulation['home_score_residual_sd_pre'].min():.2f}"
    )

    print()

    print(
        "AWAY SCORE RESIDUAL SD"
    )

    print(
        f"  High:   "
        f"{simulation['away_score_residual_sd_pre'].max():.2f}"
    )

    print(
        f"  Median: "
        f"{simulation['away_score_residual_sd_pre'].median():.2f}"
    )

    print(
        f"  Low:    "
        f"{simulation['away_score_residual_sd_pre'].min():.2f}"
    )

    print()

    print(
        "HOME/AWAY RESIDUAL CORRELATION"
    )

    print(
        f"  High:   "
        f"{simulation['score_residual_correlation_pre'].max():.3f}"
    )

    print(
        f"  Median: "
        f"{simulation['score_residual_correlation_pre'].median():.3f}"
    )

    print(
        f"  Low:    "
        f"{simulation['score_residual_correlation_pre'].min():.3f}"
    )

    # ========================================================
    # SCORE-DISTRIBUTION DIAGNOSTICS
    # ========================================================

    section(
        "SCORE DISTRIBUTION DIAGNOSTICS"
    )

    simulation[
        "actual_home_margin"
    ] = (
        simulation[
            "home_points"
        ]
        -
        simulation[
            "away_points"
        ]
    )

    simulation[
        "actual_total_points"
    ] = (
        simulation[
            "home_points"
        ]
        +
        simulation[
            "away_points"
        ]
    )

    home_mae = (
        (
            simulation[
                "simulation_home_score_mean"
            ]
            -
            simulation[
                "home_points"
            ]
        )
        .abs()
        .mean()
    )

    away_mae = (
        (
            simulation[
                "simulation_away_score_mean"
            ]
            -
            simulation[
                "away_points"
            ]
        )
        .abs()
        .mean()
    )

    total_mae = (
        (
            simulation[
                "simulation_total_mean"
            ]
            -
            simulation[
                "actual_total_points"
            ]
        )
        .abs()
        .mean()
    )

    margin_mae = (
        (
            simulation[
                "simulation_margin_mean"
            ]
            -
            simulation[
                "actual_home_margin"
            ]
        )
        .abs()
        .mean()
    )

    margin_corr = (
        simulation[
            "simulation_margin_mean"
        ]
        .corr(
            simulation[
                "actual_home_margin"
            ]
        )
    )

    total_corr = (
        simulation[
            "simulation_total_mean"
        ]
        .corr(
            simulation[
                "actual_total_points"
            ]
        )
    )

    print(
        f"Home score MAE:       "
        f"{home_mae:.2f}"
    )

    print(
        f"Away score MAE:       "
        f"{away_mae:.2f}"
    )

    print(
        f"Total score MAE:      "
        f"{total_mae:.2f}"
    )

    print(
        f"Margin MAE:           "
        f"{margin_mae:.2f}"
    )

    print()

    print(
        f"Margin correlation:   "
        f"{margin_corr:.3f}"
    )

    print(
        f"Total correlation:    "
        f"{total_corr:.3f}"
    )

    # ========================================================
    # INTERVAL COVERAGE
    # ========================================================

    section(
        "PREDICTION INTERVAL COVERAGE"
    )

    home_80_coverage = (
        (
            simulation[
                "home_points"
            ]
            >=
            simulation[
                "simulation_home_score_p10"
            ]
        )
        &
        (
            simulation[
                "home_points"
            ]
            <=
            simulation[
                "simulation_home_score_p90"
            ]
        )
    ).mean()

    away_80_coverage = (
        (
            simulation[
                "away_points"
            ]
            >=
            simulation[
                "simulation_away_score_p10"
            ]
        )
        &
        (
            simulation[
                "away_points"
            ]
            <=
            simulation[
                "simulation_away_score_p90"
            ]
        )
    ).mean()

    margin_80_coverage = (
        (
            simulation[
                "actual_home_margin"
            ]
            >=
            simulation[
                "simulation_margin_p10"
            ]
        )
        &
        (
            simulation[
                "actual_home_margin"
            ]
            <=
            simulation[
                "simulation_margin_p90"
            ]
        )
    ).mean()

    margin_90_coverage = (
        (
            simulation[
                "actual_home_margin"
            ]
            >=
            simulation[
                "simulation_margin_p05"
            ]
        )
        &
        (
            simulation[
                "actual_home_margin"
            ]
            <=
            simulation[
                "simulation_margin_p95"
            ]
        )
    ).mean()

    print(
        f"Home-score 80% interval: "
        f"{home_80_coverage:.2%}"
    )

    print(
        f"Away-score 80% interval: "
        f"{away_80_coverage:.2%}"
    )

    print(
        f"Margin 80% interval:     "
        f"{margin_80_coverage:.2%}"
    )

    print(
        f"Margin 90% interval:     "
        f"{margin_90_coverage:.2%}"
    )

    print()

    print(
        "Ideal calibration would be approximately "
        "80%, 80%, 80%, and 90%."
    )

    # ========================================================
    # RAW SIMULATION WIN PROBABILITY
    # ========================================================

    section(
        "RAW SIMULATION WIN-PROBABILITY DIAGNOSTICS"
    )

    non_ties = (
        simulation[
            "home_points"
        ]
        !=
        simulation[
            "away_points"
        ]
    )

    evaluation = (
        simulation.loc[
            non_ties
        ]
        .copy()
    )

    evaluation[
        "actual_home_win"
    ] = (
        evaluation[
            "home_points"
        ]
        >
        evaluation[
            "away_points"
        ]
    ).astype(int)

    evaluation[
        "simulation_predicted_home_win"
    ] = (
        evaluation[
            "simulation_home_win_probability"
        ]
        >
        0.50
    )

    simulation_accuracy = (
        evaluation[
            "simulation_predicted_home_win"
        ]
        ==
        evaluation[
            "actual_home_win"
        ].astype(bool)
    ).mean()

    simulation_brier = np.mean(
        (
            evaluation[
                "simulation_home_win_probability"
            ]
            -
            evaluation[
                "actual_home_win"
            ]
        )
        ** 2
    )

    simulation_logloss = binary_log_loss(
        actual=(
            evaluation[
                "actual_home_win"
            ]
        ),

        probability=(
            evaluation[
                "simulation_home_win_probability"
            ]
        ),
    )

    print(
        f"Games evaluated: "
        f"{len(evaluation):,}"
    )

    print(
        f"Accuracy:        "
        f"{simulation_accuracy:.2%}"
    )

    print(
        f"Brier score:     "
        f"{simulation_brier:.4f}"
    )

    print(
        f"Log loss:        "
        f"{simulation_logloss:.4f}"
    )

    print()

    print(
        "These are RAW simulation probabilities."
    )

    print(
        "They have not been calibrated or blended with ELO."
    )

    # ========================================================
    # ELO COMPARISON
    # ========================================================

    section(
        "ELO VS RAW SIMULATION"
    )

    elo_probability = np.clip(
        evaluation[
            "home_elo_win_probability"
        ].astype(float),
        1e-6,
        1.0 - 1e-6,
    )

    elo_prediction = (
        elo_probability
        >
        0.50
    )

    elo_accuracy = (
        elo_prediction
        ==
        evaluation[
            "actual_home_win"
        ].astype(bool)
    ).mean()

    elo_brier = np.mean(
        (
            elo_probability
            -
            evaluation[
                "actual_home_win"
            ]
        )
        ** 2
    )

    elo_logloss = binary_log_loss(
        actual=(
            evaluation[
                "actual_home_win"
            ]
        ),

        probability=(
            elo_probability
        ),
    )

    print(
        f"{'Metric':<20}"
        f"{'ELO':>12}"
        f"{'Simulation':>14}"
    )

    print(
        "-" * 46
    )

    print(
        f"{'Accuracy':<20}"
        f"{elo_accuracy:>11.2%}"
        f"{simulation_accuracy:>13.2%}"
    )

    print(
        f"{'Brier':<20}"
        f"{elo_brier:>12.4f}"
        f"{simulation_brier:>14.4f}"
    )

    print(
        f"{'Log loss':<20}"
        f"{elo_logloss:>12.4f}"
        f"{simulation_logloss:>14.4f}"
    )

    print()

    print(
        "No blending is performed here."
    )

    print(
        "The later probability/calibration layer will decide "
        "whether combining these signals improves unseen "
        "performance."
    )

    # ========================================================
    # SEASON BREAKDOWN
    # ========================================================

    section(
        "RAW SIMULATION BY SEASON"
    )

    for season in sorted(
        evaluation[
            "season"
        ]
        .dropna()
        .unique()
    ):

        season_data = (
            evaluation[
                evaluation[
                    "season"
                ]
                ==
                season
            ]
        )

        accuracy = (
            season_data[
                "simulation_predicted_home_win"
            ]
            ==
            season_data[
                "actual_home_win"
            ].astype(bool)
        ).mean()

        brier = np.mean(
            (
                season_data[
                    "simulation_home_win_probability"
                ]
                -
                season_data[
                    "actual_home_win"
                ]
            )
            ** 2
        )

        logloss = binary_log_loss(
            actual=(
                season_data[
                    "actual_home_win"
                ]
            ),

            probability=(
                season_data[
                    "simulation_home_win_probability"
                ]
            ),
        )

        print(
            f"{int(season)} | "
            f"{len(season_data):>4,} games | "
            f"Accuracy {accuracy:>6.2%} | "
            f"Brier {brier:.4f} | "
            f"LogLoss {logloss:.4f}"
        )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    section(
        "RAW WIN-PROBABILITY DISTRIBUTION"
    )

    probability = (
        simulation[
            "simulation_home_win_probability"
        ]
    )

    print(
        f"Highest home probability: "
        f"{probability.max():.2%}"
    )

    print(
        f"Median home probability:  "
        f"{probability.median():.2%}"
    )

    print(
        f"Lowest home probability:  "
        f"{probability.min():.2%}"
    )

    extreme_95 = int(
        (
            (
                probability
                >=
                0.95
            )
            |
            (
                probability
                <=
                0.05
            )
        ).sum()
    )

    extreme_99 = int(
        (
            (
                probability
                >=
                0.99
            )
            |
            (
                probability
                <=
                0.01
            )
        ).sum()
    )

    print()

    print(
        f"Games at ≥95% / ≤5%: "
        f"{extreme_95:,}"
    )

    print(
        f"Games at ≥99% / ≤1%: "
        f"{extreme_99:,}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING MONTE CARLO OUTPUT"
    )

    save_csv(
        simulation,
        SIMULATION_HISTORY_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "MONTE CARLO COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {SIMULATION_HISTORY_FILE}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "  ELO"
    )

    print(
        "   ↓"
    )

    print(
        "  OFFENCE / DEFENCE"
    )

    print(
        "   ↓"
    )

    print(
        "  OPPONENT ADJUSTMENT"
    )

    print(
        "   ↓"
    )

    print(
        "  MATCHUP"
    )

    print(
        "   ↓"
    )

    print(
        "  GAME ENVIRONMENT"
    )

    print(
        "   ↓"
    )

    print(
        "  DRIVE OPPORTUNITIES"
    )

    print(
        "   ↓"
    )

    print(
        "  SCORING"
    )

    print(
        "   ↓"
    )

    print(
        "  MONTE CARLO SCORE DISTRIBUTION"
    )

    print(
        "   ↓"
    )

    print(
        "  PROBABILITY / CALIBRATION"
    )

    print()


if __name__ == "__main__":
    main()