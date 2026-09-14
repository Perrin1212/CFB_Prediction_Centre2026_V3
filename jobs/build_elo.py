from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import (
    DEFAULT_HOME_FIELD_ADVANTAGE,
    DEFAULT_K_FACTOR,
    DEFAULT_PRESEASON_REGRESSION,
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

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    / "teams.csv"
)

ELO_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "elo_history.csv"
)

CURRENT_ELO_FILE = (
    PROCESSED_DATA_DIR
    / "team_elo_latest.csv"
)

FBS_ELO_FILE = (
    PROCESSED_DATA_DIR
    / "fbs_elo_latest.csv"
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


def normalise_bool(
    series: pd.Series,
) -> pd.Series:
    """
    Robustly convert CSV-loaded boolean-like values.
    """

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


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 — ELO BUILD"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nHistorical game dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run this first:\n"
            "python -m jobs.build_data"
        )

    if not TEAMS_FILE.exists():

        raise FileNotFoundError(
            "\nTeam dataset not found:\n"
            f"{TEAMS_FILE}\n\n"
            "Run this first:\n"
            "python -m jobs.build_data"
        )

    print(
        "\nLoading historical games..."
    )

    games = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"All historical games loaded: "
        f"{len(games):,}"
    )

    # ========================================================
    # MODEL UNIVERSE
    # ========================================================

    section(
        "FILTERING MODEL UNIVERSE"
    )

    required_flags = [
        "fbs_vs_fbs",
        "fbs_vs_fcs",
    ]

    missing_flags = [
        column
        for column in required_flags
        if column not in games.columns
    ]

    if missing_flags:

        raise RuntimeError(
            "historical_games.csv is missing required "
            f"model-universe flags: {missing_flags}"
        )

    games["fbs_vs_fbs"] = normalise_bool(
        games["fbs_vs_fbs"]
    )

    games["fbs_vs_fcs"] = normalise_bool(
        games["fbs_vs_fcs"]
    )

    # --------------------------------------------------------
    # HARD V2 RULE
    #
    # INCLUDED:
    #   FBS vs FBS
    #   FBS vs FCS
    #
    # EXCLUDED:
    #   FCS vs FCS
    #   D2 vs D2
    #   D3 vs D3
    #   any lower-division-only matchup
    # --------------------------------------------------------

    model_games = games[
        games["fbs_vs_fbs"]
        |
        games["fbs_vs_fcs"]
    ].copy()

    fbs_vs_fbs_count = int(
        model_games[
            "fbs_vs_fbs"
        ].sum()
    )

    fbs_vs_fcs_count = int(
        model_games[
            "fbs_vs_fcs"
        ].sum()
    )

    excluded_count = (
        len(games)
        -
        len(model_games)
    )

    print(
        f"FBS vs FBS games: "
        f"{fbs_vs_fbs_count:,}"
    )

    print(
        f"FBS vs FCS games: "
        f"{fbs_vs_fcs_count:,}"
    )

    print(
        f"Total model games: "
        f"{len(model_games):,}"
    )

    print(
        f"Excluded lower-division-only games: "
        f"{excluded_count:,}"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if len(model_games) == 0:

        raise RuntimeError(
            "No eligible FBS model games remain "
            "after filtering."
        )

    # ========================================================
    # ELO CONFIG
    # ========================================================

    config = EloConfig(
    # --------------------------------------------------------
    # REFINED V2 ELO CONFIGURATION
    #
    # Selected using 2024 tuning data.
    # 2025 was held out from parameter selection and used
    # only for out-of-sample validation.
    #
    # 2025 holdout:
    #   Accuracy: 72.59%
    #   Brier:    0.1855
    #   LogLoss:  0.5505
    #
    # These remain provisional until the complete V2 model
    # has been built and evaluated end-to-end.
    # --------------------------------------------------------

    starting_rating=1500.0,

    home_field_advantage=75.0,

    k_factor=30.0,

    preseason_regression=0.45,

    mov_multiplier_enabled=True,

    fbs_over_fcs_weight=0.60,

    fcs_over_fbs_weight=0.70,

    fbs_fcs_mov_cap=1.80,
)

    section(
        "INITIAL ELO CONFIGURATION"
    )

    print(
        f"Starting rating:       "
        f"{config.starting_rating:.1f}"
    )

    print(
        f"Home-field advantage:  "
        f"{config.home_field_advantage:.1f}"
    )

    print(
        f"K factor:              "
        f"{config.k_factor:.1f}"
    )

    print(
        f"Preseason regression:  "
        f"{config.preseason_regression:.2%}"
    )

    print(
        "Margin-of-victory:     enabled"
    )

    print()
    print(
        "These are initial parameters only."
    )

    print(
        "We will tune them using historical "
        "walk-forward testing."
    )

    # ========================================================
    # BUILD ELO
    # ========================================================

    section(
        "BUILDING CHRONOLOGICAL ELO"
    )

    engine = EloEngine(
        config=config
    )

    elo_history = engine.process_games(
        model_games
    )

    latest_ratings = (
        engine.current_ratings()
    )

    # ========================================================
    # ADD RESULT FIELDS
    # ========================================================

    elo_history[
        "elo_predicted_home_win"
    ] = (
        elo_history[
            "home_elo_win_probability"
        ]
        >= 0.50
    ).astype(int)

    elo_history[
        "actual_home_win"
    ] = (
        elo_history[
            "actual_home_result"
        ]
        == 1.0
    ).astype(int)

    elo_history[
        "elo_prediction_correct"
    ] = (
        elo_history[
            "elo_predicted_home_win"
        ]
        ==
        elo_history[
            "actual_home_win"
        ]
    )

    tie_mask = (
        elo_history[
            "actual_home_result"
        ]
        ==
        0.5
    )

    elo_history.loc[
        tie_mask,
        "elo_prediction_correct",
    ] = pd.NA

    # ========================================================
    # BUILD FBS-ONLY LEADERBOARD
    # ========================================================

    section(
        "BUILDING FBS-ONLY LEADERBOARD"
    )

    teams["classification"] = (
        teams["classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    fbs_teams = teams[
        teams[
            "classification"
        ]
        ==
        "fbs"
    ].copy()

    fbs_team_names = set(
        fbs_teams[
            "team_name"
        ]
        .dropna()
        .astype(str)
    )

    fbs_ratings = latest_ratings[
        latest_ratings[
            "team"
        ].isin(
            fbs_team_names
        )
    ].copy()

    fbs_ratings = (
        fbs_ratings
        .sort_values(
            "elo",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    fbs_ratings["rank"] = (
        fbs_ratings.index
        +
        1
    )

    fbs_ratings = fbs_ratings[
        [
            "rank",
            "team",
            "elo",
        ]
    ]

    print(
        f"FBS teams in team master: "
        f"{len(fbs_team_names):,}"
    )

    print(
        f"FBS teams with ELO ratings: "
        f"{len(fbs_ratings):,}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING ELO OUTPUT"
    )

    save_csv(
        elo_history,
        ELO_HISTORY_FILE,
    )

    # Internal ratings include FCS where needed to process
    # FBS-vs-FCS games correctly.
    save_csv(
        latest_ratings,
        CURRENT_ELO_FILE,
    )

    # Official leaderboard is strictly FBS-only.
    save_csv(
        fbs_ratings,
        FBS_ELO_FILE,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "ELO VALIDATION"
    )

    print(
        f"Eligible games rated: "
        f"{len(elo_history):,}"
    )

    print(
        f"All internally rated teams: "
        f"{len(latest_ratings):,}"
    )

    print(
        f"Official FBS leaderboard teams: "
        f"{len(fbs_ratings):,}"
    )

    missing_pre_ratings = (
        elo_history[
            [
                "home_elo_pre",
                "away_elo_pre",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Games missing pregame ratings: "
        f"{int(missing_pre_ratings):,}"
    )

    probability_errors = (
        (
            elo_history[
                "home_elo_win_probability"
            ]
            < 0
        )
        |
        (
            elo_history[
                "home_elo_win_probability"
            ]
            > 1
        )
    ).sum()

    print(
        f"Invalid probabilities: "
        f"{int(probability_errors):,}"
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    non_ties = elo_history[
        elo_history[
            "actual_home_result"
        ]
        != 0.5
    ].copy()

    accuracy = (
        non_ties[
            "elo_prediction_correct"
        ]
        .astype(float)
        .mean()
    )

    brier = (
        (
            elo_history[
                "home_elo_win_probability"
            ]
            -
            elo_history[
                "actual_home_result"
            ]
        )
        ** 2
    ).mean()

    print(
        f"ELO straight winner accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"ELO Brier score: "
        f"{brier:.4f}"
    )

    # ========================================================
    # PERFORMANCE BY SEASON
    # ========================================================

    section(
        "ELO PERFORMANCE BY SEASON"
    )

    for season in sorted(
        elo_history[
            "season"
        ]
        .dropna()
        .astype(int)
        .unique()
    ):

        season_df = elo_history[
            elo_history[
                "season"
            ]
            ==
            season
        ]

        season_non_ties = season_df[
            season_df[
                "actual_home_result"
            ]
            != 0.5
        ]

        season_accuracy = (
            season_non_ties[
                "elo_prediction_correct"
            ]
            .astype(float)
            .mean()
        )

        season_brier = (
            (
                season_df[
                    "home_elo_win_probability"
                ]
                -
                season_df[
                    "actual_home_result"
                ]
            )
            ** 2
        ).mean()

        print(
            f"{season}: "
            f"{len(season_df):,} games | "
            f"Accuracy {season_accuracy:.2%} | "
            f"Brier {season_brier:.4f}"
        )

    # ========================================================
    # TOP 25 FBS ONLY
    # ========================================================

    section(
        "TOP 25 FBS ELO RATINGS"
    )

    top_25 = (
        fbs_ratings
        .head(
            25
        )
    )

    for row in top_25.itertuples(
        index=False
    ):

        print(
            f"{int(row.rank):>2}. "
            f"{row.team:<28} "
            f"{row.elo:>7.1f}"
        )

    # ========================================================
    # FBS RANGE
    # ========================================================

    section(
        "FBS RATING RANGE"
    )

    if not fbs_ratings.empty:

        print(
            f"Highest FBS rating: "
            f"{fbs_ratings['elo'].max():.1f}"
        )

        print(
            f"Median FBS rating:  "
            f"{fbs_ratings['elo'].median():.1f}"
        )

        print(
            f"Lowest FBS rating:  "
            f"{fbs_ratings['elo'].min():.1f}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "ELO BUILD COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {ELO_HISTORY_FILE}"
    )

    print(
        f"  {CURRENT_ELO_FILE}"
    )

    print(
        f"  {FBS_ELO_FILE}"
    )

    print()

    print(
        "Model-universe rule enforced:"
    )

    print(
        "  ✓ FBS vs FBS included"
    )

    print(
        "  ✓ FBS vs FCS included"
    )

    print(
        "  ✗ FCS vs FCS excluded"
    )

    print(
        "  ✗ lower-division-only games excluded"
    )

    print()

    print(
        "FCS teams may exist internally so FBS-vs-FCS "
        "results can be rated correctly."
    )

    print(
        "They are excluded from the official FBS leaderboard."
    )

    print()


if __name__ == "__main__":
    main()