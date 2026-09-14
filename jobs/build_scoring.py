from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from config.settings import PROCESSED_DATA_DIR

from engine.scoring import (
    ScoringConfig,
    ScoringModel,
)


# ============================================================
# FILES
# ============================================================

TEAM_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "opponent_adjusted_rating_history.csv"
)

POSSESSION_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "possession_drive_history.csv"
)

TEAMS_FILE = (
    PROCESSED_DATA_DIR
    / "teams.csv"
)

SCORING_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "scoring_history.csv"
)

TEAM_SCORING_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "team_scoring_state_latest.csv"
)

FBS_SCORING_LATEST_FILE = (
    PROCESSED_DATA_DIR
    / "fbs_scoring_state_latest.csv"
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— SCORING MODEL"
    )

    # ========================================================
    # FILE CHECK
    # ========================================================

    for path in [
        TEAM_HISTORY_FILE,
        POSSESSION_HISTORY_FILE,
        TEAMS_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired file does not exist:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "LOADING INPUTS"
    )

    team_history = pd.read_csv(
        TEAM_HISTORY_FILE,
        low_memory=False,
    )

    possession = pd.read_csv(
        POSSESSION_HISTORY_FILE,
        low_memory=False,
    )

    teams = pd.read_csv(
        TEAMS_FILE,
        low_memory=False,
    )

    print(
        f"Team-game rows loaded:      "
        f"{len(team_history):,}"
    )

    print(
        f"Historical games:           "
        f"{team_history['game_id'].nunique():,}"
    )

    print(
        f"Possession games loaded:    "
        f"{len(possession):,}"
    )

    print(
        f"Teams loaded:               "
        f"{len(teams):,}"
    )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    section(
        "SCORING INPUT VALIDATION"
    )

    required_team = [
        "game_id",
        "season",
        "week",
        "team_name",
        "home_away",

        "team_points",
        "rushing_attempts",
        "passing_attempts",
    ]

    missing_team = [
        column
        for column in required_team
        if column not in team_history.columns
    ]

    if missing_team:

        raise RuntimeError(
            "Team history is missing required "
            f"scoring fields: {missing_team}"
        )

    required_possession = [
        "game_id",

        "home_offensive_matchup_index",
        "away_offensive_matchup_index",

        "expected_home_plays",
        "expected_away_plays",

        "expected_home_plays_per_drive",
        "expected_away_plays_per_drive",

        "expected_home_drives",
        "expected_away_drives",
    ]

    missing_possession = [
        column
        for column in required_possession
        if column not in possession.columns
    ]

    if missing_possession:

        raise RuntimeError(
            "Possession history is missing "
            f"required fields: {missing_possession}"
        )

    for column in [
        "team_points",
        "rushing_attempts",
        "passing_attempts",
    ]:

        coverage = (
            team_history[
                column
            ]
            .notna()
            .mean()
        )

        print(
            f"{column:<25} "
            f"{coverage:>7.2%}"
        )

    print()

    print(
        f"Team-history games: "
        f"{team_history['game_id'].nunique():,}"
    )

    print(
        f"Possession games:   "
        f"{possession['game_id'].nunique():,}"
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    section(
        "SCORING CONFIGURATION"
    )

    config = ScoringConfig(
        update_rate=0.25,
        preseason_regression=0.40,

        starting_points_per_play=0.39,

        national_prior_team_games=100.0,

        offense_scoring_weight=0.55,
        defense_scoring_weight=0.45,

        matchup_factor_weight=0.35,

        home_scoring_multiplier=1.025,
        away_scoring_multiplier=0.985,
        neutral_scoring_multiplier=1.0,

        minimum_points_per_play=0.10,
        maximum_points_per_play=0.85,

        minimum_points_per_drive=0.50,
        maximum_points_per_drive=5.50,

        minimum_expected_points=6.0,
        maximum_expected_points=65.0,

        matchup_factor_floor=0.75,
        matchup_factor_ceiling=1.25,
    )

    print(
        f"Update rate:              "
        f"{config.update_rate:.0%}"
    )

    print(
        f"Preseason regression:     "
        f"{config.preseason_regression:.0%}"
    )

    print(
        f"Starting points/play:     "
        f"{config.starting_points_per_play:.3f}"
    )

    print(
        f"Scoring blend:            "
        f"{config.offense_scoring_weight:.0%} offence / "
        f"{config.defense_scoring_weight:.0%} defence"
    )

    print(
        f"Matchup modifier weight:  "
        f"{config.matchup_factor_weight:.0%}"
    )

    print(
        f"Home scoring multiplier:  "
        f"{config.home_scoring_multiplier:.3f}"
    )

    print(
        f"Away scoring multiplier:  "
        f"{config.away_scoring_multiplier:.3f}"
    )

    print()

    print(
        "No parameter optimisation is being run."
    )

    print(
        "This is a single chronological scoring pass."
    )

    print()

    print(
        "Historical scoring state is learned from "
        "REAL points/play."
    )

    print(
        "Expected points/drive is derived only after "
        "the pregame drive estimate."
    )

    # ========================================================
    # BUILD
    # ========================================================

    section(
        "BUILDING CHRONOLOGICAL SCORING MODEL"
    )

    model = ScoringModel(
        config=config
    )

    scoring_history = (
        model.process_history(
            team_history=team_history,
            possession_history=possession,
        )
    )

    current_states = (
        model.current_states()
    )

    print(
        f"Scoring game rows created: "
        f"{len(scoring_history):,}"
    )

    print(
        f"Teams with scoring state:  "
        f"{len(current_states):,}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    section(
        "SCORING MODEL VALIDATION"
    )

    expected_games = int(
        possession[
            "game_id"
        ]
        .nunique()
    )

    created_games = int(
        scoring_history[
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

    if created_games != expected_games:

        raise RuntimeError(
            "Scoring model failed to create "
            "all historical games."
        )

    core_columns = [
        "expected_home_points_per_play",
        "expected_away_points_per_play",

        "expected_home_points_per_drive",
        "expected_away_points_per_drive",

        "expected_home_points",
        "expected_away_points",

        "expected_total_points",
        "expected_home_margin",
    ]

    missing_core = int(
        scoring_history[
            core_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing core features: "
        f"{missing_core:,}"
    )

    if missing_core > 0:

        raise RuntimeError(
            "Scoring output contains missing "
            "pregame features."
        )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    section(
        "EXPECTED SCORING DISTRIBUTION"
    )

    print(
        "EXPECTED HOME POINTS"
    )

    print(
        f"  High:   "
        f"{scoring_history['expected_home_points'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{scoring_history['expected_home_points'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{scoring_history['expected_home_points'].min():.1f}"
    )

    print()

    print(
        "EXPECTED AWAY POINTS"
    )

    print(
        f"  High:   "
        f"{scoring_history['expected_away_points'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{scoring_history['expected_away_points'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{scoring_history['expected_away_points'].min():.1f}"
    )

    print()

    print(
        "EXPECTED TOTAL"
    )

    print(
        f"  High:   "
        f"{scoring_history['expected_total_points'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{scoring_history['expected_total_points'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{scoring_history['expected_total_points'].min():.1f}"
    )

    print()

    print(
        "EXPECTED HOME MARGIN"
    )

    print(
        f"  Largest home edge: "
        f"{scoring_history['expected_home_margin'].max():.1f}"
    )

    print(
        f"  Median:            "
        f"{scoring_history['expected_home_margin'].median():.1f}"
    )

    print(
        f"  Largest away edge: "
        f"{scoring_history['expected_home_margin'].min():.1f}"
    )

    # ========================================================
    # POINTS / PLAY DISTRIBUTION
    # ========================================================

    section(
        "EXPECTED SCORING EFFICIENCY"
    )

    combined_ppp = pd.concat(
        [
            scoring_history[
                "expected_home_points_per_play"
            ],

            scoring_history[
                "expected_away_points_per_play"
            ],
        ],
        ignore_index=True,
    )

    combined_ppd = pd.concat(
        [
            scoring_history[
                "expected_home_points_per_drive"
            ],

            scoring_history[
                "expected_away_points_per_drive"
            ],
        ],
        ignore_index=True,
    )

    print(
        "EXPECTED POINTS / PLAY"
    )

    print(
        f"  High:   "
        f"{combined_ppp.max():.3f}"
    )

    print(
        f"  Median: "
        f"{combined_ppp.median():.3f}"
    )

    print(
        f"  Low:    "
        f"{combined_ppp.min():.3f}"
    )

    print()

    print(
        "EXPECTED POINTS / ESTIMATED DRIVE"
    )

    print(
        f"  High:   "
        f"{combined_ppd.max():.2f}"
    )

    print(
        f"  Median: "
        f"{combined_ppd.median():.2f}"
    )

    print(
        f"  Low:    "
        f"{combined_ppd.min():.2f}"
    )

    # ========================================================
    # HISTORICAL RESULTS
    # ========================================================

    section(
        "SCORING DIAGNOSTICS"
    )

    scoring_history["home_points"] = pd.to_numeric(
        scoring_history["home_points"],
        errors="coerce",
    )

    scoring_history["away_points"] = pd.to_numeric(
        scoring_history["away_points"],
        errors="coerce",
    )

    scoring_history[
        "actual_total_points"
    ] = (
        scoring_history[
            "home_points"
        ]
        +
        scoring_history[
            "away_points"
        ]
    )

    scoring_history[
        "actual_home_margin"
    ] = (
        scoring_history[
            "home_points"
        ]
        -
        scoring_history[
            "away_points"
        ]
    )

    scoring_history[
        "home_score_error"
    ] = (
        scoring_history[
            "expected_home_points"
        ]
        -
        scoring_history[
            "home_points"
        ]
    )

    scoring_history[
        "away_score_error"
    ] = (
        scoring_history[
            "expected_away_points"
        ]
        -
        scoring_history[
            "away_points"
        ]
    )

    scoring_history[
        "total_score_error"
    ] = (
        scoring_history[
            "expected_total_points"
        ]
        -
        scoring_history[
            "actual_total_points"
        ]
    )

    scoring_history[
        "margin_error"
    ] = (
        scoring_history[
            "expected_home_margin"
        ]
        -
        scoring_history[
            "actual_home_margin"
        ]
    )

    home_mae = (
        scoring_history[
            "home_score_error"
        ]
        .abs()
        .mean()
    )

    away_mae = (
        scoring_history[
            "away_score_error"
        ]
        .abs()
        .mean()
    )

    total_mae = (
        scoring_history[
            "total_score_error"
        ]
        .abs()
        .mean()
    )

    margin_mae = (
        scoring_history[
            "margin_error"
        ]
        .abs()
        .mean()
    )

    home_corr = (
        scoring_history[
            "expected_home_points"
        ]
        .corr(
            scoring_history[
                "home_points"
            ]
        )
    )

    away_corr = (
        scoring_history[
            "expected_away_points"
        ]
        .corr(
            scoring_history[
                "away_points"
            ]
        )
    )

    total_corr = (
        scoring_history[
            "expected_total_points"
        ]
        .corr(
            scoring_history[
                "actual_total_points"
            ]
        )
    )

    margin_corr = (
        scoring_history[
            "expected_home_margin"
        ]
        .corr(
            scoring_history[
                "actual_home_margin"
            ]
        )
    )

    print(
        "MEAN ABSOLUTE ERROR"
    )

    print(
        f"  Home score MAE: "
        f"{home_mae:.2f}"
    )

    print(
        f"  Away score MAE: "
        f"{away_mae:.2f}"
    )

    print(
        f"  Total MAE:      "
        f"{total_mae:.2f}"
    )

    print(
        f"  Margin MAE:     "
        f"{margin_mae:.2f}"
    )

    print()

    print(
        "CORRELATION"
    )

    print(
        f"  Home predicted vs actual: "
        f"{home_corr:.3f}"
    )

    print(
        f"  Away predicted vs actual: "
        f"{away_corr:.3f}"
    )

    print(
        f"  Total predicted vs actual: "
        f"{total_corr:.3f}"
    )

    print(
        f"  Margin predicted vs actual: "
        f"{margin_corr:.3f}"
    )

    # ========================================================
    # DIRECTIONAL WIN SANITY CHECK
    # ========================================================

    section(
        "DIRECTIONAL WIN SANITY CHECK"
    )

    non_ties = (
        scoring_history[
            "home_points"
        ]
        !=
        scoring_history[
            "away_points"
        ]
    )

    valid = (
        non_ties
        &
        scoring_history[
            "home_points"
        ].notna()
        &
        scoring_history[
            "away_points"
        ].notna()
    )

    actual_home_win = (
        scoring_history.loc[
            valid,
            "home_points",
        ]
        >
        scoring_history.loc[
            valid,
            "away_points",
        ]
    )

    predicted_home_win = (
        scoring_history.loc[
            valid,
            "expected_home_margin",
        ]
        >
        0
    )

    directional_accuracy = (
        predicted_home_win
        ==
        actual_home_win
    ).mean()

    print(
        f"Games evaluated:       "
        f"{int(valid.sum()):,}"
    )

    print(
        f"Expected-score winner "
        f"accuracy: "
        f"{directional_accuracy:.2%}"
    )

    print()

    print(
        "This is NOT the final win-probability model."
    )

    print(
        "It is only a sanity test of expected scoring "
        "direction."
    )

    # ========================================================
    # CURRENT FBS STATES
    # ========================================================

    section(
        "BUILDING FBS-ONLY SCORING STATE"
    )

    if "team_name" not in teams.columns:

        raise RuntimeError(
            "teams.csv is missing team_name."
        )

    if "classification" not in teams.columns:

        raise RuntimeError(
            "teams.csv is missing classification."
        )

    teams["classification"] = (
        teams["classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    fbs_names = set(
        teams.loc[
            teams["classification"]
            ==
            "fbs",
            "team_name",
        ]
        .dropna()
        .astype(str)
    )

    fbs_states = (
        current_states[
            current_states[
                "team"
            ].isin(
                fbs_names
            )
        ]
        .copy()
    )

    print(
        f"FBS teams with scoring state: "
        f"{len(fbs_states):,}"
    )

    # ========================================================
    # TOP SCORING OFFENCES
    # ========================================================

    section(
        "TOP 15 CURRENT FBS SCORING EFFICIENCIES"
    )

    top_offense = (
        fbs_states
        .sort_values(
            "offense_points_per_play",
            ascending=False,
        )
        .head(15)
    )

    for rank, row in enumerate(
        top_offense.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.offense_points_per_play:>6.3f} "
            f"pts/play"
        )

    # ========================================================
    # TOP DEFENCES
    # ========================================================

    section(
        "TOP 15 CURRENT FBS SCORING DEFENCES"
    )

    top_defense = (
        fbs_states
        .sort_values(
            "defense_points_per_play_allowed",
            ascending=True,
        )
        .head(15)
    )

    for rank, row in enumerate(
        top_defense.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{row.team:<28} "
            f"{row.defense_points_per_play_allowed:>6.3f} "
            f"allowed/play"
        )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING SCORING OUTPUT"
    )

    save_csv(
        scoring_history,
        SCORING_HISTORY_FILE,
    )

    save_csv(
        current_states,
        TEAM_SCORING_LATEST_FILE,
    )

    save_csv(
        fbs_states,
        FBS_SCORING_LATEST_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "SCORING MODEL COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {SCORING_HISTORY_FILE}"
    )

    print(
        f"  {TEAM_SCORING_LATEST_FILE}"
    )

    print(
        f"  {FBS_SCORING_LATEST_FILE}"
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
        "  MATCHUP ENGINE"
    )

    print(
        "   ↓"
    )

    print(
        "  GAME ENVIRONMENT / PACE"
    )

    print(
        "   ↓"
    )

    print(
        "  POSSESSION / DRIVE OPPORTUNITIES"
    )

    print(
        "   ↓"
    )

    print(
        "  SCORING MODEL"
    )

    print(
        "   ↓"
    )

    print(
        "  MONTE CARLO SIMULATION"
    )

    print()


if __name__ == "__main__":
    main()