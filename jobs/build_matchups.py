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

from engine.matchup import (
    MatchupConfig,
    MatchupEngine,
)


# ============================================================
# FILES
# ============================================================

ADJUSTED_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "opponent_adjusted_rating_history.csv"
)

ELO_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "elo_history.csv"
)

MATCHUP_HISTORY_FILE = (
    PROCESSED_DATA_DIR
    / "matchup_history.csv"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title: str) -> None:
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
# ID HELPERS
# ============================================================

def normalise_game_id_series(
    series: pd.Series,
) -> pd.Series:
    """
    Standardise internal database game IDs.

    V2 historical_team_games uses the internal games.id value.

    ELO stores that same value as game_db_id.

    This converts both sides to pandas nullable integers so
    comparisons are reliable.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


# ============================================================
# STANDARDISE ELO
# ============================================================

def standardise_elo_history(
    elo: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the ELO output into the schema expected by the
    matchup engine.

    The important relationship is:

        elo_history.game_db_id
                =
        rating_history.game_id

    game_db_id is the INTERNAL SQLite games.id.

    cfbd_game_id is the EXTERNAL CFBD identifier and must not
    be used for this join.
    """

    data = elo.copy()

    # --------------------------------------------------------
    # IDENTIFIER
    # --------------------------------------------------------

    if "game_db_id" in data.columns:

        print(
            "Using ELO 'game_db_id' as internal game_id."
        )

        data["game_id"] = normalise_game_id_series(
            data["game_db_id"]
        )

    elif "game_id" in data.columns:

        print(
            "ELO already contains internal game_id."
        )

        data["game_id"] = normalise_game_id_series(
            data["game_id"]
        )

    else:

        raise RuntimeError(
            "elo_history.csv contains neither "
            "'game_db_id' nor 'game_id'."
        )

    # --------------------------------------------------------
    # STANDARDISE ADJUSTED IDS FOR VALIDATION
    # --------------------------------------------------------

    adjusted_ids = normalise_game_id_series(
        adjusted["game_id"]
    )

    adjusted_set = set(
        adjusted_ids
        .dropna()
        .astype(int)
        .tolist()
    )

    elo_set = set(
        data["game_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    matching_ids = (
        adjusted_set
        &
        elo_set
    )

    missing_from_elo = (
        adjusted_set
        -
        elo_set
    )

    extra_in_elo = (
        elo_set
        -
        adjusted_set
    )

    print()
    print(
        f"Adjusted game IDs:          "
        f"{len(adjusted_set):,}"
    )

    print(
        f"ELO internal game IDs:      "
        f"{len(elo_set):,}"
    )

    print(
        f"Matching internal game IDs: "
        f"{len(matching_ids):,}"
    )

    print(
        f"Missing from ELO:            "
        f"{len(missing_from_elo):,}"
    )

    print(
        f"Extra ELO IDs:               "
        f"{len(extra_in_elo):,}"
    )

    if len(missing_from_elo) > 0:

        examples = sorted(
            list(
                missing_from_elo
            )
        )[:10]

        raise RuntimeError(
            "Some opponent-adjusted games do not have "
            "matching ELO rows.\n"
            f"Example missing internal game IDs: "
            f"{examples}"
        )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    duplicate_ids = int(
        data["game_id"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    print(
        f"Duplicate ELO internal IDs: "
        f"{duplicate_ids:,}"
    )

    if duplicate_ids > 0:

        duplicates = (
            data.loc[
                data["game_id"].duplicated(
                    keep=False
                ),
                [
                    column
                    for column in [
                        "game_id",
                        "season",
                        "week",
                        "home_team",
                        "away_team",
                    ]
                    if column in data.columns
                ],
            ]
            .sort_values("game_id")
        )

        print()
        print(
            "Duplicate ELO game examples:"
        )
        print(
            duplicates
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "ELO history contains duplicate internal "
            "game IDs."
        )

    # --------------------------------------------------------
    # REQUIRED ELO COLUMNS
    # --------------------------------------------------------

    required_elo_columns = [
        "game_id",
        "home_elo_pre",
        "away_elo_pre",
        "home_elo_win_probability",
    ]

    missing_columns = [
        column
        for column in required_elo_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "elo_history.csv is missing required "
            f"columns: {missing_columns}"
        )

    return data


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— MATCHUP ENGINE"
    )

    # ========================================================
    # FILE CHECKS
    # ========================================================

    for path in [
        ADJUSTED_HISTORY_FILE,
        ELO_HISTORY_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired file does not exist:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD INPUTS
    # ========================================================

    section(
        "LOADING INPUTS"
    )

    adjusted = pd.read_csv(
        ADJUSTED_HISTORY_FILE,
        low_memory=False,
    )

    elo = pd.read_csv(
        ELO_HISTORY_FILE,
        low_memory=False,
    )

    print(
        f"Opponent-adjusted team rows: "
        f"{len(adjusted):,}"
    )

    print(
        f"Opponent-adjusted games:     "
        f"{adjusted['game_id'].nunique():,}"
    )

    print(
        f"ELO game rows:               "
        f"{len(elo):,}"
    )

    # ========================================================
    # ADJUSTED DATA VALIDATION
    # ========================================================

    section(
        "INPUT VALIDATION"
    )

    required_adjusted_columns = [
        "game_id",
        "season",
        "week",
        "team_name",
        "opponent_name",
        "home_away",

        "team_points",

        "team_offense_rating_pre",
        "team_defense_rating_pre",

        "team_adjusted_offense_rating_pre",
        "team_adjusted_defense_rating_pre",

        "adjusted_games_pre",
    ]

    missing_adjusted = [
        column
        for column in required_adjusted_columns
        if column not in adjusted.columns
    ]

    if missing_adjusted:

        raise RuntimeError(
            "Opponent-adjusted history is missing "
            f"required columns: {missing_adjusted}"
        )

    # --------------------------------------------------------
    # STANDARDISE INTERNAL IDs
    # --------------------------------------------------------

    adjusted["game_id"] = normalise_game_id_series(
        adjusted["game_id"]
    )

    if adjusted["game_id"].isna().any():

        raise RuntimeError(
            "Opponent-adjusted history contains invalid "
            "game_id values."
        )

    # --------------------------------------------------------
    # ROWS PER GAME
    # --------------------------------------------------------

    rows_per_game = (
        adjusted
        .groupby(
            "game_id"
        )
        .size()
    )

    complete_pairs = int(
        (
            rows_per_game
            ==
            2
        ).sum()
    )

    invalid_pairs = int(
        (
            rows_per_game
            !=
            2
        ).sum()
    )

    print(
        f"Games with two rating rows: "
        f"{complete_pairs:,}"
    )

    print(
        f"Invalid game pairs:         "
        f"{invalid_pairs:,}"
    )

    if invalid_pairs > 0:

        raise RuntimeError(
            "Matchup engine requires exactly two "
            "team rows per game."
        )

    # --------------------------------------------------------
    # HOME / AWAY
    # --------------------------------------------------------

    adjusted["home_away"] = (
        adjusted["home_away"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    home_count = int(
        (
            adjusted["home_away"]
            ==
            "home"
        ).sum()
    )

    away_count = int(
        (
            adjusted["home_away"]
            ==
            "away"
        ).sum()
    )

    print(
        f"Home rows:                  "
        f"{home_count:,}"
    )

    print(
        f"Away rows:                  "
        f"{away_count:,}"
    )

    if (
        home_count != complete_pairs
        or
        away_count != complete_pairs
    ):

        raise RuntimeError(
            "Home/away orientation does not match "
            "the number of games."
        )

    # ========================================================
    # ALIGN ELO IDENTIFIERS
    # ========================================================

    section(
        "ALIGNING ELO GAME IDENTIFIERS"
    )

    print(
        "Important identifier relationship:"
    )

    print(
        "  rating_history.game_id"
    )

    print(
        "          ="
    )

    print(
        "  elo_history.game_db_id"
    )

    print()

    print(
        "CFBD external IDs are NOT used for this join."
    )

    print()

    elo = standardise_elo_history(
        elo=elo,
        adjusted=adjusted,
    )

    # ========================================================
    # FINAL ELO COVERAGE
    # ========================================================

    section(
        "ELO COVERAGE"
    )

    adjusted_ids = set(
        adjusted["game_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    elo_ids = set(
        elo["game_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    matches = (
        adjusted_ids
        &
        elo_ids
    )

    print(
        f"Adjusted games:              "
        f"{len(adjusted_ids):,}"
    )

    print(
        f"Adjusted games matching ELO: "
        f"{len(matches):,}"
    )

    print(
        f"Coverage:                    "
        f"{len(matches) / len(adjusted_ids):.2%}"
    )

    if matches != adjusted_ids:

        raise RuntimeError(
            "ELO coverage is incomplete."
        )

    # ========================================================
    # MATCHUP CONFIG
    # ========================================================

    section(
        "MATCHUP CONFIGURATION"
    )

    config = MatchupConfig(
        matchup_index_floor=60.0,
        matchup_index_ceiling=140.0,

        strength_floor=50.0,
        strength_ceiling=150.0,

        established_games=6,
    )

    print(
        f"Matchup index range: "
        f"{config.matchup_index_floor:.0f}"
        f" – "
        f"{config.matchup_index_ceiling:.0f}"
    )

    print(
        f"Established after:    "
        f"{config.established_games} games"
    )

    print()

    print(
        "No winner model is being trained."
    )

    print(
        "No probability weights are being fitted."
    )

    print(
        "This layer creates football matchup "
        "features only."
    )

    # ========================================================
    # BUILD MATCHUPS
    # ========================================================

    section(
        "BUILDING GAME MATCHUPS"
    )

    engine = MatchupEngine(
        config=config
    )

    matchups = engine.build_history(
        adjusted_history=adjusted,
        elo_history=elo,
    )

    print(
        f"Game matchup rows created: "
        f"{len(matchups):,}"
    )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    section(
        "MATCHUP VALIDATION"
    )

    expected_games = int(
        adjusted["game_id"]
        .nunique()
    )

    created_games = len(
        matchups
    )

    missing_games = (
        expected_games
        -
        created_games
    )

    print(
        f"Expected games:          "
        f"{expected_games:,}"
    )

    print(
        f"Created games:           "
        f"{created_games:,}"
    )

    print(
        f"Missing games:           "
        f"{missing_games:,}"
    )

    if missing_games != 0:

        raise RuntimeError(
            f"Matchup engine failed to create "
            f"{missing_games:,} games."
        )

    # --------------------------------------------------------
    # REQUIRED OUTPUT FEATURES
    # --------------------------------------------------------

    required_features = [
        "home_adjusted_offense_rating_pre",
        "home_adjusted_defense_rating_pre",

        "away_adjusted_offense_rating_pre",
        "away_adjusted_defense_rating_pre",

        "home_offensive_matchup_index",
        "away_offensive_matchup_index",

        "home_adjusted_overall_strength",
        "away_adjusted_overall_strength",

        "adjusted_strength_difference",
        "offensive_matchup_difference",

        "home_elo_pre",
        "away_elo_pre",
        "elo_difference",

        "home_elo_win_probability",
    ]

    missing_output_columns = [
        column
        for column in required_features
        if column not in matchups.columns
    ]

    if missing_output_columns:

        raise RuntimeError(
            "Matchup output is missing expected "
            f"columns: {missing_output_columns}"
        )

    rows_missing_features = int(
        matchups[
            required_features
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"Rows missing core features: "
        f"{rows_missing_features:,}"
    )

    if rows_missing_features > 0:

        raise RuntimeError(
            "Some matchup rows contain missing core "
            "pregame features."
        )

    # --------------------------------------------------------
    # RANGE VALIDATION
    # --------------------------------------------------------

    invalid_home_matchup = int(
        (
            (
                matchups[
                    "home_offensive_matchup_index"
                ]
                <
                config.matchup_index_floor
            )
            |
            (
                matchups[
                    "home_offensive_matchup_index"
                ]
                >
                config.matchup_index_ceiling
            )
        ).sum()
    )

    invalid_away_matchup = int(
        (
            (
                matchups[
                    "away_offensive_matchup_index"
                ]
                <
                config.matchup_index_floor
            )
            |
            (
                matchups[
                    "away_offensive_matchup_index"
                ]
                >
                config.matchup_index_ceiling
            )
        ).sum()
    )

    print(
        f"Invalid home matchup indices: "
        f"{invalid_home_matchup:,}"
    )

    print(
        f"Invalid away matchup indices: "
        f"{invalid_away_matchup:,}"
    )

    # ========================================================
    # MATCHUP DISTRIBUTION
    # ========================================================

    section(
        "MATCHUP DISTRIBUTION"
    )

    print(
        "HOME OFFENSIVE MATCHUP INDEX"
    )

    print(
        f"  High:   "
        f"{matchups['home_offensive_matchup_index'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{matchups['home_offensive_matchup_index'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{matchups['home_offensive_matchup_index'].min():.1f}"
    )

    print()

    print(
        "AWAY OFFENSIVE MATCHUP INDEX"
    )

    print(
        f"  High:   "
        f"{matchups['away_offensive_matchup_index'].max():.1f}"
    )

    print(
        f"  Median: "
        f"{matchups['away_offensive_matchup_index'].median():.1f}"
    )

    print(
        f"  Low:    "
        f"{matchups['away_offensive_matchup_index'].min():.1f}"
    )

    print()

    print(
        "ADJUSTED TEAM-STRENGTH DIFFERENCE"
    )

    print(
        f"  Largest home edge: "
        f"{matchups['adjusted_strength_difference'].max():.1f}"
    )

    print(
        f"  Median:            "
        f"{matchups['adjusted_strength_difference'].median():.1f}"
    )

    print(
        f"  Largest away edge: "
        f"{matchups['adjusted_strength_difference'].min():.1f}"
    )

    # ========================================================
    # TOP HISTORICAL MATCHUPS
    # ========================================================

    section(
        "TOP 10 HISTORICAL HOME OFFENSIVE MATCHUPS"
    )

    top_home = (
        matchups
        .sort_values(
            "home_offensive_matchup_index",
            ascending=False,
        )
        .head(10)
    )

    for row in top_home.itertuples(
        index=False
    ):

        print(
            f"{int(row.season)} "
            f"W{int(row.week):>2} | "
            f"{row.home_team:<24} "
            f"vs "
            f"{row.away_team:<24} "
            f"{row.home_offensive_matchup_index:>6.1f}"
        )

    section(
        "TOP 10 HISTORICAL AWAY OFFENSIVE MATCHUPS"
    )

    top_away = (
        matchups
        .sort_values(
            "away_offensive_matchup_index",
            ascending=False,
        )
        .head(10)
    )

    for row in top_away.itertuples(
        index=False
    ):

        print(
            f"{int(row.season)} "
            f"W{int(row.week):>2} | "
            f"{row.away_team:<24} "
            f"at "
            f"{row.home_team:<24} "
            f"{row.away_offensive_matchup_index:>6.1f}"
        )

    # ========================================================
    # HISTORICAL SANITY CHECK
    # ========================================================

    section(
        "BASIC HISTORICAL SANITY CHECK"
    )

    matchups["home_points"] = pd.to_numeric(
        matchups["home_points"],
        errors="coerce",
    )

    matchups["away_points"] = pd.to_numeric(
        matchups["away_points"],
        errors="coerce",
    )

    matchups[
        "actual_home_margin"
    ] = (
        matchups["home_points"]
        -
        matchups["away_points"]
    )

    valid_margin = (
        matchups[
            "actual_home_margin"
        ]
        .notna()
    )

    correlation_elo = (
        matchups.loc[
            valid_margin,
            "elo_difference",
        ]
        .corr(
            matchups.loc[
                valid_margin,
                "actual_home_margin",
            ]
        )
    )

    correlation_strength = (
        matchups.loc[
            valid_margin,
            "adjusted_strength_difference",
        ]
        .corr(
            matchups.loc[
                valid_margin,
                "actual_home_margin",
            ]
        )
    )

    correlation_matchup = (
        matchups.loc[
            valid_margin,
            "offensive_matchup_difference",
        ]
        .corr(
            matchups.loc[
                valid_margin,
                "actual_home_margin",
            ]
        )
    )

    print(
        "Correlation with actual home scoring margin:"
    )

    print(
        f"  ELO difference:               "
        f"{correlation_elo:.3f}"
    )

    print(
        f"  Adjusted strength difference: "
        f"{correlation_strength:.3f}"
    )

    print(
        f"  Offensive matchup difference: "
        f"{correlation_matchup:.3f}"
    )

    print()

    print(
        "These are diagnostic correlations only."
    )

    print(
        "They are NOT fitted prediction weights."
    )

    # ========================================================
    # EARLY-SEASON MATURITY SANITY
    # ========================================================

    section(
        "RATING MATURITY"
    )

    print(
        f"Average home maturity: "
        f"{matchups['home_rating_maturity'].mean():.3f}"
    )

    print(
        f"Average away maturity: "
        f"{matchups['away_rating_maturity'].mean():.3f}"
    )

    print(
        f"Average matchup maturity: "
        f"{matchups['matchup_rating_maturity'].mean():.3f}"
    )

    low_maturity_games = int(
        (
            matchups[
                "matchup_rating_maturity"
            ]
            <
            0.50
        ).sum()
    )

    mature_games = int(
        (
            matchups[
                "matchup_rating_maturity"
            ]
            >=
            1.00
        ).sum()
    )

    print(
        f"Low-maturity games (<0.50): "
        f"{low_maturity_games:,}"
    )

    print(
        f"Fully mature games:           "
        f"{mature_games:,}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING MATCHUP OUTPUT"
    )

    save_csv(
        matchups,
        MATCHUP_HISTORY_FILE,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "MATCHUP ENGINE COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        f"  {MATCHUP_HISTORY_FILE}"
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
        "  RAW OFFENCE / DEFENCE"
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

    print()


if __name__ == "__main__":
    main()