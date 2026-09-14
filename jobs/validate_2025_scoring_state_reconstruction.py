from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENGINE IMPORTS
# ============================================================

from engine.environment import (
    EnvironmentConfig,
    GameEnvironmentEngine,
)

from engine.possessions import (
    PossessionConfig,
    PossessionModel,
)

from engine.scoring import (
    ScoringConfig,
    ScoringModel,
)

from engine.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
)


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TEAM_GAMES_PATH = (
    PROCESSED_DIR
    / "historical_team_games.csv"
)

MATCHUP_PATH = (
    PROCESSED_DIR
    / "matchup_history.csv"
)

ENVIRONMENT_PATH = (
    PROCESSED_DIR
    / "game_environment_history.csv"
)

POSSESSION_PATH = (
    PROCESSED_DIR
    / "possession_drive_history.csv"
)

SCORING_PATH = (
    PROCESSED_DIR
    / "scoring_history.csv"
)

SIMULATION_PATH = (
    PROCESSED_DIR
    / "simulation_history.csv"
)


FLOAT_TOLERANCE = 1e-8


# ============================================================
# DISPLAY
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# LOAD
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    if "start_date" in frame.columns:

        frame["start_date"] = pd.to_datetime(
            frame["start_date"],
            errors="coerce",
            utc=True,
        )

    if "game_id" in frame.columns:

        frame["game_id"] = pd.to_numeric(
            frame["game_id"],
            errors="coerce",
        ).astype("Int64")

    return frame


# ============================================================
# HISTORICAL TEAM-GAME NORMALISATION
# ============================================================

def derive_opponent_name(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct opponent_name from the paired team row.

    Historical team-game data contains one row per team per game.
    For every valid model game there should be exactly two rows:

        home team
        away team

    Therefore the opponent can be derived without introducing
    any new information or future leakage.
    """

    data = team_games.copy()

    if "opponent_name" in data.columns:

        existing = (
            data["opponent_name"]
            .astype("string")
            .str.strip()
        )

        if existing.notna().all():

            print(
                "opponent_name already present."
            )

            return data

    required = [
        "game_id",
        "team_name",
        "home_away",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Cannot derive opponent_name. "
            f"Missing columns: {missing}"
        )

    data["team_name"] = (
        data["team_name"]
        .astype("string")
        .str.strip()
    )

    data["home_away"] = (
        data["home_away"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    opponent_map: dict[
        tuple[int, str],
        str,
    ] = {}

    valid_games = 0
    invalid_games = 0

    for game_id, rows in data.groupby(
        "game_id",
        sort=False,
    ):

        if len(rows) != 2:

            invalid_games += 1
            continue

        home_rows = rows[
            rows["home_away"]
            ==
            "home"
        ]

        away_rows = rows[
            rows["home_away"]
            ==
            "away"
        ]

        if (
            len(home_rows) != 1
            or
            len(away_rows) != 1
        ):

            invalid_games += 1
            continue

        home_team = str(
            home_rows.iloc[0][
                "team_name"
            ]
        ).strip()

        away_team = str(
            away_rows.iloc[0][
                "team_name"
            ]
        ).strip()

        opponent_map[
            (
                int(game_id),
                home_team,
            )
        ] = away_team

        opponent_map[
            (
                int(game_id),
                away_team,
            )
        ] = home_team

        valid_games += 1

    data["opponent_name"] = [
        opponent_map.get(
            (
                int(game_id)
                if pd.notna(game_id)
                else -1,
                str(team).strip(),
            ),
            pd.NA,
        )
        for game_id, team
        in zip(
            data["game_id"],
            data["team_name"],
        )
    ]

    print(
        f"Games with paired opponent derivation: "
        f"{valid_games:,}"
    )

    print(
        f"Games not exactly home+away pair:      "
        f"{invalid_games:,}"
    )

    print(
        f"Rows with derived opponent_name:       "
        f"{data['opponent_name'].notna().sum():,}"
        f" / {len(data):,}"
    )

    return data


# ============================================================
# SORT / ALIGN
# ============================================================

def sort_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    columns = [
        column
        for column in [
            "season",
            "start_date",
            "week",
            "game_id",
        ]
        if column in result.columns
    ]

    if columns:

        result = (
            result
            .sort_values(
                columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    return result


# ============================================================
# EXACT COMPARISON
# ============================================================

def compare_frames(
    name: str,
    saved: pd.DataFrame,
    rebuilt: pd.DataFrame,
    important_columns: list[str],
) -> bool:

    section(
        f"COMPARE: {name}"
    )

    saved = sort_frame(
        saved
    )

    rebuilt = sort_frame(
        rebuilt
    )

    print(
        f"Saved rows:        {len(saved):,}"
    )

    print(
        f"Rebuilt rows:      {len(rebuilt):,}"
    )

    if len(saved) != len(rebuilt):

        print(
            "FAIL: row counts differ."
        )

        return False

    if "game_id" not in saved.columns:
        print(
            "FAIL: saved frame missing game_id."
        )
        return False

    if "game_id" not in rebuilt.columns:
        print(
            "FAIL: rebuilt frame missing game_id."
        )
        return False

    saved_ids = (
        saved["game_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    rebuilt_ids = (
        rebuilt["game_id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    if saved_ids != rebuilt_ids:

        print(
            "FAIL: game_id ordering/coverage differs."
        )

        print(
            f"Saved unique IDs:   "
            f"{len(set(saved_ids)):,}"
        )

        print(
            f"Rebuilt unique IDs: "
            f"{len(set(rebuilt_ids)):,}"
        )

        return False

    overall_pass = True

    for column in important_columns:

        if column not in saved.columns:

            print(
                f"{column:<45} "
                "FAIL - missing from saved history"
            )

            overall_pass = False
            continue

        if column not in rebuilt.columns:

            print(
                f"{column:<45} "
                "FAIL - missing from rebuilt history"
            )

            overall_pass = False
            continue

        saved_values = pd.to_numeric(
            saved[column],
            errors="coerce",
        )

        rebuilt_values = pd.to_numeric(
            rebuilt[column],
            errors="coerce",
        )

        nan_mismatch = int(
            (
                saved_values.isna()
                !=
                rebuilt_values.isna()
            ).sum()
        )

        valid = (
            saved_values.notna()
            &
            rebuilt_values.notna()
        )

        if valid.any():

            differences = np.abs(
                saved_values[valid].to_numpy(
                    dtype=float
                )
                -
                rebuilt_values[valid].to_numpy(
                    dtype=float
                )
            )

            max_difference = float(
                differences.max()
            )

            mismatch_count = int(
                (
                    differences
                    >
                    FLOAT_TOLERANCE
                ).sum()
            )

        else:

            max_difference = 0.0
            mismatch_count = 0

        mismatch_count += (
            nan_mismatch
        )

        passed = (
            mismatch_count == 0
        )

        print(
            f"{column:<45} "
            f"{'PASS' if passed else 'FAIL':<5} "
            f"max diff={max_difference:.12g} "
            f"mismatches={mismatch_count:,}"
        )

        if not passed:

            overall_pass = False

    print()

    print(
        f"{name}: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    return overall_pass


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- SCORING STATE RECONSTRUCTION"
    )

    print(
        "READ ONLY validation."
    )

    print()

    print(
        "Historical pipeline:"
    )

    print(
        "team games"
    )

    print(
        "  -> opponent derivation"
    )

    print(
        "  -> environment"
    )

    print(
        "  -> possession / drives"
    )

    print(
        "  -> scoring"
    )

    print(
        "  -> Monte Carlo uncertainty"
    )

    # ========================================================
    # LOAD
    # ========================================================

    section(
        "1. LOAD HISTORICAL DATA"
    )

    team_games = load_csv(
        TEAM_GAMES_PATH
    )

    matchup_history = load_csv(
        MATCHUP_PATH
    )

    saved_environment = load_csv(
        ENVIRONMENT_PATH
    )

    saved_possession = load_csv(
        POSSESSION_PATH
    )

    saved_scoring = load_csv(
        SCORING_PATH
    )

    saved_simulation = load_csv(
        SIMULATION_PATH
    )

    print(
        f"Historical team-game rows: "
        f"{len(team_games):,}"
    )

    print(
        f"Matchup history rows:       "
        f"{len(matchup_history):,}"
    )

    print(
        f"Saved environment rows:     "
        f"{len(saved_environment):,}"
    )

    print(
        f"Saved possession rows:      "
        f"{len(saved_possession):,}"
    )

    print(
        f"Saved scoring rows:         "
        f"{len(saved_scoring):,}"
    )

    print(
        f"Saved simulation rows:      "
        f"{len(saved_simulation):,}"
    )

    # ========================================================
    # DERIVE OPPONENT
    # ========================================================

    section(
        "2. DERIVE HISTORICAL OPPONENT IDENTITY"
    )

    team_games = derive_opponent_name(
        team_games
    )

    # Only games appearing in matchup history belong to the
    # scoring/model universe.

    model_game_ids = set(
        matchup_history[
            "game_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    team_games_model = (
        team_games[
            team_games[
                "game_id"
            ]
            .isin(
                model_game_ids
            )
        ]
        .copy()
    )

    print()
    print(
        f"Model-universe game IDs:   "
        f"{len(model_game_ids):,}"
    )

    print(
        f"Model-universe team rows:  "
        f"{len(team_games_model):,}"
    )

    missing_opponents = int(
        team_games_model[
            "opponent_name"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing opponent_name:     "
        f"{missing_opponents:,}"
    )

    if missing_opponents:

        raise ValueError(
            "Model-universe historical rows "
            "still contain missing opponent_name."
        )

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    section(
        "3. REPLAY ENVIRONMENT"
    )

    environment_engine = (
        GameEnvironmentEngine(
            EnvironmentConfig()
        )
    )

    rebuilt_environment = (
        environment_engine
        .process_history(
            team_history=(
                team_games_model
            ),
            matchup_history=(
                matchup_history
            ),
        )
    )

    print(
        f"Environment rows rebuilt: "
        f"{len(rebuilt_environment):,}"
    )

    print(
        f"Environment team states:  "
        f"{len(environment_engine.states):,}"
    )

    # ========================================================
    # POSSESSION
    # ========================================================

    section(
        "4. REPLAY POSSESSION / DRIVE MODEL"
    )

    possession_engine = (
        PossessionModel(
            PossessionConfig()
        )
    )

    rebuilt_possession = (
        possession_engine
        .process_history(
            team_history=(
                team_games_model
            ),
            environment_history=(
                rebuilt_environment
            ),
        )
    )

    print(
        f"Possession rows rebuilt: "
        f"{len(rebuilt_possession):,}"
    )

    print(
        f"Possession team states:  "
        f"{len(possession_engine.states):,}"
    )

    # ========================================================
    # SCORING
    # ========================================================

    section(
        "5. REPLAY SCORING MODEL"
    )

    scoring_engine = (
        ScoringModel(
            ScoringConfig()
        )
    )

    rebuilt_scoring = (
        scoring_engine
        .process_history(
            team_history=(
                team_games_model
            ),
            possession_history=(
                rebuilt_possession
            ),
        )
    )

    print(
        f"Scoring rows rebuilt: "
        f"{len(rebuilt_scoring):,}"
    )

    print(
        f"Scoring team states:  "
        f"{len(scoring_engine.states):,}"
    )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    section(
        "6. REPLAY MONTE CARLO"
    )

    monte_carlo_engine = (
        MonteCarloEngine(
            MonteCarloConfig()
        )
    )

    rebuilt_simulation = (
        monte_carlo_engine
        .process_history(
            rebuilt_scoring
        )
    )

    print(
        f"Simulation rows rebuilt: "
        f"{len(rebuilt_simulation):,}"
    )

    print()

    print(
        "Final uncertainty:"
    )

    for key, value in (
        monte_carlo_engine
        .current_uncertainty()
        .items()
    ):

        print(
            f"  {key:<35} "
            f"{value}"
        )

    # ========================================================
    # COMPARISONS
    # ========================================================

    environment_pass = compare_frames(
        "Environment",
        saved_environment,
        rebuilt_environment,
        [
            "expected_home_plays",
            "expected_away_plays",
            "expected_total_plays",
            "expected_home_possession_share",
            "expected_away_possession_share",
            "expected_home_possession_minutes",
            "expected_away_possession_minutes",
            "game_pace_index",
        ],
    )

    possession_pass = compare_frames(
        "Possession / drive",
        saved_possession,
        rebuilt_possession,
        [
            "expected_home_plays_per_drive",
            "expected_away_plays_per_drive",
            "expected_home_drives",
            "expected_away_drives",
            "expected_total_drives",
            "home_offensive_sustainability_index",
            "away_offensive_sustainability_index",
            "home_defensive_resistance_index",
            "away_defensive_resistance_index",
        ],
    )

    scoring_pass = compare_frames(
        "Scoring",
        saved_scoring,
        rebuilt_scoring,
        [
            "expected_home_points_per_play",
            "expected_away_points_per_play",
            "expected_home_points_per_drive",
            "expected_away_points_per_drive",
            "expected_home_points",
            "expected_away_points",
            "expected_total_points",
            "expected_home_margin",
        ],
    )

    simulation_columns = [
        column
        for column in [
            "simulated_home_win_probability",
            "simulated_away_win_probability",
            "simulated_home_score_mean",
            "simulated_away_score_mean",
            "simulated_margin_mean",
            "simulated_total_mean",
            "home_score_p10",
            "home_score_p50",
            "home_score_p90",
            "away_score_p10",
            "away_score_p50",
            "away_score_p90",
        ]
        if (
            column
            in
            saved_simulation.columns
            and
            column
            in
            rebuilt_simulation.columns
        )
    ]

    simulation_pass = compare_frames(
        "Monte Carlo",
        saved_simulation,
        rebuilt_simulation,
        simulation_columns,
    )

    # ========================================================
    # RESULT
    # ========================================================

    section(
        "7. FINAL RESULT"
    )

    results = [
        (
            "Environment",
            environment_pass,
        ),
        (
            "Possession / drive",
            possession_pass,
        ),
        (
            "Scoring",
            scoring_pass,
        ),
        (
            "Monte Carlo",
            simulation_pass,
        ),
    ]

    for name, passed in results:

        print(
            f"{'PASS' if passed else 'FAIL':<6} "
            f"{name}"
        )

    print()

    if all(
        passed
        for _, passed in results
    ):

        print(
            "✓ HISTORICAL SCORING STATE "
            "RECONSTRUCTION IS EXACT."
        )

        print()

        print(
            "2023-2025 scoring state can now "
            "be safely continued into 2026."
        )

    else:

        print(
            "✗ HISTORICAL SCORING STATE "
            "RECONSTRUCTION FAILED."
        )

        print()

        print(
            "Do not build the live 2026 scoring "
            "runner until the failed layer is "
            "understood."
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()