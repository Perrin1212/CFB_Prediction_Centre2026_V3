from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# ALLOW RUNNING:
#
# python -m jobs.build_data
#
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import (
    HISTORICAL_SEASONS,
    PROCESSED_DATA_DIR,
    V1_DATABASE_PATH,
)

from engine.data_loader import (
    CFBDataLoader,
)


# ============================================================
# OUTPUT FILES
# ============================================================

GAMES_OUTPUT = (
    PROCESSED_DATA_DIR
    / "historical_games.csv"
)

TEAM_GAMES_OUTPUT = (
    PROCESSED_DATA_DIR
    / "historical_team_games.csv"
)

TEAMS_OUTPUT = (
    PROCESSED_DATA_DIR
    / "teams.csv"
)


# ============================================================
# HELPERS
# ============================================================

def save_dataframe(
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
        f"  ✓ Saved {len(df):,} rows → {path}"
    )


def print_dataset_summary(
    name: str,
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    if "season" in df.columns:

        seasons = sorted(
            df["season"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        print(
            f"Seasons: {seasons}"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("CFB PREDICTION CENTRE 2026 V2")
    print("DATA FOUNDATION BUILD")
    print("=" * 70)

    print()
    print(
        "V1 database:"
    )

    print(
        f"  {V1_DATABASE_PATH}"
    )

    print()
    print(
        "Historical seasons:"
    )

    print(
        f"  {list(HISTORICAL_SEASONS)}"
    )

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------

    loader = CFBDataLoader(
        V1_DATABASE_PATH
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print(
        "[1/5] Validating V1 database..."
    )

    loader.validate_schema()

    tables = loader.get_tables()

    print(
        "  ✓ Database found"
    )

    print(
        "  ✓ Required tables available"
    )

    print(
        f"  Tables: {', '.join(tables)}"
    )

    # --------------------------------------------------------
    # Teams
    # --------------------------------------------------------

    print()
    print(
        "[2/5] Loading teams..."
    )

    teams = loader.load_teams()

    save_dataframe(
        teams,
        TEAMS_OUTPUT,
    )

    print_dataset_summary(
        "TEAM DATA",
        teams,
    )

    # --------------------------------------------------------
    # Games
    # --------------------------------------------------------

    print()
    print(
        "[3/5] Building historical game dataset..."
    )

    games = loader.build_game_dataset(
        seasons=HISTORICAL_SEASONS,
        completed_only=True,
    )

    save_dataframe(
        games,
        GAMES_OUTPUT,
    )

    print_dataset_summary(
        "HISTORICAL GAME DATA",
        games,
    )

    # --------------------------------------------------------
    # Team/game data
    # --------------------------------------------------------

    print()
    print(
        "[4/5] Building historical team-game dataset..."
    )

    team_games = (
        loader.build_team_game_dataset(
            seasons=HISTORICAL_SEASONS,
        )
    )

    save_dataframe(
        team_games,
        TEAM_GAMES_OUTPUT,
    )

    print_dataset_summary(
        "HISTORICAL TEAM-GAME DATA",
        team_games,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print(
        "[5/5] Running basic data validation..."
    )

    errors: list[str] = []

    if games.empty:

        errors.append(
            "Historical game dataset is empty."
        )

    if team_games.empty:

        errors.append(
            "Historical team-game dataset is empty."
        )

    if teams.empty:

        errors.append(
            "Team dataset is empty."
        )

    # --------------------------------------------------------
    # Game uniqueness
    # --------------------------------------------------------

    if (
        not games.empty
        and "cfbd_id" in games.columns
    ):

        duplicate_games = (
            games["cfbd_id"]
            .duplicated()
            .sum()
        )

        if duplicate_games:

            errors.append(
                f"{duplicate_games:,} duplicate "
                "game IDs detected."
            )

    # --------------------------------------------------------
    # Team-game uniqueness
    # --------------------------------------------------------

    if (
        not team_games.empty
        and "team_game_id"
        in team_games.columns
    ):

        duplicate_team_games = (
            team_games[
                "team_game_id"
            ]
            .duplicated()
            .sum()
        )

        if duplicate_team_games:

            errors.append(
                f"{duplicate_team_games:,} duplicate "
                "team-game IDs detected."
            )

    # --------------------------------------------------------
    # Missing scores
    # --------------------------------------------------------

    if not games.empty:

        missing_scores = (
            games[
                [
                    "home_points",
                    "away_points",
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        if missing_scores:

            errors.append(
                f"{missing_scores:,} games have "
                "missing scores."
            )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()

    if errors:

        print(
            "✗ DATA FOUNDATION FAILED"
        )

        print()

        for error in errors:

            print(
                f"  - {error}"
            )

        raise SystemExit(1)

    print(
        "✓ DATA FOUNDATION BUILD COMPLETE"
    )

    print()
    print(
        "V2 datasets created:"
    )

    print(
        f"  {GAMES_OUTPUT}"
    )

    print(
        f"  {TEAM_GAMES_OUTPUT}"
    )

    print(
        f"  {TEAMS_OUTPUT}"
    )

    print()
    print(
        "The V1 database has NOT been modified."
    )

    print()


if __name__ == "__main__":

    main()