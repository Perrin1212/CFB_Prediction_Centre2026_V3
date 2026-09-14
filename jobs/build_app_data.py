from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

PREDICTIONS_DIR = (
    DATA_DIR
    / "predictions"
)

APP_DATA_DIR = (
    DATA_DIR
    / "app"
)

MARKET_DIR = (
    DATA_DIR
    / "market"
)

MARKET_CURRENT = (
    MARKET_DIR
    / "market_lines_current.csv"
)


TRACKER_CURRENT = (
    PREDICTIONS_DIR
    / "prediction_tracker_current.csv"
)

TRACKER_HISTORY = (
    PREDICTIONS_DIR
    / "prediction_tracker_history.csv"
)

LOCKED = (
    PREDICTIONS_DIR
    / "official_locked_predictions.csv"
)

LOCK_STATUS = (
    PREDICTIONS_DIR
    / "official_lock_status.csv"
)


OUT_GAMES = (
    APP_DATA_DIR
    / "games.csv"
)

OUT_ELO_WEEKLY = (
    APP_DATA_DIR
    / "elo_weekly.csv"
)

OUT_ELO_RANKINGS = (
    APP_DATA_DIR
    / "elo_rankings_current.csv"
)

OUT_TEAMS = (
    APP_DATA_DIR
    / "teams.csv"
)

OUT_TEAM_PROFILES = (
    APP_DATA_DIR
    / "team_profiles.csv"
)

OUT_PERFORMANCE = (
    APP_DATA_DIR
    / "performance.json"
)

OUT_MANIFEST = (
    APP_DATA_DIR
    / "app_data_manifest.json"
)


# ============================================================
# DISPLAY
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print(
        "=" * 78
    )
    print(
        title
    )
    print(
        "=" * 78
    )


# ============================================================
# TIME
# ============================================================

def utc_now() -> pd.Timestamp:

    return pd.Timestamp(
        datetime.now(
            timezone.utc
        )
    )


def iso_utc(
    value: pd.Timestamp,
) -> str:

    return (
        value
        .isoformat(
            timespec="seconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def pick_col(
    frame: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> str | None:

    exact = {
        str(column): str(column)
        for column in frame.columns
    }

    lower = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:

        if candidate in exact:

            return exact[
                candidate
            ]

        if candidate.lower() in lower:

            return lower[
                candidate.lower()
            ]

    if required:

        raise RuntimeError(
            "Could not find any of these required columns:\n"
            +
            "\n".join(
                f"  - {candidate}"
                for candidate in candidates
            )
            +
            "\n\nAvailable columns:\n"
            +
            ", ".join(
                map(
                    str,
                    frame.columns,
                )
            )
        )

    return None


def numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def clean_text(
    value: Any,
) -> str:

    if value is None:

        return ""

    try:

        if pd.isna(
            value
        ):

            return ""

    except Exception:

        pass

    return str(
        value
    ).strip()


def read_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


# ============================================================
# V1 TEAM METADATA
# ============================================================

def resolve_v1_db() -> Path:

    try:

        from config.settings import V1_DB_PATH  # type: ignore

        path = Path(
            V1_DB_PATH
        )

        if path.exists():

            return path

    except Exception:

        pass

    candidates = [
        (
            PROJECT_ROOT.parent
            / "CFB_Prediction_Centre2026"
            / "data"
            / "cfb_prediction.db"
        ),
        (
            DATA_DIR
            / "cfb_prediction.db"
        ),
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "Could not locate the V1 SQLite database.\n"
        "Expected sibling project database at:\n"
        f"{PROJECT_ROOT.parent / 'CFB_Prediction_Centre2026' / 'data' / 'cfb_prediction.db'}"
    )


def load_team_metadata() -> pd.DataFrame:

    db_path = (
        resolve_v1_db()
    )

    uri = (
        f"file:{db_path.as_posix()}?mode=ro"
    )

    with sqlite3.connect(
        uri,
        uri=True,
    ) as connection:

        teams = pd.read_sql_query(
            """
            SELECT
                cfbd_id,
                school,
                abbreviation,
                mascot,
                conference,
                classification,
                color,
                logo_url
            FROM teams
            """,
            connection,
        )

    teams[
        "school"
    ] = (
        teams[
            "school"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    teams[
        "classification"
    ] = (
        teams[
            "classification"
        ]
        .astype(
            "string"
        )
        .str.lower()
        .str.strip()
    )

    return teams


# ============================================================
# GAME DATASET
# ============================================================

def probability_col(
    frame: pd.DataFrame,
    side: str,
) -> str:

    return pick_col(
        frame,
        [
            f"{side}_win_probability",
            f"{side}_probability",
            f"{side}_win_prob",
            f"official_{side}_win_probability",
        ],
        required=True,
    )


def create_games_dataset(
    current: pd.DataFrame,
    locks: pd.DataFrame,
    lock_status: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:

    game_id_col = pick_col(
        current,
        [
            "cfbd_game_id",
            "tracker_game_id",
            "cfbd_id",
            "game_id",
        ],
        required=True,
    )

    current = (
        current.copy()
    )

    current[
        "cfbd_game_id"
    ] = (
        numeric(
            current[
                game_id_col
            ]
        )
        .astype(
            "Int64"
        )
    )

    home_team_col = pick_col(
        current,
        [
            "home_team",
        ],
        required=True,
    )

    away_team_col = pick_col(
        current,
        [
            "away_team",
        ],
        required=True,
    )

    start_date_col = pick_col(
        current,
        [
            "start_date",
            "kickoff",
            "start_time",
        ],
        required=True,
    )

    home_probability_col = (
        probability_col(
            current,
            "home",
        )
    )

    away_probability_col = (
        probability_col(
            current,
            "away",
        )
    )

    rename_map: dict[
        str,
        str,
    ] = {}

    if home_team_col != "home_team":

        rename_map[
            home_team_col
        ] = "home_team"

    if away_team_col != "away_team":

        rename_map[
            away_team_col
        ] = "away_team"

    if start_date_col != "start_date":

        rename_map[
            start_date_col
        ] = "start_date"

    if home_probability_col != "home_win_probability":

        rename_map[
            home_probability_col
        ] = "home_win_probability"

    if away_probability_col != "away_win_probability":

        rename_map[
            away_probability_col
        ] = "away_win_probability"

    current = current.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # OFFICIAL LOCK VALUES
    # --------------------------------------------------------

    if not locks.empty:

        lock_id_col = pick_col(
            locks,
            [
                "cfbd_game_id",
                "tracker_game_id",
                "cfbd_id",
                "game_id",
            ],
            required=True,
        )

        locks = (
            locks.copy()
        )

        locks[
            "cfbd_game_id"
        ] = (
            numeric(
                locks[
                    lock_id_col
                ]
            )
            .astype(
                "Int64"
            )
        )

        lock_keep = [
            "cfbd_game_id",
        ]

        for column in [
            "home_win_probability",
            "away_win_probability",
            "predicted_winner",
            "projected_home_score",
            "projected_away_score",
            "final_expected_home_margin",
            "final_expected_total_points",
            "simulation_home_win_probability",
            "simulation_away_win_probability",
            "official_lock_quality",
            "official_lock_cutoff_utc",
            "official_locked_at_utc",
            "official_prediction_is_strict_24h",
            "snapshot_id",
        ]:

            if column in locks.columns:

                lock_keep.append(
                    column
                )

        lock_view = (
            locks[
                lock_keep
            ]
            .copy()
        )

        lock_view = lock_view.rename(
            columns={
                column: (
                    f"locked_{column}"
                )
                for column in lock_view.columns
                if column != "cfbd_game_id"
            }
        )

        current = current.merge(
            lock_view,
            on="cfbd_game_id",
            how="left",
        )

    # --------------------------------------------------------
    # LOCK STATUS
    # --------------------------------------------------------

    if not lock_status.empty:

        status_id_col = pick_col(
            lock_status,
            [
                "cfbd_game_id",
                "tracker_game_id",
                "cfbd_id",
                "game_id",
            ],
            required=True,
        )

        lock_status = (
            lock_status.copy()
        )

        lock_status[
            "cfbd_game_id"
        ] = (
            numeric(
                lock_status[
                    status_id_col
                ]
            )
            .astype(
                "Int64"
            )
        )

        status_keep = [
            column
            for column in [
                "cfbd_game_id",
                "lock_status",
                "hours_to_kickoff",
                "hours_until_lock",
                "lock_cutoff_utc",
            ]
            if column in lock_status.columns
        ]

        current = current.merge(
            lock_status[
                status_keep
            ],
            on="cfbd_game_id",
            how="left",
            suffixes=(
                "",
                "_status",
            ),
        )

    # --------------------------------------------------------
    # DISPLAY VALUES
    # --------------------------------------------------------

    for column in [
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "projected_home_score",
        "projected_away_score",
        "final_expected_home_margin",
        "final_expected_total_points",
        "simulation_home_win_probability",
        "simulation_away_win_probability",
    ]:

        locked_column = (
            f"locked_{column}"
        )

        if (
            locked_column
            in current.columns
            and column
            in current.columns
        ):

            current[
                f"display_{column}"
            ] = (
                current[
                    locked_column
                ]
                .combine_first(
                    current[
                        column
                    ]
                )
            )

        elif column in current.columns:

            current[
                f"display_{column}"
            ] = current[
                column
            ]

        elif locked_column in current.columns:

            current[
                f"display_{column}"
            ] = current[
                locked_column
            ]

    if (
        "locked_official_lock_quality"
        in current.columns
    ):

        current[
            "official_lock_quality"
        ] = current[
            "locked_official_lock_quality"
        ]

    elif (
        "official_lock_quality"
        not in current.columns
    ):

        current[
            "official_lock_quality"
        ] = pd.NA

    if (
        "lock_status"
        not in current.columns
    ):

        current[
            "lock_status"
        ] = "dynamic"

    current[
        "is_locked"
    ] = (
        current[
            "lock_status"
        ]
        .astype(
            "string"
        )
        .eq(
            "locked"
        )
    )

    current[
        "is_strict_24h_lock"
    ] = (
        current[
            "official_lock_quality"
        ]
        .astype(
            "string"
        )
        .eq(
            "on_time_snapshot"
        )
    )

    current[
        "is_late_lock"
    ] = (
        current[
            "official_lock_quality"
        ]
        .astype(
            "string"
        )
        .eq(
            "late_initial_capture"
        )
    )

    display_home = numeric(
        current[
            "display_home_win_probability"
        ]
    )

    display_away = numeric(
        current[
            "display_away_win_probability"
        ]
    )

    current[
        "model_favourite_probability"
    ] = pd.concat(
        [
            display_home,
            display_away,
        ],
        axis=1,
    ).max(
        axis=1
    )

    current[
        "confidence_bucket"
    ] = pd.cut(
        current[
            "model_favourite_probability"
        ],
        bins=[
            -np.inf,
            0.55,
            0.60,
            0.70,
            0.80,
            np.inf,
        ],
        labels=[
            "Toss-up",
            "Lean",
            "Solid",
            "Strong",
            "Elite",
        ],
        right=False,
    ).astype(
        "string"
    )

    current[
        "start_date_utc"
    ] = pd.to_datetime(
        current[
            "start_date"
        ],
        errors="coerce",
        utc=True,
    )

    if (
        "season"
        not in current.columns
    ):

        current[
            "season"
        ] = 2026

    current[
        "season"
    ] = (
        numeric(
            current[
                "season"
            ]
        )
        .fillna(
            2026
        )
        .astype(
            "Int64"
        )
    )

    if (
        "week"
        not in current.columns
    ):

        current[
            "week"
        ] = pd.NA

    current[
        "week"
    ] = (
        numeric(
            current[
                "week"
            ]
        )
        .astype(
            "Int64"
        )
    )

    # --------------------------------------------------------
    # TEAM METADATA
    # --------------------------------------------------------

    home_metadata = teams.rename(
        columns={
            "school": "home_team",
            "abbreviation": "home_abbreviation",
            "mascot": "home_mascot",
            "conference": "home_conference",
            "classification": "home_classification",
            "color": "home_color",
            "logo_url": "home_logo_url",
        }
    )

    away_metadata = teams.rename(
        columns={
            "school": "away_team",
            "abbreviation": "away_abbreviation",
            "mascot": "away_mascot",
            "conference": "away_conference",
            "classification": "away_classification",
            "color": "away_color",
            "logo_url": "away_logo_url",
        }
    )

    home_columns = [
        column
        for column in home_metadata.columns
        if column != "cfbd_id"
    ]

    away_columns = [
        column
        for column in away_metadata.columns
        if column != "cfbd_id"
    ]

    current = current.merge(
        home_metadata[
            home_columns
        ],
        on="home_team",
        how="left",
    )

    current = current.merge(
        away_metadata[
            away_columns
        ],
        on="away_team",
        how="left",
    )

    return (
        current
        .sort_values(
            [
                "start_date_utc",
                "cfbd_game_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )



# ============================================================
# MARKET DISPLAY DATA (UI ONLY)
# ============================================================

def merge_market_display_data(
    games: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the latest captured market line to the UI game dataset.

    IMPORTANT:
    This is a display-only merge. Market data does not feed ELO,
    probabilities, score projections, locks, or any frozen model feature.
    """
    if games.empty or market.empty:
        return games

    market = market.copy()

    game_id_col = pick_col(
        market,
        [
            "cfbd_game_id",
            "cfbd_id",
            "game_id",
        ],
        required=True,
    )

    market["cfbd_game_id"] = (
        numeric(market[game_id_col])
        .astype("Int64")
    )

    if "captured_at_utc" in market.columns:
        market["_market_captured"] = pd.to_datetime(
            market["captured_at_utc"],
            errors="coerce",
            utc=True,
        )
        market = market.sort_values(
            ["cfbd_game_id", "_market_captured"],
            kind="stable",
            na_position="last",
        )
    else:
        market["_market_captured"] = pd.NaT

    # market_lines_current should already be one row/game, but this protects
    # the UI if a provider duplication ever appears.
    market = market.drop_duplicates(
        subset=["cfbd_game_id"],
        keep="last",
    )

    keep_map = {
        "provider": "market_provider",
        "captured_at_utc": "market_captured_at_utc",
        "capture_timing": "market_capture_timing",
        "spread": "market_spread",
        "spread_favourite": "market_spread_favourite",
        "formatted_spread": "market_formatted_spread",
        "spread_open": "market_spread_open",
        "over_under": "market_over_under",
        "over_under_open": "market_over_under_open",
        "home_moneyline": "market_home_moneyline",
        "away_moneyline": "market_away_moneyline",
    }

    keep = ["cfbd_game_id"]
    rename: dict[str, str] = {}

    for source, target in keep_map.items():
        if source in market.columns:
            keep.append(source)
            rename[source] = target

    market_view = (
        market[keep]
        .copy()
        .rename(columns=rename)
    )

    # Avoid stale duplicate market columns if this builder is run on a tracker
    # that later begins carrying them itself.
    overlapping = [
        column
        for column in market_view.columns
        if column != "cfbd_game_id"
        and column in games.columns
    ]
    if overlapping:
        games = games.drop(columns=overlapping)

    return games.merge(
        market_view,
        on="cfbd_game_id",
        how="left",
        validate="m:1",
    )


# ============================================================
# LIVE 2026 WEEKLY ELO
# ============================================================

def detect_live_elo_columns(
    frame: pd.DataFrame,
) -> dict[
    str,
    str | None,
]:

    """
    Detect the actual ELO columns carried by the live 2026
    production/tracker output.

    Current production columns are:

        home_pregame_elo
        away_pregame_elo

    Additional aliases are retained for future compatibility.
    """

    return {
        "week": pick_col(
            frame,
            [
                "week",
            ],
            required=True,
        ),
        "start_date": pick_col(
            frame,
            [
                "start_date_utc",
                "start_date",
                "kickoff",
            ],
        ),
        "home_team": pick_col(
            frame,
            [
                "home_team",
            ],
            required=True,
        ),
        "away_team": pick_col(
            frame,
            [
                "away_team",
            ],
            required=True,
        ),
        "home_pre": pick_col(
            frame,
            [
                "home_pregame_elo",
                "home_pre_elo",
                "pregame_home_elo",
                "home_elo_pre",
                "home_pre_rating",
                "home_rating_pre",
                "home_elo",
            ],
            required=True,
        ),
        "away_pre": pick_col(
            frame,
            [
                "away_pregame_elo",
                "away_pre_elo",
                "pregame_away_elo",
                "away_elo_pre",
                "away_pre_rating",
                "away_rating_pre",
                "away_elo",
            ],
            required=True,
        ),
    }


def build_elo_weekly_from_live_games(
    games: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:

    """
    Build the application's 2026 ELO history without leaking future
    schedule weeks into the historical chart.

    IMPORTANT SEMANTICS
    -------------------

    Preseason
        The earliest 2026 pregame ELO observed for each FBS team.

    Week N
        The team's ELO ENTERING Week N, after every completed game
        from earlier weeks has been applied.

    Why this reconstruction is necessary
    -------------------------------------

    The production prediction files contain pregame ELO values for the
    entire future schedule.  A future game's pregame ELO can already
    include updates from games that have completed today.

    Example:
        Stanford plays in Week 0 and its ELO rises from 1450 to 1488.
        If Stanford's next scheduled game were Week 3, that Week 3 row
        would carry 1488.  We must NOT interpret that as a Week 3 ELO
        change.  The change happened after the completed Week 0 game and
        therefore belongs to the Week 1 entering snapshot.

    To reconstruct that correctly we:
        1. Take each completed game's pregame state.
        2. Find that team's next chronological pregame observation.
           In a chronological production replay, that next observation
           contains the postgame ELO after the completed game.
        3. Attach that inferred postgame ELO to Week (game_week + 1).
        4. Carry the rating through bye weeks.
        5. Stop the app history at the CURRENT schedule week.  Future
           weeks are never rendered as historical ELO snapshots.

    This keeps future predictions available for game forecasting while
    preventing future schedule rows from creating fake future ELO moves.
    """

    output_columns = [
        "team",
        "season",
        "week",
        "week_label",
        "elo",
        "previous_elo",
        "elo_change",
        "rank",
        "previous_rank",
        "rank_change",
        "conference",
        "classification",
        "logo_url",
        "color",
    ]

    if games.empty:
        return pd.DataFrame(columns=output_columns)

    columns = detect_live_elo_columns(games)
    data = games.copy()

    data["_week"] = numeric(
        data[columns["week"]]
    ).astype("Int64")

    if columns["start_date"]:
        data["_start_date"] = pd.to_datetime(
            data[columns["start_date"]],
            errors="coerce",
            utc=True,
        )
    else:
        data["_start_date"] = pd.NaT

    # --------------------------------------------------------
    # COMPLETED FLAG
    # --------------------------------------------------------

    if "game_status" in data.columns:
        data["_completed"] = (
            data["game_status"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(
                [
                    "completed",
                    "complete",
                    "final",
                    "closed",
                ]
            )
        )
    elif (
        "actual_home_points" in data.columns
        and "actual_away_points" in data.columns
    ):
        data["_completed"] = (
            numeric(data["actual_home_points"]).notna()
            & numeric(data["actual_away_points"]).notna()
        )
    elif (
        "final_home_score" in data.columns
        and "final_away_score" in data.columns
    ):
        data["_completed"] = (
            numeric(data["final_home_score"]).notna()
            & numeric(data["final_away_score"]).notna()
        )
    else:
        data["_completed"] = False

    # --------------------------------------------------------
    # PREGAME OBSERVATIONS FOR BOTH SIDES
    # --------------------------------------------------------

    observations: list[dict[str, Any]] = []

    ordered_games = data.sort_values(
        [
            "_start_date",
            "_week",
        ],
        kind="stable",
        na_position="last",
    )

    for game_index, row in ordered_games.iterrows():
        week = row["_week"]

        if pd.isna(week):
            continue

        for side in ["home", "away"]:
            team = clean_text(
                row[columns[f"{side}_team"]]
            )

            pregame_elo = pd.to_numeric(
                pd.Series(
                    [
                        row[columns[f"{side}_pre"]]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if not team or pd.isna(pregame_elo):
                continue

            observations.append(
                {
                    "game_index": game_index,
                    "team": team,
                    "week": int(week),
                    "start_date": row["_start_date"],
                    "elo": float(pregame_elo),
                    "completed": bool(row["_completed"]),
                }
            )

    observations_frame = pd.DataFrame(observations)

    if observations_frame.empty:
        return pd.DataFrame(columns=output_columns)

    # --------------------------------------------------------
    # FBS UNIVERSE ONLY
    # --------------------------------------------------------

    fbs_metadata = (
        teams.loc[
            teams["classification"].eq("fbs"),
            [
                "school",
                "conference",
                "classification",
                "logo_url",
                "color",
            ],
        ]
        .rename(columns={"school": "team"})
        .copy()
    )

    fbs_names = set(
        fbs_metadata["team"]
        .dropna()
        .astype(str)
        .tolist()
    )

    observations_frame = observations_frame[
        observations_frame["team"].isin(fbs_names)
    ].copy()

    if observations_frame.empty:
        return pd.DataFrame(columns=output_columns)

    observations_frame = observations_frame.sort_values(
        [
            "team",
            "start_date",
            "week",
            "game_index",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # PRESEASON SNAPSHOT
    # --------------------------------------------------------

    preseason = (
        observations_frame
        .groupby("team", as_index=False)
        .first()[["team", "elo"]]
    )

    preseason["week"] = 0

    # --------------------------------------------------------
    # DETERMINE THE CURRENT SCHEDULE WEEK
    # --------------------------------------------------------
    #
    # Use the next chronological uncompleted game, not the maximum week
    # found in the season schedule.  Therefore a September Week 1 app
    # cannot suddenly render Week 3/10/15 just because those future rows
    # already exist in the prediction file.
    # --------------------------------------------------------

    now = pd.Timestamp.now(tz="UTC")

    upcoming = data[
        (~data["_completed"])
        & data["_week"].notna()
        & data["_start_date"].notna()
    ].copy()

    upcoming_future = upcoming[
        upcoming["_start_date"] >= (now - pd.Timedelta(hours=12))
    ].copy()

    if not upcoming_future.empty:
        next_game = (
            upcoming_future
            .sort_values("_start_date", kind="stable")
            .iloc[0]
        )
        current_week = int(next_game["_week"])

    elif not upcoming.empty:
        next_game = (
            upcoming
            .sort_values("_start_date", kind="stable")
            .iloc[0]
        )
        current_week = int(next_game["_week"])

    else:
        completed_weeks = data.loc[
            data["_completed"] & data["_week"].notna(),
            "_week",
        ]

        if completed_weeks.empty:
            current_week = 1
        else:
            current_week = int(completed_weeks.max()) + 1

    # Week 0 is represented separately by the Preseason row in the app.
    current_week = max(1, current_week)

    # --------------------------------------------------------
    # INFER POSTGAME ELO CHANGES FROM THE NEXT PREGAME STATE
    # --------------------------------------------------------
    #
    # For a completed game, the team's next chronological pregame row
    # contains the rating after that result has been applied.  Crucially,
    # we attach that value to game_week + 1 rather than to the week number
    # of the future game from which the value was observed.
    # --------------------------------------------------------

    changes: list[dict[str, Any]] = []

    for team, team_obs in observations_frame.groupby(
        "team",
        sort=False,
    ):
        team_obs = team_obs.sort_values(
            [
                "start_date",
                "week",
                "game_index",
            ],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

        for position in range(len(team_obs)):
            row = team_obs.iloc[position]

            if not bool(row["completed"]):
                continue

            if position + 1 >= len(team_obs):
                # No later scheduled observation exists from which we can
                # infer the postgame rating safely.  Do not invent one.
                continue

            next_row = team_obs.iloc[position + 1]

            if pd.isna(next_row["elo"]):
                continue

            effective_week = int(row["week"]) + 1

            # Never create a future historical snapshot.
            if effective_week > current_week:
                continue

            changes.append(
                {
                    "team": team,
                    "week": effective_week,
                    "elo": float(next_row["elo"]),
                    "source_game_week": int(row["week"]),
                    "source_game_date": row["start_date"],
                }
            )

    changes_frame = pd.DataFrame(changes)

    if not changes_frame.empty:
        # If an unusual schedule gives a team multiple completed games that
        # map into the same entering-week snapshot, keep the last completed
        # game's resulting state.
        changes_frame = (
            changes_frame
            .sort_values(
                [
                    "team",
                    "week",
                    "source_game_date",
                ],
                kind="stable",
                na_position="last",
            )
            .drop_duplicates(
                subset=["team", "week"],
                keep="last",
            )
        )

        snapshots = pd.concat(
            [
                preseason[["team", "week", "elo"]],
                changes_frame[["team", "week", "elo"]],
            ],
            ignore_index=True,
        )
    else:
        snapshots = preseason[["team", "week", "elo"]].copy()

    snapshots = snapshots.drop_duplicates(
        subset=["team", "week"],
        keep="last",
    )

    # --------------------------------------------------------
    # CARRY RATINGS THROUGH BYE WEEKS — ONLY TO CURRENT WEEK
    # --------------------------------------------------------

    all_fbs_teams = sorted(fbs_names)

    grid = (
        pd.MultiIndex
        .from_product(
            [
                all_fbs_teams,
                range(0, current_week + 1),
            ],
            names=["team", "week"],
        )
        .to_frame(index=False)
    )

    weekly = grid.merge(
        snapshots[["team", "week", "elo"]],
        on=["team", "week"],
        how="left",
    )

    weekly = weekly.sort_values(
        ["team", "week"],
        kind="stable",
    )

    weekly["elo"] = (
        weekly
        .groupby("team")["elo"]
        .ffill()
    )

    weekly = weekly.dropna(
        subset=["elo"]
    ).copy()

    weekly["season"] = 2026

    weekly["week_label"] = np.where(
        weekly["week"].eq(0),
        "Preseason",
        "Week " + weekly["week"].astype(int).astype(str),
    )

    weekly = weekly.merge(
        fbs_metadata,
        on="team",
        how="left",
    )

    # --------------------------------------------------------
    # RANKS / MOVEMENT
    # --------------------------------------------------------

    weekly["rank"] = (
        weekly
        .groupby("week")["elo"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype("Int64")
    )

    weekly["previous_elo"] = (
        weekly
        .groupby("team")["elo"]
        .shift(1)
    )

    weekly["elo_change"] = (
        weekly["elo"]
        - weekly["previous_elo"]
    )

    weekly["previous_rank"] = (
        weekly
        .groupby("team")["rank"]
        .shift(1)
    )

    weekly["rank_change"] = (
        weekly["previous_rank"]
        - weekly["rank"]
    )

    weekly = (
        weekly
        .sort_values(
            ["week", "rank"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return weekly[output_columns]


# ============================================================
# CURRENT ELO RANKINGS
# ============================================================

def build_current_rankings(
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    if weekly.empty:

        return pd.DataFrame(
            columns=[
                "team",
                "season",
                "week",
                "week_label",
                "elo",
                "previous_elo",
                "elo_change",
                "rank",
                "previous_rank",
                "rank_change",
                "conference",
                "classification",
                "logo_url",
                "color",
            ]
        )

    current_week = int(
        weekly[
            "week"
        ]
        .max()
    )

    rankings = (
        weekly[
            weekly[
                "week"
            ]
            .eq(
                current_week
            )
        ]
        .copy()
    )

    rankings[
        "current_week"
    ] = current_week

    return (
        rankings
        .sort_values(
            "rank",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TEAM PROFILE ANALYTICS (UI ONLY)
# ============================================================

def first_existing_numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first real numeric production field found, otherwise NaN."""
    for column in candidates:
        if column in frame.columns:
            return numeric(frame[column])
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def build_team_profiles(
    games: pd.DataFrame,
    weekly: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one UI-ready row per FBS team.

    This is an analytics/presentation layer only. It does not alter ELO,
    probabilities, score predictions, locks, or any frozen production model.
    Momentum is deliberately descriptive: recent ELO movement plus performance
    versus the model's expected margin. It is NOT a prediction-model feature.
    """
    fbs = teams.loc[teams["classification"].eq("fbs")].copy()
    latest = (
        weekly.sort_values(["team", "week"], kind="stable")
        .groupby("team", as_index=False)
        .tail(1)
        .set_index("team")
        if not weekly.empty else pd.DataFrame()
    )

    current_elo = {}
    current_rank = {}
    if not latest.empty:
        current_elo = numeric(latest["elo"]).to_dict()
        current_rank = numeric(latest["rank"]).to_dict()

    # Optional production ratings: use only genuine columns when present.
    home_off = first_existing_numeric(games, [
        "home_offensive_rating", "home_offense_rating", "home_adjusted_offense",
        "home_offensive_strength", "home_offensive_matchup_rating",
    ])
    away_off = first_existing_numeric(games, [
        "away_offensive_rating", "away_offense_rating", "away_adjusted_offense",
        "away_offensive_strength", "away_offensive_matchup_rating",
    ])
    home_def = first_existing_numeric(games, [
        "home_defensive_rating", "home_defense_rating", "home_adjusted_defense",
        "home_defensive_strength", "home_defensive_matchup_rating",
    ])
    away_def = first_existing_numeric(games, [
        "away_defensive_rating", "away_defense_rating", "away_adjusted_defense",
        "away_defensive_strength", "away_defensive_matchup_rating",
    ])

    working = games.copy()
    working["_home_off"] = home_off
    working["_away_off"] = away_off
    working["_home_def"] = home_def
    working["_away_def"] = away_def

    home_actual = first_existing_numeric(working, ["actual_home_points", "final_home_score", "home_points"])
    away_actual = first_existing_numeric(working, ["actual_away_points", "final_away_score", "away_points"])
    working["_home_actual"] = home_actual
    working["_away_actual"] = away_actual
    working["_completed"] = home_actual.notna() & away_actual.notna()

    expected_home_margin = first_existing_numeric(working, [
        "display_final_expected_home_margin", "final_expected_home_margin", "expected_home_margin"
    ])
    working["_margin_vs_expectation_home"] = (
        (home_actual - away_actual) - expected_home_margin
    )

    rows: list[dict[str, Any]] = []
    for _, meta in fbs.iterrows():
        team = clean_text(meta.get("school"))
        tg = working.loc[
            working["home_team"].astype(str).eq(team)
            | working["away_team"].astype(str).eq(team)
        ].copy()
        completed = tg.loc[tg["_completed"]].sort_values("start_date_utc", kind="stable")

        wins = losses = 0
        pf: list[float] = []
        pa: list[float] = []
        expectation: list[float] = []
        opponents: list[str] = []
        off_values: list[float] = []
        def_values: list[float] = []

        for _, game in completed.iterrows():
            is_home = clean_text(game.get("home_team")) == team
            team_pts = float(game["_home_actual"] if is_home else game["_away_actual"])
            opp_pts = float(game["_away_actual"] if is_home else game["_home_actual"])
            wins += int(team_pts > opp_pts)
            losses += int(team_pts < opp_pts)
            pf.append(team_pts); pa.append(opp_pts)
            exp = game.get("_margin_vs_expectation_home", np.nan)
            if pd.notna(exp):
                expectation.append(float(exp) if is_home else -float(exp))
            opponents.append(clean_text(game.get("away_team" if is_home else "home_team")))
            ov = game.get("_home_off" if is_home else "_away_off", np.nan)
            dv = game.get("_home_def" if is_home else "_away_def", np.nan)
            if pd.notna(ov): off_values.append(float(ov))
            if pd.notna(dv): def_values.append(float(dv))

        team_weekly = weekly.loc[weekly["team"].astype(str).eq(team)].sort_values("week", kind="stable")
        elo_now = float(current_elo.get(team, np.nan)) if team in current_elo else np.nan
        rank_now = float(current_rank.get(team, np.nan)) if team in current_rank else np.nan
        elo_change_recent = np.nan
        if len(team_weekly) >= 2:
            vals = numeric(team_weekly["elo"]).dropna()
            if len(vals) >= 2:
                elo_change_recent = float(vals.iloc[-1] - vals.iloc[max(0, len(vals)-4)])

        avg_vs_expectation = float(np.mean(expectation[-3:])) if expectation else np.nan
        momentum_score = 0.0
        evidence = 0
        if np.isfinite(elo_change_recent):
            momentum_score += elo_change_recent / 20.0; evidence += 1
        if np.isfinite(avg_vs_expectation):
            momentum_score += avg_vs_expectation / 4.0; evidence += 1

        if (wins + losses) < 2 or evidence == 0:
            trend = "Building sample"
        elif momentum_score >= 2.0:
            trend = "Surging"
        elif momentum_score >= 0.65:
            trend = "Improving"
        elif momentum_score <= -2.0:
            trend = "Struggling"
        elif momentum_score <= -0.65:
            trend = "Declining"
        else:
            trend = "Stable"

        opp_elos = [current_elo[o] for o in opponents if o in current_elo and pd.notna(current_elo[o])]
        sos = float(np.mean(opp_elos)) if opp_elos else np.nan
        if np.isfinite(sos) and (wins + losses) >= 2:
            all_elos = np.array([v for v in current_elo.values() if pd.notna(v)], dtype=float)
            pctile = float((all_elos <= sos).mean()) if len(all_elos) else np.nan
            if pctile >= .80: sos_label = "Very tough"
            elif pctile >= .60: sos_label = "Tough"
            elif pctile <= .20: sos_label = "Very soft"
            elif pctile <= .40: sos_label = "Soft"
            else: sos_label = "Average"
        else:
            sos_label = "Building sample"

        rows.append({
            "team": team,
            "abbreviation": clean_text(meta.get("abbreviation")),
            "mascot": clean_text(meta.get("mascot")),
            "conference": clean_text(meta.get("conference")),
            "logo_url": clean_text(meta.get("logo_url")),
            "color": clean_text(meta.get("color")),
            "record": f"{wins}-{losses}",
            "wins": wins, "losses": losses, "games_played": wins + losses,
            "current_elo": elo_now, "model_rank": rank_now,
            "recent_elo_change": elo_change_recent,
            "points_for_per_game": float(np.mean(pf)) if pf else np.nan,
            "points_against_per_game": float(np.mean(pa)) if pa else np.nan,
            "recent_margin_vs_expectation": avg_vs_expectation,
            "momentum_score": momentum_score if evidence else np.nan,
            "momentum_label": trend,
            "schedule_strength_elo": sos,
            "schedule_strength_label": sos_label,
            "offensive_rating": float(np.mean(off_values[-3:])) if off_values else np.nan,
            "defensive_rating": float(np.mean(def_values[-3:])) if def_values else np.nan,
            "rating_note": "Production rating" if off_values or def_values else "Not yet exported by production model",
        })

    profiles = pd.DataFrame(rows)
    if not profiles.empty:
        profiles = profiles.sort_values(["model_rank", "team"], kind="stable", na_position="last")
    return profiles.reset_index(drop=True)


# ============================================================
# PERFORMANCE
# ============================================================

def performance_block(
    frame: pd.DataFrame,
    mask: pd.Series,
    label: str,
) -> dict[
    str,
    Any,
]:

    subset = (
        frame.loc[
            mask
        ]
        .copy()
    )

    result: dict[
        str,
        Any,
    ] = {
        "label": label,
        "games": int(
            len(
                subset
            )
        ),
        "accuracy": None,
        "brier": None,
        "record": None,
    }

    required = [
        "actual_home_points",
        "actual_away_points",
        "display_home_win_probability",
    ]

    if any(
        column
        not in subset.columns
        for column in required
    ):

        return result

    home_actual = numeric(
        subset[
            "actual_home_points"
        ]
    )

    away_actual = numeric(
        subset[
            "actual_away_points"
        ]
    )

    valid = (
        home_actual.notna()
        &
        away_actual.notna()
        &
        home_actual.ne(
            away_actual
        )
    )

    subset = (
        subset.loc[
            valid
        ]
        .copy()
    )

    home_actual = (
        home_actual.loc[
            valid
        ]
    )

    away_actual = (
        away_actual.loc[
            valid
        ]
    )

    if subset.empty:

        result[
            "games"
        ] = 0

        return result

    actual_home_win = (
        home_actual
        .gt(
            away_actual
        )
        .astype(
            int
        )
    )

    probability = (
        numeric(
            subset[
                "display_home_win_probability"
            ]
        )
        .clip(
            0,
            1,
        )
    )

    predicted_home_win = (
        probability
        .ge(
            0.5
        )
        .astype(
            int
        )
    )

    correct = (
        predicted_home_win
        .eq(
            actual_home_win
        )
    )

    wins = int(
        correct.sum()
    )

    losses = int(
        (
            ~correct
        )
        .sum()
    )

    result[
        "games"
    ] = int(
        len(
            subset
        )
    )

    result[
        "accuracy"
    ] = float(
        correct.mean()
    )

    result[
        "brier"
    ] = float(
        np.mean(
            (
                probability
                -
                actual_home_win
            )
            ** 2
        )
    )

    result[
        "record"
    ] = (
        f"{wins}-{losses}"
    )

    return result


def build_performance(
    games: pd.DataFrame,
) -> dict[
    str,
    Any,
]:

    if (
        "game_status"
        in games.columns
    ):

        completed = (
            games[
                "game_status"
            ]
            .astype(
                "string"
            )
            .eq(
                "completed"
            )
        )

    else:

        completed = (
            numeric(
                games[
                    "actual_home_points"
                ]
            )
            .notna()
            &
            numeric(
                games[
                    "actual_away_points"
                ]
            )
            .notna()
        )

    official = (
        completed
        &
        games[
            "is_locked"
        ]
        .fillna(
            False
        )
    )

    reconstructed = (
        completed
        &
        games[
            "capture_mode"
        ]
        .astype(
            "string"
        )
        .eq(
            "historical_reconstruction"
        )
    )

    return {
        "generated_at_utc": (
            iso_utc(
                utc_now()
            )
        ),
        "official_locked": (
            performance_block(
                games,
                official,
                "Official locked",
            )
        ),
        "historical_reconstruction": (
            performance_block(
                games,
                reconstructed,
                "2026 reconstruction",
            )
        ),
        "all_completed": (
            performance_block(
                games,
                completed,
                "All completed",
            )
        ),
        "note": (
            "Historical reconstruction is diagnostic and must not be presented "
            "as genuine live/locked model performance."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 - BUILD APP DATA"
    )

    APP_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    section(
        "1. LOAD PRODUCTION OUTPUTS"
    )

    current = (
        read_csv(
            TRACKER_CURRENT
        )
    )

    history = (
        read_csv(
            TRACKER_HISTORY
        )
    )

    locks = (
        read_csv(
            LOCKED
        )
        if LOCKED.exists()
        else pd.DataFrame()
    )

    lock_status = (
        read_csv(
            LOCK_STATUS
        )
        if LOCK_STATUS.exists()
        else pd.DataFrame()
    )

    market = (
        read_csv(
            MARKET_CURRENT
        )
        if MARKET_CURRENT.exists()
        else pd.DataFrame()
    )

    teams = (
        load_team_metadata()
    )

    print(
        f"Current tracker games: "
        f"{len(current):,}"
    )

    print(
        f"Tracker history rows:  "
        f"{len(history):,}"
    )

    print(
        f"Official locks:        "
        f"{len(locks):,}"
    )

    print(
        f"Team metadata rows:    "
        f"{len(teams):,}"
    )

    print(
        f"Current market rows:   "
        f"{len(market):,}"
    )

    # --------------------------------------------------------
    # GAMES
    # --------------------------------------------------------

    section(
        "2. BUILD APP GAME DATASET"
    )

    games = create_games_dataset(
        current=(
            current
        ),
        locks=(
            locks
        ),
        lock_status=(
            lock_status
        ),
        teams=(
            teams
        ),
    )

    games = merge_market_display_data(
        games=games,
        market=market,
    )

    games.to_csv(
        OUT_GAMES,
        index=False,
    )

    print(
        f"App game rows:         "
        f"{len(games):,}"
    )

    print(
        f"Saved -> "
        f"{OUT_GAMES}"
    )

    # --------------------------------------------------------
    # WEEKLY ELO
    # --------------------------------------------------------

    section(
        "3. BUILD WEEKLY ELO"
    )

    print(
        "Source -> chronological completed-game ELO reconstruction"
    )

    print(
        "Detected production fields -> "
        "home_pregame_elo / away_pregame_elo"
    )

    weekly = (
        build_elo_weekly_from_live_games(
            games=(
                games
            ),
            teams=(
                teams
            ),
        )
    )

    weekly.to_csv(
        OUT_ELO_WEEKLY,
        index=False,
    )

    rankings = (
        build_current_rankings(
            weekly
        )
    )

    rankings.to_csv(
        OUT_ELO_RANKINGS,
        index=False,
    )

    print(
        f"Weekly ELO rows:       "
        f"{len(weekly):,}"
    )

    print(
        f"Current rankings:      "
        f"{len(rankings):,}"
    )

    if not weekly.empty:

        print(
            f"ELO weeks available:   "
            f"{int(weekly['week'].min())}"
            f" -> "
            f"{int(weekly['week'].max())}"
        )

        print(
            f"FBS teams represented: "
            f"{weekly['team'].nunique():,}"
        )

    print(
        f"Saved -> "
        f"{OUT_ELO_WEEKLY}"
    )

    print(
        f"Saved -> "
        f"{OUT_ELO_RANKINGS}"
    )

    # --------------------------------------------------------
    # TEAMS
    # --------------------------------------------------------

    section(
        "4. SAVE TEAM DIRECTORY"
    )

    fbs_teams = (
        teams[
            teams[
                "classification"
            ]
            .eq(
                "fbs"
            )
        ]
        .copy()
    )

    fbs_teams.to_csv(
        OUT_TEAMS,
        index=False,
    )

    print(
        f"FBS teams:             "
        f"{len(fbs_teams):,}"
    )

    print(
        f"Saved -> "
        f"{OUT_TEAMS}"
    )

    # --------------------------------------------------------
    # TEAM PROFILES
    # --------------------------------------------------------

    section(
        "5. BUILD TEAM PROFILES"
    )

    team_profiles = build_team_profiles(
        games=games,
        weekly=weekly,
        teams=teams,
    )

    team_profiles.to_csv(
        OUT_TEAM_PROFILES,
        index=False,
    )

    print(
        f"Team profiles:         {len(team_profiles):,}"
    )
    print(
        f"Saved -> {OUT_TEAM_PROFILES}"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    section(
        "6. BUILD PERFORMANCE SUMMARY"
    )

    performance = (
        build_performance(
            games
        )
    )

    OUT_PERFORMANCE.write_text(
        json.dumps(
            performance,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            performance,
            indent=2,
        )
    )

    print(
        f"Saved -> "
        f"{OUT_PERFORMANCE}"
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    section(
        "7. APP DATA MANIFEST"
    )

    manifest = {
        "generated_at_utc": (
            iso_utc(
                utc_now()
            )
        ),
        "app_data_version": (
            "v2_app_data_5_team_profiles_market"
        ),
        "games": int(
            len(
                games
            )
        ),
        "official_locks": int(
            games[
                "is_locked"
            ]
            .sum()
        ),
        "weekly_elo_rows": int(
            len(
                weekly
            )
        ),
        "elo_rankings_rows": int(
            len(
                rankings
            )
        ),
        "fbs_team_rows": int(
            len(
                fbs_teams
            )
        ),
        "market_rows": int(
            len(
                market
            )
        ),
        "weekly_elo_source": (
            "2026 completed-game chronology with next-pregame postgame inference"
        ),
        "weekly_elo_home_column": (
            "home_pregame_elo"
        ),
        "weekly_elo_away_column": (
            "away_pregame_elo"
        ),
        "files": {
            "games": (
                str(
                    OUT_GAMES
                )
            ),
            "elo_weekly": (
                str(
                    OUT_ELO_WEEKLY
                )
            ),
            "elo_rankings_current": (
                str(
                    OUT_ELO_RANKINGS
                )
            ),
            "teams": (
                str(
                    OUT_TEAMS
                )
            ),
            "team_profiles": (
                str(
                    OUT_TEAM_PROFILES
                )
            ),
            "performance": (
                str(
                    OUT_PERFORMANCE
                )
            ),
        },
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved -> "
        f"{OUT_MANIFEST}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    section(
        "APP DATA BUILD COMPLETE"
    )

    print(
        "✓ Production predictions converted to UI-ready data."
    )

    print(
        "✓ Official locked values override dynamic values for display."
    )

    print(
        "✓ Historical reconstructions remain explicitly identifiable."
    )

    print(
        "✓ Live 2026 pregame ELO snapshots drive weekly rankings."
    )

    print(
        "✓ Bye-week ratings carry forward."
    )

    print(
        "✓ Weekly ELO change and rank movement are calculated."
    )

    print(
        "✓ Team logos and metadata included."
    )

    print(
        "✓ Current market spread / moneyline data attached for display only."
    )


if __name__ == "__main__":

    main()
