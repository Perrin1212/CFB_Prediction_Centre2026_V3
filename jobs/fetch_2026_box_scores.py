from __future__ import annotations

from pathlib import Path
import json
import os
import sqlite3
import sys
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


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
# PATHS
# ============================================================

V1_PROJECT_ROOT = (
    PROJECT_ROOT.parent
    / "CFB_Prediction_Centre2026"
)

DEFAULT_V1_DB = (
    V1_PROJECT_ROOT
    / "data"
    / "cfb_prediction.db"
)

V1_DB_PATH = Path(
    os.getenv(
        "CFB_V1_DB_PATH",
        str(DEFAULT_V1_DB),
    )
)

RAW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2026_box_scores"
)

PROCESSED_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "2026_team_game_stats.csv"
)

GAME_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "2026_completed_game_stats_audit.csv"
)


# ============================================================
# CFBD
# ============================================================

CFBD_BASE_URL = (
    "https://api.collegefootballdata.com"
)

CFBD_TEAM_STATS_ENDPOINT = (
    f"{CFBD_BASE_URL}/games/teams"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

load_dotenv(
    V1_PROJECT_ROOT / ".env",
    override=False,
)

CFBD_API_KEY = (
    os.getenv(
        "CFBD_API_KEY"
    )
    or
    os.getenv(
        "COLLEGE_FOOTBALL_DATA_API_KEY"
    )
    or
    os.getenv(
        "CFBD_API_TOKEN"
    )
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


def normalise_classification(
    value: Any,
) -> str:

    if value is None:
        return "missing"

    text = (
        str(value)
        .strip()
        .lower()
    )

    aliases = {
        "fbs": "fbs",
        "fcs": "fcs",

        "i-a": "fbs",
        "ia": "fbs",
        "division i-a": "fbs",

        "i-aa": "fcs",
        "iaa": "fcs",
        "division i-aa": "fcs",

        "ii": "ii",
        "division ii": "ii",

        "iii": "iii",
        "division iii": "iii",
    }

    return aliases.get(
        text,
        text,
    )


def bool_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.fillna(
            False
        )

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return text.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


def safe_int(
    value: Any,
) -> int | None:

    if value is None:
        return None

    try:

        text = str(
            value
        ).strip()

        if text == "":
            return None

        return int(
            float(
                text
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:

        text = str(
            value
        ).strip()

        if text == "":
            return None

        return float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def normalise_stat_name(
    value: Any,
) -> str:

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace("%", "")
    )


def build_stat_lookup(
    stats: list[dict[str, Any]],
) -> dict[str, Any]:

    lookup: dict[str, Any] = {}

    for item in stats:

        category = (
            item.get(
                "category"
            )
            or
            item.get(
                "name"
            )
            or
            ""
        )

        stat = item.get(
            "stat"
        )

        key = normalise_stat_name(
            category
        )

        if key:

            lookup[
                key
            ] = stat

    return lookup


def get_first_stat(
    lookup: dict[str, Any],
    names: list[str],
) -> Any:

    for name in names:

        key = normalise_stat_name(
            name
        )

        if key in lookup:

            return lookup[
                key
            ]

    return None


# ============================================================
# STAT PARSERS
# ============================================================

def parse_fraction_stat(
    value: Any,
) -> tuple[
    int | None,
    int | None,
]:

    """
    Examples:

        6-15
        6/15

    Returns:
        first number,
        second number
    """

    if value is None:

        return (
            None,
            None,
        )

    text = str(
        value
    ).strip()

    for separator in [
        "-",
        "/",
    ]:

        if separator in text:

            pieces = text.split(
                separator
            )

            if len(pieces) >= 2:

                return (
                    safe_int(
                        pieces[0]
                    ),

                    safe_int(
                        pieces[1]
                    ),
                )

    return (
        None,
        None,
    )


def parse_completion_attempts(
    value: Any,
) -> tuple[
    int | None,
    int | None,
]:

    """
    CFBD commonly supplies:

        completionAttempts = "20-31"

    Returns:
        completions,
        passing attempts
    """

    return parse_fraction_stat(
        value
    )


def parse_combined_passing(
    value: Any,
) -> tuple[
    int | None,
    int | None,
    int | None,
    int | None,
]:

    """
    Defensive fallback for combined passing values.

    Possible form:

        completions-attempts-yards-touchdowns

    Example:

        20-31-275-3
    """

    if value is None:

        return (
            None,
            None,
            None,
            None,
        )

    text = str(
        value
    ).strip()

    pieces = text.split(
        "-"
    )

    values = [
        safe_int(
            piece
        )
        for piece in pieces
    ]

    if len(values) >= 4:

        return (
            values[0],
            values[1],
            values[2],
            values[3],
        )

    if len(values) == 3:

        return (
            values[0],
            values[1],
            values[2],
            None,
        )

    if len(values) == 2:

        return (
            values[0],
            values[1],
            None,
            None,
        )

    return (
        None,
        None,
        None,
        None,
    )


def parse_combined_rushing(
    value: Any,
) -> tuple[
    int | None,
    int | None,
    int | None,
]:

    if value is None:

        return (
            None,
            None,
            None,
        )

    text = str(
        value
    ).strip()

    pieces = text.split(
        "-"
    )

    values = [
        safe_int(
            piece
        )
        for piece in pieces
    ]

    if len(values) >= 3:

        return (
            values[0],
            values[1],
            values[2],
        )

    if len(values) == 2:

        return (
            values[0],
            values[1],
            None,
        )

    return (
        None,
        None,
        None,
    )


# ============================================================
# DATABASE
# ============================================================

def load_eligible_completed_games(
) -> pd.DataFrame:

    if not V1_DB_PATH.exists():

        raise FileNotFoundError(
            "\nV1 database not found:\n"
            f"{V1_DB_PATH}"
        )

    uri = (
        f"file:{V1_DB_PATH.as_posix()}"
        "?mode=ro"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
    )

    try:

        games = pd.read_sql_query(
            """
            SELECT
                id AS game_db_id,
                cfbd_id AS game_id,
                season,
                week,
                season_type,
                start_date,
                home_team,
                away_team,
                home_points,
                away_points,
                completed,
                neutral_site
            FROM games
            WHERE season = 2026
            ORDER BY
                start_date,
                id
            """,
            connection,
        )

        teams = pd.read_sql_query(
            """
            SELECT
                school,
                classification
            FROM teams
            """,
            connection,
        )

    finally:

        connection.close()

    teams[
        "classification"
    ] = (
        teams[
            "classification"
        ]
        .apply(
            normalise_classification
        )
    )

    teams = (
        teams
        .drop_duplicates(
            subset=[
                "school"
            ]
        )
    )

    home_lookup = (
        teams
        .rename(
            columns={
                "school":
                    "home_team",

                "classification":
                    "home_classification",
            }
        )
    )

    away_lookup = (
        teams
        .rename(
            columns={
                "school":
                    "away_team",

                "classification":
                    "away_classification",
            }
        )
    )

    games = (
        games
        .merge(
            home_lookup,
            on="home_team",
            how="left",
        )
        .merge(
            away_lookup,
            on="away_team",
            how="left",
        )
    )

    games[
        "home_classification"
    ] = (
        games[
            "home_classification"
        ]
        .fillna(
            "missing"
        )
    )

    games[
        "away_classification"
    ] = (
        games[
            "away_classification"
        ]
        .fillna(
            "missing"
        )
    )

    games[
        "completed_bool"
    ] = bool_series(
        games[
            "completed"
        ]
    )

    games[
        "home_points"
    ] = pd.to_numeric(
        games[
            "home_points"
        ],
        errors="coerce",
    )

    games[
        "away_points"
    ] = pd.to_numeric(
        games[
            "away_points"
        ],
        errors="coerce",
    )

    fbs_vs_fbs = (
        (
            games[
                "home_classification"
            ]
            ==
            "fbs"
        )
        &
        (
            games[
                "away_classification"
            ]
            ==
            "fbs"
        )
    )

    fbs_vs_fcs = (
        (
            (
                games[
                    "home_classification"
                ]
                ==
                "fbs"
            )
            &
            (
                games[
                    "away_classification"
                ]
                ==
                "fcs"
            )
        )
        |
        (
            (
                games[
                    "home_classification"
                ]
                ==
                "fcs"
            )
            &
            (
                games[
                    "away_classification"
                ]
                ==
                "fbs"
            )
        )
    )

    games[
        "model_eligible"
    ] = (
        fbs_vs_fbs
        |
        fbs_vs_fcs
    )

    games[
        "has_score"
    ] = (
        games[
            "home_points"
        ].notna()
        &
        games[
            "away_points"
        ].notna()
    )

    eligible = (
        games[
            games[
                "model_eligible"
            ]
            &
            games[
                "completed_bool"
            ]
            &
            games[
                "has_score"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return eligible


# ============================================================
# API
# ============================================================

def fetch_game_team_stats(
    session: requests.Session,
    game_id: int,
) -> list[dict[str, Any]]:

    # ========================================================
    # RATE-LIMIT SAFE REQUEST
    #
    # Preserve the existing API behaviour, but do not abort the
    # entire production run immediately when CFBD returns HTTP 429.
    #
    # Retry-After is honoured when supplied by CFBD. Otherwise an
    # exponential backoff is used. The same game is retried, so no
    # completed game is silently skipped.
    # ========================================================

    max_rate_limit_retries = 8
    base_backoff_seconds = 15.0
    max_backoff_seconds = 120.0
    minimum_retry_wait_seconds = 10.0

    for attempt in range(
        max_rate_limit_retries + 1
    ):

        response = session.get(
            CFBD_TEAM_STATS_ENDPOINT,

            params={
                "id": game_id,
            },

            timeout=30,
        )

        if response.status_code == 401:

            raise RuntimeError(
                "CFBD returned HTTP 401.\n"
                "Check CFBD_API_KEY."
            )

        if response.status_code == 403:

            raise RuntimeError(
                "CFBD returned HTTP 403."
            )

        if response.status_code == 429:

            if attempt >= max_rate_limit_retries:

                raise RuntimeError(
                    "CFBD returned HTTP 429 rate limit exceeded "
                    f"for game {game_id} after "
                    f"{max_rate_limit_retries} retries."
                )

            retry_after_header = (
                response.headers.get(
                    "Retry-After"
                )
            )

            retry_after_seconds = None

            if retry_after_header is not None:

                try:

                    retry_after_seconds = float(
                        retry_after_header
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    retry_after_seconds = None

            if (
                retry_after_seconds is None
                or
                retry_after_seconds <= 0
            ):

                retry_after_seconds = min(
                    base_backoff_seconds
                    *
                    (2 ** attempt),
                    max_backoff_seconds,
                )

            retry_after_seconds = max(
                retry_after_seconds,
                minimum_retry_wait_seconds,
            )

            print(
                "  ⚠ CFBD HTTP 429 rate limit exceeded. "
                f"Waiting {retry_after_seconds:.1f}s "
                f"before retry "
                f"{attempt + 1}/"
                f"{max_rate_limit_retries}..."
            )

            time.sleep(
                retry_after_seconds
            )

            continue

        response.raise_for_status()

        payload = response.json()

        if not isinstance(
            payload,
            list,
        ):

            raise RuntimeError(
                f"Unexpected CFBD response for game "
                f"{game_id}."
            )

        return payload

    raise RuntimeError(
        f"Unable to fetch CFBD team stats for game "
        f"{game_id}."
    )


# ============================================================
# PARSE TEAM
# ============================================================

def parse_team_row(
    game: pd.Series,
    team_data: dict[str, Any],
) -> dict[str, Any]:

    stats = (
        team_data.get(
            "stats"
        )
        or
        []
    )

    lookup = build_stat_lookup(
        stats
    )

    # ========================================================
    # PASSING
    #
    # IMPORTANT:
    #
    # Current CFBD team box scores provide completions and
    # attempts through completionAttempts, e.g.:
    #
    #     "18-29"
    #
    # This was the missing parser in the previous version.
    # ========================================================

    completion_attempts = get_first_stat(
        lookup,
        [
            "completionAttempts",
            "completionsAttempts",
            "completionAttempt",
            "passingCompletionAttempts",
        ],
    )

    (
        passing_completions,
        passing_attempts,
    ) = parse_completion_attempts(
        completion_attempts
    )

    # --------------------------------------------------------
    # Explicit fallback categories
    # --------------------------------------------------------

    explicit_completions = get_first_stat(
        lookup,
        [
            "passingCompletions",
            "completions",
        ],
    )

    if explicit_completions is not None:

        passing_completions = safe_int(
            explicit_completions
        )

    explicit_attempts = get_first_stat(
        lookup,
        [
            "passingAttempts",
            "passAttempts",
            "attempts",
        ],
    )

    if explicit_attempts is not None:

        passing_attempts = safe_int(
            explicit_attempts
        )

    passing_yards = safe_int(
        get_first_stat(
            lookup,
            [
                "netPassingYards",
                "passingYards",
                "netPassYards",
            ],
        )
    )

    passing_tds = safe_int(
        get_first_stat(
            lookup,
            [
                "passingTDs",
                "passingTouchdowns",
            ],
        )
    )

    # --------------------------------------------------------
    # Combined fallback
    # --------------------------------------------------------

    combined_passing = get_first_stat(
        lookup,
        [
            "passing",
            "passingCAYTD",
            "passingC-A-Y-TD",
        ],
    )

    if combined_passing is not None:

        (
            combined_completions,
            combined_attempts,
            combined_yards,
            combined_tds,
        ) = parse_combined_passing(
            combined_passing
        )

        if passing_completions is None:
            passing_completions = (
                combined_completions
            )

        if passing_attempts is None:
            passing_attempts = (
                combined_attempts
            )

        if passing_yards is None:
            passing_yards = (
                combined_yards
            )

        if passing_tds is None:
            passing_tds = (
                combined_tds
            )

    # ========================================================
    # RUSHING
    # ========================================================

    rushing_attempts = safe_int(
        get_first_stat(
            lookup,
            [
                "rushingAttempts",
                "rushAttempts",
            ],
        )
    )

    rushing_yards = safe_int(
        get_first_stat(
            lookup,
            [
                "rushingYards",
                "rushYards",
            ],
        )
    )

    rushing_tds = safe_int(
        get_first_stat(
            lookup,
            [
                "rushingTDs",
                "rushingTouchdowns",
            ],
        )
    )

    combined_rushing = get_first_stat(
        lookup,
        [
            "rushing",
            "rushingAttemptsYardsTD",
        ],
    )

    if combined_rushing is not None:

        (
            combined_attempts,
            combined_yards,
            combined_tds,
        ) = parse_combined_rushing(
            combined_rushing
        )

        if rushing_attempts is None:
            rushing_attempts = (
                combined_attempts
            )

        if rushing_yards is None:
            rushing_yards = (
                combined_yards
            )

        if rushing_tds is None:
            rushing_tds = (
                combined_tds
            )

    # ========================================================
    # DOWNS
    # ========================================================

    third_down_value = get_first_stat(
        lookup,
        [
            "thirdDownEff",
            "thirdDownEfficiency",
            "thirdDownConversions",
        ],
    )

    (
        third_down_made,
        third_down_attempts,
    ) = parse_fraction_stat(
        third_down_value
    )

    fourth_down_value = get_first_stat(
        lookup,
        [
            "fourthDownEff",
            "fourthDownEfficiency",
            "fourthDownConversions",
        ],
    )

    (
        fourth_down_made,
        fourth_down_attempts,
    ) = parse_fraction_stat(
        fourth_down_value
    )

    # ========================================================
    # CORE STATS
    # ========================================================

    total_yards = safe_int(
        get_first_stat(
            lookup,
            [
                "totalYards",
                "totalOffense",
            ],
        )
    )

    first_downs = safe_int(
        get_first_stat(
            lookup,
            [
                "firstDowns",
            ],
        )
    )

    turnovers = safe_int(
        get_first_stat(
            lookup,
            [
                "turnovers",
            ],
        )
    )

    possession_time = get_first_stat(
        lookup,
        [
            "possessionTime",
            "timeOfPossession",
        ],
    )

    interceptions = safe_int(
        get_first_stat(
            lookup,
            [
                "interceptions",
                "interceptionsThrown",
            ],
        )
    )

    fumbles_lost = safe_int(
        get_first_stat(
            lookup,
            [
                "fumblesLost",
            ],
        )
    )

    # ========================================================
    # TURNOVER FALLBACK
    # ========================================================

    if turnovers is None:

        turnover_parts = [
            value
            for value in [
                interceptions,
                fumbles_lost,
            ]
            if value is not None
        ]

        if turnover_parts:

            turnovers = sum(
                turnover_parts
            )

    # ========================================================
    # PLAY COUNT
    # ========================================================

    total_plays = None

    if (
        rushing_attempts is not None
        and
        passing_attempts is not None
    ):

        total_plays = (
            rushing_attempts
            +
            passing_attempts
        )

    # ========================================================
    # YARDS PER PLAY
    # ========================================================

    yards_per_play_proxy = None

    if (
        total_yards is not None
        and
        total_plays is not None
        and
        total_plays > 0
    ):

        yards_per_play_proxy = (
            total_yards
            /
            total_plays
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    return {
        "game_db_id": safe_int(
            game[
                "game_db_id"
            ]
        ),

        "game_id": safe_int(
            game[
                "game_id"
            ]
        ),

        "season": 2026,

        "week": safe_int(
            game[
                "week"
            ]
        ),

        "season_type": game[
            "season_type"
        ],

        "start_date": game[
            "start_date"
        ],

        "home_team": game[
            "home_team"
        ],

        "away_team": game[
            "away_team"
        ],

        "neutral_site": game[
            "neutral_site"
        ],

        # TEAM
        "team_id": safe_int(
            team_data.get(
                "teamId"
            )
        ),

        "team_name": team_data.get(
            "team"
        ),

        "conference": team_data.get(
            "conference"
        ),

        "home_away": team_data.get(
            "homeAway"
        ),

        "points": safe_int(
            team_data.get(
                "points"
            )
        ),

        # RUSHING
        "rushing_attempts":
            rushing_attempts,

        "rushing_yards":
            rushing_yards,

        "rushing_tds":
            rushing_tds,

        # PASSING
        "passing_completions":
            passing_completions,

        "passing_attempts":
            passing_attempts,

        "net_passing_yards":
            passing_yards,

        "passing_tds":
            passing_tds,

        # GENERAL
        "total_yards":
            total_yards,

        "first_downs":
            first_downs,

        "third_down_made":
            third_down_made,

        "third_down_attempts":
            third_down_attempts,

        "fourth_down_made":
            fourth_down_made,

        "fourth_down_attempts":
            fourth_down_attempts,

        "turnovers":
            turnovers,

        "interceptions":
            interceptions,

        "fumbles_lost":
            fumbles_lost,

        "possession_time":
            possession_time,

        # DERIVED
        "total_plays_proxy":
            total_plays,

        "yards_per_play_proxy":
            yards_per_play_proxy,

        # DIAGNOSTICS
        "completion_attempts_raw":
            completion_attempts,

        "raw_stat_categories":
            json.dumps(
                sorted(
                    lookup.keys()
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— FETCH 2026 BOX SCORES"
    )

    # ========================================================
    # KEY
    # ========================================================

    if not CFBD_API_KEY:

        raise RuntimeError(
            "\nCFBD API key was not found.\n\n"
            "Checked:\n"
            f"  {PROJECT_ROOT / '.env'}\n"
            f"  {V1_PROJECT_ROOT / '.env'}"
        )

    print(
        "CFBD API key: found ✓"
    )

    print(
        "API key will not be printed or saved."
    )

    # ========================================================
    # GAMES
    # ========================================================

    section(
        "LOADING COMPLETED 2026 MODEL GAMES"
    )

    games = (
        load_eligible_completed_games()
    )

    print(
        f"Completed eligible games: "
        f"{len(games):,}"
    )

    if games.empty:

        print(
            "No completed eligible games."
        )

        return

    print()

    print(
        games[
            [
                "week",
                "away_team",
                "away_points",
                "home_team",
                "home_points",
                "game_id",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SESSION
    # ========================================================

    session = requests.Session()

    session.headers.update(
        {
            "Authorization":
                f"Bearer {CFBD_API_KEY}",

            "Accept":
                "application/json",

            "User-Agent":
                "CFB-Prediction-Centre-V2/2026",
        }
    )

    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    audit_rows: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # FETCH
    # ========================================================

    section(
        "FETCHING CFBD TEAM BOX SCORES"
    )

    for index, game in games.iterrows():

        game_id = safe_int(
            game[
                "game_id"
            ]
        )

        if game_id is None:

            continue

        print(
            f"[{index + 1}/{len(games)}] "
            f"Game {game_id}: "
            f"{game['away_team']} @ "
            f"{game['home_team']}"
        )

        payload = (
            fetch_game_team_stats(
                session=session,
                game_id=game_id,
            )
        )

        raw_file = (
            RAW_OUTPUT_DIR
            /
            f"{game_id}.json"
        )

        with open(
            raw_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )

        matching_game = None

        for item in payload:

            if safe_int(
                item.get(
                    "id"
                )
            ) == game_id:

                matching_game = item

                break

        if (
            matching_game is None
            and
            len(payload) == 1
        ):

            matching_game = payload[
                0
            ]

        if matching_game is None:

            print(
                "  ✗ Game missing from API payload"
            )

            audit_rows.append(
                {
                    "game_id":
                        game_id,

                    "status":
                        "game_not_found",

                    "team_rows":
                        0,
                }
            )

            continue

        teams = (
            matching_game.get(
                "teams"
            )
            or
            []
        )

        for team_data in teams:

            rows.append(
                parse_team_row(
                    game=game,
                    team_data=team_data,
                )
            )

        audit_rows.append(
            {
                "game_id":
                    game_id,

                "week":
                    game[
                        "week"
                    ],

                "away_team":
                    game[
                        "away_team"
                    ],

                "home_team":
                    game[
                        "home_team"
                    ],

                "status":
                    (
                        "ok"
                        if len(teams) == 2
                        else
                        "unexpected_team_count"
                    ),

                "team_rows":
                    len(
                        teams
                    ),
            }
        )

        print(
            f"  ✓ {len(teams)} team rows"
        )

        # Pace requests between games to reduce the chance of
        # hitting CFBD burst/rate limits during large refreshes.
        time.sleep(
            1.00
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    stats = pd.DataFrame(
        rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    section(
        "VALIDATING FETCHED BOX SCORES"
    )

    print(
        f"Team-stat rows fetched: "
        f"{len(stats):,}"
    )

    print(
        f"Games represented:      "
        f"{stats['game_id'].nunique():,}"
    )

    rows_per_game = (
        stats
        .groupby(
            "game_id"
        )
        .size()
    )

    exactly_two = int(
        (
            rows_per_game
            ==
            2
        ).sum()
    )

    print()

    print(
        f"Games with exactly 2 team rows: "
        f"{exactly_two:,}"
    )

    print(
        f"Games without exactly 2 rows:   "
        f"{int((rows_per_game != 2).sum()):,}"
    )

    # ========================================================
    # CORE COVERAGE
    # ========================================================

    section(
        "CORE STAT COVERAGE"
    )

    core_columns = [
        "points",

        "rushing_attempts",
        "rushing_yards",

        "passing_completions",
        "passing_attempts",
        "net_passing_yards",

        "total_yards",
        "first_downs",

        "turnovers",
        "possession_time",

        "total_plays_proxy",
        "yards_per_play_proxy",
    ]

    for column in core_columns:

        coverage = (
            stats[
                column
            ]
            .notna()
            .mean()
        )

        print(
            f"{column:<28} "
            f"{coverage:>7.2%}"
        )

    # ========================================================
    # PASSING DIAGNOSTIC
    # ========================================================

    section(
        "PASSING ATTEMPT DIAGNOSTIC"
    )

    print(
        stats[
            [
                "team_name",
                "completion_attempts_raw",
                "passing_completions",
                "passing_attempts",
                "net_passing_yards",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SCORE CHECK
    # ========================================================

    section(
        "SCORE VALIDATION"
    )

    score_mismatches = 0

    for (
        game_id,
        game_stats,
    ) in stats.groupby(
        "game_id"
    ):

        source = games[
            games[
                "game_id"
            ]
            ==
            game_id
        ]

        if source.empty:
            continue

        source = source.iloc[
            0
        ]

        home = game_stats[
            game_stats[
                "home_away"
            ]
            .astype(str)
            .str.lower()
            ==
            "home"
        ]

        away = game_stats[
            game_stats[
                "home_away"
            ]
            .astype(str)
            .str.lower()
            ==
            "away"
        ]

        if (
            len(home) != 1
            or
            len(away) != 1
        ):

            score_mismatches += 1
            continue

        api_home = safe_int(
            home.iloc[
                0
            ][
                "points"
            ]
        )

        api_away = safe_int(
            away.iloc[
                0
            ][
                "points"
            ]
        )

        db_home = safe_int(
            source[
                "home_points"
            ]
        )

        db_away = safe_int(
            source[
                "away_points"
            ]
        )

        if (
            api_home != db_home
            or
            api_away != db_away
        ):

            score_mismatches += 1

            print(
                f"Mismatch {game_id}: "
                f"DB {db_away}-{db_home} | "
                f"API {api_away}-{api_home}"
            )

    print(
        f"Score mismatches: "
        f"{score_mismatches:,}"
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    section(
        "FETCHED TEAM STAT SAMPLE"
    )

    sample_columns = [
        "week",
        "team_name",
        "home_away",

        "points",

        "rushing_attempts",
        "rushing_yards",

        "passing_completions",
        "passing_attempts",
        "net_passing_yards",

        "total_yards",
        "total_plays_proxy",
        "yards_per_play_proxy",

        "first_downs",
        "turnovers",
        "possession_time",
    ]

    print(
        stats[
            sample_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    section(
        "SAVING 2026 BOX SCORES"
    )

    PROCESSED_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats.to_csv(
        PROCESSED_OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        GAME_OUTPUT_FILE,
        index=False,
    )

    print(
        f"✓ Saved {len(stats):,} team-stat rows → "
        f"{PROCESSED_OUTPUT_FILE}"
    )

    print(
        f"✓ Saved {len(audit):,} game audit rows → "
        f"{GAME_OUTPUT_FILE}"
    )

    # ========================================================
    # FINAL READINESS
    # ========================================================

    section(
        "2026 BOX-SCORE READINESS"
    )

    expected_games = len(
        games
    )

    fetched_games = int(
        stats[
            "game_id"
        ].nunique()
    )

    print(
        f"Completed eligible games expected: "
        f"{expected_games:,}"
    )

    print(
        f"Games returned by CFBD:             "
        f"{fetched_games:,}"
    )

    print(
        f"Games with two team-stat rows:      "
        f"{exactly_two:,}"
    )

    print(
        f"Score mismatches:                   "
        f"{score_mismatches:,}"
    )

    required_for_state_update = [
        "points",
        "rushing_attempts",
        "rushing_yards",
        "passing_attempts",
        "net_passing_yards",
        "total_yards",
        "first_downs",
        "turnovers",
        "possession_time",
        "yards_per_play_proxy",
    ]

    missing_columns = []

    for column in required_for_state_update:

        coverage = (
            stats[
                column
            ]
            .notna()
            .mean()
        )

        if coverage < 1.0:

            missing_columns.append(
                (
                    column,
                    coverage,
                )
            )

    print()

    if (
        fetched_games
        ==
        expected_games
        and
        exactly_two
        ==
        expected_games
        and
        score_mismatches
        ==
        0
        and
        not missing_columns
    ):

        print(
            "✓ 2026 completed-game box scores are READY."
        )

        print()

        print(
            "All state-update fields have 100% coverage."
        )

        print()

        print(
            "Next:"
        )

        print(
            "  Build the frozen chronological "
            "2026 forward runner."
        )

    else:

        print(
            "⚠ 2026 box scores still require review."
        )

        if missing_columns:

            print()

            print(
                "Incomplete state-update fields:"
            )

            for (
                column,
                coverage,
            ) in missing_columns:

                print(
                    f"  {column:<28} "
                    f"{coverage:>7.2%}"
                )


if __name__ == "__main__":
    main()