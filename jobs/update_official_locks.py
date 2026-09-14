from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT / PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

TRACKER_HISTORY_PATH = (
    PREDICTIONS_DIR
    / "prediction_tracker_history.csv"
)

TRACKER_CURRENT_PATH = (
    PREDICTIONS_DIR
    / "prediction_tracker_current.csv"
)

LOCKED_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "official_locked_predictions.csv"
)

LOCK_STATUS_PATH = (
    PREDICTIONS_DIR
    / "official_lock_status.csv"
)

LOCK_MANIFEST_PATH = (
    PREDICTIONS_DIR
    / "official_lock_manifest.json"
)


# ============================================================
# LOCK CONFIGURATION
# ============================================================

LOCK_HOURS_BEFORE_KICKOFF = 24.0
LOCK_VERSION = "v2_official_lock_1"


# ============================================================
# DISPLAY
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# HELPERS
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


def safe_float(
    value: Any,
    fallback: float = np.nan,
) -> float:

    try:

        number = float(value)

        if np.isfinite(
            number
        ):

            return number

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(
        fallback
    )


def safe_int(
    value: Any,
    fallback: int = 0,
) -> int:

    try:

        number = float(value)

        if np.isfinite(
            number
        ):

            return int(
                number
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return int(
        fallback
    )


def clean_string(
    value: Any,
) -> str:

    if value is None:
        return ""

    try:

        if pd.isna(
            value
        ):
            return ""

    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(
        value
    ).strip()


def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def parse_timestamp_series(
    series: pd.Series,
) -> pd.Series:

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


# ============================================================
# LOAD / NORMALISE
# ============================================================

def prepare_history(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "cfbd_game_id",
        "captured_at_utc",
        "capture_mode",
        "start_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "snapshot_id",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Prediction tracker history is missing "
            "required columns:\n"
            + "\n".join(
                missing
            )
        )

    data = frame.copy()

    data[
        "cfbd_game_id"
    ] = pd.to_numeric(
        data[
            "cfbd_game_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    data[
        "_captured_at"
    ] = parse_timestamp_series(
        data[
            "captured_at_utc"
        ]
    )

    data[
        "_kickoff"
    ] = parse_timestamp_series(
        data[
            "start_date"
        ]
    )

    data[
        "capture_mode"
    ] = (
        data[
            "capture_mode"
        ]
        .astype("string")
        .str.strip()
    )

    data = data[
        data[
            "cfbd_game_id"
        ]
        .notna()
        &
        data[
            "_captured_at"
        ]
        .notna()
        &
        data[
            "_kickoff"
        ]
        .notna()
    ].copy()

    return data


def prepare_current(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "cfbd_game_id",
        "start_date",
        "home_team",
        "away_team",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Prediction tracker current view is missing "
            "required columns:\n"
            + "\n".join(
                missing
            )
        )

    data = frame.copy()

    data[
        "cfbd_game_id"
    ] = pd.to_numeric(
        data[
            "cfbd_game_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    data[
        "_kickoff"
    ] = parse_timestamp_series(
        data[
            "start_date"
        ]
    )

    return data


def load_existing_locks() -> pd.DataFrame:

    if not LOCKED_PREDICTIONS_PATH.exists():

        return pd.DataFrame()

    data = pd.read_csv(
        LOCKED_PREDICTIONS_PATH,
        low_memory=False,
    )

    if data.empty:
        return data

    if "cfbd_game_id" not in data.columns:

        raise RuntimeError(
            "Existing locked prediction file does not "
            "contain cfbd_game_id."
        )

    data[
        "cfbd_game_id"
    ] = pd.to_numeric(
        data[
            "cfbd_game_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    if (
        data[
            "cfbd_game_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Existing locked prediction file contains "
            "duplicate game IDs."
        )

    return data


# ============================================================
# LOCK SELECTION
# ============================================================

def select_lock_snapshot(
    game_history: pd.DataFrame,
    kickoff: pd.Timestamp,
    lock_cutoff: pd.Timestamp,
) -> tuple[pd.Series | None, str, float]:

    genuine = game_history[
        (
            game_history[
                "capture_mode"
            ]
            ==
            "live_dynamic"
        )
        &
        (
            game_history[
                "_captured_at"
            ]
            <
            kickoff
        )
    ].copy()

    if genuine.empty:

        return (
            None,
            "no_genuine_pregame_snapshot",
            np.nan,
        )

    on_or_before_cutoff = genuine[
        genuine[
            "_captured_at"
        ]
        <=
        lock_cutoff
    ].copy()

    if not on_or_before_cutoff.empty:

        chosen = (
            on_or_before_cutoff
            .sort_values(
                "_captured_at",
                kind="stable",
            )
            .iloc[
                -1
            ]
        )

        lag_hours = (
            (
                lock_cutoff
                -
                chosen[
                    "_captured_at"
                ]
            )
            .total_seconds()
            /
            3600.0
        )

        return (
            chosen,
            "on_time_snapshot",
            -float(
                lag_hours
            ),
        )

    after_cutoff = genuine[
        genuine[
            "_captured_at"
        ]
        >
        lock_cutoff
    ].copy()

    if not after_cutoff.empty:

        chosen = (
            after_cutoff
            .sort_values(
                "_captured_at",
                kind="stable",
            )
            .iloc[
                0
            ]
        )

        late_hours = (
            (
                chosen[
                    "_captured_at"
                ]
                -
                lock_cutoff
            )
            .total_seconds()
            /
            3600.0
        )

        return (
            chosen,
            "late_initial_capture",
            float(
                late_hours
            ),
        )

    return (
        None,
        "no_eligible_snapshot",
        np.nan,
    )


# ============================================================
# BUILD LOCK ROW
# ============================================================

def build_lock_row(
    snapshot: pd.Series,
    *,
    locked_at: pd.Timestamp,
    lock_cutoff: pd.Timestamp,
    lock_quality: str,
    lock_timing_delta_hours: float,
) -> dict[str, Any]:

    row = snapshot.to_dict()

    # Remove internal helper columns.
    row.pop(
        "_captured_at",
        None,
    )

    row.pop(
        "_kickoff",
        None,
    )

    row[
        "official_lock_version"
    ] = LOCK_VERSION

    row[
        "official_lock_hours_before_kickoff"
    ] = LOCK_HOURS_BEFORE_KICKOFF

    row[
        "official_lock_cutoff_utc"
    ] = iso_utc(
        lock_cutoff
    )

    row[
        "official_locked_at_utc"
    ] = iso_utc(
        locked_at
    )

    row[
        "official_lock_quality"
    ] = lock_quality

    row[
        "lock_snapshot_timing_delta_hours"
    ] = lock_timing_delta_hours

    row[
        "official_prediction_is_strict_24h"
    ] = (
        lock_quality
        ==
        "on_time_snapshot"
    )

    return row


# ============================================================
# LOCK STATUS
# ============================================================

def build_lock_status(
    current: pd.DataFrame,
    locked: pd.DataFrame,
    now: pd.Timestamp,
) -> pd.DataFrame:

    locked_ids = set()

    if not locked.empty:

        locked_ids = set(
            pd.to_numeric(
                locked[
                    "cfbd_game_id"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

    rows = []

    for _, game in current.iterrows():

        game_id = safe_int(
            game.get(
                "cfbd_game_id"
            ),
            0,
        )

        kickoff = game.get(
            "_kickoff"
        )

        if pd.isna(
            kickoff
        ):

            rows.append(
                {
                    "cfbd_game_id": game_id,
                    "home_team": clean_string(
                        game.get(
                            "home_team"
                        )
                    ),
                    "away_team": clean_string(
                        game.get(
                            "away_team"
                        )
                    ),
                    "start_date": clean_string(
                        game.get(
                            "start_date"
                        )
                    ),
                    "lock_status": (
                        "invalid_kickoff"
                    ),
                    "hours_to_kickoff": np.nan,
                    "hours_until_lock": np.nan,
                }
            )

            continue

        lock_cutoff = (
            kickoff
            -
            pd.Timedelta(
                hours=(
                    LOCK_HOURS_BEFORE_KICKOFF
                )
            )
        )

        hours_to_kickoff = (
            (
                kickoff
                -
                now
            )
            .total_seconds()
            /
            3600.0
        )

        hours_until_lock = (
            (
                lock_cutoff
                -
                now
            )
            .total_seconds()
            /
            3600.0
        )

        if game_id in locked_ids:

            status = (
                "locked"
            )

        elif kickoff <= now:

            status = (
                "expired_unlocked"
            )

        elif now >= lock_cutoff:

            status = (
                "due_for_lock"
            )

        else:

            status = (
                "dynamic"
            )

        rows.append(
            {
                "cfbd_game_id": game_id,
                "home_team": clean_string(
                    game.get(
                        "home_team"
                    )
                ),
                "away_team": clean_string(
                    game.get(
                        "away_team"
                    )
                ),
                "start_date": clean_string(
                    game.get(
                        "start_date"
                    )
                ),
                "lock_cutoff_utc": (
                    iso_utc(
                        lock_cutoff
                    )
                ),
                "lock_status": status,
                "hours_to_kickoff": (
                    hours_to_kickoff
                ),
                "hours_until_lock": (
                    hours_until_lock
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- 24-HOUR OFFICIAL PREDICTION LOCK"
    )

    print(
        "Rule:"
    )

    print(
        "Dynamic predictions remain live until "
        "24 hours before kickoff."
    )

    print(
        "Once an official lock is written, that game "
        "is never replaced or rewritten."
    )

    print()

    print(
        "Historical reconstructions are never eligible "
        "to become official predictions."
    )

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    section(
        "1. REQUIRED FILES"
    )

    for path in [
        TRACKER_HISTORY_PATH,
        TRACKER_CURRENT_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        print(
            f"FOUND  {path.name}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    section(
        "2. LOAD TRACKER"
    )

    history = prepare_history(
        load_csv(
            TRACKER_HISTORY_PATH
        )
    )

    current = prepare_current(
        load_csv(
            TRACKER_CURRENT_PATH
        )
    )

    existing_locks = (
        load_existing_locks()
    )

    print(
        f"History rows:       "
        f"{len(history):,}"
    )

    print(
        f"Current games:       "
        f"{len(current):,}"
    )

    print(
        f"Existing locks:      "
        f"{len(existing_locks):,}"
    )

    # --------------------------------------------------------
    # FIND GAMES DUE
    # --------------------------------------------------------

    section(
        "3. FIND GAMES DUE FOR 24-HOUR LOCK"
    )

    now = utc_now()

    print(
        f"Lock run UTC:        "
        f"{iso_utc(now)}"
    )

    existing_locked_ids = set()

    if not existing_locks.empty:

        existing_locked_ids = set(
            existing_locks[
                "cfbd_game_id"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

    new_lock_rows = []

    due_games = 0
    skipped_started = 0
    skipped_no_snapshot = 0

    for _, game in current.iterrows():

        game_id = safe_int(
            game.get(
                "cfbd_game_id"
            ),
            0,
        )

        if game_id in existing_locked_ids:
            continue

        kickoff = game.get(
            "_kickoff"
        )

        if pd.isna(
            kickoff
        ):
            continue

        if kickoff <= now:

            skipped_started += 1
            continue

        lock_cutoff = (
            kickoff
            -
            pd.Timedelta(
                hours=(
                    LOCK_HOURS_BEFORE_KICKOFF
                )
            )
        )

        if now < lock_cutoff:
            continue

        due_games += 1

        game_history = history[
            history[
                "cfbd_game_id"
            ]
            ==
            game_id
        ].copy()

        (
            snapshot,
            lock_quality,
            timing_delta,
        ) = select_lock_snapshot(
            game_history=(
                game_history
            ),
            kickoff=(
                kickoff
            ),
            lock_cutoff=(
                lock_cutoff
            ),
        )

        if snapshot is None:

            skipped_no_snapshot += 1

            print(
                f"SKIP   {game_id}  "
                f"{clean_string(game.get('away_team'))} "
                f"@ "
                f"{clean_string(game.get('home_team'))}  "
                f"reason={lock_quality}"
            )

            continue

        lock_row = build_lock_row(
            snapshot,
            locked_at=(
                now
            ),
            lock_cutoff=(
                lock_cutoff
            ),
            lock_quality=(
                lock_quality
            ),
            lock_timing_delta_hours=(
                timing_delta
            ),
        )

        new_lock_rows.append(
            lock_row
        )

        strict_label = (
            "STRICT"
            if (
                lock_quality
                ==
                "on_time_snapshot"
            )
            else
            "LATE"
        )

        print(
            f"LOCK   {game_id}  "
            f"{clean_string(game.get('away_team'))} "
            f"@ "
            f"{clean_string(game.get('home_team'))}  "
            f"{strict_label}  "
            f"{clean_string(snapshot.get('predicted_winner'))} "
            f"{safe_float(snapshot.get('prediction_probability')):.2%}"
        )

    print()

    print(
        f"Games due now:       "
        f"{due_games:,}"
    )

    print(
        f"New locks created:   "
        f"{len(new_lock_rows):,}"
    )

    print(
        f"Started/unlocked:    "
        f"{skipped_started:,}"
    )

    print(
        f"No valid snapshot:   "
        f"{skipped_no_snapshot:,}"
    )

    # --------------------------------------------------------
    # APPEND LOCKS
    # --------------------------------------------------------

    section(
        "4. APPEND OFFICIAL LOCK FILE"
    )

    new_locks = pd.DataFrame(
        new_lock_rows
    )

    if existing_locks.empty:

        combined = (
            new_locks.copy()
        )

    elif new_locks.empty:

        combined = (
            existing_locks.copy()
        )

    else:

        all_columns = list(
            dict.fromkeys(
                list(
                    existing_locks.columns
                )
                +
                list(
                    new_locks.columns
                )
            )
        )

        combined = pd.concat(
            [
                existing_locks.reindex(
                    columns=all_columns
                ),
                new_locks.reindex(
                    columns=all_columns
                ),
            ],
            ignore_index=True,
        )

    if not combined.empty:

        combined[
            "cfbd_game_id"
        ] = pd.to_numeric(
            combined[
                "cfbd_game_id"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

        if (
            combined[
                "cfbd_game_id"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                "Lock safety failure: duplicate "
                "official game lock detected."
            )

        combined = (
            combined
            .sort_values(
                [
                    "start_date",
                    "cfbd_game_id",
                ],
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    LOCKED_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        LOCKED_PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"Official locked rows: "
        f"{len(combined):,}"
    )

    print(
        f"Saved -> "
        f"{LOCKED_PREDICTIONS_PATH}"
    )

    # --------------------------------------------------------
    # LOCK STATUS
    # --------------------------------------------------------

    section(
        "5. BUILD LOCK STATUS"
    )

    status = build_lock_status(
        current=(
            current
        ),
        locked=(
            combined
        ),
        now=(
            now
        ),
    )

    status.to_csv(
        LOCK_STATUS_PATH,
        index=False,
    )

    print(
        status[
            "lock_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print(
        f"Saved -> "
        f"{LOCK_STATUS_PATH}"
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    strict_locks = 0
    late_locks = 0

    if not combined.empty:

        if (
            "official_lock_quality"
            in
            combined.columns
        ):

            strict_locks = int(
                (
                    combined[
                        "official_lock_quality"
                    ]
                    ==
                    "on_time_snapshot"
                )
                .sum()
            )

            late_locks = int(
                (
                    combined[
                        "official_lock_quality"
                    ]
                    ==
                    "late_initial_capture"
                )
                .sum()
            )

    manifest = {
        "lock_version": (
            LOCK_VERSION
        ),
        "lock_hours_before_kickoff": (
            LOCK_HOURS_BEFORE_KICKOFF
        ),
        "last_run_utc": (
            iso_utc(
                now
            )
        ),
        "total_official_locks": int(
            len(
                combined
            )
        ),
        "new_locks_this_run": int(
            len(
                new_locks
            )
        ),
        "strict_24h_locks": (
            strict_locks
        ),
        "late_initial_capture_locks": (
            late_locks
        ),
        "historical_reconstruction_allowed": (
            False
        ),
        "immutability_rule": (
            "Once a cfbd_game_id exists in "
            "official_locked_predictions.csv, "
            "future runs never replace it."
        ),
        "late_lock_rule": (
            "If tracking began after the 24-hour "
            "cutoff but before kickoff, the earliest "
            "genuine pregame live snapshot may be "
            "locked with quality=late_initial_capture. "
            "It must not be treated as a strict 24-hour "
            "prediction in evaluation."
        ),
    }

    LOCK_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved manifest -> "
        f"{LOCK_MANIFEST_PATH}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    section(
        "24-HOUR OFFICIAL LOCK RUN COMPLETE"
    )

    print(
        "✓ Historical reconstructions cannot be locked."
    )

    print(
        "✓ Games more than 24h away remain dynamic."
    )

    print(
        "✓ Due games lock from genuine pregame history."
    )

    print(
        "✓ Existing official locks are immutable."
    )

    print(
        "✓ Late first captures are explicitly flagged."
    )

    print(
        "✓ Strict and late locks can be evaluated separately."
    )


if __name__ == "__main__":
    main()
