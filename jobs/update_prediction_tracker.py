from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT / PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

WINNER_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "2026_forward_predictions.csv"
)

SCORING_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "2026_scoring_predictions.csv"
)

WINNER_MODEL_MANIFEST_PATH = (
    PROCESSED_DIR
    / "v2_probability_model_manifest.json"
)

SCORE_MODEL_MANIFEST_PATH = (
    PROCESSED_DIR
    / "v2_score_model_manifest.json"
)

UNCERTAINTY_MANIFEST_PATH = (
    PROCESSED_DIR
    / "v2_corrected_score_uncertainty.json"
)

TRACKER_HISTORY_PATH = (
    PREDICTIONS_DIR
    / "prediction_tracker_history.csv"
)

TRACKER_CURRENT_PATH = (
    PREDICTIONS_DIR
    / "prediction_tracker_current.csv"
)

TRACKER_MANIFEST_PATH = (
    PREDICTIONS_DIR
    / "prediction_tracker_manifest.json"
)


# ============================================================
# TRACKER VERSION
# ============================================================

TRACKER_VERSION = "v2_prediction_tracker_1"


# ============================================================
# DISPLAY
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(
    value: Any,
    fallback: float = np.nan,
) -> float:

    try:

        number = float(value)

        if np.isfinite(number):
            return number

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(fallback)


def safe_int(
    value: Any,
    fallback: int = 0,
) -> int:

    try:

        number = float(value)

        if np.isfinite(number):
            return int(number)

    except (
        TypeError,
        ValueError,
    ):
        pass

    return int(fallback)


def clean_string(
    value: Any,
) -> str:

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(value).strip()


def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    return frame


def load_json_if_exists(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():
        return {}

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):

        return {}


def first_value(
    row: pd.Series,
    candidates: list[str],
    fallback: Any = np.nan,
) -> Any:

    for column in candidates:

        if column not in row.index:
            continue

        value = row.get(
            column
        )

        if value is None:
            continue

        try:

            if pd.isna(value):
                continue

        except (
            TypeError,
            ValueError,
        ):
            pass

        return value

    return fallback


def find_game_id_column(
    frame: pd.DataFrame,
) -> str:

    candidates = [
        "cfbd_game_id",
        "_cfbd_game_id",
        "cfbd_id",
        "game_id",
    ]

    for column in candidates:

        if column not in frame.columns:
            continue

        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if numeric.notna().any():
            return column

    raise RuntimeError(
        "Could not identify a game ID column."
    )


def iso_utc_now() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


def parse_utc(
    value: Any,
) -> pd.Timestamp | None:

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return None

    return parsed


def serialise_for_hash(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        if not np.isfinite(
            value
        ):
            return ""

        return f"{float(value):.10f}"

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):

        return str(
            int(value)
        )

    text = clean_string(
        value
    )

    return text


# ============================================================
# MODEL METADATA
# ============================================================

def model_metadata() -> dict[str, Any]:

    winner = load_json_if_exists(
        WINNER_MODEL_MANIFEST_PATH
    )

    score = load_json_if_exists(
        SCORE_MODEL_MANIFEST_PATH
    )

    uncertainty = load_json_if_exists(
        UNCERTAINTY_MANIFEST_PATH
    )

    winner_name = (
        winner.get(
            "model_name"
        )
        or
        winner.get(
            "architecture"
        )
        or
        "elo_matchup"
    )

    winner_version = (
        winner.get(
            "model_version"
        )
        or
        winner.get(
            "version"
        )
        or
        ""
    )

    score_name = (
        score.get(
            "model_name"
        )
        or
        score.get(
            "architecture"
        )
        or
        "compact_full"
    )

    score_version = (
        score.get(
            "model_version"
        )
        or
        score.get(
            "version"
        )
        or
        ""
    )

    uncertainty_type = (
        uncertainty.get(
            "artifact_type"
        )
        or
        "corrected_score_uncertainty"
    )

    uncertainty_oos_rows = (
        uncertainty.get(
            "oos_rows"
        )
    )

    return {
        "winner_model_name": (
            winner_name
        ),
        "winner_model_version": (
            winner_version
        ),
        "score_model_name": (
            score_name
        ),
        "score_model_version": (
            score_version
        ),
        "uncertainty_model_name": (
            uncertainty_type
        ),
        "uncertainty_oos_rows": (
            uncertainty_oos_rows
        ),
    }


# ============================================================
# SOURCE PREPARATION
# ============================================================

def prepare_winner_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    data = frame.copy()

    game_id_column = (
        find_game_id_column(
            data
        )
    )

    data[
        "tracker_game_id"
    ] = pd.to_numeric(
        data[
            game_id_column
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    if (
        data[
            "tracker_game_id"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Winner prediction file contains "
            "missing game IDs."
        )

    if (
        data[
            "tracker_game_id"
        ]
        .duplicated()
        .any()
    ):

        duplicates = (
            data.loc[
                data[
                    "tracker_game_id"
                ]
                .duplicated(
                    keep=False
                ),
                "tracker_game_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Winner prediction file contains "
            "duplicate game IDs:\n"
            + ", ".join(
                duplicates[:20]
            )
        )

    return data


def prepare_scoring_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    data = frame.copy()

    game_id_column = (
        find_game_id_column(
            data
        )
    )

    data[
        "tracker_game_id"
    ] = pd.to_numeric(
        data[
            game_id_column
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    data = (
        data[
            data[
                "tracker_game_id"
            ]
            .notna()
        ]
        .copy()
    )

    if (
        data[
            "tracker_game_id"
        ]
        .duplicated()
        .any()
    ):

        duplicates = (
            data.loc[
                data[
                    "tracker_game_id"
                ]
                .duplicated(
                    keep=False
                ),
                "tracker_game_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Scoring prediction file contains "
            "duplicate game IDs:\n"
            + ", ".join(
                duplicates[:20]
            )
        )

    return data


# ============================================================
# SNAPSHOT BUILD
# ============================================================

def build_snapshot_rows(
    winner_predictions: pd.DataFrame,
    scoring_predictions: pd.DataFrame,
    captured_at_utc: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:

    score_lookup = (
        scoring_predictions
        .set_index(
            "tracker_game_id",
            drop=False,
        )
    )

    captured_timestamp = pd.to_datetime(
        captured_at_utc,
        utc=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    for _, winner_row in (
        winner_predictions
        .sort_values(
            "tracker_game_id"
        )
        .iterrows()
    ):

        game_id = safe_int(
            winner_row[
                "tracker_game_id"
            ],
            0,
        )

        if (
            game_id
            in
            score_lookup.index
        ):

            score_row = (
                score_lookup.loc[
                    game_id
                ]
            )

            if isinstance(
                score_row,
                pd.DataFrame,
            ):

                score_row = (
                    score_row.iloc[
                        0
                    ]
                )

        else:

            score_row = pd.Series(
                dtype=object
            )

        start_date_raw = (
            first_value(
                winner_row,
                [
                    "start_date",
                ],
                "",
            )
        )

        kickoff = parse_utc(
            start_date_raw
        )

        if kickoff is None:

            hours_to_kickoff = (
                np.nan
            )

            capture_mode = (
                "unknown_time"
            )

        else:

            hours_to_kickoff = (
                (
                    kickoff
                    -
                    captured_timestamp
                )
                .total_seconds()
                /
                3600.0
            )

            if (
                kickoff
                >
                captured_timestamp
            ):

                capture_mode = (
                    "live_dynamic"
                )

            else:

                capture_mode = (
                    "historical_reconstruction"
                )

        actual_home_points = (
            safe_float(
                first_value(
                    score_row,
                    [
                        "actual_home_points",
                    ],
                    np.nan,
                )
            )
        )

        actual_away_points = (
            safe_float(
                first_value(
                    score_row,
                    [
                        "actual_away_points",
                    ],
                    np.nan,
                )
            )
        )

        has_result = (
            np.isfinite(
                actual_home_points
            )
            and
            np.isfinite(
                actual_away_points
            )
        )

        if has_result:

            game_status = (
                "completed"
            )

        elif (
            kickoff is not None
            and
            kickoff
            <=
            captured_timestamp
        ):

            game_status = (
                "started_or_result_pending"
            )

        else:

            game_status = (
                "upcoming"
            )

        home_team = clean_string(
            first_value(
                winner_row,
                [
                    "home_team",
                ],
                "",
            )
        )

        away_team = clean_string(
            first_value(
                winner_row,
                [
                    "away_team",
                ],
                "",
            )
        )

        home_probability = (
            safe_float(
                first_value(
                    winner_row,
                    [
                        "home_win_probability",
                        "home_probability",
                    ],
                    np.nan,
                )
            )
        )

        away_probability = (
            safe_float(
                first_value(
                    winner_row,
                    [
                        "away_win_probability",
                        "away_probability",
                    ],
                    np.nan,
                )
            )
        )

        predicted_winner = (
            clean_string(
                first_value(
                    winner_row,
                    [
                        "predicted_winner",
                    ],
                    "",
                )
            )
        )

        row = {
            "tracker_version": (
                TRACKER_VERSION
            ),

            "captured_at_utc": (
                captured_at_utc
            ),

            "capture_mode": (
                capture_mode
            ),

            "game_status": (
                game_status
            ),

            "season": (
                safe_int(
                    first_value(
                        winner_row,
                        [
                            "season",
                        ],
                        2026,
                    ),
                    2026,
                )
            ),

            "week": (
                safe_int(
                    first_value(
                        winner_row,
                        [
                            "week",
                        ],
                        0,
                    ),
                    0,
                )
            ),

            "season_type": (
                clean_string(
                    first_value(
                        winner_row,
                        [
                            "season_type",
                        ],
                        "",
                    )
                )
            ),

            "cfbd_game_id": (
                game_id
            ),

            "start_date": (
                clean_string(
                    start_date_raw
                )
            ),

            "hours_to_kickoff_at_capture": (
                hours_to_kickoff
            ),

            "home_team": (
                home_team
            ),

            "away_team": (
                away_team
            ),

            "neutral_site": (
                first_value(
                    winner_row,
                    [
                        "neutral_site",
                    ],
                    False,
                )
            ),

            # --------------------------------------------
            # OFFICIAL FROZEN WINNER MODEL
            # --------------------------------------------

            "home_win_probability": (
                home_probability
            ),

            "away_win_probability": (
                away_probability
            ),

            "predicted_winner": (
                predicted_winner
            ),

            "prediction_probability": (
                max(
                    home_probability,
                    away_probability,
                )
                if (
                    np.isfinite(
                        home_probability
                    )
                    and
                    np.isfinite(
                        away_probability
                    )
                )
                else
                np.nan
            ),

            # --------------------------------------------
            # CORE WINNER FEATURES / EXPLANATION INPUTS
            # --------------------------------------------

            "home_pregame_elo": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "home_pregame_elo",
                            "home_elo_pre",
                        ],
                        np.nan,
                    )
                )
            ),

            "away_pregame_elo": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "away_pregame_elo",
                            "away_elo_pre",
                        ],
                        np.nan,
                    )
                )
            ),

            "elo_difference": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "elo_difference",
                        ],
                        np.nan,
                    )
                )
            ),

            "home_offensive_matchup_index": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "home_offensive_matchup_index",
                        ],
                        np.nan,
                    )
                )
            ),

            "away_offensive_matchup_index": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "away_offensive_matchup_index",
                        ],
                        np.nan,
                    )
                )
            ),

            "offensive_matchup_difference": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "offensive_matchup_difference",
                        ],
                        np.nan,
                    )
                )
            ),

            "adjusted_strength_difference": (
                safe_float(
                    first_value(
                        winner_row,
                        [
                            "adjusted_strength_difference",
                        ],
                        np.nan,
                    )
                )
            ),

            # --------------------------------------------
            # FROZEN SCORE MODEL
            # --------------------------------------------

            "projected_home_score": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "projected_home_score",
                        ],
                        np.nan,
                    )
                )
            ),

            "projected_away_score": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "projected_away_score",
                        ],
                        np.nan,
                    )
                )
            ),

            "final_expected_home_points": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "final_expected_home_points",
                            "expected_home_points",
                        ],
                        np.nan,
                    )
                )
            ),

            "final_expected_away_points": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "final_expected_away_points",
                            "expected_away_points",
                        ],
                        np.nan,
                    )
                )
            ),

            "final_expected_home_margin": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "final_expected_home_margin",
                            "expected_home_margin",
                        ],
                        np.nan,
                    )
                )
            ),

            "final_expected_total_points": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "final_expected_total_points",
                            "expected_total_points",
                        ],
                        np.nan,
                    )
                )
            ),

            "score_model_winner": (
                clean_string(
                    first_value(
                        score_row,
                        [
                            "score_model_winner",
                        ],
                        "",
                    )
                )
            ),

            # --------------------------------------------
            # MONTE CARLO DIAGNOSTIC
            # --------------------------------------------

            "simulation_count": (
                safe_int(
                    first_value(
                        score_row,
                        [
                            "simulation_count",
                        ],
                        0,
                    ),
                    0,
                )
            ),

            "simulation_home_win_probability": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_home_win_probability",
                        ],
                        np.nan,
                    )
                )
            ),

            "simulation_away_win_probability": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_away_win_probability",
                        ],
                        np.nan,
                    )
                )
            ),

            "simulation_home_score_p10": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_home_score_p10",
                        ],
                        np.nan,
                    )
                )
            ),

            "simulation_home_score_p90": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_home_score_p90",
                        ],
                        np.nan,
                    )
                )
            ),

            "simulation_away_score_p10": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_away_score_p10",
                        ],
                        np.nan,
                    )
                )
            ),

            "simulation_away_score_p90": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "simulation_away_score_p90",
                        ],
                        np.nan,
                    )
                )
            ),

            "home_score_residual_sd_pre": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "home_score_residual_sd_pre",
                        ],
                        np.nan,
                    )
                )
            ),

            "away_score_residual_sd_pre": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "away_score_residual_sd_pre",
                        ],
                        np.nan,
                    )
                )
            ),

            "score_residual_correlation_pre": (
                safe_float(
                    first_value(
                        score_row,
                        [
                            "score_residual_correlation_pre",
                        ],
                        np.nan,
                    )
                )
            ),

            # --------------------------------------------
            # RESULT - PRESENT ONLY WHEN KNOWN
            # --------------------------------------------

            "actual_home_points": (
                actual_home_points
            ),

            "actual_away_points": (
                actual_away_points
            ),

            # --------------------------------------------
            # MODEL METADATA
            # --------------------------------------------

            "winner_model_name": (
                metadata[
                    "winner_model_name"
                ]
            ),

            "winner_model_version": (
                metadata[
                    "winner_model_version"
                ]
            ),

            "score_model_name": (
                metadata[
                    "score_model_name"
                ]
            ),

            "score_model_version": (
                metadata[
                    "score_model_version"
                ]
            ),

            "uncertainty_model_name": (
                metadata[
                    "uncertainty_model_name"
                ]
            ),

            "uncertainty_oos_rows": (
                metadata[
                    "uncertainty_oos_rows"
                ]
            ),
        }

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# FINGERPRINT
# ============================================================

FINGERPRINT_COLUMNS = [
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "projected_home_score",
    "projected_away_score",
    "final_expected_home_points",
    "final_expected_away_points",
    "final_expected_home_margin",
    "final_expected_total_points",
    "simulation_home_win_probability",
    "simulation_away_win_probability",
    "simulation_home_score_p10",
    "simulation_home_score_p90",
    "simulation_away_score_p10",
    "simulation_away_score_p90",
    "home_score_residual_sd_pre",
    "away_score_residual_sd_pre",
    "score_residual_correlation_pre",
    "game_status",
    "actual_home_points",
    "actual_away_points",
]


def add_fingerprints(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    result = frame.copy()

    fingerprints = []

    snapshot_ids = []

    for _, row in result.iterrows():

        payload_parts = [
            str(
                safe_int(
                    row.get(
                        "cfbd_game_id"
                    ),
                    0,
                )
            )
        ]

        for column in (
            FINGERPRINT_COLUMNS
        ):

            payload_parts.append(
                serialise_for_hash(
                    row.get(
                        column
                    )
                )
            )

        payload = "|".join(
            payload_parts
        )

        fingerprint = (
            hashlib.sha256(
                payload.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        snapshot_payload = (
            f"{payload}|"
            f"{row.get('captured_at_utc', '')}"
        )

        snapshot_id = (
            hashlib.sha256(
                snapshot_payload.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        fingerprints.append(
            fingerprint
        )

        snapshot_ids.append(
            snapshot_id
        )

    result[
        "prediction_fingerprint"
    ] = fingerprints

    result[
        "snapshot_id"
    ] = snapshot_ids

    return result


# ============================================================
# HISTORY
# ============================================================

def load_existing_history() -> pd.DataFrame:

    if not TRACKER_HISTORY_PATH.exists():

        return pd.DataFrame()

    history = pd.read_csv(
        TRACKER_HISTORY_PATH,
        low_memory=False,
    )

    return history


def identify_new_snapshots(
    snapshots: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:

    if history.empty:

        result = snapshots.copy()

        result[
            "snapshot_reason"
        ] = "first_capture"

        return result

    if (
        "cfbd_game_id"
        not in history.columns
        or
        "prediction_fingerprint"
        not in history.columns
    ):

        raise RuntimeError(
            "Existing prediction tracker history "
            "does not have the expected schema."
        )

    previous = history.copy()

    previous[
        "cfbd_game_id"
    ] = pd.to_numeric(
        previous[
            "cfbd_game_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    previous = (
        previous
        .dropna(
            subset=[
                "cfbd_game_id",
            ]
        )
        .sort_values(
            [
                "cfbd_game_id",
                "captured_at_utc",
            ],
            kind="stable",
        )
    )

    latest = (
        previous
        .groupby(
            "cfbd_game_id",
            sort=False,
        )
        .tail(
            1
        )
        .set_index(
            "cfbd_game_id"
        )
    )

    rows = []

    for _, row in snapshots.iterrows():

        game_id = safe_int(
            row[
                "cfbd_game_id"
            ],
            0,
        )

        if game_id not in latest.index:

            output = row.to_dict()

            output[
                "snapshot_reason"
            ] = "first_capture"

            rows.append(
                output
            )

            continue

        previous_row = (
            latest.loc[
                game_id
            ]
        )

        if isinstance(
            previous_row,
            pd.DataFrame,
        ):

            previous_row = (
                previous_row.iloc[
                    -1
                ]
            )

        old_fingerprint = clean_string(
            previous_row.get(
                "prediction_fingerprint"
            )
        )

        new_fingerprint = clean_string(
            row.get(
                "prediction_fingerprint"
            )
        )

        if (
            old_fingerprint
            ==
            new_fingerprint
        ):

            continue

        output = row.to_dict()

        output[
            "snapshot_reason"
        ] = "prediction_changed"

        rows.append(
            output
        )

    if not rows:

        return pd.DataFrame(
            columns=(
                list(
                    snapshots.columns
                )
                +
                [
                    "snapshot_reason",
                ]
            )
        )

    return pd.DataFrame(
        rows
    )


def append_history(
    history: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:

    if history.empty:

        combined = (
            new_rows.copy()
        )

    elif new_rows.empty:

        combined = (
            history.copy()
        )

    else:

        all_columns = list(
            dict.fromkeys(
                list(
                    history.columns
                )
                +
                list(
                    new_rows.columns
                )
            )
        )

        combined = pd.concat(
            [
                history.reindex(
                    columns=all_columns
                ),
                new_rows.reindex(
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

        combined = (
            combined
            .sort_values(
                [
                    "cfbd_game_id",
                    "captured_at_utc",
                ],
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    return combined


def build_current_view(
    history: pd.DataFrame,
) -> pd.DataFrame:

    if history.empty:
        return history.copy()

    current = (
        history
        .sort_values(
            [
                "cfbd_game_id",
                "captured_at_utc",
            ],
            kind="stable",
        )
        .groupby(
            "cfbd_game_id",
            sort=False,
        )
        .tail(
            1
        )
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

    return current


# ============================================================
# VALIDATION
# ============================================================

def validate_snapshot_frame(
    frame: pd.DataFrame,
) -> None:

    required = [
        "cfbd_game_id",
        "start_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "prediction_fingerprint",
        "snapshot_id",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Tracker snapshot is missing columns:\n"
            + "\n".join(
                missing
            )
        )

    if (
        frame[
            "cfbd_game_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Tracker snapshot contains duplicate "
            "game IDs."
        )

    probability_sample = frame[
        frame[
            "home_win_probability"
        ]
        .notna()
        &
        frame[
            "away_win_probability"
        ]
        .notna()
    ].copy()

    if not probability_sample.empty:

        sums = (
            probability_sample[
                "home_win_probability"
            ]
            +
            probability_sample[
                "away_win_probability"
            ]
        )

        invalid = (
            (
                sums
                -
                1.0
            )
            .abs()
            >
            1e-6
        )

        if invalid.any():

            raise RuntimeError(
                "Winner probabilities do not "
                "sum to 1.0 for every game."
            )


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    *,
    captured_at_utc: str,
    snapshots: pd.DataFrame,
    new_rows: pd.DataFrame,
    history: pd.DataFrame,
    current: pd.DataFrame,
) -> None:

    capture_mode_counts = (
        snapshots[
            "capture_mode"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    game_status_counts = (
        snapshots[
            "game_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    manifest = {
        "tracker_version": (
            TRACKER_VERSION
        ),
        "last_run_utc": (
            captured_at_utc
        ),
        "source_winner_predictions": (
            str(
                WINNER_PREDICTIONS_PATH
            )
        ),
        "source_scoring_predictions": (
            str(
                SCORING_PREDICTIONS_PATH
            )
        ),
        "snapshot_games_seen_this_run": int(
            len(
                snapshots
            )
        ),
        "new_history_rows_this_run": int(
            len(
                new_rows
            )
        ),
        "total_history_rows": int(
            len(
                history
            )
        ),
        "current_games": int(
            len(
                current
            )
        ),
        "capture_mode_counts": {
            str(
                key
            ): int(
                value
            )
            for key, value
            in capture_mode_counts.items()
        },
        "game_status_counts": {
            str(
                key
            ): int(
                value
            )
            for key, value
            in game_status_counts.items()
        },
        "locking_enabled": False,
        "important_note": (
            "This tracker records immutable dynamic "
            "snapshots. Games already started before "
            "the tracker first observed them are marked "
            "historical_reconstruction and must never "
            "be treated as genuine lock-time predictions."
        ),
    }

    TRACKER_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "- IMMUTABLE PREDICTION TRACKER"
    )

    print(
        "Purpose:"
    )

    print(
        "Append genuine prediction changes to an "
        "immutable history without overwriting "
        "earlier snapshots."
    )

    print()

    print(
        "Games already started before first capture "
        "are labelled historical_reconstruction."
    )

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    section(
        "1. REQUIRED FILES"
    )

    required_files = [
        WINNER_PREDICTIONS_PATH,
        SCORING_PREDICTIONS_PATH,
    ]

    for path in required_files:

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
        "2. LOAD CURRENT PRODUCTION PREDICTIONS"
    )

    winner_predictions = (
        prepare_winner_predictions(
            load_csv(
                WINNER_PREDICTIONS_PATH
            )
        )
    )

    scoring_predictions = (
        prepare_scoring_predictions(
            load_csv(
                SCORING_PREDICTIONS_PATH
            )
        )
    )

    print(
        f"Winner prediction rows: "
        f"{len(winner_predictions):,}"
    )

    print(
        f"Scoring prediction rows: "
        f"{len(scoring_predictions):,}"
    )

    winner_ids = set(
        winner_predictions[
            "tracker_game_id"
        ]
        .astype(int)
        .tolist()
    )

    scoring_ids = set(
        scoring_predictions[
            "tracker_game_id"
        ]
        .astype(int)
        .tolist()
    )

    overlap = len(
        winner_ids
        &
        scoring_ids
    )

    print(
        f"Winner/scoring game overlap: "
        f"{overlap:,}"
    )

    if overlap != len(
        winner_predictions
    ):

        missing_scoring = sorted(
            winner_ids
            -
            scoring_ids
        )

        print(
            f"WARNING - games without scoring output: "
            f"{len(missing_scoring):,}"
        )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    section(
        "3. BUILD CURRENT SNAPSHOT"
    )

    captured_at_utc = (
        iso_utc_now()
    )

    metadata = (
        model_metadata()
    )

    snapshots = (
        build_snapshot_rows(
            winner_predictions=(
                winner_predictions
            ),
            scoring_predictions=(
                scoring_predictions
            ),
            captured_at_utc=(
                captured_at_utc
            ),
            metadata=(
                metadata
            ),
        )
    )

    snapshots = (
        add_fingerprints(
            snapshots
        )
    )

    validate_snapshot_frame(
        snapshots
    )

    print(
        f"Captured at UTC: "
        f"{captured_at_utc}"
    )

    print(
        f"Snapshot games: "
        f"{len(snapshots):,}"
    )

    print()

    print(
        "Capture modes:"
    )

    print(
        snapshots[
            "capture_mode"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print(
        "Game statuses:"
    )

    print(
        snapshots[
            "game_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # --------------------------------------------------------
    # APPEND IMMUTABLE HISTORY
    # --------------------------------------------------------

    section(
        "4. APPEND IMMUTABLE HISTORY"
    )

    existing_history = (
        load_existing_history()
    )

    print(
        f"Existing history rows: "
        f"{len(existing_history):,}"
    )

    new_rows = (
        identify_new_snapshots(
            snapshots,
            existing_history,
        )
    )

    print(
        f"New snapshot rows: "
        f"{len(new_rows):,}"
    )

    if not new_rows.empty:

        print()

        print(
            "Snapshot reasons:"
        )

        print(
            new_rows[
                "snapshot_reason"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    combined_history = (
        append_history(
            existing_history,
            new_rows,
        )
    )

    # --------------------------------------------------------
    # CURRENT VIEW
    # --------------------------------------------------------

    section(
        "5. BUILD CURRENT TRACKER VIEW"
    )

    current = (
        build_current_view(
            combined_history
        )
    )

    print(
        f"Total history rows: "
        f"{len(combined_history):,}"
    )

    print(
        f"Current games: "
        f"{len(current):,}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    section(
        "6. SAVE TRACKER"
    )

    TRACKER_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_history.to_csv(
        TRACKER_HISTORY_PATH,
        index=False,
    )

    current.to_csv(
        TRACKER_CURRENT_PATH,
        index=False,
    )

    save_manifest(
        captured_at_utc=(
            captured_at_utc
        ),
        snapshots=(
            snapshots
        ),
        new_rows=(
            new_rows
        ),
        history=(
            combined_history
        ),
        current=(
            current
        ),
    )

    print(
        f"Saved history -> "
        f"{TRACKER_HISTORY_PATH}"
    )

    print(
        f"Saved current -> "
        f"{TRACKER_CURRENT_PATH}"
    )

    print(
        f"Saved manifest -> "
        f"{TRACKER_MANIFEST_PATH}"
    )

    # --------------------------------------------------------
    # SAFETY SUMMARY
    # --------------------------------------------------------

    section(
        "PREDICTION TRACKER RUN COMPLETE"
    )

    print(
        "✓ Existing history was never rewritten "
        "from current prediction values."
    )

    print(
        "✓ Identical reruns do not create duplicate "
        "prediction snapshots."
    )

    print(
        "✓ Prediction changes create a new immutable "
        "history row."
    )

    print(
        "✓ Already-started games are explicitly marked "
        "historical_reconstruction."
    )

    print(
        "✓ Future games are tracked as genuine "
        "live_dynamic snapshots."
    )

    print(
        "✓ 24-hour locking is NOT enabled yet."
    )

    print()

    print(
        "Next layer after this validates successfully:"
    )

    print(
        "24-hour official prediction locking."
    )


if __name__ == "__main__":
    main()
