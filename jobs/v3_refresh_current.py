from __future__ import annotations

"""Rebuild only the live-season portion of the canonical structural V3 data."""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.v3_possession_engine import aggregate_team_games, reconstruct_drives
from jobs.v3_build_canonical import (
    build_pbp_integrity_audit,
    canonical_games,
    load_jsons,
    print_pbp_integrity_report,
    validate_pbp_integrity,
)


FROZEN = SETTINGS.processed_root / "frozen_2021_2025"
REQUIRED_FROZEN = (
    "manifest.json",
    "games.csv",
    "drives.csv",
    "drives_clean.csv",
    "team_game_drive_metrics.csv",
    "team_game_drive_metrics_clean.csv",
    "chronological_training_rows.csv",
)


def _frozen_csv(name: str) -> pd.DataFrame:
    path = FROZEN / name
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen history is incomplete: {path}. Run the pipeline with --rebuild-history."
        )
    return pd.read_csv(path, low_memory=False)


def _combine(history: pd.DataFrame, current: pd.DataFrame, sort: list[str] | None = None) -> pd.DataFrame:
    combined = pd.concat([history, current], ignore_index=True, sort=False)
    if sort:
        available = [column for column in sort if column in combined.columns]
        if available:
            combined = combined.sort_values(available, kind="stable")
    return combined.reset_index(drop=True)


def main() -> None:
    missing = [name for name in REQUIRED_FROZEN if not (FROZEN / name).exists()]
    if missing:
        raise RuntimeError(
            "Frozen 2021-2025 structural data is missing. Run: "
            "python -m jobs.run_pipeline --refresh-current --rebuild-history"
        )

    games_all = canonical_games()
    current_games = games_all[games_all["season"].eq(SETTINGS.live_season)].copy()
    if current_games.empty:
        raise RuntimeError(f"No {SETTINGS.live_season} canonical games were found")
    if current_games["game_id"].duplicated().any():
        raise RuntimeError("Duplicate live-season canonical game IDs")

    plays, files = load_jsons(f"plays/{SETTINGS.live_season}/*.json")
    play_frame = pd.json_normalize(plays)
    if play_frame.empty:
        raise RuntimeError(f"No {SETTINGS.live_season} play-by-play is cached")

    current_drives = reconstruct_drives(play_frame)
    current_ids = set(current_games["game_id"].astype("int64"))
    current_drives = current_drives[current_drives["game_id"].isin(current_ids)].copy()
    current_team_games = aggregate_team_games(current_drives)

    audit = build_pbp_integrity_audit(current_drives, current_games)
    print_pbp_integrity_report(audit)
    validate_pbp_integrity(audit)
    quarantine = audit[audit["quarantine"]].copy()
    quarantine_ids = set(quarantine["game_id"].astype("int64"))
    current_drives_clean = current_drives[~current_drives["game_id"].isin(quarantine_ids)].copy()
    current_team_games_clean = current_team_games[
        ~current_team_games["game_id"].isin(quarantine_ids)
    ].copy()

    completed = current_games.home_points.notna() & current_games.away_points.notna()
    completed_ids = set(current_games.loc[completed, "game_id"].astype("int64"))
    covered_ids = set(current_team_games_clean["game_id"].astype("int64"))
    result_only_ids = completed_ids - covered_ids

    history_games = _frozen_csv("games.csv")
    games = _combine(history_games, current_games, ["season", "start_date", "game_id"])
    drives = _combine(_frozen_csv("drives.csv"), current_drives, ["game_id", "drive_id"])
    drives_clean = _combine(
        _frozen_csv("drives_clean.csv"), current_drives_clean, ["game_id", "drive_id"]
    )
    team_games = _combine(
        _frozen_csv("team_game_drive_metrics.csv"), current_team_games, ["game_id", "team"]
    )
    team_games_clean = _combine(
        _frozen_csv("team_game_drive_metrics_clean.csv"),
        current_team_games_clean,
        ["game_id", "team"],
    )

    games.to_csv(SETTINGS.processed_root / "games.csv", index=False)
    drives.to_csv(SETTINGS.processed_root / "drives.csv", index=False)
    drives_clean.to_csv(SETTINGS.processed_root / "drives_clean.csv", index=False)
    team_games.to_csv(SETTINGS.processed_root / "team_game_drive_metrics.csv", index=False)
    team_games_clean.to_csv(
        SETTINGS.processed_root / "team_game_drive_metrics_clean.csv", index=False
    )
    _frozen_csv("chronological_training_rows.csv").to_csv(
        SETTINGS.processed_root / "chronological_training_rows.csv", index=False
    )

    historical_audit_path = FROZEN / "v3_pbp_integrity_audit.csv"
    if historical_audit_path.exists():
        historical_audit = pd.read_csv(historical_audit_path, low_memory=False)
        combined_audit = _combine(historical_audit, audit, ["season", "game_id"])
    else:
        combined_audit = audit
    combined_audit.to_csv(SETTINGS.audit_root / "v3_pbp_integrity_audit.csv", index=False)

    historical_quarantine_path = FROZEN / "v3_pbp_quarantine.csv"
    quarantine_frames = []
    if historical_quarantine_path.exists():
        quarantine_frames.append(pd.read_csv(historical_quarantine_path, low_memory=False))
    quarantine_frames.append(quarantine)
    combined_quarantine = pd.concat(quarantine_frames, ignore_index=True, sort=False)
    combined_quarantine.to_csv(SETTINGS.audit_root / "v3_pbp_quarantine.csv", index=False)

    season_counts = (
        games[games.home_points.notna() & games.away_points.notna()]
        .groupby("season").game_id.nunique().astype(int).to_dict()
    )
    canonical_audit = {
        "mode": "frozen_history_plus_current_refresh",
        "games": int(len(games)),
        "live_season_games": int(len(current_games)),
        "live_season_completed_games": int(completed.sum()),
        "live_season_pbp_games": int(current_drives.game_id.nunique()),
        "live_season_clean_pbp_games": int(current_drives_clean.game_id.nunique()),
        "live_season_result_only_games": int(len(result_only_ids)),
        "plays": int(len(play_frame)),
        "current_cached_week_files": int(len(files)),
        "drives_raw": int(len(drives)),
        "drives_clean": int(len(drives_clean)),
        "team_games_raw": int(len(team_games)),
        "team_games_clean": int(len(team_games_clean)),
        "training_rows": int(len(_frozen_csv("chronological_training_rows.csv"))),
        "completed_game_counts": {str(int(k)): int(v) for k, v in season_counts.items()},
        "duplicate_game_ids": int(games["game_id"].duplicated().sum()),
    }
    (SETTINGS.audit_root / "v3_canonical_audit.json").write_text(
        json.dumps(canonical_audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(canonical_audit, indent=2))
    print("Current-season canonical refresh complete; frozen history was reused.")


if __name__ == "__main__":
    main()
