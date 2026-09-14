from __future__ import annotations

"""Freeze the validated 2021-2025 structural V3 datasets for weekly reuse."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS


FROZEN = SETTINGS.processed_root / "frozen_2021_2025"
SCHEMA_VERSION = 1


def _read(name: str) -> pd.DataFrame:
    path = SETTINGS.processed_root / name
    if not path.exists():
        raise FileNotFoundError(f"Required canonical file is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(FROZEN / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    manifest_path = FROZEN / "manifest.json"
    if manifest_path.exists() and not args.replace:
        print(f"Historical structural snapshot already frozen: {FROZEN}")
        return

    games = _read("games.csv")
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    history = games[games["season"].between(2021, SETTINGS.holdout_season)].copy()
    if history["game_id"].duplicated().any():
        raise RuntimeError("Cannot freeze history: duplicate canonical game IDs")

    for column in ("home_classification", "away_classification"):
        history[column] = history[column].astype(str).str.strip().str.casefold()
    allowed = (
        (history.home_classification.eq("fbs") & history.away_classification.isin({"fbs", "fcs"}))
        | (history.away_classification.eq("fbs") & history.home_classification.isin({"fbs", "fcs"}))
    )
    if not allowed.all():
        raise RuntimeError(f"Cannot freeze history: {(~allowed).sum()} games violate the FBS universe")

    holdout_games = int(history.loc[history.season.eq(SETTINGS.holdout_season), "game_id"].nunique())
    if not 400 <= holdout_games <= SETTINGS.max_reasonable_games_per_season:
        raise RuntimeError(f"Cannot freeze history: suspicious 2025 game count {holdout_games}")

    historical_ids = set(pd.to_numeric(history["game_id"], errors="coerce").dropna().astype("int64"))
    FROZEN.mkdir(parents=True, exist_ok=True)
    _write(history, "games.csv")

    counts: dict[str, int] = {"games": int(len(history))}
    for name in (
        "drives.csv",
        "drives_clean.csv",
        "team_game_drive_metrics.csv",
        "team_game_drive_metrics_clean.csv",
    ):
        frame = _read(name)
        game_ids = pd.to_numeric(frame["game_id"], errors="coerce")
        frame = frame[game_ids.isin(historical_ids)].copy()
        _write(frame, name)
        counts[name.removesuffix(".csv")] = int(len(frame))

    training = _read("chronological_training_rows.csv")
    training["season"] = pd.to_numeric(training["season"], errors="coerce")
    training = training[training["season"].between(2021, SETTINGS.holdout_season)].copy()
    if training["game_id"].duplicated().any():
        raise RuntimeError("Cannot freeze history: duplicate chronological training IDs")
    _write(training, "chronological_training_rows.csv")
    counts["chronological_training_rows"] = int(len(training))

    for name in ("v3_pbp_integrity_audit.csv", "v3_pbp_quarantine.csv"):
        source = SETTINGS.audit_root / name
        if not source.exists():
            continue
        frame = pd.read_csv(source, low_memory=False)
        game_ids = pd.to_numeric(frame["game_id"], errors="coerce")
        frame = frame[game_ids.isin(historical_ids)].copy()
        frame.to_csv(FROZEN / name, index=False)
        counts[name.removesuffix(".csv")] = int(len(frame))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seasons": [2021, 2022, 2023, 2024, 2025],
        "universe": ["FBS_vs_FBS", "FBS_vs_FCS"],
        "current_season_weights": list(SETTINGS.current_season_weights),
        "holdout_games": holdout_games,
        **counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Frozen structural history -> {FROZEN}")


if __name__ == "__main__":
    main()
