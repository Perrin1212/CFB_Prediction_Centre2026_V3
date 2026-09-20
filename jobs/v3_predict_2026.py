from __future__ import annotations

"""Replay current knowledge chronologically and publish immutable V3 forecasts."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.v3_matchup_engine import build_matchup
from engine.v3_prediction import predict_from_matchup
from engine.v3_team_state import ChronologicalStateEngine

PREDICTION_PATH = SETTINGS.prediction_root / "v3_2026_predictions.csv"
LOCKED_PATH = SETTINGS.prediction_root / "v3_2026_predictions_locked.csv"
MANIFEST_PATH = SETTINGS.audit_root / "v3_prediction_run.json"


def _metric(index: pd.DataFrame, game_id: int, team: str) -> dict | None:
    try:
        row = index.loc[(game_id, team)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            return None
        row = row.iloc[0]
    return row.to_dict()


def _classification(row: object, side: str) -> str:
    return str(getattr(row, f"{side}_classification", "")).strip().casefold()


def replay_completed(games: pd.DataFrame, team_games: pd.DataFrame) -> ChronologicalStateEngine:
    """Build the exact state available after every completed kickoff batch."""
    completed = games[games.home_points.notna() & games.away_points.notna()].copy()
    completed = completed.sort_values(["season", "start_date", "game_id"], kind="stable")
    index = team_games.set_index(["game_id", "team"])
    engine = ChronologicalStateEngine()
    last_season: int | None = None

    for (season, _kickoff), batch in completed.groupby(
        ["season", "start_date"], sort=True, dropna=False
    ):
        season = int(season)
        if last_season is not None and season != last_season:
            engine.regress_season()
        last_season = season
        pending: list[tuple[object, dict | None, dict | None]] = []
        for row in batch.itertuples(index=False):
            game_id = int(row.game_id)
            pending.append(
                (row, _metric(index, game_id, row.home_team), _metric(index, game_id, row.away_team))
            )

        # Apply only after the whole batch: simultaneous games cannot leak.
        for row, home_observation, away_observation in pending:
            common = dict(
                home=row.home_team,
                away=row.away_team,
                home_points=float(row.home_points),
                away_points=float(row.away_points),
                neutral=bool(getattr(row, "neutral_site", False)),
                home_classification=_classification(row, "home"),
                away_classification=_classification(row, "away"),
            )
            if home_observation is not None and away_observation is not None:
                engine.update_game(h_obs=home_observation, a_obs=away_observation, **common)
            else:
                engine.update_result_only(**common)
    return engine


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if "game_id" in frame:
        frame["game_id"] = pd.to_numeric(frame.game_id, errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["game_id"]).copy()
        frame["game_id"] = frame.game_id.astype("int64")
    return frame


def update_locks(games: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Preserve every forecast that had already crossed kickoff or gone final."""
    existing = _read_existing(PREDICTION_PATH)
    locked = _read_existing(LOCKED_PATH)
    if existing.empty:
        return locked
    status = games[["game_id", "start_date", "home_points", "away_points"]].copy()
    candidates = existing.merge(status, on="game_id", how="left", suffixes=("", "_game"))
    crossed = candidates.start_date_game.le(now)
    completed = candidates.home_points.notna() & candidates.away_points.notna()
    candidates = candidates[crossed | completed].copy()[existing.columns]
    locked = pd.concat([locked, candidates], ignore_index=True, sort=False)
    if not locked.empty:
        locked = locked.drop_duplicates("game_id", keep="first").sort_values("game_id")
        locked.to_csv(LOCKED_PATH, index=False)
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=SETTINGS.simulations)
    args = parser.parse_args()
    if args.simulations < 100:
        raise ValueError("--simulations must be at least 100")

    games = pd.read_csv(SETTINGS.processed_root / "games.csv", low_memory=False)
    team_games = pd.read_csv(
        SETTINGS.processed_root / "team_game_drive_metrics_clean.csv", low_memory=False
    )
    bundle = joblib.load(SETTINGS.model_root / "v3_structural_models.joblib")
    games["game_id"] = pd.to_numeric(games.game_id, errors="raise").astype("int64")
    games["season"] = pd.to_numeric(games.season, errors="raise").astype("int64")
    games["start_date"] = pd.to_datetime(games.start_date, utc=True, errors="coerce")
    team_games["game_id"] = pd.to_numeric(team_games.game_id, errors="raise").astype("int64")

    now = pd.Timestamp(datetime.now(timezone.utc))
    locked = update_locks(games, now)
    locked_ids = set(locked.game_id.astype(int)) if not locked.empty else set()
    engine = replay_completed(games, team_games)
    future = games[
        games.season.eq(SETTINGS.live_season)
        & games.home_points.isna()
        & games.away_points.isna()
        & games.start_date.gt(now)
        & ~games.game_id.isin(locked_ids)
    ].sort_values(["start_date", "game_id"], kind="stable")

    rows: list[dict] = []
    for row in future.itertuples(index=False):
        home_state = engine.snapshot(
            row.home_team,
            _classification(row, "home"),
        )
        away_state = engine.snapshot(
            row.away_team,
            _classification(row, "away"),
        )
        matchup = build_matchup(home_state, away_state, bool(getattr(row, "neutral_site", False)))
        seed = SETTINGS.seed + int(row.game_id) % 100_000
        prediction = predict_from_matchup(matchup, bundle, args.simulations, seed)
        simulation = prediction.pop("simulation")
        output = {
            "game_id": int(row.game_id), "season": int(row.season),
            "week": int(row.week) if pd.notna(row.week) else -1,
            "start_date": row.start_date, "home_team": row.home_team,
            "away_team": row.away_team,
            "neutral_site": bool(getattr(row, "neutral_site", False)),
            "v3_home_win_probability": prediction.pop("home_win_probability"),
            "v3_away_win_probability": prediction.pop("away_win_probability"),
            "v3_projected_home_points": prediction.pop("projected_home_points"),
            "v3_projected_away_points": prediction.pop("projected_away_points"),
            "v3_projected_margin": prediction.pop("projected_margin"),
            "v3_projected_total": prediction.pop("projected_total"),
            "state_maturity": float(matchup["state_maturity"]),
        }
        output.update({f"v3_{key}": value for key, value in prediction.items()})
        output.update({f"sim_{key}": value for key, value in simulation.items()})
        output.update({f"matchup_{key}": value for key, value in matchup.items()})
        rows.append(output)

    dynamic = pd.DataFrame(rows)
    combined = pd.concat([locked, dynamic], ignore_index=True, sort=False)
    if not combined.empty:
        combined = combined.drop_duplicates("game_id", keep="first")
        combined = combined.sort_values(["start_date", "game_id"], kind="stable")
    combined.to_csv(PREDICTION_PATH, index=False)
    manifest = {
        "generated_at_utc": now.isoformat(),
        "simulations_per_new_game": int(args.simulations),
        "locked_predictions": int(len(locked)),
        "new_forward_predictions": int(len(dynamic)),
        "published_predictions": int(len(combined)),
        "production_training_games": int(bundle.get("production_games", 0)),
        "production_training_seasons": [
            int(season) for season in bundle.get("production_seasons", ())
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved V3 predictions -> {PREDICTION_PATH}")


if __name__ == "__main__":
    main()
