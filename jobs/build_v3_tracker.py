from __future__ import annotations

"""Build the app's V3-first tracker from immutable forecasts and final scores."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "data" / "app"
PREDICTIONS = ROOT / "data" / "predictions"
OUT = APP / "v3_tracker.csv"


def main() -> None:
    games = pd.read_csv(APP / "games_v3.csv", low_memory=False)
    predictions = pd.read_csv(PREDICTIONS / "v3_2026_predictions.csv", low_memory=False)
    locks_path = PREDICTIONS / "v3_2026_predictions_locked.csv"
    locked = pd.read_csv(locks_path, low_memory=False) if locks_path.exists() else pd.DataFrame()
    games["cfbd_game_id"] = pd.to_numeric(games.cfbd_game_id, errors="coerce").astype("int64")
    predictions["game_id"] = pd.to_numeric(predictions.game_id, errors="coerce").astype("int64")
    p = predictions.rename(columns={"game_id": "cfbd_game_id"})
    keep = ["cfbd_game_id"] + [c for c in p.columns if c != "cfbd_game_id" and (c.startswith(("v3_", "sim_", "matchup_", "state_")) or c not in games.columns)]
    tracker = games.merge(p[keep].drop_duplicates("cfbd_game_id"), on="cfbd_game_id", how="left", suffixes=("", "_prediction"))
    prob = pd.to_numeric(tracker.get("v3_home_win_probability"), errors="coerce")
    has_v3 = prob.notna()
    tracker["v3_predicted_winner"] = np.where(prob.ge(.5), tracker.home_team, tracker.away_team)
    tracker["v3_prediction_probability"] = np.maximum(prob, 1 - prob)
    locked_ids = set(pd.to_numeric(locked.game_id, errors="coerce").dropna().astype("int64")) if not locked.empty else set()
    tracker["v3_is_locked"] = tracker.cfbd_game_id.isin(locked_ids) & has_v3
    aliases = {"home_win_probability":"v3_home_win_probability", "away_win_probability":"v3_away_win_probability", "predicted_winner":"v3_predicted_winner", "prediction_probability":"v3_prediction_probability", "projected_home_score":"v3_projected_home_points", "projected_away_score":"v3_projected_away_points", "final_expected_home_margin":"v3_projected_margin", "final_expected_total_points":"v3_projected_total", "simulation_home_win_probability":"sim_home_win_probability", "simulation_away_win_probability":"sim_away_win_probability", "simulation_count":"sim_simulations"}
    for target, source in aliases.items():
        if source in tracker:
            # A row without a V3 forecast must remain absent from the V3 view;
            # falling back to V2 here falsely reports V3 coverage.
            tracker[target] = tracker[source].where(has_v3, np.nan)
    tracker["is_locked"] = tracker["v3_is_locked"]
    kickoff = pd.to_datetime(
        tracker.get("start_date_utc", tracker.get("start_date")), errors="coerce", utc=True
    )
    tracker["hours_until_lock"] = (kickoff - pd.Timestamp(datetime.now(timezone.utc))).dt.total_seconds() / 3600.0
    tracker.loc[tracker["v3_is_locked"], "hours_until_lock"] = 0.0
    tracker.loc[tracker["v3_is_locked"], "official_lock_quality"] = "v3_pregame_lock"
    tracker["lock_status"] = np.where(tracker["v3_is_locked"], "v3_locked", tracker.get("lock_status", "v3_pending"))
    tracker["confidence_bucket"] = pd.cut(tracker["v3_prediction_probability"], [-np.inf, .55, .60, .70, .80, .90, np.inf], labels=["Coin Flip", "Lean", "Moderate", "Strong", "Very Strong", "Elite"]).astype(object)
    tracker.to_csv(OUT, index=False)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": int(len(tracker)), "v3_forecast_rows": int(has_v3.sum()), "v3_locked_rows": int(tracker.v3_is_locked.sum())}
    (APP / "v3_tracker_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print(f"Saved V3-first tracker -> {OUT}")


if __name__ == "__main__":
    main()
