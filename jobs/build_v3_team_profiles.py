from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from jobs.v3_predict_2026 import replay_completed
from config.v3_settings import SETTINGS

APP = ROOT / "data" / "app"

def main() -> None:
    games = pd.read_csv(SETTINGS.processed_root / "games.csv", low_memory=False)
    metrics = pd.read_csv(SETTINGS.processed_root / "team_game_drive_metrics_clean.csv", low_memory=False)
    games["game_id"] = pd.to_numeric(games.game_id, errors="coerce").astype("Int64")
    games["season"] = pd.to_numeric(games.season, errors="coerce").astype("Int64")
    games["start_date"] = pd.to_datetime(games.start_date, utc=True, errors="coerce")
    metrics["game_id"] = pd.to_numeric(metrics.game_id, errors="coerce").astype("Int64")
    engine = replay_completed(games.dropna(subset=["game_id"]), metrics.dropna(subset=["game_id"]))
    live = games[games.season.eq(SETTINGS.live_season)].copy()
    meta_path = ROOT / "data" / "processed" / "teams.csv"
    meta = pd.read_csv(meta_path, low_memory=False) if meta_path.exists() else pd.DataFrame()
    meta = meta.set_index("team_name") if not meta.empty and "team_name" in meta else pd.DataFrame()
    teams = sorted(set(live.home_team.dropna()) | set(live.away_team.dropna()))
    rows = []
    for team in teams:
        s = engine.snapshot(team)
        tg = live[(live.home_team.eq(team) | live.away_team.eq(team)) & live.home_points.notna() & live.away_points.notna()]
        pf=[]; pa=[]
        for r in tg.itertuples(index=False):
            home = r.home_team == team; pf.append(float(r.home_points if home else r.away_points)); pa.append(float(r.away_points if home else r.home_points))
        m = meta.loc[team] if not meta.empty and team in meta.index else pd.Series(dtype=object)
        rows.append({
            "team": team, "conference": m.get("conference", ""), "classification": m.get("classification", ""),
            "logo_url": m.get("logo_url", ""), "color": m.get("color", ""), "abbreviation": m.get("abbreviation", ""),
            "current_elo": s.get("elo"), "offensive_rating": s.get("off_rating"), "defensive_rating": s.get("def_rating"),
            "success_rate": s.get("off_success_rate"), "explosive_rate": s.get("off_explosive_rate"),
            "points_per_drive": s.get("off_points_per_drive"), "drives_per_game": s.get("off_drives"),
            "avg_start_field_position": s.get("off_avg_start_field"), "third_down_rate": s.get("off_third_down_rate"),
            "def_success_rate": s.get("def_success_rate"), "explosives_allowed_rate": s.get("def_explosive_rate"),
            "def_epa_per_play": s.get("def_ppa_per_play"), "points_for_per_game": np.mean(pf) if pf else np.nan,
            "points_against_per_game": np.mean(pa) if pa else np.nan, "schedule_strength_elo": s.get("schedule_def"),
            "games_played": len(tg), "momentum_label": "Current-season V3 state",
        })
    out = pd.DataFrame(rows)
    fbs = out.classification.astype(str).str.casefold().eq("fbs")
    out.loc[fbs, "model_rank"] = out.loc[fbs, "current_elo"].rank(method="min", ascending=False)
    out.to_csv(APP / "v3_team_profiles.csv", index=False)
    print(f"Saved {len(out):,} V3 team profiles -> {APP / 'v3_team_profiles.csv'}")

if __name__ == "__main__": main()
