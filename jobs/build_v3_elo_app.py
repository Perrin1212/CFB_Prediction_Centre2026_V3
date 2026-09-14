from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "v3" / "games.csv"
APP = ROOT / "data" / "app"

def main() -> None:
    g = pd.read_csv(SRC, low_memory=False)
    g = g[pd.to_numeric(g.get("season"), errors="coerce").eq(2026)].copy()
    g["homePregameElo"] = pd.to_numeric(g.get("homePregameElo"), errors="coerce")
    g["awayPregameElo"] = pd.to_numeric(g.get("awayPregameElo"), errors="coerce")
    g["homePostgameElo"] = pd.to_numeric(g.get("homePostgameElo"), errors="coerce")
    g["awayPostgameElo"] = pd.to_numeric(g.get("awayPostgameElo"), errors="coerce")
    g = g[g["homePostgameElo"].notna() & g["awayPostgameElo"].notna()].copy()
    rows = []
    for r in g.itertuples(index=False):
        for side, team, opp in [("home", r.home_team, r.away_team), ("away", r.away_team, r.home_team)]:
            pre = float(getattr(r, f"{side}PregameElo")); post = float(getattr(r, f"{side}PostgameElo"))
            rows.append({"game_id": r.game_id, "season": r.season, "week": r.week, "start_date": r.start_date,
                         "team": team, "opponent": opp, "venue": side, "elo": post,
                         "previous_elo": pre, "elo_change": post - pre,
                         "result": "W" if (r.home_points > r.away_points) == (side == "home") else "L" if r.home_points != r.away_points else "T",
                         "classification": getattr(r, f"{side}_classification", "")})
    out = pd.DataFrame(rows).sort_values(["start_date", "game_id", "team"], kind="stable")
    out["rank"] = out.groupby(["season", "week"])["elo"].rank(method="min", ascending=False)
    out["previous_rank"] = out.groupby(["season", "week"])["previous_elo"].rank(method="min", ascending=False)
    out["rank_change"] = out["previous_rank"] - out["rank"]
    out["week_label"] = "Week " + out["week"].astype(int).astype(str)
    cols = ["team","season","week","week_label","elo","previous_elo","elo_change","rank","previous_rank","rank_change","opponent","venue","result","classification","game_id"]
    out[cols].to_csv(APP / "v3_elo_weekly.csv", index=False)
    pairs = out.groupby("game_id")["elo_change"].sum()
    audit = {"games": int(len(pairs)), "zero_sum_failures": int((pairs.abs() > 0.01).sum()), "max_pair_sum_abs": float(pairs.abs().max()) if len(pairs) else 0.0}
    (APP / "v3_elo_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2)); print(f"Saved V3 ELO -> {APP / 'v3_elo_weekly.csv'}")

if __name__ == "__main__": main()
