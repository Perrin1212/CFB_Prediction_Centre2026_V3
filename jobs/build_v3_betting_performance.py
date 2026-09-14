from __future__ import annotations

"""Evaluate settled V3 forecasts against display-only market prices."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "data" / "app"
OUT = APP / "v3_betting_performance.csv"


def decimal_odds(value: object) -> float:
    try:
        x = float(value)
        if x > 1: return x if x < 100 else np.nan
        if x < 0: return 1 + 100 / abs(x)
        return 1 + x / 100
    except (TypeError, ValueError): return np.nan


def main() -> None:
    games = pd.read_csv(APP / "games_v3.csv", low_memory=False)
    rows: list[dict] = []
    for _, g in games.iterrows():
        prob = pd.to_numeric(pd.Series([g.get("v3_home_win_probability")]), errors="coerce").iloc[0]
        home, away = g.get("home_team"), g.get("away_team")
        hs, as_ = g.get("actual_home_points"), g.get("actual_away_points")
        if pd.isna(prob) or pd.isna(hs) or pd.isna(as_): continue
        pick_home = float(prob) >= .5
        pick = home if pick_home else away
        odds = decimal_odds(g.get("market_home_moneyline" if pick_home else "market_away_moneyline"))
        if not np.isfinite(odds): continue
        won = (float(hs) > float(as_)) if pick_home else (float(as_) > float(hs))
        stake = 100.0
        rows.append({"bet_number": len(rows) + 1, "bet_type": "Moneyline", "game_id": g.get("cfbd_game_id"), "week": g.get("week"), "start_date": g.get("start_date_utc", g.get("start_date")), "away_team": away, "home_team": home, "selection": pick, "result": "WIN" if won else "LOSS", "stake": stake, "profit_loss": stake * (odds - 1) if won else -stake, "model_probability": max(float(prob), 1 - float(prob)), "market_odds": odds, "source_model": "V3"})
        spread = pd.to_numeric(pd.Series([g.get("market_spread")]), errors="coerce").iloc[0]
        margin = pd.to_numeric(pd.Series([g.get("v3_projected_margin")]), errors="coerce").iloc[0]
        if pd.notna(spread) and pd.notna(margin):
            # CFBD's home spread convention is negative for a home favourite.
            pick_home_spread = float(margin) + float(spread) > 0
            covered = (float(hs) - float(as_) + float(spread)) * (1 if pick_home_spread else -1)
            result = "WIN" if covered > 0 else "LOSS" if covered < 0 else "PUSH"
            rows.append({"bet_number": len(rows) + 1, "bet_type": "Spread", "game_id": g.get("cfbd_game_id"), "week": g.get("week"), "start_date": g.get("start_date_utc", g.get("start_date")), "away_team": away, "home_team": home, "selection": home if pick_home_spread else away, "market_line": spread, "result": result, "stake": 100.0, "profit_loss": 100.0 if result == "WIN" else -100.0 if result == "LOSS" else 0.0, "model_probability": max(float(prob), 1 - float(prob)), "market_odds": 2.0, "source_model": "V3"})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["cumulative_profit_loss"] = out.groupby("bet_type")["profit_loss"].cumsum()
        out.to_csv(OUT, index=False)
    else: pd.DataFrame(columns=["bet_number","bet_type","game_id","week","start_date","away_team","home_team","selection","market_line","result","stake","profit_loss","cumulative_profit_loss","model_probability","market_odds","source_model"]).to_csv(OUT, index=False)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "model": "V3", "settled_moneyline_bets": int(len(out))}
    (APP / "v3_betting_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print(f"Saved V3 betting performance -> {OUT}")


if __name__ == "__main__": main()
