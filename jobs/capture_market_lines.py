from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
OUT_CURRENT = MARKET_DIR / "market_lines_current.csv"
OUT_HISTORY = MARKET_DIR / "market_lines_history.csv"
OUT_MANIFEST = MARKET_DIR / "market_lines_manifest.json"
SEASON = 2026

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = os.getenv("CFBD_BASE_URL", "https://api.collegefootballdata.com").rstrip("/")

PROVIDER_PRIORITY = ["DraftKings", "FanDuel", "ESPN BET", "Caesars", "Bovada", "Consensus"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def clean(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_datetime(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def provider_rank(name: str) -> int:
    name_clean = clean(name).lower()
    for index, provider in enumerate(PROVIDER_PRIORITY):
        if name_clean == provider.lower():
            return index
    return len(PROVIDER_PRIORITY) + 1


def api_get_lines() -> list[dict[str, Any]]:
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY was not found in the V2 .env file.")
    response = requests.get(
        f"{BASE_URL}/lines",
        headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"},
        params={"year": SEASON},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("CFBD /lines did not return a list.")
    return payload


def normalise_provider_line(game: dict[str, Any], provider_line: dict[str, Any], captured_at: datetime) -> dict[str, Any]:
    kickoff_raw = game.get("startDate") or game.get("start_date") or game.get("kickoff")
    kickoff = parse_datetime(kickoff_raw)
    captured_ts = pd.Timestamp(captured_at)
    if kickoff is None:
        capture_timing = "unknown"
        hours_to_kickoff = np.nan
    else:
        hours_to_kickoff = (kickoff - captured_ts).total_seconds() / 3600.0
        capture_timing = "pregame" if hours_to_kickoff > 0 else "historical_market_backfill"

    spread = safe_float(provider_line.get("spread"))
    home_team = clean(game.get("homeTeam") or game.get("home_team"))
    away_team = clean(game.get("awayTeam") or game.get("away_team"))
    if spread is None:
        spread_favourite = ""
    elif spread < 0:
        spread_favourite = home_team
    elif spread > 0:
        spread_favourite = away_team
    else:
        spread_favourite = ""

    return {
        "captured_at_utc": iso_utc(captured_at),
        "capture_timing": capture_timing,
        "hours_to_kickoff_at_capture": round(hours_to_kickoff, 4) if np.isfinite(hours_to_kickoff) else np.nan,
        "season": game.get("season", SEASON),
        "week": game.get("week"),
        "season_type": game.get("seasonType") or game.get("season_type"),
        "cfbd_game_id": game.get("id") or game.get("gameId") or game.get("game_id"),
        "start_date_utc": kickoff_raw,
        "home_team": home_team,
        "away_team": away_team,
        "provider": clean(provider_line.get("provider")),
        "spread": spread,
        "spread_favourite": spread_favourite,
        "formatted_spread": clean(provider_line.get("formattedSpread")),
        "spread_open": safe_float(provider_line.get("spreadOpen")),
        "over_under": safe_float(provider_line.get("overUnder")),
        "over_under_open": safe_float(provider_line.get("overUnderOpen")),
        "home_moneyline": safe_float(provider_line.get("homeMoneyline")),
        "away_moneyline": safe_float(provider_line.get("awayMoneyline")),
    }


def flatten_lines(payload: list[dict[str, Any]], captured_at: datetime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in payload:
        provider_lines = game.get("lines") or game.get("providers") or []
        if isinstance(provider_lines, dict):
            provider_lines = [provider_lines]
        if not provider_lines:
            candidate = {
                "provider": game.get("provider"),
                "spread": game.get("spread"),
                "formattedSpread": game.get("formattedSpread"),
                "spreadOpen": game.get("spreadOpen"),
                "overUnder": game.get("overUnder"),
                "overUnderOpen": game.get("overUnderOpen"),
                "homeMoneyline": game.get("homeMoneyline"),
                "awayMoneyline": game.get("awayMoneyline"),
            }
            if any(v is not None for k, v in candidate.items() if k != "provider"):
                provider_lines = [candidate]
        for provider_line in provider_lines:
            if isinstance(provider_line, dict):
                rows.append(normalise_provider_line(game, provider_line, captured_at))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["cfbd_game_id"] = pd.to_numeric(frame["cfbd_game_id"], errors="coerce").astype("Int64")
    frame["start_date_utc"] = pd.to_datetime(frame["start_date_utc"], errors="coerce", utc=True)
    frame["_provider_rank"] = frame["provider"].fillna("").astype(str).map(provider_rank)
    frame["_has_moneyline"] = frame[["home_moneyline", "away_moneyline"]].notna().any(axis=1).astype(int)
    frame["_has_spread"] = frame["spread"].notna().astype(int)
    return frame


def select_current(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work["_coverage"] = work["_has_moneyline"] + work["_has_spread"]
    work = work.sort_values(["cfbd_game_id", "_coverage", "_provider_rank"], ascending=[True, False, True], kind="stable")
    current = work.drop_duplicates(subset=["cfbd_game_id"], keep="first").copy()
    return current.drop(columns=["_provider_rank", "_has_moneyline", "_has_spread", "_coverage"], errors="ignore")


def append_history(latest: pd.DataFrame) -> pd.DataFrame:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    clean_latest = latest.drop(columns=["_provider_rank", "_has_moneyline", "_has_spread"], errors="ignore").copy()
    if OUT_HISTORY.exists():
        old = pd.read_csv(OUT_HISTORY, low_memory=False)
        history = pd.concat([old, clean_latest], ignore_index=True, sort=False)
    else:
        history = clean_latest
    if not history.empty:
        dedupe_columns = [c for c in ["cfbd_game_id", "provider", "spread", "home_moneyline", "away_moneyline", "over_under", "capture_timing"] if c in history.columns]
        history = history.drop_duplicates(subset=dedupe_columns, keep="first").sort_values(["start_date_utc", "cfbd_game_id", "captured_at_utc"], kind="stable", na_position="last").reset_index(drop=True)
    history.to_csv(OUT_HISTORY, index=False)
    return history


def main() -> None:
    print("=" * 78)
    print("CFB PREDICTION CENTRE 2026 V2 - CAPTURE MARKET LINES")
    print("=" * 78)
    captured_at = utc_now()
    payload = api_get_lines()
    print(f"CFBD games with line payloads: {len(payload):,}")
    all_lines = flatten_lines(payload, captured_at)
    print(f"Provider line rows:           {len(all_lines):,}")
    if all_lines.empty:
        raise RuntimeError("No usable market-line rows were returned by CFBD.")
    current = select_current(all_lines)
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_CURRENT, index=False)
    history = append_history(all_lines)
    ml_games = int(current[["home_moneyline", "away_moneyline"]].notna().any(axis=1).sum())
    spread_games = int(current["spread"].notna().sum())
    pregame_games = int(current["capture_timing"].eq("pregame").sum())
    backfill_games = int(current["capture_timing"].eq("historical_market_backfill").sum())
    manifest = {
        "generated_at_utc": iso_utc(captured_at),
        "season": SEASON,
        "current_games": int(len(current)),
        "history_rows": int(len(history)),
        "moneyline_games_current": ml_games,
        "spread_games_current": spread_games,
        "pregame_games_current": pregame_games,
        "historical_backfill_games_current": backfill_games,
        "source": "CollegeFootballData /lines",
        "current_file": str(OUT_CURRENT),
        "history_file": str(OUT_HISTORY),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print("-" * 78)
    print(f"Current games:               {len(current):,}")
    print(f"Moneyline coverage:          {ml_games:,}")
    print(f"Spread coverage:             {spread_games:,}")
    print(f"Captured before kickoff:     {pregame_games:,}")
    print(f"Historical/backfill rows:    {backfill_games:,}")
    print(f"Total history rows:          {len(history):,}")
    print("-" * 78)
    print(f"Saved -> {OUT_CURRENT}")
    print(f"Saved -> {OUT_HISTORY}")
    print("\nDONE")


if __name__ == "__main__":
    main()
