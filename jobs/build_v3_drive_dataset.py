from __future__ import annotations

"""Build the V3 drive dataset with a frozen historical base.

The first successful run builds and freezes the 2021-2025 FBS universe.
Later runs reuse that immutable historical base and rebuild only 2026 drives.

Allowed games:
    * FBS vs FBS
    * FBS vs FCS

Excluded games:
    * FCS vs FCS
    * Any Division II / Division III / NAIA-only matchup
    * Any game without exactly one usable drive row for each team

Use ``--rebuild-history`` only after deliberately changing historical feature
engineering or correcting the historical source data.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.v3_drive_engine import DriveStateEngine, matchup_features, save_states


RAW = ROOT / "data" / "raw" / "v3_cfbd" / "plays"
OUT = ROOT / "data" / "v3"
FROZEN = OUT / "frozen_2021_2025"
HISTORICAL_GAMES_PATH = ROOT / "data" / "processed" / "historical_games.csv"
TEAMS_PATH = ROOT / "data" / "processed" / "teams.csv"

FREEZE_SCHEMA_VERSION = 2
HISTORICAL_YEARS = set(range(2021, 2026))
CURRENT_YEAR = 2026

FROZEN_DRIVES = FROZEN / "drives_2021_2025.csv"
FROZEN_TEAM_GAMES = FROZEN / "team_game_drive_metrics_2021_2025.csv"
FROZEN_TRAINING = FROZEN / "v3_drive_training_rows_2021_2025.csv"
FROZEN_STATES = FROZEN / "v3_end_2025_drive_states.json"
FROZEN_MANIFEST = FROZEN / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Discard the reusable processed-history snapshot and rebuild 2021-2025.",
    )
    return parser.parse_args()


def normalise_team(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip().casefold()


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.casefold().isin({"1", "true", "t", "yes", "y"})


def numeric_year_directories(years: set[int]) -> list[Path]:
    directories: list[Path] = []
    if not RAW.exists():
        return directories
    for path in RAW.iterdir():
        if path.is_dir() and re.fullmatch(r"\d{4}", path.name):
            year = int(path.name)
            if year in years:
                directories.append(path)
    return sorted(directories, key=lambda item: int(item.name))


def load_plays(years: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files: list[Path] = []

    for year_dir in numeric_year_directories(years):
        for path in sorted(year_dir.glob("week_*.json")):
            files.append(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"WARNING: skipping unreadable cache file {path}: {exc}")
                continue

            if not isinstance(payload, list) or not payload:
                continue

            frame = pd.json_normalize(payload)
            frame["_cache_season"] = int(year_dir.name)
            week_match = re.search(r"(\d+)", path.stem)
            frame["_cache_week"] = int(week_match.group(1)) if week_match else np.nan
            frames.append(frame)

    if not frames:
        print(f"Loaded 0 plays from {len(files)} cached week files")
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Loaded {len(data):,} plays from {len(files)} cached week files")
    return data


def classify_result(text: object) -> str:
    value = str(text).lower()
    if "touchdown" in value:
        return "TD"
    if "field goal" in value and ("good" in value or "made" in value):
        return "FG"
    if "turnover on downs" in value or value.strip() == "downs":
        return "DOWNS"
    if any(token in value for token in ("interception", "fumble")):
        return "TO"
    if "punt" in value:
        return "PUNT"
    return "OTHER"


def drive_table(plays: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_id", "drive_id", "offense", "defense", "drive_number",
        "start_field_position", "plays", "yards", "points", "result",
        "turnover", "touchdown", "field_goal", "punt", "success_rate",
        "ppa", "explosiveness",
    ]
    if plays.empty:
        return pd.DataFrame(columns=columns)

    rename = {
        "gameId": "game_id",
        "driveId": "drive_id",
        "driveNumber": "drive_number",
        "yardsToGoal": "yards_to_goal",
        "yardsGained": "yards_gained",
        "playType": "play_type",
        "playText": "play_text",
    }
    data = plays.rename(columns=rename).copy()
    required = {"game_id", "drive_id", "offense", "defense"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"Cached play data is missing required columns: {missing}")

    data["game_id"] = pd.to_numeric(data["game_id"], errors="coerce")
    data = data.dropna(subset=["game_id", "drive_id", "offense", "defense"])
    data["game_id"] = data["game_id"].astype("int64")

    for column in (
        "yards_to_goal", "yards_gained", "ppa", "period",
        "offenseScore", "defenseScore", "playNumber",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    sort_columns = [
        column for column in ("game_id", "drive_id", "period", "playNumber")
        if column in data.columns
    ]
    data = data.sort_values(sort_columns, kind="stable")

    output: list[dict[str, object]] = []
    for (game_id, drive_id), group in data.groupby(["game_id", "drive_id"], sort=False):
        first = group.iloc[0]
        last = group.iloc[-1]
        play_text = group.get("play_text", pd.Series(index=group.index, dtype=str))
        play_type = group.get("play_type", pd.Series(index=group.index, dtype=str))
        result_text = " ".join(play_text.fillna("").astype(str).tail(3))
        result_text += " " + " ".join(play_type.fillna("").astype(str).tail(3))

        yards_to_goal = first.get("yards_to_goal", np.nan)
        start_yards_to_goal = float(yards_to_goal) if pd.notna(yards_to_goal) else 72.5
        start_field_position = float(np.clip(100.0 - start_yards_to_goal, 1.0, 99.0))

        if "ppa" in group.columns:
            ppa = pd.to_numeric(group["ppa"], errors="coerce")
        else:
            ppa = pd.Series(index=group.index, dtype=float)

        result = classify_result(result_text)
        points = 7 if result == "TD" else 3 if result == "FG" else 0

        try:
            score_delta = max(0.0, float(last["offenseScore"]) - float(first["offenseScore"]))
            if score_delta in {2.0, 3.0, 6.0, 7.0, 8.0}:
                points = int(score_delta)
        except (KeyError, TypeError, ValueError):
            pass

        if "yards_gained" in group.columns:
            yards = float(pd.to_numeric(group["yards_gained"], errors="coerce").fillna(0).sum())
        else:
            yards = 0.0

        valid_ppa = ppa.dropna()
        positive_ppa = ppa[ppa > 0]
        output.append(
            {
                "game_id": int(game_id),
                "drive_id": drive_id,
                "offense": first["offense"],
                "defense": first["defense"],
                "drive_number": first.get("drive_number", np.nan),
                "start_field_position": start_field_position,
                "plays": int(len(group)),
                "yards": yards,
                "points": points,
                "result": result,
                "turnover": int(result == "TO"),
                "touchdown": int(result == "TD"),
                "field_goal": int(result == "FG"),
                "punt": int(result == "PUNT"),
                "success_rate": float((ppa > 0).mean()) if len(valid_ppa) else 0.42,
                "ppa": float(valid_ppa.mean()) if len(valid_ppa) else 0.0,
                "explosiveness": float(positive_ppa.mean()) if len(positive_ppa) else 0.9,
            }
        )

    return pd.DataFrame(output, columns=columns)


def team_games(drives: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "drives", "ppd", "avg_start", "td_rate", "fg_rate", "to_rate",
        "punt_rate", "success_rate", "ppa", "explosiveness",
    ]
    if drives.empty:
        return pd.DataFrame(columns=["game_id", "offense", "defense"])

    grouped = drives.groupby(["game_id", "offense", "defense"], as_index=False).agg(
        drives=("drive_id", "nunique"),
        points=("points", "sum"),
        avg_start=("start_field_position", "mean"),
        td=("touchdown", "sum"),
        fg=("field_goal", "sum"),
        to=("turnover", "sum"),
        punt=("punt", "sum"),
        success_rate=("success_rate", "mean"),
        ppa=("ppa", "mean"),
        explosiveness=("explosiveness", "mean"),
    )

    for output_column, source_column in (
        ("td_rate", "td"),
        ("fg_rate", "fg"),
        ("to_rate", "to"),
        ("punt_rate", "punt"),
    ):
        grouped[output_column] = grouped[source_column] / grouped["drives"].clip(lower=1)
    grouped["ppd"] = grouped["points"] / grouped["drives"].clip(lower=1)

    opponent = grouped.rename(
        columns={
            "offense": "defense",
            "defense": "offense",
            **{column: f"opp_{column}" for column in metric_columns},
        }
    )
    keep = ["game_id", "offense", "defense"] + [f"opp_{column}" for column in metric_columns]
    return grouped.merge(opponent[keep], on=["game_id", "offense", "defense"], how="left")


def eligible_historical_games() -> tuple[pd.DataFrame, dict[str, int]]:
    games = pd.read_csv(HISTORICAL_GAMES_PATH, low_memory=False)
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games = games[games["season"].isin(HISTORICAL_YEARS)].copy()

    completed = bool_series(games, "completed")
    home_fbs = bool_series(games, "home_is_fbs")
    away_fbs = bool_series(games, "away_is_fbs")
    home_fcs = bool_series(games, "home_is_fcs")
    away_fcs = bool_series(games, "away_is_fcs")

    fbs_vs_fbs = home_fbs & away_fbs
    fbs_vs_fcs = (home_fbs & away_fcs) | (away_fbs & home_fcs)
    eligible = completed & (fbs_vs_fbs | fbs_vs_fcs)

    games = games[eligible].copy()
    games["matchup_class"] = np.where(
        fbs_vs_fbs.loc[games.index], "FBS_vs_FBS", "FBS_vs_FCS"
    )
    games["cfbd_id"] = pd.to_numeric(games["cfbd_id"], errors="coerce")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["away_points"] = pd.to_numeric(games["away_points"], errors="coerce")
    games = games.dropna(subset=["cfbd_id", "home_team", "away_team", "home_points", "away_points"])
    games["cfbd_id"] = games["cfbd_id"].astype("int64")
    games = games.drop_duplicates(subset=["cfbd_id"], keep="last")

    invalid = games[~games["matchup_class"].isin({"FBS_vs_FBS", "FBS_vs_FCS"})]
    if not invalid.empty:
        raise RuntimeError(f"Division filter failed for {len(invalid)} historical games")

    audit = {
        "eligible_historical_games": int(len(games)),
        "fbs_vs_fbs": int((games["matchup_class"] == "FBS_vs_FBS").sum()),
        "fbs_vs_fcs": int((games["matchup_class"] == "FBS_vs_FCS").sum()),
    }
    return games, audit


def filter_historical_plays(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if plays.empty:
        return plays
    game_column = "gameId" if "gameId" in plays.columns else "game_id"
    game_ids = pd.to_numeric(plays[game_column], errors="coerce")
    allowed_ids = set(games["cfbd_id"].astype("int64"))
    mask = plays["_cache_season"].isin(HISTORICAL_YEARS) & game_ids.isin(allowed_ids)
    return plays[mask].copy()


def current_team_classifications() -> dict[str, str]:
    teams = pd.read_csv(TEAMS_PATH, low_memory=False)
    teams["classification"] = teams["classification"].astype(str).str.strip().str.casefold()
    return {
        normalise_team(name): classification
        for name, classification in zip(teams["team_name"], teams["classification"])
        if classification in {"fbs", "fcs"}
    }


def filter_current_plays(plays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if plays.empty:
        return plays, {"current_games": 0, "unclassified_current_games": 0}

    game_column = "gameId" if "gameId" in plays.columns else "game_id"
    if game_column not in plays.columns:
        raise RuntimeError("2026 cached plays do not contain gameId")
    if "offense" not in plays.columns or "defense" not in plays.columns:
        raise RuntimeError("2026 cached plays do not contain offense/defense team names")

    classifications = current_team_classifications()
    offense_class = plays["offense"].map(lambda value: classifications.get(normalise_team(value)))
    defense_class = plays["defense"].map(lambda value: classifications.get(normalise_team(value)))
    allowed_row = (
        ((offense_class == "fbs") & defense_class.isin({"fbs", "fcs"}))
        | ((defense_class == "fbs") & offense_class.isin({"fbs", "fcs"}))
    )

    numeric_game_ids = pd.to_numeric(plays[game_column], errors="coerce")
    allowed_game_ids = set(numeric_game_ids[allowed_row].dropna().astype("int64"))
    all_game_ids = set(numeric_game_ids.dropna().astype("int64"))
    filtered = plays[numeric_game_ids.isin(allowed_game_ids)].copy()
    audit = {
        "current_games": int(len(allowed_game_ids)),
        "unclassified_or_excluded_current_games": int(len(all_game_ids - allowed_game_ids)),
    }
    return filtered, audit


def build_training_rows(
    games: pd.DataFrame,
    historical_team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, DriveStateEngine, int]:
    team_game_data = historical_team_games.copy()
    team_game_data["game_id_key"] = pd.to_numeric(team_game_data["game_id"], errors="coerce")

    ordered_games = games.copy()
    ordered_games["start_date"] = pd.to_datetime(ordered_games["start_date"], utc=True, errors="coerce")
    ordered_games = ordered_games.sort_values(["season", "start_date", "cfbd_id"])

    engine = DriveStateEngine()
    rows: list[dict[str, object]] = []
    skipped = 0
    last_season: int | None = None

    for game in ordered_games.itertuples(index=False):
        pair = team_game_data[team_game_data["game_id_key"].eq(game.cfbd_id)]
        home = pair[pair["offense"].eq(game.home_team)]
        away = pair[pair["offense"].eq(game.away_team)]
        if len(home) != 1 or len(away) != 1:
            skipped += 1
            continue

        season = int(game.season)
        if last_season is not None and season != last_season:
            engine.regress_season()
        last_season = season

        home_state = engine.snapshot(game.home_team)
        away_state = engine.snapshot(game.away_team)
        features = matchup_features(home_state, away_state)
        base: dict[str, object] = {
            "game_id": int(game.cfbd_id),
            "season": season,
            "week": game.week,
            "start_date": game.start_date,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_points": game.home_points,
            "away_points": game.away_points,
            "matchup_class": game.matchup_class,
        }
        base.update({f"home_{key}": value for key, value in home_state.items()})
        base.update({f"away_{key}": value for key, value in away_state.items()})
        base.update(features)
        rows.append(base)

        engine.update_game(
            game.home_team,
            game.away_team,
            home.iloc[0].to_dict(),
            away.iloc[0].to_dict(),
        )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("No valid historical training games remained after drive matching")
    if dataset["game_id"].duplicated().any():
        raise RuntimeError("Duplicate game IDs were created in the historical training dataset")

    dataset["actual_home_margin"] = (
        pd.to_numeric(dataset["home_points"], errors="coerce")
        - pd.to_numeric(dataset["away_points"], errors="coerce")
    )
    dataset["actual_total"] = (
        pd.to_numeric(dataset["home_points"], errors="coerce")
        + pd.to_numeric(dataset["away_points"], errors="coerce")
    )
    return dataset, engine, skipped


def frozen_history_is_valid() -> bool:
    required = (
        FROZEN_DRIVES,
        FROZEN_TEAM_GAMES,
        FROZEN_TRAINING,
        FROZEN_STATES,
        FROZEN_MANIFEST,
    )
    if not all(path.exists() and path.stat().st_size > 2 for path in required):
        return False
    try:
        manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("freeze_schema_version") == FREEZE_SCHEMA_VERSION


def build_and_freeze_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    games, game_audit = eligible_historical_games()
    all_plays = load_plays(HISTORICAL_YEARS)
    historical_plays = filter_historical_plays(all_plays, games)
    historical_drives = drive_table(historical_plays)
    historical_team_games = team_games(historical_drives)
    training_rows, engine, skipped = build_training_rows(games, historical_team_games)

    FROZEN.mkdir(parents=True, exist_ok=True)
    historical_drives.to_csv(FROZEN_DRIVES, index=False)
    historical_team_games.to_csv(FROZEN_TEAM_GAMES, index=False)
    training_rows.to_csv(FROZEN_TRAINING, index=False)
    save_states(engine, FROZEN_STATES)

    manifest = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "years": [2021, 2022, 2023, 2024, 2025],
        "universe": ["FBS_vs_FBS", "FBS_vs_FCS"],
        **game_audit,
        "games_without_complete_drive_pairs": int(skipped),
        "training_games": int(len(training_rows)),
        "historical_drives": int(len(historical_drives)),
        "historical_team_games": int(len(historical_team_games)),
    }
    FROZEN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Frozen historical base: "
        f"{len(training_rows):,} training games "
        f"({game_audit['fbs_vs_fbs']:,} eligible FBS-FBS; "
        f"{game_audit['fbs_vs_fcs']:,} eligible FBS-FCS; "
        f"{skipped:,} skipped for incomplete drive pairs)."
    )
    return historical_drives, historical_team_games, training_rows, manifest


def load_frozen_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    historical_drives = pd.read_csv(FROZEN_DRIVES, low_memory=False)
    historical_team_games = pd.read_csv(FROZEN_TEAM_GAMES, low_memory=False)
    training_rows = pd.read_csv(FROZEN_TRAINING, low_memory=False)
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    print(
        "Reusing frozen 2021-2025 base: "
        f"{len(training_rows):,} training games and {len(historical_drives):,} drives."
    )
    return historical_drives, historical_team_games, training_rows, manifest


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.rebuild_history or not frozen_history_is_valid():
        historical_drives, historical_team_games, training_rows, historical_audit = (
            build_and_freeze_history()
        )
    else:
        historical_drives, historical_team_games, training_rows, historical_audit = (
            load_frozen_history()
        )

    current_plays = load_plays({CURRENT_YEAR})
    current_plays, current_audit = filter_current_plays(current_plays)
    current_drives = drive_table(current_plays)
    current_team_games = team_games(current_drives)

    all_drives = pd.concat([historical_drives, current_drives], ignore_index=True, sort=False)
    all_team_games = pd.concat(
        [historical_team_games, current_team_games], ignore_index=True, sort=False
    )

    all_drives.to_csv(OUT / "drives.csv", index=False)
    all_team_games.to_csv(OUT / "team_game_drive_metrics.csv", index=False)
    training_rows.to_csv(OUT / "v3_drive_training_rows.csv", index=False)
    shutil.copy2(FROZEN_STATES, OUT / "v3_end_2025_drive_states.json")

    audit = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "historical": historical_audit,
        "current": current_audit,
        "combined_drives": int(len(all_drives)),
        "combined_team_games": int(len(all_team_games)),
        "historical_training_games": int(len(training_rows)),
        "current_drives": int(len(current_drives)),
        "current_team_games": int(len(current_team_games)),
    }
    (OUT / "v3_drive_dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(
        f"Saved {len(all_drives):,} allowed drives, "
        f"{len(all_team_games):,} allowed team-games, "
        f"and {len(training_rows):,} frozen chronological training games."
    )
    print(
        f"2026 refresh: {current_audit['current_games']:,} FBS-involved games, "
        f"{len(current_drives):,} drives."
    )
    print("Next: python -m jobs.train_v3_drive_models")


if __name__ == "__main__":
    main()
