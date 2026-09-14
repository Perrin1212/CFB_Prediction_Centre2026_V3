from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.cfbd_client import CFBDClient


def is_week_zero_bad_request(exc: Exception, week: int) -> bool:
    """
    CFBD does not consistently support week=0 across every historical
    endpoint/year combination.

    Only suppress a 400 response when the requested week is exactly 0.
    All other errors remain fatal so genuine acquisition problems are
    not silently hidden.
    """
    if week != 0:
        return False

    text = str(exc).lower()

    return (
        "400" in text
        and (
            "bad request" in text
            or "request failed" in text
        )
    )


def fetch_week_data(
    client: CFBDClient,
    year: int,
    week: int,
    force: bool,
) -> tuple[list[Any], list[Any]]:
    """
    Fetch play-by-play and team-game box data for one season/week.

    Week 0 is treated specially because some CFBD endpoints reject it
    for historical seasons even though our global season configuration
    includes week 0.

    A week-0 400 becomes an empty dataset. Any other API failure is
    raised immediately.
    """

    try:
        plays = client.get_plays(year, week, force)
    except Exception as exc:
        if is_week_zero_bad_request(exc, week):
            print(
                f"  W{week:02d}: plays endpoint does not support "
                f"this week/year -> skipped"
            )
            plays = []
        else:
            raise

    try:
        stats = client.get_team_game_stats(year, week, force)
    except Exception as exc:
        if is_week_zero_bad_request(exc, week):
            print(
                f"  W{week:02d}: team-box endpoint does not support "
                f"this week/year -> skipped"
            )
            stats = []
        else:
            raise

    return plays, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire and cache the independent CFBD historical "
            "foundation used by V3."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache files and request data again.",
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Also acquire advanced box-score data for completed games.",
    )

    parser.add_argument(
        "--refresh-current",
        action="store_true",
        help=(
            "Refresh the live-season schedule/results and every week that "
            "currently contains completed games. Historical seasons remain cached."
        ),
    )

    args = parser.parse_args()

    client = CFBDClient(SETTINGS.raw_root)

    manifest: dict[str, Any] = {
        "seasons": {},
        "advanced_requested": args.advanced,
    }

    for year in SETTINGS.seasons:
        print()
        print("=" * 70)
        print(f"ACQUIRING {year}")
        print("=" * 70)

        refresh_live_year = bool(
            args.refresh_current
            and year == SETTINGS.live_season
        )
        games = client.get_games(
            year,
            force=bool(args.force or refresh_live_year),
        )

        print(f"{year}: games={len(games):,}")

        year_manifest: dict[str, Any] = {
            "games": len(games),
            "plays": 0,
            "team_game_stats": 0,
            "advanced": 0,
            "weeks_with_data": [],
            "week_zero_skipped": False,
        }

        completed_weeks = [
            int(game.get("week"))
            for game in games
            if game.get("week") is not None
            and game.get("homePoints") is not None
            and game.get("awayPoints") is not None
        ]
        latest_completed_week = max(completed_weeks, default=0)
        completed_week_set = set(completed_weeks)

        for week in SETTINGS.weeks:
            # Future live-season week files are often the literal payload [] and
            # therefore bypass the client's non-empty cache check.  Do not spend
            # two API calls per future week on every update.
            if (
                year == SETTINGS.live_season
                and (week > latest_completed_week or week not in completed_week_set)
            ):
                continue

            week_force = bool(
                args.force
                or (refresh_live_year and week <= latest_completed_week)
            )
            try:
                plays, stats = fetch_week_data(
                    client=client,
                    year=year,
                    week=week,
                    force=week_force,
                )

            except Exception as exc:
                print()
                print(
                    f"FATAL acquisition error: "
                    f"year={year} week={week}"
                )
                print(str(exc))
                print()
                print(
                    "Existing successful downloads remain cached. "
                    "Fix the error and rerun WITHOUT --force."
                )
                raise

            year_manifest["plays"] += len(plays)
            year_manifest["team_game_stats"] += len(stats)

            if week == 0 and not plays and not stats:
                year_manifest["week_zero_skipped"] = True

            if plays or stats:
                year_manifest["weeks_with_data"].append(week)

                print(
                    f"  W{week:02d}: "
                    f"plays={len(plays):,} "
                    f"team-box={len(stats):,}"
                )

        if args.advanced:
            completed = [
                game
                for game in games
                if game.get("homePoints") is not None
                and game.get("awayPoints") is not None
            ]

            print()
            print(
                f"  Advanced boxes: "
                f"{len(completed):,} completed games"
            )

            for index, game in enumerate(completed, start=1):
                game_id = game.get("id")

                if game_id is None:
                    print("  advanced skip: game missing id")
                    continue

                try:
                    client.get_advanced_box(
                        int(game_id),
                        year,
                        args.force,
                    )
                    year_manifest["advanced"] += 1

                except Exception as exc:
                    # Advanced coverage can legitimately be absent for
                    # individual historical/FCS games. Do not throw away
                    # an otherwise healthy season acquisition because one
                    # advanced box is unavailable.
                    print(
                        f"  advanced skip {game_id}: {exc}"
                    )

                if index % 100 == 0:
                    print(
                        f"  advanced "
                        f"{index:,}/{len(completed):,}"
                    )

            if completed:
                print(
                    f"  advanced complete: "
                    f"{year_manifest['advanced']:,}/"
                    f"{len(completed):,}"
                )

        manifest["seasons"][str(year)] = year_manifest

        # Write a checkpoint after every season. If acquisition is
        # interrupted later, we still retain a useful audit trail.
        manifest["new_api_calls"] = client.new_calls

        SETTINGS.audit_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            SETTINGS.audit_root
            / "v3_acquisition_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print()
        print(
            f"{year} complete: "
            f"plays={year_manifest['plays']:,}, "
            f"team-box={year_manifest['team_game_stats']:,}, "
            f"advanced={year_manifest['advanced']:,}"
        )

    manifest["new_api_calls"] = client.new_calls

    SETTINGS.audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        SETTINGS.audit_root
        / "v3_acquisition_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("V3 HISTORICAL ACQUISITION COMPLETE")
    print("=" * 70)
    print(json.dumps(manifest, indent=2))
    print(f"NEW API CALLS: {client.new_calls}")


if __name__ == "__main__":
    main()
