from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.v3_settings import SETTINGS
from engine.v3_possession_engine import (
    reconstruct_drives,
    aggregate_team_games,
)
from engine.v3_team_state import ChronologicalStateEngine
from engine.v3_matchup_engine import build_matchup


# =============================================================================
# V3 CANONICAL BUILD
# =============================================================================
#
# Pipeline:
#
# CFBD /games
#     -> authoritative game universe/results
#
# CFBD /plays
#     -> possession reconstruction
#     -> PBP integrity gate
#     -> clean possession/team-game metrics
#
# clean possession metrics
#     -> opponent-adjusted chronological team state
#
# authoritative results
#     -> ELO/result updates even when PBP is quarantined
#
#
# IMPORTANT
# ---------
#
# Quarantining PBP does NOT delete a football game.
#
# The authoritative:
#   - teams
#   - kickoff
#   - score
#   - winner/result
#
# remain valid.
#
# Quarantine only prevents unreliable possession-derived observations from
# entering future offensive/defensive team state.
# =============================================================================


# =============================================================================
# PBP INTEGRITY CONFIGURATION
# =============================================================================

FRAGMENTED_MIN_DRIVE_CONTAINERS = 10
FRAGMENTED_ONE_PLAY_SHARE = 0.30

INCOMPLETE_MIN_DRIVE_CONTAINERS = 10
INCOMPLETE_MIN_TEAM_DRIVES = 5
INCOMPLETE_MIN_MAX_PERIOD = 3

EXTREME_MAX_TEAM_DRIVES = 20

# Audited historical baseline after possession reconstruction fixes:
#
# approximately 0.5% of games quarantined.
#
# 2% is intentionally generous. If corruption suddenly rises above this,
# assume a source/provider/schema problem and stop the build.
MAX_QUARANTINE_RATE = 0.02
MAX_QUARANTINE_GAMES = 100


# =============================================================================
# RAW DATA LOADING
# =============================================================================


def load_jsons(pattern):
    rows = []
    files = sorted(
        SETTINGS.raw_root.glob(pattern)
    )

    for p in files:
        try:
            x = json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )

            rows.extend(
                x
                if isinstance(x, list)
                else [x]
            )

        except Exception as e:
            print(
                "SKIP",
                p,
                e,
            )

    return rows, files


# =============================================================================
# AUTHORITATIVE GAME UNIVERSE
# =============================================================================


def canonical_games():
    rows, _ = load_jsons(
        "games/*.json"
    )

    g = pd.json_normalize(rows)

    ren = {
        "id": "game_id",
        "season": "season",
        "week": "week",
        "startDate": "start_date",
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "homePoints": "home_points",
        "awayPoints": "away_points",
        "neutralSite": "neutral_site",
        "seasonType": "season_type",
        "homeClassification": "home_classification",
        "awayClassification": "away_classification",
    }

    g = g.rename(
        columns={
            k: v
            for k, v in ren.items()
            if k in g
        }
    )

    need = [
        "game_id",
        "season",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
    ]

    miss = [
        c
        for c in need
        if c not in g
    ]

    if miss:
        raise ValueError(
            f"/games schema missing {miss}"
        )

    g["game_id"] = pd.to_numeric(
        g["game_id"],
        errors="coerce",
    )

    g = g.dropna(
        subset=[
            "game_id",
            "home_team",
            "away_team",
        ]
    )

    g["game_id"] = (
        g["game_id"]
        .astype("int64")
    )

    g["season"] = pd.to_numeric(
        g["season"],
        errors="coerce",
    ).astype("Int64")

    g["week"] = pd.to_numeric(
        g["week"],
        errors="coerce",
    ).astype("Int64")

    g["start_date"] = pd.to_datetime(
        g["start_date"],
        utc=True,
        errors="coerce",
    )

    g["home_points"] = pd.to_numeric(
        g["home_points"],
        errors="coerce",
    )

    g["away_points"] = pd.to_numeric(
        g["away_points"],
        errors="coerce",
    )

    # Do not trust the endpoint classification query by itself.  Enforce the
    # production universe in the application layer using the season-specific
    # classifications returned on each game.
    for column in ("home_classification", "away_classification"):
        if column not in g.columns:
            raise ValueError(f"/games schema missing {column}")
        g[column] = g[column].astype(str).str.strip().str.casefold()

    allowed = (
        (
            g["home_classification"].eq("fbs")
            & g["away_classification"].isin({"fbs", "fcs"})
        )
        |
        (
            g["away_classification"].eq("fbs")
            & g["home_classification"].isin({"fbs", "fcs"})
        )
    )
    g = g[allowed].copy()

    if "neutral_site" in g.columns:
        g["neutral_site"] = (
            g["neutral_site"]
            .fillna(False)
            .astype(bool)
        )
    else:
        g["neutral_site"] = False

    return (
        g.drop_duplicates(
            "game_id",
            keep="last",
        )
        .sort_values(
            [
                "season",
                "start_date",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# PBP INTEGRITY GATE
# =============================================================================


def build_pbp_integrity_audit(
    drives: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one structural PBP integrity record per reconstructed game.

    Four independent corruption/failure modes are detected:

    1. FRAGMENTED
       A normal possession has been split into many tiny drive containers.

    2. INCOMPLETE
       Only part of the game's PBP is available.

    3. EXTREME_POSSESSIONS
       Reconstructed team possession counts remain implausibly large.

    4. BAD_TEAM_STRUCTURE
       PBP does not contain exactly the two canonical teams as both
       offensive and defensive participants.

    No hard-coded game IDs are used.
    """

    if drives.empty:
        raise RuntimeError(
            "STOP: no reconstructed drives available "
            "for PBP integrity audit."
        )

    required = {
        "game_id",
        "drive_id",
        "offense",
        "defense",
        "plays",
        "result",
        "period_start",
        "period_end",
    }

    missing = sorted(
        required.difference(
            drives.columns
        )
    )

    if missing:
        raise ValueError(
            "Drive data missing integrity columns: "
            f"{missing}"
        )

    d = drives.copy()

    d["game_id"] = pd.to_numeric(
        d["game_id"],
        errors="coerce",
    )

    d["plays"] = pd.to_numeric(
        d["plays"],
        errors="coerce",
    ).fillna(0)

    d["period_start"] = pd.to_numeric(
        d["period_start"],
        errors="coerce",
    )

    d["period_end"] = pd.to_numeric(
        d["period_end"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Basic game-level possession structure
    # -------------------------------------------------------------------------

    game_level = (
        d.groupby("game_id")
        .agg(
            drive_containers=(
                "drive_id",
                "count",
            ),
            offense_team_count=(
                "offense",
                "nunique",
            ),
            defense_team_count=(
                "defense",
                "nunique",
            ),
            min_period=(
                "period_start",
                "min",
            ),
            max_period=(
                "period_end",
                "max",
            ),
            one_play_drives=(
                "plays",
                lambda x: int(
                    (x <= 1).sum()
                ),
            ),
            other_drives=(
                "result",
                lambda x: int(
                    (x == "OTHER").sum()
                ),
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Possessions by offense
    # -------------------------------------------------------------------------

    team_counts = (
        d.groupby(
            [
                "game_id",
                "offense",
            ]
        )
        .size()
        .rename("drives")
        .reset_index()
    )

    possession_range = (
        team_counts.groupby(
            "game_id"
        )["drives"]
        .agg(
            ["min", "max"]
        )
        .rename(
            columns={
                "min": "min_team_drives",
                "max": "max_team_drives",
            }
        )
        .reset_index()
    )

    audit = game_level.merge(
        possession_range,
        on="game_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Actual PBP team sets
    # -------------------------------------------------------------------------

    offense_sets = (
        d.groupby("game_id")["offense"]
        .apply(
            lambda x: frozenset(
                str(v).strip()
                for v in x.dropna().unique()
            )
        )
        .rename("pbp_offense_set")
        .reset_index()
    )

    defense_sets = (
        d.groupby("game_id")["defense"]
        .apply(
            lambda x: frozenset(
                str(v).strip()
                for v in x.dropna().unique()
            )
        )
        .rename("pbp_defense_set")
        .reset_index()
    )

    audit = audit.merge(
        offense_sets,
        on="game_id",
        how="left",
    )

    audit = audit.merge(
        defense_sets,
        on="game_id",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Canonical metadata
    # -------------------------------------------------------------------------

    meta_cols = [
        "game_id",
        "season",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
    ]

    meta = (
        games[meta_cols]
        .drop_duplicates("game_id")
        .copy()
    )

    audit = audit.merge(
        meta,
        on="game_id",
        how="left",
    )

    audit["canonical_team_set"] = (
        audit.apply(
            lambda r: frozenset(
                {
                    str(r["home_team"]).strip(),
                    str(r["away_team"]).strip(),
                }
            ),
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # Shares
    # -------------------------------------------------------------------------

    audit["one_play_share"] = (
        audit["one_play_drives"]
        / audit["drive_containers"].clip(
            lower=1
        )
    )

    audit["other_share"] = (
        audit["other_drives"]
        / audit["drive_containers"].clip(
            lower=1
        )
    )

    # =========================================================================
    # RULE 1: FRAGMENTED
    # =========================================================================

    audit["fragmented"] = (
        (
            audit["drive_containers"]
            >= FRAGMENTED_MIN_DRIVE_CONTAINERS
        )
        &
        (
            audit["one_play_share"]
            >= FRAGMENTED_ONE_PLAY_SHARE
        )
    )

    # =========================================================================
    # RULE 2: INCOMPLETE
    # =========================================================================

    audit["incomplete"] = (
        (
            audit["max_period"]
            < INCOMPLETE_MIN_MAX_PERIOD
        )
        |
        (
            audit["drive_containers"]
            < INCOMPLETE_MIN_DRIVE_CONTAINERS
        )
        |
        (
            audit["min_team_drives"]
            < INCOMPLETE_MIN_TEAM_DRIVES
        )
    )

    # =========================================================================
    # RULE 3: EXTREME POSSESSION COUNT
    # =========================================================================

    audit["extreme_possessions"] = (
        audit["max_team_drives"]
        > EXTREME_MAX_TEAM_DRIVES
    )

    # =========================================================================
    # RULE 4: BAD TEAM STRUCTURE
    # =========================================================================
    #
    # A completed football game must contain exactly the canonical two teams
    # in both offensive and defensive PBP roles.
    #
    # This catches source failures such as:
    #
    # 401645328 Army 37 - Rice 14
    #
    # where all reconstructed drives were labelled:
    #
    #     offense = Rice
    #     defense = Army
    #
    # despite the game containing 18 apparent possessions and four periods.
    # =========================================================================

    audit["bad_offense_count"] = (
        audit["offense_team_count"]
        != 2
    )

    audit["bad_defense_count"] = (
        audit["defense_team_count"]
        != 2
    )

    audit["offense_team_mismatch"] = (
        audit.apply(
            lambda r: (
                r["pbp_offense_set"]
                != r["canonical_team_set"]
            ),
            axis=1,
        )
    )

    audit["defense_team_mismatch"] = (
        audit.apply(
            lambda r: (
                r["pbp_defense_set"]
                != r["canonical_team_set"]
            ),
            axis=1,
        )
    )

    audit["bad_team_structure"] = (
        audit["bad_offense_count"]
        |
        audit["bad_defense_count"]
        |
        audit["offense_team_mismatch"]
        |
        audit["defense_team_mismatch"]
    )

    # =========================================================================
    # FINAL QUARANTINE
    # =========================================================================

    audit["quarantine"] = (
        audit["fragmented"]
        |
        audit["incomplete"]
        |
        audit["extreme_possessions"]
        |
        audit["bad_team_structure"]
    )

    def reason(row):
        reasons = []

        if bool(
            row["fragmented"]
        ):
            reasons.append(
                "FRAGMENTED"
            )

        if bool(
            row["incomplete"]
        ):
            reasons.append(
                "INCOMPLETE"
            )

        if bool(
            row["extreme_possessions"]
        ):
            reasons.append(
                "EXTREME_POSSESSIONS"
            )

        if bool(
            row["bad_team_structure"]
        ):
            reasons.append(
                "BAD_TEAM_STRUCTURE"
            )

        return (
            "|".join(reasons)
            if reasons
            else "HEALTHY"
        )

    audit["quarantine_reason"] = (
        audit.apply(
            reason,
            axis=1,
        )
    )

    # Convert sets into readable audit text before CSV persistence.
    audit["pbp_offense_teams"] = (
        audit["pbp_offense_set"]
        .apply(
            lambda x: " | ".join(
                sorted(x)
            )
        )
    )

    audit["pbp_defense_teams"] = (
        audit["pbp_defense_set"]
        .apply(
            lambda x: " | ".join(
                sorted(x)
            )
        )
    )

    audit = audit.drop(
        columns=[
            "pbp_offense_set",
            "pbp_defense_set",
            "canonical_team_set",
        ]
    )

    preferred = [
        "game_id",
        "season",
        "week",
        "start_date",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "drive_containers",
        "offense_team_count",
        "defense_team_count",
        "pbp_offense_teams",
        "pbp_defense_teams",
        "min_period",
        "max_period",
        "min_team_drives",
        "max_team_drives",
        "one_play_drives",
        "one_play_share",
        "other_drives",
        "other_share",
        "fragmented",
        "incomplete",
        "extreme_possessions",
        "bad_offense_count",
        "bad_defense_count",
        "offense_team_mismatch",
        "defense_team_mismatch",
        "bad_team_structure",
        "quarantine",
        "quarantine_reason",
    ]

    extra = [
        c
        for c in audit.columns
        if c not in preferred
    ]

    audit = audit[
        preferred + extra
    ]

    return (
        audit.sort_values(
            [
                "quarantine",
                "season",
                "game_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# INTEGRITY VALIDATION
# =============================================================================


def validate_pbp_integrity(
    audit: pd.DataFrame,
) -> None:

    total = len(audit)

    if total == 0:
        raise RuntimeError(
            "STOP: PBP integrity audit "
            "contains zero games."
        )

    quarantined = int(
        audit["quarantine"].sum()
    )

    rate = (
        quarantined
        / total
    )

    if (
        quarantined
        > MAX_QUARANTINE_GAMES
    ):
        raise RuntimeError(
            "STOP: PBP quarantine count "
            "is unexpectedly high. "
            f"{quarantined:,}/{total:,} "
            "games were flagged. "
            "Investigate source structure "
            "before training."
        )

    if rate > MAX_QUARANTINE_RATE:
        raise RuntimeError(
            "STOP: PBP quarantine rate "
            "is unexpectedly high. "
            f"{quarantined:,}/{total:,} "
            f"({rate * 100:.3f}%) "
            "games were flagged. "
            "Investigate source structure "
            "before training."
        )


# =============================================================================
# INTEGRITY REPORT
# =============================================================================


def print_pbp_integrity_report(
    audit: pd.DataFrame,
) -> None:

    total = len(audit)

    fragmented = int(
        audit["fragmented"].sum()
    )

    incomplete = int(
        audit["incomplete"].sum()
    )

    extreme = int(
        audit["extreme_possessions"].sum()
    )

    bad_team_structure = int(
        audit["bad_team_structure"].sum()
    )

    quarantined = int(
        audit["quarantine"].sum()
    )

    healthy = (
        total
        - quarantined
    )

    rate = (
        quarantined / total
        if total
        else np.nan
    )

    print()
    print("=" * 75)
    print("V3 PBP INTEGRITY GATE")
    print("=" * 75)

    print(
        f"Audited PBP games:       "
        f"{total:,}"
    )

    print(
        f"Healthy PBP games:       "
        f"{healthy:,}"
    )

    print(
        f"Quarantined PBP games:   "
        f"{quarantined:,}"
    )

    print(
        f"Quarantine rate:         "
        f"{rate * 100:.3f}%"
    )

    print()

    print(
        f"Fragmented:              "
        f"{fragmented:,}"
    )

    print(
        f"Incomplete:              "
        f"{incomplete:,}"
    )

    print(
        f"Extreme possessions:     "
        f"{extreme:,}"
    )

    print(
        f"Bad team structure:      "
        f"{bad_team_structure:,}"
    )

    if quarantined:

        print()
        print(
            "QUARANTINE REASONS:"
        )

        print(
            audit.loc[
                audit["quarantine"],
                "quarantine_reason",
            ]
            .value_counts()
            .to_string()
        )

        print()
        print(
            "QUARANTINED BY SEASON:"
        )

        print(
            audit.loc[
                audit["quarantine"]
            ]
            .groupby("season")
            .size()
            .to_string()
        )

    healthy_rows = audit[
        ~audit["quarantine"]
    ].copy()

    if not healthy_rows.empty:

        team_drive_values = pd.concat(
            [
                healthy_rows[
                    "min_team_drives"
                ],
                healthy_rows[
                    "max_team_drives"
                ],
            ],
            ignore_index=True,
        )

        print()
        print(
            "HEALTHY TEAM POSSESSION "
            "DISTRIBUTION:"
        )

        print(
            team_drive_values.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            .round(3)
            .to_string()
        )

    print("=" * 75)
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():

    SETTINGS.processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    SETTINGS.audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # CANONICAL GAMES
    # =========================================================================

    games = canonical_games()

    print(
        "Canonical games by season:"
    )

    print(
        games.groupby("season")
        .game_id
        .nunique()
        .to_string()
    )

    # =========================================================================
    # PBP / POSSESSION RECONSTRUCTION
    # =========================================================================

    plays, files = load_jsons(
        "plays/*/*.json"
    )

    p = pd.json_normalize(
        plays
    )

    print(
        f"Loaded {len(p):,} plays "
        f"from {len(files)} files"
    )

    drives = reconstruct_drives(
        p
    )

    # The /plays endpoint has historically returned lower-division-only rows
    # even when classification=fbs was requested.  Filter by authoritative
    # canonical game IDs before any drive data can be persisted or modelled.
    canonical_game_ids = set(games["game_id"].astype("int64"))
    drives = drives[drives["game_id"].isin(canonical_game_ids)].copy()

    tg = aggregate_team_games(
        drives
    )

    print(
        f"Reconstructed "
        f"{len(drives):,} drives "
        f"and {len(tg):,} "
        f"team-game rows"
    )

    # =========================================================================
    # INTEGRITY GATE
    # =========================================================================

    pbp_audit = (
        build_pbp_integrity_audit(
            drives=drives,
            games=games,
        )
    )

    print_pbp_integrity_report(
        pbp_audit
    )

    # Persist diagnostics before the safety gate so failures remain auditable.
    pbp_audit.to_csv(
        SETTINGS.audit_root
        / "v3_pbp_integrity_audit.csv",
        index=False,
    )

    quarantine = (
        pbp_audit[
            pbp_audit["quarantine"]
        ]
        .copy()
    )

    quarantine.to_csv(
        SETTINGS.audit_root
        / "v3_pbp_quarantine.csv",
        index=False,
    )

    validate_pbp_integrity(
        pbp_audit
    )

    quarantine_ids = set(
        quarantine["game_id"]
        .astype("int64")
        .tolist()
    )

    healthy_pbp_ids = set(
        pbp_audit.loc[
            ~pbp_audit["quarantine"],
            "game_id",
        ]
        .astype("int64")
        .tolist()
    )

    # =========================================================================
    # PERSIST RAW + CLEAN DERIVED DATA
    # =========================================================================
    #
    # Raw reconstructed datasets remain available for auditing.
    #
    # Clean datasets are the only possession-derived inputs allowed into the
    # chronological state engine.
    # =========================================================================

    games.to_csv(
        SETTINGS.processed_root
        / "games.csv",
        index=False,
    )

    drives.to_csv(
        SETTINGS.processed_root
        / "drives.csv",
        index=False,
    )

    tg.to_csv(
        SETTINGS.processed_root
        / "team_game_drive_metrics.csv",
        index=False,
    )

    clean_drives = (
        drives[
            drives["game_id"].isin(
                healthy_pbp_ids
            )
        ]
        .copy()
    )

    clean_tg = (
        tg[
            tg["game_id"].isin(
                healthy_pbp_ids
            )
        ]
        .copy()
    )

    clean_drives.to_csv(
        SETTINGS.processed_root
        / "drives_clean.csv",
        index=False,
    )

    clean_tg.to_csv(
        SETTINGS.processed_root
        / "team_game_drive_metrics_clean.csv",
        index=False,
    )

    # =========================================================================
    # STRONG CLEAN TEAM-GAME INVARIANT
    # =========================================================================
    #
    # Every healthy game must produce exactly TWO team-game aggregate rows.
    #
    # This would have caught Army-Rice automatically even without inspecting
    # the detailed game.
    # =========================================================================

    clean_counts = (
        clean_tg.groupby(
            "game_id"
        )
        .size()
    )

    missing_clean_games = (
        healthy_pbp_ids
        - set(
            clean_counts.index
            .astype("int64")
        )
    )

    wrong_clean_counts = (
        clean_counts[
            clean_counts != 2
        ]
    )

    if missing_clean_games:
        raise RuntimeError(
            "STOP: healthy PBP games "
            "missing clean team-game rows: "
            f"{sorted(missing_clean_games)}"
        )

    if not wrong_clean_counts.empty:
        raise RuntimeError(
            "STOP: healthy PBP games "
            "do not contain exactly two "
            "team-game aggregate rows:\n"
            f"{wrong_clean_counts.to_string()}"
        )

    # =========================================================================
    # COMPLETED GAME UNIVERSE SANITY CHECK
    # =========================================================================

    completed_mask = (
        games["home_points"].notna()
        &
        games["away_points"].notna()
    )

    counts = (
        games.loc[
            completed_mask
        ]
        .groupby("season")
        .game_id
        .nunique()
    )

    bad = {
        int(y): int(n)
        for y, n in counts.items()
        if (
            y < SETTINGS.live_season
            and (
                n
                > SETTINGS.max_reasonable_games_per_season
                or
                n
                < SETTINGS.min_reasonable_games_per_full_season
            )
        )
    }

    if bad:
        raise RuntimeError(
            "STOP: implausible "
            "completed-game universe: "
            f"{bad}. Training blocked."
        )

    # =========================================================================
    # CHRONOLOGICAL STATE CONSTRUCTION
    # =========================================================================

    tgi = clean_tg.set_index(
        [
            "game_id",
            "team",
        ]
    )

    engine = (
        ChronologicalStateEngine()
    )

    rows = []

    last_season = None

    completed = (
        games[
            completed_mask
        ]
        .copy()
    )

    # Equal kickoff timestamps are processed as a batch.
    #
    # Every game at a given kickoff receives its PRE-BATCH state before any
    # result from that kickoff is allowed to update the engine.
    for (
        season,
        kick,
    ), batch in completed.groupby(
        [
            "season",
            "start_date",
        ],
        sort=True,
        dropna=False,
    ):

        season = int(
            season
        )

        if (
            last_season is not None
            and season != last_season
        ):
            engine.regress_season()

        last_season = season

        # Healthy possession updates.
        pending_full = []

        # Quarantined result-only updates.
        pending_result_only = []

        for r in batch.itertuples(
            index=False
        ):

            game_id = int(
                r.game_id
            )

            hs = engine.snapshot(
                r.home_team,
                getattr(r, "home_classification", None),
            )

            as_ = engine.snapshot(
                r.away_team,
                getattr(r, "away_classification", None),
            )

            m = build_matchup(
                hs,
                as_,
                bool(
                    r.neutral_site
                ),
            )

            row = {
                "game_id": game_id,
                "season": season,
                "week": (
                    int(r.week)
                    if pd.notna(r.week)
                    else -1
                ),
                "start_date": (
                    r.start_date
                ),
                "home_team": (
                    r.home_team
                ),
                "away_team": (
                    r.away_team
                ),
                "neutral_site": bool(
                    r.neutral_site
                ),
                "home_points": float(
                    r.home_points
                ),
                "away_points": float(
                    r.away_points
                ),
                "pbp_quarantined": (
                    game_id
                    in quarantine_ids
                ),
            }

            row.update(
                {
                    "home_state_" + k: v
                    for k, v
                    in hs.items()
                }
            )

            row.update(
                {
                    "away_state_" + k: v
                    for k, v
                    in as_.items()
                }
            )

            row.update(m)

            rows.append(
                row
            )

            # -----------------------------------------------------------------
            # QUARANTINED PBP
            # -----------------------------------------------------------------
            #
            # Keep the authoritative result for ELO/context.
            #
            # Do NOT pass possession observations to the engine.
            # -----------------------------------------------------------------

            if game_id in quarantine_ids:

                pending_result_only.append(
                    r
                )

                continue

            # -----------------------------------------------------------------
            # HEALTHY PBP
            # -----------------------------------------------------------------

            try:
                h = (
                    tgi.loc[
                        (
                            game_id,
                            r.home_team,
                        )
                    ]
                    .to_dict()
                )

                a = (
                    tgi.loc[
                        (
                            game_id,
                            r.away_team,
                        )
                    ]
                    .to_dict()
                )

            except KeyError as e:
                # At this point the integrity gate has explicitly certified
                # the game healthy and the clean-team-game invariant has
                # certified two rows exist.
                #
                # A missing canonical team therefore indicates a real pipeline
                # inconsistency and must stop the build.
                raise RuntimeError(
                    "STOP: healthy game missing "
                    "canonical team-game observation. "
                    f"game_id={game_id}, "
                    f"home={r.home_team}, "
                    f"away={r.away_team}"
                ) from e

            pending_full.append(
                (
                    r,
                    h,
                    a,
                )
            )

        # =====================================================================
        # APPLY BATCH UPDATES
        # =====================================================================

        for r, h, a in pending_full:

            engine.update_game(
                r.home_team,
                r.away_team,
                h,
                a,
                float(
                    r.home_points
                ),
                float(
                    r.away_points
                ),
                bool(
                    r.neutral_site
                ),
                home_classification=getattr(r, "home_classification", None),
                away_classification=getattr(r, "away_classification", None),
            )

        for r in pending_result_only:

            engine.update_result_only(
                r.home_team,
                r.away_team,
                float(
                    r.home_points
                ),
                float(
                    r.away_points
                ),
                bool(
                    r.neutral_site
                ),
                home_classification=getattr(r, "home_classification", None),
                away_classification=getattr(r, "away_classification", None),
            )

    # =========================================================================
    # CHRONOLOGICAL MODELLING DATASET
    # =========================================================================

    ds = pd.DataFrame(
        rows
    )

    ds["actual_margin"] = (
        ds["home_points"]
        - ds["away_points"]
    )

    ds["actual_total"] = (
        ds["home_points"]
        + ds["away_points"]
    )

    ds["home_win"] = (
        ds["actual_margin"]
        > 0
    ).astype(int)

    ds.to_csv(
        SETTINGS.processed_root
        / "chronological_training_rows.csv",
        index=False,
    )

    # =========================================================================
    # COVERAGE
    # =========================================================================

    coverage = (
        ds.groupby("season")
        .agg(
            games=(
                "game_id",
                "nunique",
            ),
            home_score=(
                "home_points",
                "mean",
            ),
            away_score=(
                "away_points",
                "mean",
            ),
            total=(
                "actual_total",
                "mean",
            ),
            pbp_quarantined=(
                "pbp_quarantined",
                "sum",
            ),
        )
    )

    # =========================================================================
    # AUDIT JSON
    # =========================================================================

    quarantined_by_season = {}

    if not quarantine.empty:

        qseason = (
            quarantine.dropna(
                subset=["season"]
            )
            .groupby("season")
            .size()
        )

        quarantined_by_season = {
            str(int(k)): int(v)
            for k, v
            in qseason.items()
        }

    audit = {
        "plays": int(
            len(p)
        ),
        "drives_raw": int(
            len(drives)
        ),
        "drives_clean": int(
            len(clean_drives)
        ),
        "team_games_raw": int(
            len(tg)
        ),
        "team_games_clean": int(
            len(clean_tg)
        ),
        "training_rows": int(
            len(ds)
        ),
        "season_counts": {
            str(k): int(v)
            for k, v
            in (
                ds.groupby("season")
                .game_id
                .nunique()
                .items()
            )
        },
        "drive_coverage_games_raw": int(
            tg.game_id.nunique()
        ),
        "drive_coverage_games_clean": int(
            clean_tg.game_id.nunique()
        ),
        "pbp_integrity_games": int(
            len(pbp_audit)
        ),
        "pbp_healthy_games": int(
            (
                ~pbp_audit[
                    "quarantine"
                ]
            ).sum()
        ),
        "pbp_quarantined_games": int(
            pbp_audit[
                "quarantine"
            ].sum()
        ),
        "pbp_quarantine_rate": float(
            pbp_audit[
                "quarantine"
            ].mean()
        ),
        "pbp_fragmented_games": int(
            pbp_audit[
                "fragmented"
            ].sum()
        ),
        "pbp_incomplete_games": int(
            pbp_audit[
                "incomplete"
            ].sum()
        ),
        "pbp_extreme_possession_games": int(
            pbp_audit[
                "extreme_possessions"
            ].sum()
        ),
        "pbp_bad_team_structure_games": int(
            pbp_audit[
                "bad_team_structure"
            ].sum()
        ),
        "pbp_quarantined_by_season": (
            quarantined_by_season
        ),
        "duplicate_game_ids": int(
            ds["game_id"]
            .duplicated()
            .sum()
        ),
    }

    (
        SETTINGS.audit_root
        / "v3_canonical_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print(
        "CANONICAL COVERAGE:"
    )

    print(
        coverage.to_string()
    )

    print()
    print(
        "CANONICAL AUDIT:"
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print()

    print(
        "PBP integrity audit:",
        SETTINGS.audit_root
        / "v3_pbp_integrity_audit.csv",
    )

    print(
        "PBP quarantine:",
        SETTINGS.audit_root
        / "v3_pbp_quarantine.csv",
    )

    print(
        "Clean drive metrics:",
        SETTINGS.processed_root
        / "team_game_drive_metrics_clean.csv",
    )

    print()

    print(
        "Canonical build complete."
    )

    print(
        "Do NOT train until the "
        "final clean-build audit "
        "has been reviewed."
    )


if __name__ == "__main__":
    main()
