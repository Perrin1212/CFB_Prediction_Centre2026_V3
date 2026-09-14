from __future__ import annotations

"""
V3 CFBD possession / drive reconstruction.

Purpose
-------
Convert CFBD play-by-play into a reliable possession-level dataset.

Important CFBD behaviour
------------------------
A CFBD driveId is useful as a possession container, but the rows inside a
driveId can contain offense labels for BOTH teams.

Typical example:

    kickoff                  -> kicking team labelled offense
    kickoff return           -> receiving team labelled offense
    scrimmage plays          -> actual possession offense

Therefore the first row's offense MUST NOT be treated as the possession owner.

This engine:

1. Uses gameId + driveId as the source drive container.
2. Determines the true possession offense from football-relevant play types.
3. Prevents kickoffs, returns, timeouts and period markers from deciding
   possession ownership.
4. Uses the resolved possession offense when calculating offensive metrics.
5. Preserves drive-ending events such as punts, field goals and turnovers.
6. Separates offensive efficiency plays from transition/admin plays.
7. Produces diagnostics for mixed and ambiguous drives.
8. Performs conservative sanity checks without silently fabricating data.

No market information is used anywhere in this module.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RESULTS = (
    "TD",
    "FG",
    "TURNOVER",
    "PUNT",
    "DOWNS",
    "SAFETY",
    "END_HALF",
    "OTHER",
)


# ---------------------------------------------------------------------------
# CFBD PLAY-TYPE TAXONOMY
# ---------------------------------------------------------------------------

# Plays that strongly identify which team owns the offensive possession.
#
# These are normal offensive snaps or events arising directly from an
# offensive snap.
POSSESSION_TYPES = {
    "Rush",
    "Pass Reception",
    "Pass Incompletion",
    "Pass Completion",
    "Pass",
    "Sack",
    "Rushing Touchdown",
    "Passing Touchdown",
    "Fumble Recovery (Own)",
    "Fumble Recovery (Opponent)",
    "Fumble",
    "Interception",
    "Pass Interception Return",
    "Interception Return Touchdown",
    "Fumble Return Touchdown",
    "Safety",
    "Two Point Pass",
    "Two Point Rush",
    "Defensive 2pt Conversion",
}


# Plays that are part of offensive efficiency / drive production.
#
# Defensive returns are deliberately excluded from yards-per-drive,
# success-rate, PPA etc. They may identify a drive outcome, but they should
# not be treated as normal offensive production.
EFFICIENCY_TYPES = {
    "Rush",
    "Pass Reception",
    "Pass Incompletion",
    "Pass Completion",
    "Pass",
    "Sack",
    "Rushing Touchdown",
    "Passing Touchdown",
    "Fumble Recovery (Own)",
    "Fumble",
    "Interception",
    "Safety",
    "Two Point Pass",
    "Two Point Rush",
}


# Special-teams plays that can legitimately terminate an offensive drive.
DRIVE_END_SPECIAL_TEAMS = {
    "Punt",
    "Field Goal Good",
    "Field Goal Missed",
    "Blocked Punt",
    "Blocked Field Goal",
    "Blocked Punt Touchdown",
    "Blocked Field Goal Touchdown",
}


# Plays that should not determine possession ownership.
TRANSITION_ADMIN_TYPES = {
    "Kickoff",
    "Kickoff Return (Offense)",
    "Kickoff Return Touchdown",
    "Punt Return",
    "Punt Return Touchdown",
    "Missed Field Goal Return",
    "Timeout",
    "End Period",
    "End of Half",
    "End of Game",
    "End of Regulation",
    "placeholder",
}


@dataclass(frozen=True)
class DriveBuildConfig:
    default_start_yards_to_goal: float = 72.5
    explosive_yards: float = 20.0

    garbage_margin_q4: int = 24
    garbage_margin_q3: int = 35

    # Minimum dominance required when two teams receive meaningful
    # possession-identifying votes.
    possession_vote_ratio: float = 0.60

    # Extremely unusual team-game possession counts are flagged rather than
    # silently accepted.
    warning_low_drives: int = 4
    warning_high_drives: int = 20


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------


def _num(x: Any, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _clean_team(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _clean_type(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _garbage(
    period: Any,
    off_score: Any,
    def_score: Any,
    cfg: DriveBuildConfig,
) -> int:
    p = int(_num(period, 0))
    margin = abs(
        _num(off_score, 0)
        - _num(def_score, 0)
    )

    return int(
        (p >= 4 and margin >= cfg.garbage_margin_q4)
        or
        (p == 3 and margin >= cfg.garbage_margin_q3)
    )


# ---------------------------------------------------------------------------
# POSSESSION RESOLUTION
# ---------------------------------------------------------------------------


def _is_possession_vote(play_type: str) -> bool:
    return play_type in POSSESSION_TYPES


def _is_efficiency_play(play_type: str) -> bool:
    return play_type in EFFICIENCY_TYPES


def _resolve_possession(
    g: pd.DataFrame,
    cfg: DriveBuildConfig,
) -> tuple[str, str, str, float, int]:
    """
    Resolve the true offensive possession owner for a CFBD driveId.

    Returns
    -------
    offense
    defense
    method
    confidence
    mixed_offense_labels
    """

    raw_teams = (
        g["offense"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    raw_teams = raw_teams[raw_teams.ne("")]

    unique_raw = list(dict.fromkeys(raw_teams.tolist()))
    mixed = int(len(set(unique_raw)) > 1)

    # ---------------------------------------------------------------
    # 1. Strongest evidence:
    #    normal offensive / scrimmage plays.
    # ---------------------------------------------------------------

    vote_mask = g["play_type"].map(_is_possession_vote)

    vote_rows = g.loc[
        vote_mask & g["offense"].astype(str).str.strip().ne("")
    ].copy()

    if not vote_rows.empty:
        counts = (
            vote_rows["offense"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        offense = str(counts.index[0]).strip()
        top_votes = int(counts.iloc[0])
        total_votes = int(counts.sum())

        confidence = (
            top_votes / total_votes
            if total_votes
            else 0.0
        )

        # In most CFBD drives this will be essentially 100%.
        if (
            len(counts) == 1
            or confidence >= cfg.possession_vote_ratio
        ):
            defense_candidates = (
                vote_rows.loc[
                    vote_rows["offense"]
                    .astype(str)
                    .str.strip()
                    .eq(offense),
                    "defense",
                ]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            defense_candidates = defense_candidates[
                defense_candidates.ne("")
                & defense_candidates.ne(offense)
            ]

            if not defense_candidates.empty:
                defense = str(
                    defense_candidates.value_counts().index[0]
                ).strip()
            else:
                others = [
                    t
                    for t in unique_raw
                    if t != offense
                ]
                defense = others[0] if others else ""

            return (
                offense,
                defense,
                "scrimmage_vote",
                float(confidence),
                mixed,
            )

    # ---------------------------------------------------------------
    # 2. Field-goal / punt fallback.
    #
    # Some extremely short drives may contain no normal scrimmage
    # play but still have a valid possession-ending event.
    # ---------------------------------------------------------------

    terminal_mask = g["play_type"].isin(
        DRIVE_END_SPECIAL_TEAMS
    )

    terminal_rows = g.loc[
        terminal_mask
        & g["offense"].astype(str).str.strip().ne("")
    ]

    if not terminal_rows.empty:
        counts = (
            terminal_rows["offense"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        offense = str(counts.index[0]).strip()

        defense_candidates = (
            terminal_rows.loc[
                terminal_rows["offense"]
                .astype(str)
                .str.strip()
                .eq(offense),
                "defense",
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        defense_candidates = defense_candidates[
            defense_candidates.ne("")
            & defense_candidates.ne(offense)
        ]

        if not defense_candidates.empty:
            defense = str(
                defense_candidates.value_counts().index[0]
            ).strip()
        else:
            others = [
                t
                for t in unique_raw
                if t != offense
            ]
            defense = others[0] if others else ""

        return (
            offense,
            defense,
            "terminal_event",
            1.0,
            mixed,
        )

    # ---------------------------------------------------------------
    # 3. Majority-label fallback.
    #
    # Used only when the drive contains no recognised scrimmage or
    # terminal play. This keeps unusual historical rows visible while
    # explicitly flagging the weaker resolution method.
    # ---------------------------------------------------------------

    if not raw_teams.empty:
        counts = raw_teams.value_counts()

        offense = str(counts.index[0]).strip()
        top_votes = int(counts.iloc[0])
        total_votes = int(counts.sum())

        confidence = (
            top_votes / total_votes
            if total_votes
            else 0.0
        )

        defense_candidates = (
            g.loc[
                g["offense"]
                .astype(str)
                .str.strip()
                .eq(offense),
                "defense",
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        defense_candidates = defense_candidates[
            defense_candidates.ne("")
            & defense_candidates.ne(offense)
        ]

        if not defense_candidates.empty:
            defense = str(
                defense_candidates.value_counts().index[0]
            ).strip()
        else:
            others = [
                t
                for t in unique_raw
                if t != offense
            ]
            defense = others[0] if others else ""

        return (
            offense,
            defense,
            "majority_fallback",
            float(confidence),
            mixed,
        )

    return (
        "",
        "",
        "unresolved",
        0.0,
        mixed,
    )


# ---------------------------------------------------------------------------
# DRIVE OUTCOME
# ---------------------------------------------------------------------------


def _drive_result(
    drive_rows: pd.DataFrame,
    offense: str,
) -> str:
    """
    Determine drive outcome from the resolved possession owner's events.

    Priority matters because return plays can occur inside the same CFBD
    drive container.
    """

    if drive_rows.empty:
        return "OTHER"

    own = drive_rows.loc[
        drive_rows["offense"]
        .astype(str)
        .str.strip()
        .eq(offense)
    ].copy()

    if own.empty:
        own = drive_rows.copy()

    types = own["play_type"].fillna("").astype(str)

    # Offensive TD first.
    if types.isin(
        {
            "Rushing Touchdown",
            "Passing Touchdown",
        }
    ).any():
        return "TD"

    # Made FG.
    if types.eq("Field Goal Good").any():
        return "FG"

    # Safety.
    if types.eq("Safety").any():
        return "SAFETY"

    # Turnovers.
    if types.isin(
        {
            "Interception",
            "Pass Interception Return",
            "Interception Return Touchdown",
            "Fumble Recovery (Opponent)",
            "Fumble Return Touchdown",
        }
    ).any():
        return "TURNOVER"

    # Punt.
    if types.isin(
        {
            "Punt",
            "Blocked Punt",
            "Blocked Punt Touchdown",
        }
    ).any():
        return "PUNT"

    # Missed / blocked field goal behaves as a failed scoring possession.
    if types.isin(
        {
            "Field Goal Missed",
            "Blocked Field Goal",
            "Blocked Field Goal Touchdown",
        }
    ).any():
        return "FG_MISSED"

    # End-of-half/game.
    if types.isin(
        {
            "End of Half",
            "End of Game",
            "End of Regulation",
        }
    ).any():
        return "END_HALF"

    # Text fallback for turnover on downs.
    text = " ".join(
        own["play_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .tolist()
    )

    if (
        "turnover on downs" in text
        or "failed fourth" in text
    ):
        return "DOWNS"

    return "OTHER"


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------


def _drive_points(
    drive_rows: pd.DataFrame,
    offense: str,
    result: str,
) -> int:
    """
    Estimate offensive points produced by the possession.

    Explicit scoring play types are preferred because score fields can be
    awkward around kickoffs and transition events.
    """

    own = drive_rows.loc[
        drive_rows["offense"]
        .astype(str)
        .str.strip()
        .eq(offense)
    ].copy()

    if own.empty:
        own = drive_rows.copy()

    types = set(
        own["play_type"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if (
        "Rushing Touchdown" in types
        or "Passing Touchdown" in types
    ):
        # Most TD drives are 7. We retain the structural convention used by
        # the existing model. Two-point modelling can be expanded separately.
        return 7

    if "Field Goal Good" in types:
        return 3

    if "Safety" in types:
        return 2

    if result == "TD":
        return 7

    if result == "FG":
        return 3

    if result == "SAFETY":
        return 2

    return 0


# ---------------------------------------------------------------------------
# FIELD POSITION
# ---------------------------------------------------------------------------


def _first_valid_number(
    series: pd.Series,
    default=np.nan,
) -> float:
    vals = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if vals.empty:
        return default

    return float(vals.iloc[0])


def _last_valid_number(
    series: pd.Series,
    default=np.nan,
) -> float:
    vals = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if vals.empty:
        return default

    return float(vals.iloc[-1])


# ---------------------------------------------------------------------------
# MAIN DRIVE RECONSTRUCTION
# ---------------------------------------------------------------------------


def reconstruct_drives(
    plays: pd.DataFrame,
    cfg: DriveBuildConfig | None = None,
) -> pd.DataFrame:
    """
    Reconstruct one row per offensive possession from CFBD PBP.

    CFBD driveId remains the authoritative drive container, but possession
    ownership is resolved from the actual football plays inside that
    container rather than from its first row.
    """

    cfg = cfg or DriveBuildConfig()

    p = plays.copy()

    rename = {
        "gameId": "game_id",
        "driveId": "drive_id",
        "driveNumber": "drive_number",
        "playNumber": "play_number",
        "yardsToGoal": "yards_to_goal",
        "yardsGained": "yards_gained",
        "playType": "play_type",
        "playText": "play_text",
        "offenseScore": "offense_score",
        "defenseScore": "defense_score",
    }

    p = p.rename(
        columns={
            k: v
            for k, v in rename.items()
            if k in p.columns
        }
    )

    required = (
        "game_id",
        "drive_id",
        "offense",
        "defense",
    )

    missing = [
        c
        for c in required
        if c not in p.columns
    ]

    if missing:
        raise ValueError(
            f"PBP missing required columns: {missing}"
        )

    p = p.dropna(
        subset=["game_id", "drive_id"]
    ).copy()

    p["game_id"] = pd.to_numeric(
        p["game_id"],
        errors="coerce",
    )

    p = p.dropna(
        subset=["game_id"]
    ).copy()

    p["game_id"] = p["game_id"].astype("int64")

    numeric_cols = (
        "period",
        "play_number",
        "drive_number",
        "yards_to_goal",
        "yards_gained",
        "ppa",
        "down",
        "distance",
        "offense_score",
        "defense_score",
    )

    for c in numeric_cols:
        if c not in p.columns:
            p[c] = np.nan

        p[c] = pd.to_numeric(
            p[c],
            errors="coerce",
        )

    if "play_text" not in p.columns:
        p["play_text"] = ""

    if "play_type" not in p.columns:
        p["play_type"] = ""

    p["play_text"] = (
        p["play_text"]
        .fillna("")
        .astype(str)
    )

    p["play_type"] = (
        p["play_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    p["offense"] = (
        p["offense"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    p["defense"] = (
        p["defense"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Derived play-level metrics
    # ------------------------------------------------------------------

    p["success"] = np.where(
        p["ppa"].notna(),
        (p["ppa"] > 0).astype(float),
        np.nan,
    )

    p["explosive_play"] = np.where(
        p["yards_gained"].notna(),
        (
            p["yards_gained"]
            >= cfg.explosive_yards
        ).astype(float),
        np.nan,
    )

    p["third_down"] = (
        p["down"] == 3
    ).astype(int)

    p["third_success"] = (
        (p["down"] == 3)
        & p["distance"].notna()
        & p["yards_gained"].notna()
        & (
            p["yards_gained"]
            >= p["distance"]
        )
    ).astype(int)

    p["fourth_down"] = (
        p["down"] == 4
    ).astype(int)

    p["fourth_success"] = (
        (p["down"] == 4)
        & p["distance"].notna()
        & p["yards_gained"].notna()
        & (
            p["yards_gained"]
            >= p["distance"]
        )
    ).astype(int)

    p["garbage_time"] = [
        _garbage(
            period,
            off_score,
            def_score,
            cfg,
        )
        for period, off_score, def_score
        in zip(
            p["period"],
            p["offense_score"],
            p["defense_score"],
        )
    ]

    # Preserve source ordering as much as possible.
    p["_source_order"] = np.arange(len(p))

    p = p.sort_values(
        [
            "game_id",
            "drive_id",
            "period",
            "play_number",
            "_source_order",
        ],
        kind="stable",
        na_position="last",
    )

    rows: list[dict[str, Any]] = []

    for (gid, did), g in p.groupby(
        ["game_id", "drive_id"],
        sort=False,
        observed=True,
    ):
        g = g.copy()

        (
            offense,
            defense,
            possession_method,
            possession_confidence,
            mixed_offense_labels,
        ) = _resolve_possession(
            g,
            cfg,
        )

        if not offense:
            # Keep the historical source auditable but do not fabricate a
            # possession owner.
            continue

        # --------------------------------------------------------------
        # Rows belonging to resolved possession offense
        # --------------------------------------------------------------

        own_rows = g.loc[
            g["offense"].eq(offense)
        ].copy()

        # Football-efficiency subset.
        efficiency = own_rows.loc[
            own_rows["play_type"].map(
                _is_efficiency_play
            )
        ].copy()

        # If a strange historical drive contains no recognised efficiency
        # play, retain own-team non-admin rows as a conservative fallback.
        if efficiency.empty:
            efficiency = own_rows.loc[
                ~own_rows["play_type"].isin(
                    TRANSITION_ADMIN_TYPES
                )
            ].copy()

        if efficiency.empty:
            efficiency = own_rows.copy()

        # --------------------------------------------------------------
        # Competitive efficiency subset
        # --------------------------------------------------------------

        competitive = efficiency.loc[
            efficiency["garbage_time"].eq(0)
        ].copy()

        if competitive.empty:
            competitive = efficiency.copy()

        # --------------------------------------------------------------
        # Drive outcome
        # --------------------------------------------------------------

        result = _drive_result(
            g,
            offense,
        )

        points = _drive_points(
            g,
            offense,
            result,
        )

        # --------------------------------------------------------------
        # Start / end field position
        #
        # Use the possession offense's football rows rather than the
        # transition kickoff row from the other team.
        # --------------------------------------------------------------

        field_rows = efficiency

        start_ytg = _first_valid_number(
            field_rows["yards_to_goal"],
            cfg.default_start_yards_to_goal,
        )

        end_ytg = _last_valid_number(
            field_rows["yards_to_goal"],
            np.nan,
        )

        start_field = float(
            np.clip(
                100 - start_ytg,
                1,
                99,
            )
        )

        if np.isfinite(end_ytg):
            end_field = float(
                np.clip(
                    100 - end_ytg,
                    1,
                    99,
                )
            )
        else:
            end_field = np.nan

        # --------------------------------------------------------------
        # Periods / drive number
        # --------------------------------------------------------------

        period_start = int(
            _num(
                _first_valid_number(
                    own_rows["period"],
                    0,
                ),
                0,
            )
        )

        period_end = int(
            _num(
                _last_valid_number(
                    own_rows["period"],
                    0,
                ),
                0,
            )
        )

        drive_number = _first_valid_number(
            g["drive_number"],
            np.nan,
        )

        # --------------------------------------------------------------
        # Score state
        #
        # Again use the resolved offense's rows, not the first raw row.
        # --------------------------------------------------------------

        score_row_source = (
            efficiency
            if not efficiency.empty
            else own_rows
        )

        start_off_score = _first_valid_number(
            score_row_source["offense_score"],
            0,
        )

        start_def_score = _first_valid_number(
            score_row_source["defense_score"],
            0,
        )

        start_score_margin = (
            start_off_score
            - start_def_score
        )

        # --------------------------------------------------------------
        # Production
        # --------------------------------------------------------------

        yards = float(
            pd.to_numeric(
                efficiency["yards_gained"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        valid_yards = pd.to_numeric(
            efficiency["yards_gained"],
            errors="coerce",
        ).dropna()

        yards_per_play = (
            float(valid_yards.mean())
            if not valid_yards.empty
            else 0.0
        )

        # --------------------------------------------------------------
        # Success / PPA / explosiveness
        # --------------------------------------------------------------

        success_values = pd.to_numeric(
            competitive["success"],
            errors="coerce",
        ).dropna()

        success_rate = (
            float(success_values.mean())
            if not success_values.empty
            else 0.0
        )

        ppa_values = pd.to_numeric(
            competitive["ppa"],
            errors="coerce",
        ).dropna()

        ppa = (
            float(ppa_values.mean())
            if not ppa_values.empty
            else 0.0
        )

        explosive_values = pd.to_numeric(
            competitive["explosive_play"],
            errors="coerce",
        ).dropna()

        explosive_rate = (
            float(explosive_values.mean())
            if not explosive_values.empty
            else 0.0
        )

        positive_ppa_values = ppa_values.loc[
            ppa_values > 0
        ]

        positive_ppa_mean = (
            float(positive_ppa_values.mean())
            if not positive_ppa_values.empty
            else 0.0
        )

        # --------------------------------------------------------------
        # Downs
        # --------------------------------------------------------------

        third_down_attempts = int(
            efficiency["third_down"].sum()
        )

        third_down_conversions = int(
            efficiency["third_success"].sum()
        )

        fourth_down_attempts = int(
            efficiency["fourth_down"].sum()
        )

        fourth_down_conversions = int(
            efficiency["fourth_success"].sum()
        )

        # --------------------------------------------------------------
        # Garbage-time share
        # --------------------------------------------------------------

        garbage_play_share = (
            float(
                efficiency["garbage_time"].mean()
            )
            if len(efficiency)
            else 0.0
        )

        raw_offense_labels = int(
            g["offense"]
            .replace("", np.nan)
            .dropna()
            .nunique()
        )

        rows.append(
            {
                "game_id": int(gid),
                "drive_id": str(did),

                "offense": offense,
                "defense": defense,

                "drive_number": drive_number,

                "period_start": period_start,
                "period_end": period_end,

                "start_field_position": start_field,
                "end_field_position": end_field,

                # 'plays' now means offensive football plays used for
                # possession efficiency, not raw rows inside driveId.
                "plays": int(len(efficiency)),
                "raw_rows": int(len(g)),
                "competitive_plays": int(len(competitive)),

                "yards": yards,
                "yards_per_play": yards_per_play,

                "points": int(points),
                "result": result,

                "touchdown": int(
                    result == "TD"
                ),
                "field_goal": int(
                    result == "FG"
                ),
                "turnover": int(
                    result == "TURNOVER"
                ),
                "punt": int(
                    result == "PUNT"
                ),
                "downs": int(
                    result == "DOWNS"
                ),
                "safety": int(
                    result == "SAFETY"
                ),

                "success_rate": success_rate,
                "ppa": ppa,
                "explosive_rate": explosive_rate,
                "positive_ppa_mean": positive_ppa_mean,

                "third_down_attempts": third_down_attempts,
                "third_down_conversions": third_down_conversions,

                "fourth_down_attempts": fourth_down_attempts,
                "fourth_down_conversions": fourth_down_conversions,

                "garbage_play_share": garbage_play_share,
                "start_score_margin": float(
                    start_score_margin
                ),

                # Diagnostics
                "possession_method": possession_method,
                "possession_confidence": float(
                    possession_confidence
                ),
                "mixed_offense_labels": int(
                    mixed_offense_labels
                ),
                "raw_offense_label_count": raw_offense_labels,
            }
        )

    drives = pd.DataFrame(rows)

    if drives.empty:
        return drives

    # --------------------------------------------------------------
    # Structural diagnostics
    # --------------------------------------------------------------

    drives["possession_warning"] = ""

    counts = (
        drives.groupby(
            ["game_id", "offense"],
            observed=True,
        )
        .size()
        .rename("team_game_drives")
        .reset_index()
    )

    drives = drives.merge(
        counts,
        on=["game_id", "offense"],
        how="left",
        validate="many_to_one",
    )

    low_mask = (
        drives["team_game_drives"]
        < cfg.warning_low_drives
    )

    high_mask = (
        drives["team_game_drives"]
        > cfg.warning_high_drives
    )

    drives.loc[
        low_mask,
        "possession_warning",
    ] = "LOW_DRIVE_COUNT"

    drives.loc[
        high_mask,
        "possession_warning",
    ] = "HIGH_DRIVE_COUNT"

    unresolved_method = drives[
        "possession_method"
    ].eq("majority_fallback")

    drives.loc[
        unresolved_method
        & drives["possession_warning"].eq(""),
        "possession_warning",
    ] = "WEAK_POSSESSION_RESOLUTION"

    return drives


# ---------------------------------------------------------------------------
# TEAM-GAME AGGREGATION
# ---------------------------------------------------------------------------


def aggregate_team_games(
    drives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate reconstructed possessions to one row per team-game.

    The public column names remain compatible with the existing V3
    chronological state and matchup pipeline.
    """

    if drives.empty:
        return pd.DataFrame()

    d = drives.copy()

    d["competitive_weight"] = (
        1
        - 0.65
        * pd.to_numeric(
            d["garbage_play_share"],
            errors="coerce",
        ).fillna(0)
    ).clip(
        0.25,
        1.0,
    )

    rows: list[dict[str, Any]] = []

    for (gid, offense, defense), g in d.groupby(
        ["game_id", "offense", "defense"],
        sort=False,
        observed=True,
    ):
        w = pd.to_numeric(
            g["competitive_weight"],
            errors="coerce",
        ).fillna(1.0).to_numpy(float)

        if (
            not np.isfinite(w).all()
            or w.sum() <= 0
        ):
            w = np.ones(
                len(g),
                dtype=float,
            )

        def weighted_average(column: str) -> float:
            values = pd.to_numeric(
                g[column],
                errors="coerce",
            )

            valid = (
                values.notna()
                & np.isfinite(values)
            )

            if not valid.any():
                return 0.0

            return float(
                np.average(
                    values.loc[valid].to_numpy(float),
                    weights=w[valid.to_numpy()],
                )
            )

        n = max(
            int(len(g)),
            1,
        )

        points = float(
            pd.to_numeric(
                g["points"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        third_attempts = int(
            pd.to_numeric(
                g["third_down_attempts"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        third_conversions = int(
            pd.to_numeric(
                g["third_down_conversions"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        fourth_attempts = int(
            pd.to_numeric(
                g["fourth_down_attempts"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        fourth_conversions = int(
            pd.to_numeric(
                g["fourth_down_conversions"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        rows.append(
            {
                "game_id": int(gid),
                "team": str(offense),
                "opponent": str(defense),

                "drives": n,
                "points": points,
                "points_per_drive": (
                    points / n
                ),

                "td_rate": float(
                    g["touchdown"].sum()
                    / n
                ),

                "fg_rate": float(
                    g["field_goal"].sum()
                    / n
                ),

                "turnover_rate": float(
                    g["turnover"].sum()
                    / n
                ),

                "punt_rate": float(
                    g["punt"].sum()
                    / n
                ),

                "downs_rate": float(
                    g["downs"].sum()
                    / n
                ),

                "avg_start_field": weighted_average(
                    "start_field_position"
                ),

                "plays_per_drive": weighted_average(
                    "plays"
                ),

                "yards_per_drive": weighted_average(
                    "yards"
                ),

                "success_rate": weighted_average(
                    "success_rate"
                ),

                "ppa_per_play": weighted_average(
                    "ppa"
                ),

                "explosive_rate": weighted_average(
                    "explosive_rate"
                ),

                "positive_ppa": weighted_average(
                    "positive_ppa_mean"
                ),

                "third_down_rate": float(
                    third_conversions
                    / max(
                        third_attempts,
                        1,
                    )
                ),

                "fourth_down_rate": float(
                    fourth_conversions
                    / max(
                        fourth_attempts,
                        1,
                    )
                ),

                "garbage_share": float(
                    pd.to_numeric(
                        g["garbage_play_share"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                ),

                # Additional audit information. Existing consumers can
                # ignore these columns.
                "mixed_drive_share": float(
                    pd.to_numeric(
                        g["mixed_offense_labels"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                ),

                "weak_resolution_share": float(
                    g["possession_method"]
                    .eq("majority_fallback")
                    .mean()
                ),

                "mean_possession_confidence": float(
                    pd.to_numeric(
                        g["possession_confidence"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                ),
            }
        )

    x = pd.DataFrame(rows)

    if x.empty:
        return x

    # --------------------------------------------------------------
    # Opponent-side merge
    # --------------------------------------------------------------

    opponent_metric_columns = [
        c
        for c in x.columns
        if c
        not in (
            "game_id",
            "team",
            "opponent",
        )
    ]

    rename_map = {
        "team": "opponent",
        "opponent": "team",
    }

    rename_map.update(
        {
            c: f"opp_{c}"
            for c in opponent_metric_columns
        }
    )

    opp = x.rename(
        columns=rename_map
    )

    out = x.merge(
        opp,
        on=[
            "game_id",
            "team",
            "opponent",
        ],
        how="left",
        validate="one_to_one",
    )

    return out