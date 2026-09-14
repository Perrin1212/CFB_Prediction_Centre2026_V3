from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


APP_GAMES_PATH = (
    ROOT
    / "data"
    / "app"
    / "games.csv"
)

V3_PREDICTIONS_PATH = (
    ROOT
    / "data"
    / "predictions"
    / "v3_2026_predictions.csv"
)

CANONICAL_V3_GAMES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "v3"
    / "games.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "app"
    / "games_v3.csv"
)


# ============================================================
# DISPLAY
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# HELPERS
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def numeric_game_id(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .astype("Int64")
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V3 - BUILD APP OVERLAY"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    section(
        "1. LOAD APP + V3 DATA"
    )

    games = read_csv(
        APP_GAMES_PATH
    )

    v3 = read_csv(
        V3_PREDICTIONS_PATH
    )

    # The structural V3 schedule is refreshed directly from CFBD.  Use its
    # authoritative final scores to prevent a missing V2 control artifact from
    # leaving the app stuck on an already-completed week.
    if CANONICAL_V3_GAMES_PATH.exists():
        canonical = read_csv(CANONICAL_V3_GAMES_PATH)
        canonical = canonical.rename(columns={
            "game_id": "cfbd_game_id",
            "home_points": "_canonical_home_points",
            "away_points": "_canonical_away_points",
            "start_date": "_canonical_start_date",
        })
        canonical["cfbd_game_id"] = numeric_game_id(canonical["cfbd_game_id"])
        canonical = canonical[
            ["cfbd_game_id", "_canonical_home_points", "_canonical_away_points", "_canonical_start_date"]
        ].drop_duplicates("cfbd_game_id", keep="last")
        games["cfbd_game_id"] = numeric_game_id(games["cfbd_game_id"])
        games = games.merge(canonical, on="cfbd_game_id", how="left")
        final = games["_canonical_home_points"].notna() & games["_canonical_away_points"].notna()
        games.loc[final, "actual_home_points"] = games.loc[final, "_canonical_home_points"]
        games.loc[final, "actual_away_points"] = games.loc[final, "_canonical_away_points"]
        games.loc[final, "game_status"] = "final"
        games.loc[games["_canonical_start_date"].notna(), "start_date"] = games.loc[
            games["_canonical_start_date"].notna(), "_canonical_start_date"
        ]
        games = games.drop(columns=[
            "_canonical_home_points", "_canonical_away_points", "_canonical_start_date"
        ])
        print(f"Authoritative V3 finals: {int(final.sum()):,}")

    print(
        f"Existing app games:     {len(games):,}"
    )

    print(
        f"V3 predictions:         {len(v3):,}"
    )

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    section(
        "2. VALIDATE GAME IDS"
    )

    if "cfbd_game_id" not in games.columns:
        raise RuntimeError(
            "Existing app games.csv does not contain "
            "'cfbd_game_id'."
        )

    if "game_id" not in v3.columns:
        raise RuntimeError(
            "V3 predictions do not contain 'game_id'."
        )

    games = games.copy()
    v3 = v3.copy()

    games["cfbd_game_id"] = numeric_game_id(
        games["cfbd_game_id"]
    )

    v3["game_id"] = numeric_game_id(
        v3["game_id"]
    )

    if games["cfbd_game_id"].isna().any():
        bad = int(
            games["cfbd_game_id"]
            .isna()
            .sum()
        )

        raise RuntimeError(
            f"App games contain {bad:,} invalid game IDs."
        )

    if v3["game_id"].isna().any():
        bad = int(
            v3["game_id"]
            .isna()
            .sum()
        )

        raise RuntimeError(
            f"V3 predictions contain {bad:,} invalid game IDs."
        )

    app_duplicates = int(
        games["cfbd_game_id"]
        .duplicated()
        .sum()
    )

    v3_duplicates = int(
        v3["game_id"]
        .duplicated()
        .sum()
    )

    print(
        f"App duplicate IDs:      {app_duplicates:,}"
    )

    print(
        f"V3 duplicate IDs:       {v3_duplicates:,}"
    )

    if app_duplicates:
        raise RuntimeError(
            "Existing app games contain duplicate game IDs."
        )

    if v3_duplicates:
        raise RuntimeError(
            "V3 prediction file contains duplicate game IDs."
        )

    # --------------------------------------------------------
    # CHECK V3 MATCH COVERAGE
    # --------------------------------------------------------

    section(
        "3. CHECK V3 MATCH COVERAGE"
    )

    app_ids = set(
        games["cfbd_game_id"]
        .dropna()
        .astype(int)
    )

    v3_ids = set(
        v3["game_id"]
        .dropna()
        .astype(int)
    )

    matched_ids = (
        app_ids
        & v3_ids
    )

    missing_from_app = (
        v3_ids
        - app_ids
    )

    app_without_v3 = (
        app_ids
        - v3_ids
    )

    print(
        f"Matched V3 games:       {len(matched_ids):,}"
    )

    print(
        f"V3 missing from app:    {len(missing_from_app):,}"
    )

    print(
        f"App games without V3:   {len(app_without_v3):,}"
    )

    if missing_from_app:
        sample = sorted(
            missing_from_app
        )[:10]

        print()
        print(
            "WARNING - sample V3 IDs missing from app:"
        )

        for game_id in sample:
            print(
                f"  {game_id}"
            )

    # --------------------------------------------------------
    # PREPARE V3 VIEW
    # --------------------------------------------------------

    section(
        "4. PREPARE V3 APP FIELDS"
    )

    rename_map = {
        "game_id": "cfbd_game_id",
    }

    v3 = v3.rename(
        columns=rename_map
    )

    # Every field below remains explicitly V3-prefixed.
    # Nothing can overwrite V2 / official / locked values.

    preferred_fields = [
        "cfbd_game_id",

        "v3_home_win_probability",
        "v3_away_win_probability",
        "v3_projected_home_points",
        "v3_projected_away_points",
        "v3_projected_margin",
        "v3_projected_total",
        "state_maturity",

        "sim_home_win_probability",
        "sim_away_win_probability",
        "sim_projected_home_points",
        "sim_projected_away_points",
        "sim_projected_margin",
        "sim_projected_total",
        "sim_expected_home_drives",
        "sim_expected_away_drives",

        "sim_home_p10",
        "sim_home_p90",
        "sim_away_p10",
        "sim_away_p90",

        "sim_margin_p10",
        "sim_margin_p90",
        "sim_total_p10",
        "sim_total_p90",

        "sim_home_40_plus_probability",
        "sim_away_40_plus_probability",
        "sim_home_10_or_less_probability",
        "sim_away_10_or_less_probability",

        "sim_simulations",

        "matchup_home_drives",
        "matchup_away_drives",

        "matchup_home_ppd",
        "matchup_away_ppd",

        "matchup_home_td_rate",
        "matchup_away_td_rate",

        "matchup_home_fg_rate",
        "matchup_away_fg_rate",

        "matchup_home_turnover_rate",
        "matchup_away_turnover_rate",

        "matchup_home_start_field",
        "matchup_away_start_field",

        "matchup_home_success_rate",
        "matchup_away_success_rate",

        "matchup_home_ppa",
        "matchup_away_ppa",

        "matchup_home_explosive_rate",
        "matchup_away_explosive_rate",

        "matchup_home_yards_per_drive",
        "matchup_away_yards_per_drive",

        "matchup_home_plays_per_drive",
        "matchup_away_plays_per_drive",

        "matchup_home_third_down_rate",
        "matchup_away_third_down_rate",

        "matchup_home_matchup_ratio",
        "matchup_away_matchup_ratio",

        "matchup_elo_difference",
        "matchup_rating_difference",
        "matchup_home_field_points",
        "matchup_state_maturity",
    ]

    keep = [
        column
        for column in preferred_fields
        if column in v3.columns
    ]

    missing_expected = [
        column
        for column in preferred_fields
        if column not in v3.columns
    ]

    if missing_expected:
        print(
            "Optional V3 fields not present:"
        )

        for column in missing_expected:
            print(
                f"  - {column}"
            )

    v3_view = (
        v3[
            keep
        ]
        .copy()
    )

    # --------------------------------------------------------
    # PROTECT EXISTING APP DATA
    # --------------------------------------------------------

    section(
        "5. PROTECT EXISTING PRODUCTION FIELDS"
    )

    collisions = [
        column
        for column in v3_view.columns
        if (
            column != "cfbd_game_id"
            and column in games.columns
        )
    ]

    if collisions:
        raise RuntimeError(
            "Refusing to overwrite existing app fields:\n"
            + "\n".join(
                f"  - {column}"
                for column in collisions
            )
        )

    print(
        "Existing V2 / lock / market fields protected."
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    section(
        "6. MERGE V3 INTO APP DATA"
    )

    enriched = games.merge(
        v3_view,
        on="cfbd_game_id",
        how="left",
        validate="1:1",
    )

    if len(enriched) != len(games):
        raise RuntimeError(
            "Merge changed app game row count. "
            "This should never happen."
        )

    # --------------------------------------------------------
    # V3 DISPLAY HELPERS
    # --------------------------------------------------------

    if (
        "v3_home_win_probability"
        in enriched.columns
        and
        "v3_away_win_probability"
        in enriched.columns
    ):

        home_prob = pd.to_numeric(
            enriched[
                "v3_home_win_probability"
            ],
            errors="coerce",
        )

        away_prob = pd.to_numeric(
            enriched[
                "v3_away_win_probability"
            ],
            errors="coerce",
        )

        enriched[
            "v3_predicted_winner"
        ] = pd.NA

        home_mask = (
            home_prob.notna()
            & away_prob.notna()
            & home_prob.ge(
                away_prob
            )
        )

        away_mask = (
            home_prob.notna()
            & away_prob.notna()
            & away_prob.gt(
                home_prob
            )
        )

        enriched.loc[
            home_mask,
            "v3_predicted_winner",
        ] = enriched.loc[
            home_mask,
            "home_team",
        ]

        enriched.loc[
            away_mask,
            "v3_predicted_winner",
        ] = enriched.loc[
            away_mask,
            "away_team",
        ]

        enriched[
            "v3_favourite_probability"
        ] = pd.concat(
            [
                home_prob,
                away_prob,
            ],
            axis=1,
        ).max(
            axis=1
        )

        enriched[
            "v3_confidence_bucket"
        ] = pd.cut(
            enriched[
                "v3_favourite_probability"
            ],
            bins=[
                -float("inf"),
                0.55,
                0.60,
                0.70,
                0.80,
                float("inf"),
            ],
            labels=[
                "Toss-up",
                "Lean",
                "Solid",
                "Strong",
                "Elite",
            ],
            right=False,
        ).astype(
            "string"
        )

    # V2-versus-V3 comparison fields.
    if (
        "home_win_probability"
        in enriched.columns
        and
        "v3_home_win_probability"
        in enriched.columns
    ):

        enriched[
            "v3_vs_v2_home_probability_delta"
        ] = (
            pd.to_numeric(
                enriched[
                    "v3_home_win_probability"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                enriched[
                    "home_win_probability"
                ],
                errors="coerce",
            )
        )

    if (
        "projected_home_score"
        in enriched.columns
        and
        "v3_projected_home_points"
        in enriched.columns
    ):

        enriched[
            "v3_vs_v2_home_score_delta"
        ] = (
            pd.to_numeric(
                enriched[
                    "v3_projected_home_points"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                enriched[
                    "projected_home_score"
                ],
                errors="coerce",
            )
        )

    if (
        "projected_away_score"
        in enriched.columns
        and
        "v3_projected_away_points"
        in enriched.columns
    ):

        enriched[
            "v3_vs_v2_away_score_delta"
        ] = (
            pd.to_numeric(
                enriched[
                    "v3_projected_away_points"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                enriched[
                    "projected_away_score"
                ],
                errors="coerce",
            )
        )

    if (
        "final_expected_home_margin"
        in enriched.columns
        and
        "v3_projected_margin"
        in enriched.columns
    ):

        enriched[
            "v3_vs_v2_margin_delta"
        ] = (
            pd.to_numeric(
                enriched[
                    "v3_projected_margin"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                enriched[
                    "final_expected_home_margin"
                ],
                errors="coerce",
            )
        )

    if (
        "final_expected_total_points"
        in enriched.columns
        and
        "v3_projected_total"
        in enriched.columns
    ):

        enriched[
            "v3_vs_v2_total_delta"
        ] = (
            pd.to_numeric(
                enriched[
                    "v3_projected_total"
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                enriched[
                    "final_expected_total_points"
                ],
                errors="coerce",
            )
        )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    section(
        "7. FINAL VALIDATION"
    )

    v3_available = (
        enriched[
            "v3_home_win_probability"
        ]
        .notna()
        if "v3_home_win_probability" in enriched.columns
        else pd.Series(
            False,
            index=enriched.index,
        )
    )

    v3_rows = int(
        v3_available.sum()
    )

    print(
        f"Output game rows:       {len(enriched):,}"
    )

    print(
        f"Output columns:         {len(enriched.columns):,}"
    )

    print(
        f"Rows with V3:           {v3_rows:,}"
    )

    print(
        f"Rows without V3:        {len(enriched) - v3_rows:,}"
    )

    if v3_rows != len(matched_ids):
        raise RuntimeError(
            "Unexpected V3 coverage after merge."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    section(
        "8. SAVE V3 APP DATASET"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved -> {OUTPUT_PATH}"
    )

    section(
        "V3 APP OVERLAY COMPLETE"
    )

    print(
        "✓ Existing V2 production fields preserved."
    )

    print(
        "✓ Official locks preserved."
    )

    print(
        "✓ Market fields remain display-only."
    )

    print(
        "✓ V3 challenger fields added alongside V2."
    )

    print(
        "✓ Original data/app/games.csv was NOT overwritten."
    )


if __name__ == "__main__":
    main()
