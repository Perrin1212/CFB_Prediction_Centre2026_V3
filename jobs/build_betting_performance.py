from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATHS / SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
APP_DIR = DATA_DIR / "app"
MARKET_DIR = DATA_DIR / "market"
PREDICTIONS_DIR = DATA_DIR / "predictions"

V2_GAMES = APP_DIR / "games.csv"
V3_GAMES = APP_DIR / "games_v3.csv"

# IMPORTANT:
# The app-facing games.csv is intentionally presentation-oriented and
# does not reliably carry completed final scores. The frozen scoring
# output DOES carry the genuine completed result fields:
#
#   actual_home_points
#   actual_away_points
#
# Therefore betting settlement must use scoring predictions for results.
SCORING_ALL = PREDICTIONS_DIR / "2026_scoring_predictions.csv"
SCORING_COMPLETED = (
    PREDICTIONS_DIR
    / "2026_completed_scoring_predictions.csv"
)

# V3 is the authoritative live-season result source.  The V2 score output is
# retained when available, but a weekly V3-only production refresh must still
# be able to settle betting rows from official CFBD finals.
V3_CANONICAL_RESULTS = (
    DATA_DIR
    / "processed"
    / "v3"
    / "games.csv"
)

MARKET_CURRENT = MARKET_DIR / "market_lines_current.csv"
MARKET_HISTORY = MARKET_DIR / "market_lines_history.csv"

OUT_FILE = APP_DIR / "betting_performance.csv"

BET_STAKE = 100.0
ATS_AMERICAN_ODDS = -110.0
MONEYLINE_MIN_AMERICAN_ODDS = -350.0
MONEYLINE_MAX_AMERICAN_ODDS = 350.0
MONEYLINE_MIN_MODEL_CONFIDENCE = 0.60
ATS_MIN_COVER_PROBABILITY = 0.55
MONEYLINE_SELECTION_RULE = (
    "moneyline_odds_-350_to_+350_and_"
    "win_probability_at_least_60pct"
)
ATS_SELECTION_RULE = "ats_cover_probability_at_least_55pct"


# ============================================================
# HELPERS
# ============================================================

def pick_col(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    lower = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:

        if candidate in frame.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def safe_float(
    value: Any,
) -> float | None:

    try:

        if value is None or pd.isna(value):
            return None

        number = float(value)

        if np.isfinite(number):
            return number

    except Exception:
        pass

    return None


def safe_probability(
    value: Any,
) -> float | None:

    number = safe_float(value)

    if number is None:
        return None

    # Support both canonical decimal probabilities and legacy percentage
    # displays without silently accepting invalid values.
    if number > 1.0 and number <= 100.0:
        number /= 100.0

    if 0.0 <= number <= 1.0:
        return number

    return None


def model_confidence_for_pick(
    game: pd.Series,
    *,
    model_home: bool,
) -> float | None:

    side = "home" if model_home else "away"
    candidates = [
        f"display_{side}_win_probability",
        f"locked_{side}_win_probability",
        f"{side}_win_probability",
        "model_favourite_probability",
        "prediction_probability",
    ]

    for column in candidates:
        probability = safe_probability(
            game.get(column)
        )
        if probability is not None:
            return probability

    return None


def clean(
    value: Any,
) -> str:

    try:

        if value is None or pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


def normalise_team(
    value: Any,
) -> str:

    return (
        clean(value)
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


def american_profit(
    odds: float,
    stake: float = BET_STAKE,
) -> float:

    if odds > 0:
        return stake * odds / 100.0

    return stake * 100.0 / abs(odds)


def first_number(
    row: pd.Series,
    candidates: list[str],
) -> float | None:
    """Return the first finite numeric value available on a game row."""
    for column in candidates:
        number = safe_float(row.get(column))
        if number is not None:
            return number
    return None


def projected_home_margin_and_sd(
    game: pd.Series,
) -> tuple[float | None, float | None]:
    """Recover the simulated home-margin distribution from the app schema."""
    mean = first_number(
        game,
        [
            "sim_projected_margin",
            "v3_projected_margin",
            "display_final_expected_home_margin",
            "locked_final_expected_home_margin",
            "final_expected_home_margin",
        ],
    )

    if mean is None:
        home_points = first_number(
            game,
            [
                "display_projected_home_score",
                "locked_projected_home_score",
                "projected_home_score",
            ],
        )
        away_points = first_number(
            game,
            [
                "display_projected_away_score",
                "locked_projected_away_score",
                "projected_away_score",
            ],
        )
        if home_points is not None and away_points is not None:
            mean = home_points - away_points

    margin_p10 = first_number(game, ["sim_margin_p10", "simulation_margin_p10"])
    margin_p90 = first_number(game, ["sim_margin_p90", "simulation_margin_p90"])
    sd: float | None = None

    # The V3 Monte Carlo output publishes the 10th and 90th percentiles of
    # home margin.  Their width is approximately 2.5631 standard deviations,
    # allowing the captured market spread to be evaluated independently from
    # the straight-up winner probability.
    if (
        margin_p10 is not None
        and margin_p90 is not None
        and margin_p90 > margin_p10
    ):
        sd = (margin_p90 - margin_p10) / 2.5631031311

    if sd is None:
        home_sd = first_number(game, ["home_score_residual_sd_pre"])
        away_sd = first_number(game, ["away_score_residual_sd_pre"])
        correlation = first_number(game, ["score_residual_correlation_pre"])
        if home_sd is not None and away_sd is not None:
            correlation = 0.0 if correlation is None else correlation
            variance = (
                home_sd**2
                + away_sd**2
                - 2.0 * correlation * home_sd * away_sd
            )
            if variance > 0:
                sd = float(np.sqrt(variance))

    return mean, sd


def ats_projection(
    game: pd.Series,
    home_spread: float,
) -> dict[str, float | bool] | None:
    """Choose the projected cover side and calculate its cover probability."""
    mean, sd = projected_home_margin_and_sd(game)
    if mean is None or sd is None or sd <= 0:
        return None

    home_edge = mean + home_spread
    home_cover_probability = float(
        NormalDist().cdf(home_edge / sd)
    )
    home_cover_probability = float(
        np.clip(home_cover_probability, 0.0, 1.0)
    )
    bet_home = home_cover_probability >= 0.5
    selected_probability = (
        home_cover_probability
        if bet_home
        else 1.0 - home_cover_probability
    )

    return {
        "bet_home": bet_home,
        "cover_probability": selected_probability,
        "home_cover_probability": home_cover_probability,
        "projected_home_margin": mean,
        "margin_sd": sd,
        "edge_points": abs(home_edge),
    }


def game_id_column(
    frame: pd.DataFrame,
) -> str:

    column = pick_col(
        frame,
        [
            "cfbd_game_id",
            "cfbd_id",
            "game_id",
            "tracker_game_id",
        ],
    )

    if column is None:

        raise RuntimeError(
            "Could not identify CFBD game ID column.\n"
            f"Available columns: "
            f"{', '.join(map(str, frame.columns))}"
        )

    return column


def load_market_history() -> pd.DataFrame:

    if MARKET_HISTORY.exists():

        market = pd.read_csv(
            MARKET_HISTORY,
            low_memory=False,
        )

        if not market.empty:
            return market

    if MARKET_CURRENT.exists():

        return pd.read_csv(
            MARKET_CURRENT,
            low_memory=False,
        )

    return pd.DataFrame()


def load_scoring_results() -> pd.DataFrame:
    """
    Load authoritative completed results and expose one row per game.

    V2 scoring output remains supported for backwards compatibility.  V3's
    canonical CFBD schedule is loaded afterwards and therefore takes
    precedence whenever both sources contain the same game.
    """

    sources: list[tuple[str, Path]] = []

    if SCORING_ALL.exists():
        sources.append(("v2_scoring", SCORING_ALL))
    elif SCORING_COMPLETED.exists():
        sources.append(("v2_scoring", SCORING_COMPLETED))

    if V3_CANONICAL_RESULTS.exists():
        sources.append(("v3_canonical_cfbd", V3_CANONICAL_RESULTS))

    if not sources:
        raise FileNotFoundError(
            "Could not find an authoritative scoring result source.\n"
            f"Expected V2 scoring at:\n{SCORING_ALL}\n"
            f"or V3 canonical results at:\n{V3_CANONICAL_RESULTS}"
        )

    normalised: list[pd.DataFrame] = []

    for source_name, path in sources:
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue

        scoring_id = game_id_column(frame)
        home_score_col = pick_col(
            frame,
            [
                "actual_home_points",
                "final_home_score",
                "home_points",
            ],
        )
        away_score_col = pick_col(
            frame,
            [
                "actual_away_points",
                "final_away_score",
                "away_points",
            ],
        )

        if home_score_col is None or away_score_col is None:
            raise RuntimeError(
                f"{source_name} does not contain usable actual-score "
                "columns.\n"
                f"Available columns: {', '.join(map(str, frame.columns))}"
            )

        result = frame.copy()
        result["_game_id"] = pd.to_numeric(
            result[scoring_id], errors="coerce"
        ).astype("Int64")
        result["_actual_home_score"] = pd.to_numeric(
            result[home_score_col], errors="coerce"
        )
        result["_actual_away_score"] = pd.to_numeric(
            result[away_score_col], errors="coerce"
        )
        result["_result_source"] = source_name
        normalised.append(result)

    if not normalised:
        return pd.DataFrame(
            columns=[
                "_game_id",
                "_actual_home_score",
                "_actual_away_score",
                "_result_source",
            ]
        )

    return (
        pd.concat(normalised, ignore_index=True, sort=False)
        .dropna(subset=["_game_id"])
        .drop_duplicates(subset=["_game_id"], keep="last")
        .reset_index(drop=True)
    )


def select_market_for_game(
    market_rows: pd.DataFrame,
) -> pd.Series | None:
    """
    Market selection rules.

    1. Prefer genuine V2 snapshots captured BEFORE kickoff.
    2. Within pregame snapshots, prefer better market coverage.
    3. Then use the latest capture.
    4. If no pregame V2 snapshot exists, allow CFBD historical/backfill
       data for the reconstructed early-season games.
    """

    if market_rows.empty:
        return None

    rows = market_rows.copy()

    rows["_captured"] = pd.to_datetime(
        rows.get(
            "captured_at_utc",
            pd.Series(
                pd.NaT,
                index=rows.index,
            ),
        ),
        errors="coerce",
        utc=True,
    )

    capture_timing = (
        rows.get(
            "capture_timing",
            pd.Series(
                "",
                index=rows.index,
            ),
        )
        .astype(str)
    )

    rows["_pregame"] = (
        capture_timing
        .eq("pregame")
    )

    spread_series = pd.to_numeric(
        rows.get(
            "spread",
            pd.Series(
                np.nan,
                index=rows.index,
            ),
        ),
        errors="coerce",
    )

    home_ml_series = pd.to_numeric(
        rows.get(
            "home_moneyline",
            pd.Series(
                np.nan,
                index=rows.index,
            ),
        ),
        errors="coerce",
    )

    away_ml_series = pd.to_numeric(
        rows.get(
            "away_moneyline",
            pd.Series(
                np.nan,
                index=rows.index,
            ),
        ),
        errors="coerce",
    )

    rows["_coverage"] = (
        spread_series.notna().astype(int)
        +
        (
            home_ml_series.notna()
            | away_ml_series.notna()
        ).astype(int)
    )

    pregame = (
        rows[
            rows["_pregame"]
        ]
        .copy()
    )

    if not pregame.empty:

        pregame = pregame.sort_values(
            [
                "_coverage",
                "_captured",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )

        return pregame.iloc[0]

    # Historical/backfill only when we do not possess a genuine
    # pregame V2 capture for this game.
    rows = rows.sort_values(
        [
            "_coverage",
            "_captured",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    )

    return rows.iloc[0]


def legacy_rows_from_existing_output(
    rebuilt_game_ids: set[int],
) -> pd.DataFrame:
    """
    Migration safety net.

    Preserve any old settled betting rows that V2 cannot yet rebuild
    from its own market store. This protects the already-displayed
    historical chart during the V1 -> V2 migration.

    Once a game CAN be rebuilt natively by V2, the native row replaces
    the legacy one.
    """

    if not OUT_FILE.exists():
        return pd.DataFrame()

    old = pd.read_csv(
        OUT_FILE,
        low_memory=False,
    )

    if old.empty:
        return old

    # A row selected under an obsolete betting rule must never leak into the
    # current public record.  Preserve only rows already produced by the exact
    # active rules when their source game cannot currently be reconstructed.
    if "selection_rule" not in old.columns:
        return pd.DataFrame()

    old = old[
        old["selection_rule"].isin(
            [
                MONEYLINE_SELECTION_RULE,
                ATS_SELECTION_RULE,
            ]
        )
    ].copy()

    if old.empty:
        return old

    old_id_col = game_id_column(
        old
    )

    old["_game_id"] = pd.to_numeric(
        old[old_id_col],
        errors="coerce",
    ).astype("Int64")

    old_numeric_ids = pd.to_numeric(
        old["_game_id"],
        errors="coerce",
    )

    keep = (
        old_numeric_ids.notna()
        &
        ~old_numeric_ids.isin(
            list(rebuilt_game_ids)
        )
    )

    return (
        old.loc[
            keep
        ]
        .drop(
            columns=[
                "_game_id",
            ],
            errors="ignore",
        )
        .copy()
    )


# ============================================================
# BUILD
# ============================================================

def main() -> None:

    print("=" * 78)
    print(
        "CFB PREDICTION CENTRE 2026 "
        "- BUILD BETTING PERFORMANCE"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    games_path = V3_GAMES if V3_GAMES.exists() else V2_GAMES

    if not games_path.exists():

        raise FileNotFoundError(
            f"Missing app game data:\n{games_path}\n\n"
            "Run: python -m jobs.build_app_data"
        )

    # --------------------------------------------------------
    # LOAD THE RICHEST APP GAME SCHEMA AVAILABLE
    # --------------------------------------------------------

    games = pd.read_csv(
        games_path,
        low_memory=False,
    )

    games_id = game_id_column(
        games
    )

    games = games.copy()

    games["_game_id"] = pd.to_numeric(
        games[games_id],
        errors="coerce",
    ).astype("Int64")

    # --------------------------------------------------------
    # LOAD V2 COMPLETED RESULTS FROM SCORE PIPELINE
    # --------------------------------------------------------

    scoring = load_scoring_results()

    completed_scoring = (
        scoring[
            scoring[
                [
                    "_actual_home_score",
                    "_actual_away_score",
                ]
            ]
            .notna()
            .all(
                axis=1
            )
        ]
        .copy()
    )

    # Only bring the settlement/result fields into app games.
    result_columns = [
        "_game_id",
        "_actual_home_score",
        "_actual_away_score",
    ]

    joined_games = games.merge(
        completed_scoring[
            result_columns
        ],
        on="_game_id",
        how="left",
    )

    # --------------------------------------------------------
    # LOAD V2 MARKET HISTORY
    # --------------------------------------------------------

    market = load_market_history()

    print(
        f"App games:             "
        f"{len(games):,}"
    )

    print(
        f"App game source:       "
        f"{games_path}"
    )

    print(
        f"Result source rows:    "
        f"{len(scoring):,}"
    )

    print(
        f"Completed results:     "
        f"{len(completed_scoring):,}"
    )

    print(
        f"Market history rows:   "
        f"{len(market):,}"
    )

    print(
        f"Market source:         "
        f"{MARKET_HISTORY}"
    )

    if market.empty:

        print()

        print(
            "No market data exists yet."
        )

        print(
            "Run: python -m jobs.capture_market_lines"
        )

        print(
            "Existing settled betting output was left unchanged."
        )

        return

    market_id = game_id_column(
        market
    )

    market = market.copy()

    market["_game_id"] = pd.to_numeric(
        market[market_id],
        errors="coerce",
    ).astype("Int64")

    market = (
        market[
            market["_game_id"]
            .notna()
        ]
        .copy()
    )

    # --------------------------------------------------------
    # SETTLE COMPLETED GAMES
    # --------------------------------------------------------

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    rebuilt_game_ids: set[
        int
    ] = set()

    completed_count = 0
    matched_market_count = 0

    for _, game in joined_games.iterrows():

        game_id_value = game.get(
            "_game_id"
        )

        if pd.isna(
            game_id_value
        ):
            continue

        game_id = int(
            game_id_value
        )

        home_score = safe_float(
            game.get(
                "_actual_home_score"
            )
        )

        away_score = safe_float(
            game.get(
                "_actual_away_score"
            )
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        completed_count += 1

        home_team = clean(
            game.get(
                "home_team"
            )
        )

        away_team = clean(
            game.get(
                "away_team"
            )
        )

        predicted_winner = clean(
            game.get(
                "display_predicted_winner",
                game.get(
                    "predicted_winner"
                ),
            )
        )

        if not predicted_winner:

            predicted_winner = clean(
                game.get(
                    "predicted_winner"
                )
            )

        model_home = (
            normalise_team(
                predicted_winner
            )
            ==
            normalise_team(
                home_team
            )
        )

        model_away = (
            normalise_team(
                predicted_winner
            )
            ==
            normalise_team(
                away_team
            )
        )

        if not (
            model_home
            or model_away
        ):

            print(
                f"WARNING game {game_id}: "
                f"V2 pick '{predicted_winner}' "
                "did not match either team."
            )

            continue

        model_confidence = model_confidence_for_pick(
            game,
            model_home=model_home,
        )

        game_market = (
            market[
                market[
                    "_game_id"
                ]
                .eq(
                    game_id
                )
            ]
            .copy()
        )

        selected = (
            select_market_for_game(
                game_market
            )
        )

        if selected is None:
            continue

        matched_market_count += 1

        # This game has now been evaluated using the current rules.  Mark it
        # rebuilt even when neither bet qualifies so an old legacy row cannot
        # survive after the eligibility rules change.
        rebuilt_game_ids.add(
            game_id
        )

        if home_score > away_score:

            actual_winner = home_team

        elif away_score > home_score:

            actual_winner = away_team

        else:

            actual_winner = "Tie"

        capture_timing = clean(
            selected.get(
                "capture_timing"
            )
        )

        game_capture_mode = clean(
            game.get(
                "capture_mode"
            )
        )

        if (
            capture_timing
            ==
            "pregame"
        ):

            source_type = (
                "official_locked"
                if bool(
                    game.get(
                        "is_locked",
                        False,
                    )
                )
                else
                "live_completed"
            )

        else:

            source_type = (
                "historical_reconstruction"
                if (
                    game_capture_mode
                    ==
                    "historical_reconstruction"
                )
                else
                "historical_market_backfill"
            )

        base = {
            "cfbd_game_id": game_id,
            "start_date": game.get(
                "start_date_utc",
                game.get(
                    "start_date"
                ),
            ),
            "week": game.get(
                "week"
            ),
            "away_team": away_team,
            "home_team": home_team,
            "predicted_winner": (
                predicted_winner
            ),
            "model_confidence": (
                model_confidence
            ),
            "winner_confidence": (
                model_confidence
            ),
            "final_away_score": (
                away_score
            ),
            "final_home_score": (
                home_score
            ),
            "source_type": (
                source_type
            ),
            "market_provider": clean(
                selected.get(
                    "provider"
                )
            ),
            "market_captured_at_utc": (
                selected.get(
                    "captured_at_utc"
                )
            ),
            "market_capture_timing": (
                capture_timing
            ),
        }

        # ====================================================
        # MONEYLINE
        # ====================================================

        home_ml = safe_float(
            selected.get(
                "home_moneyline"
            )
        )

        away_ml = safe_float(
            selected.get(
                "away_moneyline"
            )
        )

        if model_home:

            ml_odds = home_ml
            bet_team = home_team

        else:

            ml_odds = away_ml
            bet_team = away_team

        if (
            ml_odds is not None
            and ml_odds >= MONEYLINE_MIN_AMERICAN_ODDS
            and ml_odds <= MONEYLINE_MAX_AMERICAN_ODDS
            and model_confidence is not None
            and model_confidence >= MONEYLINE_MIN_MODEL_CONFIDENCE
        ):

            if actual_winner == "Tie":

                result = "PUSH"
                profit_loss = 0.0

            elif (
                normalise_team(
                    bet_team
                )
                ==
                normalise_team(
                    actual_winner
                )
            ):

                result = "WIN"

                profit_loss = round(
                    american_profit(
                        ml_odds
                    ),
                    2,
                )

            else:

                result = "LOSS"
                profit_loss = (
                    -BET_STAKE
                )

            rows.append(
                {
                    **base,
                    "bet_type": (
                        "Moneyline"
                    ),
                    "bet_team": (
                        bet_team
                    ),
                    "market_line": (
                        ml_odds
                    ),
                    "market_odds": (
                        ml_odds
                    ),
                    "stake": (
                        BET_STAKE
                    ),
                    "result": (
                        result
                    ),
                    "profit_loss": (
                        profit_loss
                    ),
                    "selection_rule": (
                        MONEYLINE_SELECTION_RULE
                    ),
                }
            )

        # ====================================================
        # SPREAD / ATS
        # ====================================================

        spread = safe_float(
            selected.get(
                "spread"
            )
        )

        ats = (
            ats_projection(game, spread)
            if spread is not None
            else None
        )

        if (
            spread is not None
            and ats is not None
            and float(ats["cover_probability"])
            >= ATS_MIN_COVER_PROBABILITY
        ):

            # CFBD's existing convention in this project:
            #
            # Home -7.5 => spread = -7.5
            # Home +7.5 => spread = +7.5
            #
            # Therefore:
            #
            # adjusted_home_margin =
            #     home score - away score + home spread
            adjusted_home_margin = (
                home_score
                - away_score
                + spread
            )

            if bool(ats["bet_home"]):

                bet_team = (
                    home_team
                )

                bet_spread = (
                    spread
                )

                cover_value = (
                    adjusted_home_margin
                )

            else:

                bet_team = (
                    away_team
                )

                bet_spread = (
                    -spread
                )

                cover_value = (
                    -adjusted_home_margin
                )

            if abs(
                cover_value
            ) < 1e-9:

                result = "PUSH"
                profit_loss = 0.0

            elif cover_value > 0:

                result = "WIN"

                profit_loss = round(
                    american_profit(
                        ATS_AMERICAN_ODDS
                    ),
                    2,
                )

            else:

                result = "LOSS"

                profit_loss = (
                    -BET_STAKE
                )

            rows.append(
                {
                    **base,
                    "model_confidence": float(
                        ats["cover_probability"]
                    ),
                    "winner_confidence": model_confidence,
                    "ats_cover_probability": float(
                        ats["cover_probability"]
                    ),
                    "ats_home_cover_probability": float(
                        ats["home_cover_probability"]
                    ),
                    "ats_projected_home_margin": float(
                        ats["projected_home_margin"]
                    ),
                    "ats_margin_sd": float(ats["margin_sd"]),
                    "ats_edge_points": float(ats["edge_points"]),
                    "bet_type": (
                        "Spread"
                    ),
                    "bet_team": (
                        bet_team
                    ),
                    "market_line": (
                        bet_spread
                    ),
                    "market_odds": (
                        ATS_AMERICAN_ODDS
                    ),
                    "stake": (
                        BET_STAKE
                    ),
                    "result": (
                        result
                    ),
                    "profit_loss": (
                        profit_loss
                    ),
                    "selection_rule": (
                        ATS_SELECTION_RULE
                    ),
                }
            )

    output = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # MIGRATION SAFETY NET
    # --------------------------------------------------------

    legacy = (
        legacy_rows_from_existing_output(
            rebuilt_game_ids
        )
    )

    if not legacy.empty:

        output = pd.concat(
            [
                legacy,
                output,
            ],
            ignore_index=True,
            sort=False,
        )

    # --------------------------------------------------------
    # FINAL SCHEMA
    # --------------------------------------------------------

    expected_columns = [
        "cfbd_game_id",
        "start_date",
        "week",
        "away_team",
        "home_team",
        "predicted_winner",
        "winner_confidence",
        "model_confidence",
        "ats_cover_probability",
        "ats_home_cover_probability",
        "ats_projected_home_margin",
        "ats_margin_sd",
        "ats_edge_points",
        "final_away_score",
        "final_home_score",
        "source_type",
        "market_provider",
        "market_captured_at_utc",
        "market_capture_timing",
        "bet_type",
        "bet_team",
        "market_line",
        "market_odds",
        "stake",
        "result",
        "profit_loss",
        "selection_rule",
        "bet_number",
        "cumulative_profit_loss",
    ]

    if output.empty:

        output = pd.DataFrame(
            columns=expected_columns
        )

    else:

        for column in expected_columns:

            if column not in output.columns:
                output[column] = (
                    np.nan
                )

        output[
            "cfbd_game_id"
        ] = pd.to_numeric(
            output[
                "cfbd_game_id"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

        output[
            "start_date"
        ] = pd.to_datetime(
            output[
                "start_date"
            ],
            errors="coerce",
            utc=True,
        )

        # One settled row per game / bet type.
        # Native V2 rows are appended after any migration rows,
        # so keep="last" allows V2 to replace legacy safely.
        output = (
            output
            .sort_values(
                [
                    "bet_type",
                    "start_date",
                    "cfbd_game_id",
                ],
                kind="stable",
            )
            .drop_duplicates(
                subset=[
                    "cfbd_game_id",
                    "bet_type",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        output[
            "profit_loss"
        ] = pd.to_numeric(
            output[
                "profit_loss"
            ],
            errors="coerce",
        )

        output[
            "stake"
        ] = pd.to_numeric(
            output[
                "stake"
            ],
            errors="coerce",
        )

        output[
            "bet_number"
        ] = (
            output
            .groupby(
                "bet_type"
            )
            .cumcount()
            + 1
        )

        output[
            "cumulative_profit_loss"
        ] = (
            output
            .groupby(
                "bet_type"
            )[
                "profit_loss"
            ]
            .cumsum()
            .round(
                2
            )
        )

        output = output[
            expected_columns
            + [
                column
                for column in output.columns
                if column not in expected_columns
            ]
        ]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    APP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()

    print(
        f"Completed app games:    "
        f"{completed_count:,}"
    )

    print(
        f"Games with market:      "
        f"{matched_market_count:,}"
    )

    print(
        "Moneyline rule:         "
        "odds -350 to +350 and win probability >= 60%"
    )

    print(
        "ATS rule:               "
        "simulated cover probability >= 55%"
    )

    print()

    print(
        "-" * 78
    )

    for bet_type in [
        "Moneyline",
        "Spread",
    ]:

        view = (
            output[
                output[
                    "bet_type"
                ]
                .eq(
                    bet_type
                )
            ]
            .copy()
        )

        if view.empty:

            print(
                f"{bet_type:<12} "
                "0 bets"
            )

            continue

        result_text = (
            view[
                "result"
            ]
            .astype(
                str
            )
            .str
            .upper()
        )

        wins = int(
            result_text
            .eq(
                "WIN"
            )
            .sum()
        )

        losses = int(
            result_text
            .eq(
                "LOSS"
            )
            .sum()
        )

        pushes = int(
            result_text
            .eq(
                "PUSH"
            )
            .sum()
        )

        staked = float(
            pd.to_numeric(
                view[
                    "stake"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        total_pl = float(
            pd.to_numeric(
                view[
                    "profit_loss"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        print(
            f"{bet_type:<12} "
            f"{len(view):>3} bets | "
            f"{wins}W-"
            f"{losses}L-"
            f"{pushes}P | "
            f"staked "
            f"£{staked:,.0f} | "
            f"P/L "
            f"£{total_pl:+,.2f}"
        )

    print(
        "-" * 78
    )

    if not output.empty:

        print()
        print("Eligible tracked bets by week:")

        weekly = (
            output
            .groupby(
                [
                    "week",
                    "bet_type",
                ],
                dropna=False,
            )
            .size()
            .unstack(
                fill_value=0,
            )
            .sort_index()
        )

        print(
            weekly.to_string()
        )

        print()

    if output.empty:

        reconstructed_games = 0

    else:

        reconstructed_mask = (
            output[
                "source_type"
            ]
            .astype(
                str
            )
            .isin(
                [
                    "historical_reconstruction",
                    "historical_market_backfill",
                ]
            )
        )

        reconstructed_games = int(
            output.loc[
                reconstructed_mask,
                "cfbd_game_id",
            ]
            .nunique()
        )

    print(
        "Historical/reconstructed "
        "games represented: "
        f"{reconstructed_games:,}"
    )

    print(
        f"Saved -> "
        f"{OUT_FILE}"
    )

    print()

    print(
        "DONE"
    )


if __name__ == "__main__":
    main()
