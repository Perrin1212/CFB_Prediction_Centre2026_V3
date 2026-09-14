from __future__ import annotations

"""Feature engineering for the CFB-V3 research engine.

V3 is deliberately additive.  It consumes the frozen V2 chronological states and
adds schedule-quality, non-linear matchup, field-position and volatility terms.
Nothing in this module mutates V2 state or market data.
"""

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V3FeatureConfig:
    schedule_alpha: float = 0.28
    schedule_prior_games: float = 5.0
    schedule_prior_rating: float = 100.0
    min_rating: float = 55.0
    max_rating: float = 145.0


def safe_num(x, default=0.0) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _team_schedule_quality(scoring: pd.DataFrame, cfg: V3FeatureConfig) -> pd.DataFrame:
    """Create *pregame* opponent-quality histories with no current-game leakage.

    Offence schedule quality = quality of defences previously faced.
    Defence schedule quality = quality of offences previously faced.
    Ratings are the opponent's opponent-adjusted PRE-GAME ratings, so a 400-yard
    day against a weak defence does not earn the same schedule context as one
    against an elite defence.
    """
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = scoring[["game_id", "season", "week", "start_date", f"{side}_team",
                     f"{opp}_adjusted_defense_rating_pre", f"{opp}_adjusted_offense_rating_pre"]].copy()
        x.columns = ["game_id", "season", "week", "start_date", "team", "opp_def", "opp_off"]
        x["side"] = side
        rows.append(x)
    long = pd.concat(rows, ignore_index=True)
    long["start_date"] = pd.to_datetime(long["start_date"], errors="coerce", utc=True)
    long = long.sort_values(["start_date", "game_id", "side"], kind="stable")

    state: dict[str, dict[str, float]] = {}
    out = []
    prior_n = cfg.schedule_prior_games
    for r in long.itertuples(index=False):
        team = str(r.team)
        s = state.setdefault(team, {"n": 0.0, "def": cfg.schedule_prior_rating, "off": cfg.schedule_prior_rating})
        n = s["n"]
        # Shrink early-season schedule quality to average.
        w = n / (n + prior_n) if n > 0 else 0.0
        pre_def = (1 - w) * cfg.schedule_prior_rating + w * s["def"]
        pre_off = (1 - w) * cfg.schedule_prior_rating + w * s["off"]
        out.append({"game_id": r.game_id, "team": team, "side": r.side,
                    "off_schedule_def_quality_pre": pre_def,
                    "def_schedule_off_quality_pre": pre_off,
                    "schedule_games_pre": int(n)})
        od = np.clip(safe_num(r.opp_def, 100.0), cfg.min_rating, cfg.max_rating)
        oo = np.clip(safe_num(r.opp_off, 100.0), cfg.min_rating, cfg.max_rating)
        if n == 0:
            s["def"], s["off"] = od, oo
        else:
            a = cfg.schedule_alpha
            s["def"] = (1-a)*s["def"] + a*od
            s["off"] = (1-a)*s["off"] + a*oo
        s["n"] += 1
    return pd.DataFrame(out)


def add_v3_features(scoring: pd.DataFrame, cfg: V3FeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or V3FeatureConfig()
    df = scoring.copy()
    sched = _team_schedule_quality(df, cfg)
    for side in ("home", "away"):
        s = sched[sched["side"].eq(side)].drop(columns="side").copy()
        s = s.rename(columns={c: f"{side}_{c}" for c in s.columns if c not in {"game_id", "team"}})
        s = s.drop(columns="team")
        df = df.merge(s, on="game_id", how="left", validate="one_to_one")

    for side, opp in (("home", "away"), ("away", "home")):
        raw_o = pd.to_numeric(df[f"{side}_raw_offense_rating_pre"], errors="coerce").fillna(100.0)
        raw_d = pd.to_numeric(df[f"{side}_raw_defense_rating_pre"], errors="coerce").fillna(100.0)
        adj_o = pd.to_numeric(df[f"{side}_adjusted_offense_rating_pre"], errors="coerce").fillna(100.0)
        adj_d = pd.to_numeric(df[f"{side}_adjusted_defense_rating_pre"], errors="coerce").fillna(100.0)
        opp_d = pd.to_numeric(df[f"{opp}_adjusted_defense_rating_pre"], errors="coerce").fillna(100.0)
        opp_o = pd.to_numeric(df[f"{opp}_adjusted_offense_rating_pre"], errors="coerce").fillna(100.0)
        sched_d = pd.to_numeric(df[f"{side}_off_schedule_def_quality_pre"], errors="coerce").fillna(100.0)
        sched_o = pd.to_numeric(df[f"{side}_def_schedule_off_quality_pre"], errors="coerce").fillna(100.0)

        # Explicit schedule correction. Facing weak defences (<100) pulls raw
        # offensive production down; facing elite defences pushes it up.
        df[f"{side}_schedule_corrected_offense"] = raw_o * (sched_d / 100.0)
        df[f"{side}_schedule_corrected_defense"] = raw_d * (sched_o / 100.0)

        # Matchup is multiplicative, not merely additive.  Squared/log terms
        # let the challenger learn blowout amplification at the extremes.
        ratio = (adj_o.clip(55,145) / opp_d.clip(55,145)).clip(0.45, 2.2)
        inv_ratio = (adj_d.clip(55,145) / opp_o.clip(55,145)).clip(0.45, 2.2)
        df[f"{side}_offense_vs_defense_ratio"] = ratio
        df[f"{side}_offense_matchup_log"] = np.log(ratio)
        df[f"{side}_offense_matchup_extreme"] = np.sign(ratio-1.0) * np.square(ratio-1.0)
        df[f"{side}_defense_vs_offense_ratio"] = inv_ratio

        # Existing sustainability/turnover states provide useful drive shape.
        fd = pd.to_numeric(df[f"{side}_offense_first_down_rate_pre"] if f"{side}_offense_first_down_rate_pre" in df.columns else pd.Series(0.30, index=df.index), errors="coerce").fillna(0.30)
        tov = pd.to_numeric(df[f"{side}_offense_turnover_rate_pre"] if f"{side}_offense_turnover_rate_pre" in df.columns else pd.Series(0.022, index=df.index), errors="coerce").fillna(0.022)
        opp_take = pd.to_numeric(df[f"{opp}_defense_takeaway_rate_pre"] if f"{opp}_defense_takeaway_rate_pre" in df.columns else pd.Series(0.022, index=df.index), errors="coerce").fillna(0.022)
        df[f"{side}_expected_turnover_pressure"] = 0.55*tov + 0.45*opp_take
        df[f"{side}_sustainability_pressure"] = fd / np.maximum(
            pd.to_numeric(df[f"{opp}_defense_first_down_rate_allowed_pre"] if f"{opp}_defense_first_down_rate_allowed_pre" in df.columns else pd.Series(0.30, index=df.index), errors="coerce").fillna(0.30), 0.12)

        # Latent starting field position. It is intentionally labelled latent:
        # our current historical source has no drive start coordinates.
        ret = pd.to_numeric(df[f"{side}_return_field_position_signal_pre"] if f"{side}_return_field_position_signal_pre" in df.columns else pd.Series(0.0, index=df.index), errors="coerce").fillna(0.0)
        turnover_edge = (opp_take - tov) * 100.0
        df[f"{side}_latent_start_field_position"] = (28.0 + 0.55*turnover_edge + ret).clip(20.0, 40.0)

    df["v3_matchup_extreme_difference"] = df["home_offense_matchup_extreme"] - df["away_offense_matchup_extreme"]
    df["v3_schedule_corrected_strength_difference"] = (
        0.5*(df["home_schedule_corrected_offense"] + df["home_schedule_corrected_defense"])
        - 0.5*(df["away_schedule_corrected_offense"] + df["away_schedule_corrected_defense"])
    )
    return df


V3_MODEL_FEATURES = [
    "elo_difference", "offensive_matchup_difference", "adjusted_strength_difference",
    "expected_home_plays", "expected_away_plays", "expected_total_plays", "game_pace_index",
    "home_adjusted_offense_rating_pre", "home_adjusted_defense_rating_pre",
    "away_adjusted_offense_rating_pre", "away_adjusted_defense_rating_pre",
    "home_off_schedule_def_quality_pre", "away_off_schedule_def_quality_pre",
    "home_def_schedule_off_quality_pre", "away_def_schedule_off_quality_pre",
    "home_schedule_corrected_offense", "away_schedule_corrected_offense",
    "home_schedule_corrected_defense", "away_schedule_corrected_defense",
    "home_offense_vs_defense_ratio", "away_offense_vs_defense_ratio",
    "home_offense_matchup_log", "away_offense_matchup_log",
    "home_offense_matchup_extreme", "away_offense_matchup_extreme",
    "home_expected_turnover_pressure", "away_expected_turnover_pressure",
    "home_sustainability_pressure", "away_sustainability_pressure",
    "home_latent_start_field_position", "away_latent_start_field_position",
    "v3_matchup_extreme_difference", "v3_schedule_corrected_strength_difference",
]
