from __future__ import annotations

"""Vectorised possession-level Monte Carlo for the production V3 model."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    simulations: int = 50_000
    seed: int = 20260908
    scoring_environment_sd: float = 0.16
    common_possession_sd: float = 0.85
    side_possession_sd: float = 0.55
    min_drives: int = 6
    max_drives: int = 19
    late_lead: int = 14
    leader_td_multiplier: float = 0.94
    leader_turnover_multiplier: float = 0.90
    trailer_td_multiplier: float = 1.07
    trailer_turnover_multiplier: float = 1.10


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _base_probabilities(matchup: dict, side: str) -> tuple[float, float, float, float]:
    """Create a valid drive-outcome distribution anchored to expected PPD."""
    td = float(np.clip(_number(matchup.get(f"{side}_td_rate"), 0.30), 0.02, 0.72))
    fg = float(np.clip(_number(matchup.get(f"{side}_fg_rate"), 0.12), 0.02, 0.32))
    turnover = float(
        np.clip(_number(matchup.get(f"{side}_turnover_rate"), 0.12), 0.015, 0.35)
    )
    downs = 0.04

    desired_ppd = float(np.clip(_number(matchup.get(f"{side}_ppd"), 2.25), 0.15, 5.50))
    base_ppd = max(0.10, 7.0 * td + 3.0 * fg)
    scoring_scale = float(np.clip(desired_ppd / base_ppd, 0.55, 1.45))
    td *= scoring_scale
    fg *= scoring_scale

    # Preserve the structural PPD anchor while allowing matchup context to
    # reshape the simulated distribution. Poor field position/efficiency and
    # elevated turnover exposure must reduce scoring rather than merely
    # relabelling otherwise empty possessions.
    start = _number(matchup.get(f"{side}_start_field"), 27.5)
    success = _number(matchup.get(f"{side}_success_rate"), 0.42)
    explosive = _number(matchup.get(f"{side}_explosive_rate"), 0.10)
    context = np.clip(
        1.0 + 0.008 * (start - 27.5) + 0.35 * (success - 0.42)
        + 0.55 * (explosive - 0.10),
        0.82,
        1.22,
    )
    turnover_drag = np.clip(((1.0 - turnover) / 0.88) ** 0.45, 0.84, 1.08)
    td *= float(context * turnover_drag)
    fg *= float(context * turnover_drag)

    scoring_room = max(0.03, 0.97 - turnover - downs)
    if td + fg > scoring_room:
        scale = scoring_room / (td + fg)
        td *= scale
        fg *= scale

    return td, fg, turnover, downs


def _score_half(
    rng: np.random.Generator,
    drives: np.ndarray,
    base: tuple[float, float, float, float],
    environment: np.ndarray,
    td_multiplier: np.ndarray | float = 1.0,
    turnover_multiplier: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one half for every simulation and return points and turnovers."""
    n = len(drives)
    max_drives = int(drives.max(initial=0))
    if max_drives <= 0:
        return np.zeros(n, dtype=np.int16), np.zeros(n, dtype=np.int16)

    td0, fg0, to0, downs = base
    td = np.asarray(td0 * np.asarray(td_multiplier) * environment, dtype=float)
    fg = np.asarray(fg0 * environment, dtype=float)
    turnover = np.asarray(to0 * np.asarray(turnover_multiplier), dtype=float)
    td = np.broadcast_to(td, (n,)).copy()
    fg = np.broadcast_to(fg, (n,)).copy()
    turnover = np.broadcast_to(turnover, (n,)).copy()

    turnover = np.clip(turnover, 0.01, 0.38)
    scoring_room = np.maximum(0.02, 0.97 - turnover - downs)
    scoring_total = td + fg
    excess = scoring_total > scoring_room
    scale = np.ones(n, dtype=float)
    scale[excess] = scoring_room[excess] / scoring_total[excess]
    td = np.clip(td * scale, 0.005, 0.80)
    fg = np.clip(fg * scale, 0.005, 0.40)

    uniform = rng.random((n, max_drives))
    active = np.arange(max_drives)[None, :] < drives[:, None]
    td_cut = td[:, None]
    fg_cut = (td + fg)[:, None]
    to_cut = (td + fg + turnover)[:, None]

    touchdowns = ((uniform < td_cut) & active).sum(axis=1)
    field_goals = ((uniform >= td_cut) & (uniform < fg_cut) & active).sum(axis=1)
    turnovers = ((uniform >= fg_cut) & (uniform < to_cut) & active).sum(axis=1)
    points = 7 * touchdowns + 3 * field_goals
    return points.astype(np.int16), turnovers.astype(np.int16)


def simulate_game(
    matchup: dict,
    simulations: int | None = None,
    seed: int | None = None,
    cfg: SimulationConfig | None = None,
) -> dict[str, float | int]:
    """Simulate regulation and overtime from coupled possession distributions."""
    config = cfg or SimulationConfig()
    n = int(simulations if simulations is not None else config.simulations)
    if n < 100:
        raise ValueError("simulations must be at least 100")
    actual_seed = int(seed if seed is not None else config.seed)
    rng = np.random.default_rng(actual_seed)

    common = rng.normal(0.0, config.common_possession_sd, n)
    home_drives = np.rint(
        _number(matchup.get("home_drives"), 11.5)
        + common
        + rng.normal(0.0, config.side_possession_sd, n)
    )
    away_drives = np.rint(
        _number(matchup.get("away_drives"), 11.5)
        + common
        + rng.normal(0.0, config.side_possession_sd, n)
    )
    home_drives = np.clip(home_drives, config.min_drives, config.max_drives).astype(np.int16)
    away_drives = np.clip(away_drives, config.min_drives, config.max_drives).astype(np.int16)

    home_first = home_drives // 2
    away_first = away_drives // 2
    home_second = home_drives - home_first
    away_second = away_drives - away_first

    environment = np.exp(
        rng.normal(
            -0.5 * config.scoring_environment_sd**2,
            config.scoring_environment_sd,
            n,
        )
    )
    home_base = _base_probabilities(matchup, "home")
    away_base = _base_probabilities(matchup, "away")

    home_scores, home_turnovers = _score_half(rng, home_first, home_base, environment)
    away_scores, away_turnovers = _score_half(rng, away_first, away_base, environment)

    halftime_margin = home_scores.astype(int) - away_scores.astype(int)
    home_td_mult = np.where(
        halftime_margin <= -config.late_lead,
        config.trailer_td_multiplier,
        np.where(halftime_margin >= config.late_lead, config.leader_td_multiplier, 1.0),
    )
    home_to_mult = np.where(
        halftime_margin <= -config.late_lead,
        config.trailer_turnover_multiplier,
        np.where(halftime_margin >= config.late_lead, config.leader_turnover_multiplier, 1.0),
    )
    away_td_mult = np.where(
        halftime_margin >= config.late_lead,
        config.trailer_td_multiplier,
        np.where(halftime_margin <= -config.late_lead, config.leader_td_multiplier, 1.0),
    )
    away_to_mult = np.where(
        halftime_margin >= config.late_lead,
        config.trailer_turnover_multiplier,
        np.where(halftime_margin <= -config.late_lead, config.leader_turnover_multiplier, 1.0),
    )

    points, turnovers = _score_half(
        rng, home_second, home_base, environment, home_td_mult, home_to_mult
    )
    home_scores = home_scores + points
    home_turnovers = home_turnovers + turnovers
    points, turnovers = _score_half(
        rng, away_second, away_base, environment, away_td_mult, away_to_mult
    )
    away_scores = away_scores + points
    away_turnovers = away_turnovers + turnovers

    tied = home_scores == away_scores
    tie_count = int(tied.sum())
    if tie_count:
        elo_difference = _number(matchup.get("elo_difference"), 0.0)
        home_ot_probability = 1.0 / (1.0 + 10.0 ** (-elo_difference / 400.0))
        home_ot_win = rng.random(tie_count) < home_ot_probability
        overtime_points = np.where(rng.random(tie_count) < 0.30, 3, 7).astype(np.int16)
        tied_indices = np.flatnonzero(tied)
        home_scores[tied_indices[home_ot_win]] += overtime_points[home_ot_win]
        away_scores[tied_indices[~home_ot_win]] += overtime_points[~home_ot_win]

    margin = home_scores.astype(int) - away_scores.astype(int)
    total = home_scores.astype(int) + away_scores.astype(int)

    def quantile(values: np.ndarray, probability: float) -> float:
        return float(np.quantile(values, probability))

    return {
        "home_win_probability": float((margin > 0).mean()),
        "away_win_probability": float((margin < 0).mean()),
        "projected_home_points": float(home_scores.mean()),
        "projected_away_points": float(away_scores.mean()),
        "projected_margin": float(margin.mean()),
        "projected_total": float(total.mean()),
        "expected_home_drives": float(home_drives.mean()),
        "expected_away_drives": float(away_drives.mean()),
        "expected_home_turnovers": float(home_turnovers.mean()),
        "expected_away_turnovers": float(away_turnovers.mean()),
        "home_p10": quantile(home_scores, 0.10),
        "home_p90": quantile(home_scores, 0.90),
        "away_p10": quantile(away_scores, 0.10),
        "away_p90": quantile(away_scores, 0.90),
        "margin_p10": quantile(margin, 0.10),
        "margin_p90": quantile(margin, 0.90),
        "total_p10": quantile(total, 0.10),
        "total_p90": quantile(total, 0.90),
        "home_40_plus_probability": float((home_scores >= 40).mean()),
        "away_40_plus_probability": float((away_scores >= 40).mean()),
        "home_10_or_less_probability": float((home_scores <= 10).mean()),
        "away_10_or_less_probability": float((away_scores <= 10).mean()),
        "blowout_28_plus_probability": float((np.abs(margin) >= 28).mean()),
        "simulations": n,
        "seed": actual_seed,
    }
