from __future__ import annotations

"""Fast drive-level Monte Carlo simulator for the CFB-V3 challenger.

The simulation is possession based.  It models first-half drives, observes the
simulated score state, then changes second-half efficiency for large leads and
large deficits.  This keeps the important game-state mechanism without a slow
play-by-play loop.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class V3SimulationConfig:
    simulations: int = 20000
    seed: int = 202603
    drive_sd: float = 1.20
    garbage_time_lead: int = 21
    leader_efficiency_mult: float = 0.94
    trailer_efficiency_mult: float = 1.04
    trailer_turnover_mult: float = 1.10


def _drive_probs(ppd: float, turnover: float, field_pos: float, explosive: float = 0.0):
    ppd = float(np.clip(ppd * (1.0 + 0.010*(field_pos-28.0)), 0.20, 5.9))
    turnover = float(np.clip(turnover, 0.03, 0.23))
    # Convex TD response allows extreme mismatches to create extreme score tails.
    p_td = float(np.clip(0.025 + 0.090*ppd + 0.020*ppd*ppd + 0.025*explosive, 0.025, 0.72))
    # Choose FG probability so expected scoring remains close to requested PPD.
    p_fg = float(np.clip((ppd - 7.0*p_td)/3.0, 0.025, 0.30))
    max_score_prob = max(0.10, 0.94-turnover)
    if p_td+p_fg > max_score_prob:
        scale=max_score_prob/(p_td+p_fg); p_td*=scale; p_fg*=scale
    return p_td, p_fg


def _period_score(rng, drives, ppd, turnover, field_pos, explosive, mult=None):
    drives=np.asarray(drives,dtype=int)
    out=np.zeros(len(drives),dtype=float)
    mult=np.ones(len(drives)) if mult is None else np.asarray(mult,dtype=float)
    # only three efficiency states are used; vectorized within each state
    states=[(mult<0.98,0.94),(mult>1.02,1.04),((mult>=0.98)&(mult<=1.02),1.0)]
    for mask,m in states:
        if not mask.any(): continue
        ptd,pfg=_drive_probs(ppd*m, turnover, field_pos, explosive)
        n=drives[mask]
        td=rng.binomial(n,ptd)
        rem=n-td
        cond_fg=np.clip(pfg/max(1.0-ptd,1e-9),0,1)
        fg=rng.binomial(rem,cond_fg)
        out[mask]=7*td+3*fg
    return out


def simulate_game(home_drives: float, away_drives: float, home_ppd: float, away_ppd: float,
                  home_turnover: float, away_turnover: float, home_field_pos: float = 28.0,
                  away_field_pos: float = 28.0, home_explosive: float = 0.0,
                  away_explosive: float = 0.0, cfg: V3SimulationConfig | None = None) -> dict:
    cfg=cfg or V3SimulationConfig(); rng=np.random.default_rng(cfg.seed); n=cfg.simulations
    hd=np.clip(np.rint(rng.normal(home_drives,cfg.drive_sd,n)),6,19).astype(int)
    ad=np.clip(np.rint(rng.normal(away_drives,cfg.drive_sd,n)),6,19).astype(int)
    h1=hd//2; a1=ad//2; h2=hd-h1; a2=ad-a1
    hs=_period_score(rng,h1,home_ppd,home_turnover,home_field_pos,home_explosive)
    aw=_period_score(rng,a1,away_ppd,away_turnover,away_field_pos,away_explosive)
    diff=hs-aw
    hm=np.ones(n); am=np.ones(n)
    hm[diff>=cfg.garbage_time_lead]=cfg.leader_efficiency_mult
    hm[diff<=-cfg.garbage_time_lead]=cfg.trailer_efficiency_mult
    am[diff<=-cfg.garbage_time_lead]=cfg.leader_efficiency_mult
    am[diff>=cfg.garbage_time_lead]=cfg.trailer_efficiency_mult
    hs+=_period_score(rng,h2,home_ppd,home_turnover,home_field_pos,home_explosive,hm)
    aw+=_period_score(rng,a2,away_ppd,away_turnover,away_field_pos,away_explosive,am)
    margin=hs-aw; total=hs+aw
    return {
        'home_win_probability':float(np.mean(margin>0)+.5*np.mean(margin==0)),
        'away_win_probability':float(np.mean(margin<0)+.5*np.mean(margin==0)),
        'home_points_mean':float(hs.mean()),'away_points_mean':float(aw.mean()),
        'home_points_median':float(np.median(hs)),'away_points_median':float(np.median(aw)),
        'margin_mean':float(margin.mean()),'total_mean':float(total.mean()),
        'margin_p10':float(np.quantile(margin,.10)),'margin_p90':float(np.quantile(margin,.90)),
        'total_p10':float(np.quantile(total,.10)),'total_p90':float(np.quantile(total,.90)),
        'home_40_plus_probability':float(np.mean(hs>=40)),'away_40_plus_probability':float(np.mean(aw>=40)),
        'blowout_28_plus_probability':float(np.mean(np.abs(margin)>=28)),
    }
