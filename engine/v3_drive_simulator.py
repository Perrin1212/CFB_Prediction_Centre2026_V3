from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    simulations:int=30000; seed:int=20260908; drive_sd:float=1.05
    garbage_lead:int=21; leader_ppd_mult:float=.90; trailer_ppd_mult:float=1.05

def _probs(ppd,to_rate,start,success,explosive):
    # Calibrated shape function; trained PPD remains the anchor, context reallocates TD vs FG/punt/turnover.
    field=np.clip(1+.012*(start-27.5),.82,1.25); eff=np.clip(1+.55*(success-.42)+.06*(explosive-.9),.72,1.35)
    target=float(np.clip(ppd*field*eff,.15,6.1)); pto=float(np.clip(to_rate,.025,.27))
    ptd=float(np.clip(.018+.105*target+.014*target*target,.015,.72)); pfg=float(np.clip((target-7*ptd)/3,.025,.28))
    room=max(.05,.96-pto)
    if ptd+pfg>room: scale=room/(ptd+pfg); ptd*=scale; pfg*=scale
    return ptd,pfg,pto

def _score(rng,drives,ppd,to,start,success,explosive,mult=1.):
    ptd,pfg,pto=_probs(ppd*mult,to,start,success,explosive); u=rng.random((len(drives),int(np.max(drives))))
    idx=np.arange(u.shape[1])[None,:] < drives[:,None]
    td=((u<ptd)&idx).sum(1); fg=((u>=ptd)&(u<ptd+pfg)&idx).sum(1)
    return 7*td+3*fg

def simulate_game(home_drives,away_drives,home_ppd,away_ppd,home_to,away_to,home_start,away_start,home_success,away_success,home_explosive,away_explosive,cfg=None):
    cfg=cfg or SimulationConfig(); rng=np.random.default_rng(cfg.seed); n=cfg.simulations
    hd=np.clip(np.rint(rng.normal(home_drives,cfg.drive_sd,n)),6,19).astype(int); ad=np.clip(np.rint(rng.normal(away_drives,cfg.drive_sd,n)),6,19).astype(int)
    h1=hd//2;a1=ad//2; h2=hd-h1;a2=ad-a1
    hs=_score(rng,h1,home_ppd,home_to,home_start,home_success,home_explosive); aw=_score(rng,a1,away_ppd,away_to,away_start,away_success,away_explosive)
    diff=hs-aw; hm=np.ones(n);am=np.ones(n); hm[diff>=cfg.garbage_lead]=cfg.leader_ppd_mult; hm[diff<=-cfg.garbage_lead]=cfg.trailer_ppd_mult; am[diff<=-cfg.garbage_lead]=cfg.leader_ppd_mult; am[diff>=cfg.garbage_lead]=cfg.trailer_ppd_mult
    # grouped state multipliers for fast vectorized second half
    for mask,m in [(hm<1,.90),(hm>1,1.05),(hm==1,1.)]:
        if mask.any(): hs[mask]+=_score(rng,h2[mask],home_ppd,home_to,home_start,home_success,home_explosive,m)
    for mask,m in [(am<1,.90),(am>1,1.05),(am==1,1.)]:
        if mask.any(): aw[mask]+=_score(rng,a2[mask],away_ppd,away_to,away_start,away_success,away_explosive,m)
    margin=hs-aw; total=hs+aw
    q=lambda x,p:float(np.quantile(x,p))
    return {'v3_home_win_probability':float((margin>0).mean()+.5*(margin==0).mean()),'v3_away_win_probability':float((margin<0).mean()+.5*(margin==0).mean()),
      'v3_projected_home_points':float(hs.mean()),'v3_projected_away_points':float(aw.mean()),'v3_projected_margin':float(margin.mean()),'v3_projected_total':float(total.mean()),
      'v3_home_points_median':q(hs,.5),'v3_away_points_median':q(aw,.5),'v3_margin_p10':q(margin,.1),'v3_margin_p90':q(margin,.9),'v3_total_p10':q(total,.1),'v3_total_p90':q(total,.9),
      'v3_home_40_plus_probability':float((hs>=40).mean()),'v3_away_40_plus_probability':float((aw>=40).mean()),'v3_home_10_or_less_probability':float((hs<=10).mean()),'v3_away_10_or_less_probability':float((aw<=10).mean()),'v3_blowout_28_plus_probability':float((abs(margin)>=28).mean())}
