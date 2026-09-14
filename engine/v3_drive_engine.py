from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import json, math
import numpy as np
import pandas as pd

@dataclass
class TeamDriveState:
    games:int=0
    off_drives:float=11.5; def_drives:float=11.5
    off_ppd:float=2.25; def_ppd:float=2.25
    off_td_rate:float=.30; def_td_rate:float=.30
    off_fg_rate:float=.12; def_fg_rate:float=.12
    off_to_rate:float=.12; def_to_rate:float=.12
    off_punt_rate:float=.36; def_punt_rate:float=.36
    off_start:float=27.5; def_start:float=27.5
    off_success:float=.42; def_success:float=.42
    off_explosive:float=.90; def_explosive:float=.90
    off_ppa:float=0.0; def_ppa:float=0.0
    schedule_def:float=100.; schedule_off:float=100.

@dataclass(frozen=True)
class DriveStateConfig:
    alpha:float=.24
    early_prior_games:float=5.0
    opponent_strength:float=.55
    factor_floor:float=.72
    factor_ceiling:float=1.30
    season_regression:float=.42

class DriveStateEngine:
    """Chronological drive-level offense/defense state with opponent adjustment.

    A team's observed offense is scaled by the defense it faced, and its observed
    defense by the offense it faced. Current-game results are never used in the
    pregame snapshot.
    """
    def __init__(self,cfg:DriveStateConfig|None=None): self.cfg=cfg or DriveStateConfig(); self.states={}
    def get(self,team:str)->TeamDriveState:
        return self.states.setdefault(str(team).strip(),TeamDriveState())
    def regress_season(self):
        r=self.cfg.season_regression
        prior=TeamDriveState()
        for s in self.states.values():
            for k,v in asdict(s).items():
                if k=='games': continue
                setattr(s,k,(1-r)*float(v)+r*float(getattr(prior,k)))
            s.games=0
    def _factor(self,rating:float)->float:
        return float(np.clip((rating/100.)**self.cfg.opponent_strength,self.cfg.factor_floor,self.cfg.factor_ceiling))
    @staticmethod
    def _rating_off(s:TeamDriveState)->float:
        return float(np.clip(100*(s.off_ppd/2.25)**.45*(s.off_success/.42)**.25*(max(s.off_ppa+.18,.04)/.18)**.15*(s.off_start/27.5)**.15,55,150))
    @staticmethod
    def _rating_def(s:TeamDriveState)->float:
        # >100 means stronger defense
        return float(np.clip(100*(2.25/max(s.def_ppd,.2))**.45*(.42/max(s.def_success,.08))**.25*(.18/max(s.def_ppa+.18,.04))**.15*(27.5/max(s.def_start,15))**.15,55,150))
    def snapshot(self,team:str)->dict[str,float]:
        s=self.get(team); n=s.games; w=n/(n+self.cfg.early_prior_games) if n else 0
        return {**{f'drive_{k}_pre':v for k,v in asdict(s).items()},
                'drive_off_rating_pre':self._rating_off(s),'drive_def_rating_pre':self._rating_def(s),
                'drive_state_maturity':w}
    def update_game(self,home:str,away:str,h:dict[str,float],a:dict[str,float]):
        hs,as_=self.get(home),self.get(away)
        h_def_q=self._rating_def(as_); a_def_q=self._rating_def(hs)
        h_off_q=self._rating_off(as_); a_off_q=self._rating_off(hs)
        self._update_one(hs,h,a_def_q,a_off_q)
        self._update_one(as_,a,h_def_q,h_off_q)
    def _update_one(self,s:TeamDriveState,g:dict[str,float],opp_def_q:float,opp_off_q:float):
        a=self.cfg.alpha if s.games else .38
        def ew(k,x): setattr(s,k,(1-a)*getattr(s,k)+a*float(x))
        # Offense: strong opponent defense (>100) increases credit; weak defense reduces it.
        df=self._factor(opp_def_q); of=self._factor(opp_off_q)
        ew('off_drives',g['drives']); ew('off_ppd',g['ppd']*df); ew('off_td_rate',g['td_rate']*df)
        ew('off_fg_rate',g['fg_rate']); ew('off_to_rate',g['to_rate']/df); ew('off_punt_rate',g['punt_rate']/df)
        ew('off_start',27.5+(g['avg_start']-27.5)*df); ew('off_success',g['success_rate']*df)
        ew('off_explosive',g['explosiveness']*df); ew('off_ppa',g['ppa']*df)
        # Defense observed values are "allowed"; strong opponent offense makes allowing production less bad.
        ew('def_drives',g['opp_drives']); ew('def_ppd',g['opp_ppd']/of); ew('def_td_rate',g['opp_td_rate']/of)
        ew('def_fg_rate',g['opp_fg_rate']); ew('def_to_rate',g['opp_to_rate']*of); ew('def_punt_rate',g['opp_punt_rate']*of)
        ew('def_start',27.5+(g['opp_avg_start']-27.5)/of); ew('def_success',g['opp_success_rate']/of)
        ew('def_explosive',g['opp_explosiveness']/of); ew('def_ppa',g['opp_ppa']/of)
        ew('schedule_def',opp_def_q); ew('schedule_off',opp_off_q); s.games+=1

def matchup_features(home:dict,away:dict)->dict[str,float]:
    def v(d,k,default):
        try: x=float(d.get(k,default)); return x if np.isfinite(x) else default
        except: return default
    out={}
    # Expected possessions: both offenses and opposing defenses contribute, then common game environment couples them.
    hd=.52*v(home,'drive_off_drives_pre',11.5)+.48*v(away,'drive_def_drives_pre',11.5)
    ad=.52*v(away,'drive_off_drives_pre',11.5)+.48*v(home,'drive_def_drives_pre',11.5)
    env=np.clip((hd+ad)/2,8.0,15.5); out['v3_expected_home_drives']=.7*hd+.3*env; out['v3_expected_away_drives']=.7*ad+.3*env
    for side,o,d in [('home',home,away),('away',away,home)]:
        # harmonic/geometric blends avoid raw-average domination and allow mismatch amplification.
        off_ppd=max(v(o,'drive_off_ppd_pre',2.25),.15); def_allow=max(v(d,'drive_def_ppd_pre',2.25),.15)
        oq=v(o,'drive_off_rating_pre',100); dq=v(d,'drive_def_rating_pre',100)
        mismatch=np.clip((oq/max(100+(dq-100),55)),.5,1.8)
        ppd=np.sqrt(off_ppd*def_allow)*np.clip(mismatch**.35,.82,1.22)
        start=.55*v(o,'drive_off_start_pre',27.5)+.45*v(d,'drive_def_start_pre',27.5)
        success=.55*v(o,'drive_off_success_pre',.42)+.45*v(d,'drive_def_success_pre',.42)
        explosive=.55*v(o,'drive_off_explosive_pre',.9)+.45*v(d,'drive_def_explosive_pre',.9)
        tor=.55*v(o,'drive_off_to_rate_pre',.12)+.45*v(d,'drive_def_to_rate_pre',.12)
        out[f'v3_{side}_base_ppd']=float(np.clip(ppd,.25,5.5)); out[f'v3_{side}_start_field']=float(np.clip(start,18,48))
        out[f'v3_{side}_success']=float(np.clip(success,.15,.72)); out[f'v3_{side}_explosive']=float(np.clip(explosive,.1,3.5)); out[f'v3_{side}_turnover_rate']=float(np.clip(tor,.025,.30))
        out[f'v3_{side}_matchup_ratio']=float(mismatch)
    return out

def save_states(engine:DriveStateEngine,path):
    with open(path,'w',encoding='utf-8') as f: json.dump({k:asdict(v) for k,v in engine.states.items()},f,indent=2)
def load_states(engine:DriveStateEngine,path):
    with open(path,encoding='utf-8') as f: raw=json.load(f)
    engine.states={k:TeamDriveState(**v) for k,v in raw.items()}
