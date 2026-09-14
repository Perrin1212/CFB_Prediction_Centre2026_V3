from __future__ import annotations
import numpy as np

def _g(d,k,x):
    try:v=float(d.get(k,x));return v if np.isfinite(v) else x
    except:return x

def _blend(off,allowed,off_rating,def_rating,power=.45):
    base=np.sqrt(max(off,.001)*max(allowed,.001));ratio=np.clip(off_rating/max(def_rating,1),.45,2.2)
    return base*np.clip(ratio**power,.72,1.42)

def build_matchup(home:dict,away:dict,neutral:bool=False)->dict[str,float]:
    """Translate opponent-adjusted pregame states into this-opponent expectations."""
    out={}; hfa=0 if neutral else 2.25
    for side,o,d in (("home",home,away),("away",away,home)):
        oq=_g(o,"off_rating",100);dq=_g(d,"def_rating",100)
        drives=.52*_g(o,"off_drives",11.5)+.48*_g(d,"def_drives",11.5)
        ppd=_blend(_g(o,"off_points_per_drive",2.25),_g(d,"def_points_per_drive",2.25),oq,dq,.50)
        td=_blend(_g(o,"off_td_rate",.30),_g(d,"def_td_rate",.30),oq,dq,.55)
        fg=np.sqrt(max(_g(o,"off_fg_rate",.12),.01)*max(_g(d,"def_fg_rate",.12),.01))
        tov=np.sqrt(max(_g(o,"off_turnover_rate",.12),.01)*max(_g(d,"def_turnover_rate",.12),.01))*np.clip((dq/max(oq,1))**.30,.75,1.35)
        start=.55*_g(o,"off_avg_start_field",27.5)+.45*_g(d,"def_avg_start_field",27.5)
        success=_blend(_g(o,"off_success_rate",.42),_g(d,"def_success_rate",.42),oq,dq,.25)
        ppa=.55*_g(o,"off_ppa_per_play",0)+.45*_g(d,"def_ppa_per_play",0)+.003*(oq-dq)
        expl=_blend(_g(o,"off_explosive_rate",.10),_g(d,"def_explosive_rate",.10),oq,dq,.30)
        ypd=_blend(_g(o,"off_yards_per_drive",35),_g(d,"def_yards_per_drive",35),oq,dq,.38)
        plays=_blend(_g(o,"off_plays_per_drive",6),_g(d,"def_plays_per_drive",6),oq,dq,.15)
        third=_blend(_g(o,"off_third_down_rate",.40),_g(d,"def_third_down_rate",.40),oq,dq,.25)
        vals={"drives":np.clip(drives,7.5,17),"ppd":np.clip(ppd,.15,5.5),"td_rate":np.clip(td,.04,.68),"fg_rate":np.clip(fg,.03,.30),"turnover_rate":np.clip(tov,.025,.32),
              "start_field":np.clip(start,15,50),"success_rate":np.clip(success,.15,.72),"ppa":np.clip(ppa,-.7,1.2),"explosive_rate":np.clip(expl,.02,.35),"yards_per_drive":np.clip(ypd,10,75),"plays_per_drive":np.clip(plays,3,10),"third_down_rate":np.clip(third,.15,.70),"matchup_ratio":np.clip(oq/max(dq,1),.45,2.2)}
        out.update({f"{side}_{k}":float(v) for k,v in vals.items()})
    env=(out["home_drives"]+out["away_drives"])/2
    out["home_drives"]=.72*out["home_drives"]+.28*env;out["away_drives"]=.72*out["away_drives"]+.28*env
    out["elo_difference"]=_g(home,"elo",1500)-_g(away,"elo",1500)+(0 if neutral else 75)
    out["rating_difference"]=(home.get("off_rating",100)+home.get("def_rating",100))-(away.get("off_rating",100)+away.get("def_rating",100))
    out["home_field_points"]=hfa
    out["state_maturity"]=min(_g(home,"maturity",0),_g(away,"maturity",0))
    return out

V3_MATCHUP_FEATURES=[
 "home_drives","away_drives","home_ppd","away_ppd","home_td_rate","away_td_rate","home_fg_rate","away_fg_rate","home_turnover_rate","away_turnover_rate",
 "home_start_field","away_start_field","home_success_rate","away_success_rate","home_ppa","away_ppa","home_explosive_rate","away_explosive_rate",
 "home_yards_per_drive","away_yards_per_drive","home_plays_per_drive","away_plays_per_drive","home_third_down_rate","away_third_down_rate",
 "home_matchup_ratio","away_matchup_ratio","elo_difference","rating_difference","home_field_points","state_maturity"]
