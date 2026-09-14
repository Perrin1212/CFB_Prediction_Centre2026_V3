from __future__ import annotations
from pathlib import Path
import sys,json,joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from engine.v3_drive_engine import DriveStateEngine,load_states,matchup_features
from engine.v3_drive_simulator import simulate_game
D=ROOT/'data'/'v3'; P=ROOT/'data'/'predictions'; P.mkdir(parents=True,exist_ok=True)

def main():
 schedule=pd.read_csv(ROOT/'data'/'processed'/'2026_forward_games.csv',low_memory=False); stats=pd.read_csv(D/'team_game_drive_metrics.csv',low_memory=False); stats['game_id_key']=pd.to_numeric(stats.game_id,errors='coerce')
 pack=joblib.load(D/'v3_drive_models.joblib'); engine=DriveStateEngine();load_states(engine,D/'v3_end_2025_drive_states.json');engine.regress_season()
 schedule['start_date']=pd.to_datetime(schedule.start_date,utc=True,errors='coerce');schedule=schedule.sort_values(['start_date','cfbd_id'],kind='stable');rows=[]
 for ts,games in schedule.groupby('start_date',sort=True):
  pending=[]
  for r in games.itertuples(index=False):
   hs=engine.snapshot(r.home_team);aw=engine.snapshot(r.away_team);feat=matchup_features(hs,aw); base={'game_id':int(r.cfbd_id),'season':2026,'week':r.week,'start_date':r.start_date,'home_team':r.home_team,'away_team':r.away_team}
   base.update({'home_'+k:v for k,v in hs.items()});base.update({'away_'+k:v for k,v in aw.items()});base.update(feat)
   X=pd.DataFrame([base]).reindex(columns=pack['features']).apply(pd.to_numeric,errors='coerce').fillna(0); hp=float(pack['models']['home_points'].predict(X)[0]);ap=float(pack['models']['away_points'].predict(X)[0])
   # model score estimates anchor PPD; possession and context simulator generates distribution.
   hppd=np.clip(hp/max(feat['v3_expected_home_drives'],6),.15,6.0);appd=np.clip(ap/max(feat['v3_expected_away_drives'],6),.15,6.0)
   sim=simulate_game(feat['v3_expected_home_drives'],feat['v3_expected_away_drives'],hppd,appd,feat['v3_home_turnover_rate'],feat['v3_away_turnover_rate'],feat['v3_home_start_field'],feat['v3_away_start_field'],feat['v3_home_success'],feat['v3_away_success'],feat['v3_home_explosive'],feat['v3_away_explosive'])
   base.update(sim); rows.append(base)
   pair=stats[stats.game_id_key.eq(r.cfbd_id)];h=pair[pair.offense.eq(r.home_team)];a=pair[pair.offense.eq(r.away_team)]
   if len(h)==1 and len(a)==1: pending.append((r.home_team,r.away_team,h.iloc[0].to_dict(),a.iloc[0].to_dict()))
  # same-kickoff protection
  for x in pending:engine.update_game(*x)
 out=pd.DataFrame(rows); v2=pd.read_csv(P/'2026_forward_predictions.csv',low_memory=False); key='game_id' if 'game_id' in v2.columns else 'cfbd_game_id';
 if key in v2.columns:
  keep=[c for c in [key,'home_win_probability','away_win_probability','predicted_winner'] if c in v2.columns]; out=out.merge(v2[keep].rename(columns={key:'game_id','home_win_probability':'v2_home_win_probability','away_win_probability':'v2_away_win_probability','predicted_winner':'v2_predicted_winner'}),on='game_id',how='left')
 out['v3_predicted_winner']=np.where(out.v3_home_win_probability>=.5,out.home_team,out.away_team);out.to_csv(P/'2026_v3_drive_predictions.csv',index=False)
 print(f'Saved {len(out)} V3 drive predictions -> {P/"2026_v3_drive_predictions.csv"}');print('V2 remains unchanged and is still the production control.')
if __name__=='__main__':main()
