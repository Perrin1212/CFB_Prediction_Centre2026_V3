from __future__ import annotations
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parent.parent; D=ROOT/'data'/'v3'; D.mkdir(parents=True,exist_ok=True)
FEATURES=['v3_expected_home_drives','v3_expected_away_drives','v3_home_base_ppd','v3_away_base_ppd','v3_home_start_field','v3_away_start_field','v3_home_success','v3_away_success','v3_home_explosive','v3_away_explosive','v3_home_turnover_rate','v3_away_turnover_rate','v3_home_matchup_ratio','v3_away_matchup_ratio','home_drive_off_rating_pre','home_drive_def_rating_pre','away_drive_off_rating_pre','away_drive_def_rating_pre','home_drive_schedule_def_pre','away_drive_schedule_def_pre','home_drive_schedule_off_pre','away_drive_schedule_off_pre']

def main():
 d=pd.read_csv(D/'v3_drive_training_rows.csv',low_memory=False); d=d.dropna(subset=['home_points','away_points']); X=d.reindex(columns=FEATURES).apply(pd.to_numeric,errors='coerce').fillna(0)
 train=d.season.le(2024); test=d.season.eq(2025); models={}; report={}
 for target in ['home_points','away_points']:
  y=pd.to_numeric(d[target],errors='coerce').fillna(0); m=HistGradientBoostingRegressor(max_iter=350,learning_rate=.045,max_leaf_nodes=24,l2_regularization=2.0,random_state=2026);m.fit(X[train],y[train]);models[target]=m
  report[target+'_2025_mae']=float(mean_absolute_error(y[test],m.predict(X[test]))) if test.any() else None
 joblib.dump({'models':models,'features':FEATURES},D/'v3_drive_models.joblib'); (D/'v3_drive_model_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('Next: python -m jobs.evaluate_v3_drive')
if __name__=='__main__':main()
