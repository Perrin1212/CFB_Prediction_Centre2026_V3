from __future__ import annotations
from pathlib import Path
import sys, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from engine.v3_features import V3_MODEL_FEATURES
P=ROOT/'data'/'processed'
DATA=P/'v3_research_dataset.csv'
HOLDOUT=P/'v3_2025_holdout_predictions.csv'
OUT=P/'v3_component_models_2026.joblib'
CAL=P/'v3_win_calibrator_2026.joblib'
MAN=P/'v3_2026_manifest.json'
SIDE_EXTRA={
 'home':['home_offense_points_per_play_pre','away_defense_points_per_play_allowed_pre','home_expected_turnover_pressure','home_sustainability_pressure','home_latent_start_field_position','expected_home_plays'],
 'away':['away_offense_points_per_play_pre','home_defense_points_per_play_allowed_pre','away_expected_turnover_pressure','away_sustainability_pressure','away_latent_start_field_position','expected_away_plays']}
def cols(side): return list(dict.fromkeys(V3_MODEL_FEATURES+SIDE_EXTRA[side]))
def fit(X,y):
 return HistGradientBoostingRegressor(loss='poisson',learning_rate=.045,max_iter=350,max_leaf_nodes=18,l2_regularization=2.0,min_samples_leaf=24,random_state=2026).fit(X,y)
def main():
 if not HOLDOUT.exists(): raise FileNotFoundError('Run python -m jobs.evaluate_v3 first; 2025 is the untouched calibration season.')
 df=pd.read_csv(DATA,low_memory=False); models={}
 for side in ('home','away'):
  X=df[cols(side)].apply(pd.to_numeric,errors='coerce').fillna(0)
  for target,key in ((f'actual_{side}_drive_proxy','drives'),(f'actual_{side}_points_per_drive_proxy','ppd')):
   y=pd.to_numeric(df[target],errors='coerce'); m=y.notna(); model=fit(X.loc[m],y.loc[m]); models[f'{side}_{key}']={'model':model,'features':cols(side)}
 joblib.dump(models,OUT)
 h=pd.read_csv(HOLDOUT); p=np.clip(pd.to_numeric(h.v3_home_win_probability,errors='coerce').fillna(.5).to_numpy(),.001,.999)
 y=(pd.to_numeric(h.actual_margin,errors='coerce')>0).astype(int).to_numpy()
 logit=np.log(p/(1-p)).reshape(-1,1)
 cal=LogisticRegression(C=2.0).fit(logit,y); joblib.dump(cal,CAL)
 MAN.write_text(json.dumps({'version':'CFB-V3-2026.1-challenger','component_training':'2023-2025 after untouched 2025 evaluation','win_calibration':'logistic calibration fitted on raw 2025 simulator probabilities for FUTURE 2026 only','market_inputs':False,'v2_production_mutated':False},indent=2))
 print('✓ V3 2026 challenger components refit through 2025.')
 print('✓ 2025 raw simulation used only to fit future win calibration after holdout reporting.')
 print(f'✓ {OUT}\n✓ {CAL}\n✓ {MAN}')
if __name__=='__main__': main()
