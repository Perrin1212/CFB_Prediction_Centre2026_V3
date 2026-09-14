from __future__ import annotations
from pathlib import Path
import sys, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from engine.v3_features import V3_MODEL_FEATURES
P=ROOT/'data'/'processed'; DATA=P/'v3_research_dataset.csv'; MODEL=P/'v3_component_models.joblib'; META=P/'v3_component_model_manifest.json'

SIDE_EXTRA={
 'home':['home_offense_points_per_play_pre','away_defense_points_per_play_allowed_pre','home_expected_turnover_pressure','home_sustainability_pressure','home_latent_start_field_position','expected_home_plays'],
 'away':['away_offense_points_per_play_pre','home_defense_points_per_play_allowed_pre','away_expected_turnover_pressure','away_sustainability_pressure','away_latent_start_field_position','expected_away_plays']}

def cols(side): return list(dict.fromkeys(V3_MODEL_FEATURES+SIDE_EXTRA[side]))
def fit(X,y,loss='squared_error'):
    return HistGradientBoostingRegressor(loss=loss, learning_rate=.045, max_iter=350, max_leaf_nodes=18, l2_regularization=2.0, min_samples_leaf=24, random_state=2026).fit(X,y)

def main():
    df=pd.read_csv(DATA,low_memory=False)
    train=df[df.season<=2024].copy(); test=df[df.season==2025].copy()
    if train.empty or test.empty: raise RuntimeError('V3 expects 2023-24 train and 2025 holdout rows.')
    models={}; report={}
    for side in ('home','away'):
        Xtr=train[cols(side)].apply(pd.to_numeric,errors='coerce').fillna(0); Xte=test[cols(side)].apply(pd.to_numeric,errors='coerce').fillna(0)
        for target,key in ((f'actual_{side}_drive_proxy','drives'),(f'actual_{side}_points_per_drive_proxy','ppd')):
            ytr=pd.to_numeric(train[target],errors='coerce'); yte=pd.to_numeric(test[target],errors='coerce')
            m=ytr.notna(); mt=yte.notna()
            model=fit(Xtr.loc[m],ytr.loc[m],loss='poisson')
            models[f'{side}_{key}']={'model':model,'features':cols(side)}
            pred=model.predict(Xte.loc[mt])
            report[f'{side}_{key}_mae']=float(mean_absolute_error(yte.loc[mt],pred))
    joblib.dump(models,MODEL)
    META.write_text(json.dumps({'version':'CFB-V3-2026.1-research','training_seasons':'<=2024','holdout_season':2025,'metrics':report,'market_features_used':False,'observed_drives_used':False},indent=2))
    print('='*78); print('CFB-V3 COMPONENT MODELS TRAINED — 2025 HELD OUT'); print('='*78)
    for k,v in report.items(): print(f'{k:<28} {v:.4f}')
    print(f'✓ {MODEL}')
    print('NOTE: drive targets remain box-score-derived proxies until true drive/PBP data is ingested.')
if __name__=='__main__': main()
