from __future__ import annotations
from pathlib import Path
import sys
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss, log_loss
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from engine.v3_simulator import simulate_game, V3SimulationConfig
P=ROOT/'data'/'processed'

def main():
    df=pd.read_csv(P/'v3_research_dataset.csv',low_memory=False)
    df=df[df.season==2025].copy().reset_index(drop=True)
    models=joblib.load(P/'v3_component_models.joblib')
    preds={}
    for side in ('home','away'):
        for key in ('drives','ppd'):
            pack=models[f'{side}_{key}']
            X=df[pack['features']].apply(pd.to_numeric,errors='coerce').fillna(0)
            preds[f'{side}_{key}']=pack['model'].predict(X)
    out=[]
    for i,r in df.iterrows():
        sim=simulate_game(
            preds['home_drives'][i],preds['away_drives'][i],preds['home_ppd'][i],preds['away_ppd'][i],
            float(r.get('home_expected_turnover_pressure',.022))*4.5,float(r.get('away_expected_turnover_pressure',.022))*4.5,
            float(r.get('home_latent_start_field_position',28)),float(r.get('away_latent_start_field_position',28)),
            float(r.get('home_offense_matchup_extreme',0)),float(r.get('away_offense_matchup_extreme',0)),
            V3SimulationConfig(simulations=250,seed=2026+i),
        )
        out.append({'game_id':r.game_id,'home_team':r.home_team,'away_team':r.away_team,'actual_home':r.home_points,'actual_away':r.away_points,'actual_margin':r.actual_margin,'actual_total':r.actual_total,'v3_home':sim['home_points_mean'],'v3_away':sim['away_points_mean'],'v3_margin':sim['margin_mean'],'v3_total':sim['total_mean'],'v3_home_win_probability':sim['home_win_probability']})
    o=pd.DataFrame(out); o.to_csv(P/'v3_2025_holdout_predictions.csv',index=False)
    actual=(o.actual_margin>0).astype(int); p=o.v3_home_win_probability.clip(.001,.999)
    print('='*78); print('CFB-V3 — 2025 TRUE HOLDOUT (250-sim research pass)'); print('='*78)
    print(f"Games:       {len(o):,}")
    print(f"Winner acc:  {((p>=.5)==actual).mean():.2%}")
    print(f"Brier:       {brier_score_loss(actual,p):.4f}")
    print(f"LogLoss:     {log_loss(actual,p):.4f}")
    print(f"Margin MAE:  {mean_absolute_error(o.actual_margin,o.v3_margin):.3f}")
    print(f"Total MAE:   {mean_absolute_error(o.actual_total,o.v3_total):.3f}")
    print(f"Home MAE:    {mean_absolute_error(o.actual_home,o.v3_home):.3f}")
    print(f"Away MAE:    {mean_absolute_error(o.actual_away,o.v3_away):.3f}")
    print(f"✓ {P/'v3_2025_holdout_predictions.csv'}")
if __name__=='__main__': main()
