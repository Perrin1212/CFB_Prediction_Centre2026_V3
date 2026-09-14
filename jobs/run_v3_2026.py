from __future__ import annotations
from pathlib import Path
import sys
import joblib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from engine.v3_features import add_v3_features
from engine.v3_simulator import simulate_game, V3SimulationConfig
P=ROOT/'data'/'processed'; D=ROOT/'data'/'predictions'

def main():
    hist=pd.read_csv(P/'scoring_history.csv',low_memory=False)
    live=pd.read_csv(D/'2026_scoring_predictions.csv',low_memory=False)
    common=[c for c in hist.columns if c in live.columns]
    combo=pd.concat([hist[common],live[common]],ignore_index=True,sort=False)
    featured=add_v3_features(combo)
    x=featured[pd.to_numeric(featured.season,errors='coerce').eq(2026)].copy().reset_index(drop=True)
    # Reattach V2 identity/output fields not carried by scoring_history.
    keep=['game_id']+[c for c in live.columns if c not in x.columns and c!='game_id']
    x=x.merge(live[keep],on='game_id',how='left',validate='one_to_one')
    models=joblib.load(P/'v3_component_models_2026.joblib'); cal=joblib.load(P/'v3_win_calibrator_2026.joblib')
    preds={}
    for side in ('home','away'):
        for key in ('drives','ppd'):
            pack=models[f'{side}_{key}']; X=x[pack['features']].apply(pd.to_numeric,errors='coerce').fillna(0)
            preds[f'{side}_{key}']=pack['model'].predict(X)
    # Couple possession environment because football possessions largely alternate.
    common_dr=.5*(preds['home_drives']+preds['away_drives'])
    preds['home_drives']=.45*preds['home_drives']+.55*common_dr
    preds['away_drives']=.45*preds['away_drives']+.55*common_dr
    rows=[]
    for i,r in x.iterrows():
        sim=simulate_game(
            preds['home_drives'][i],preds['away_drives'][i],preds['home_ppd'][i],preds['away_ppd'][i],
            float(r.get('home_expected_turnover_pressure',.022))*4.5,float(r.get('away_expected_turnover_pressure',.022))*4.5,
            float(r.get('home_latent_start_field_position',28)),float(r.get('away_latent_start_field_position',28)),
            float(r.get('home_offense_matchup_extreme',0)),float(r.get('away_offense_matchup_extreme',0)),
            V3SimulationConfig(simulations=1500,seed=20260000+i),
        )
        raw=np.clip(sim['home_win_probability'],.001,.999); lp=np.log(raw/(1-raw)); calibrated=float(cal.predict_proba([[lp]])[0,1])
        rows.append({
            'model_version':'CFB-V3-2026.1-challenger','game_id':r.game_id,'cfbd_game_id':r.get('cfbd_game_id'),'season':r.season,'week':r.week,'start_date':r.start_date,'home_team':r.home_team,'away_team':r.away_team,
            'v3_home_win_probability':calibrated,'v3_away_win_probability':1-calibrated,'v3_raw_sim_home_win_probability':raw,
            'v3_projected_home_points':sim['home_points_mean'],'v3_projected_away_points':sim['away_points_mean'],'v3_projected_margin':sim['margin_mean'],'v3_projected_total':sim['total_mean'],
            'v3_expected_home_drives':preds['home_drives'][i],'v3_expected_away_drives':preds['away_drives'][i],'v3_expected_home_points_per_drive':preds['home_ppd'][i],'v3_expected_away_points_per_drive':preds['away_ppd'][i],
            'v3_home_start_field_position':r.get('home_latent_start_field_position',28),'v3_away_start_field_position':r.get('away_latent_start_field_position',28),
            'v3_home_schedule_def_quality':r.get('home_off_schedule_def_quality_pre',100),'v3_away_schedule_def_quality':r.get('away_off_schedule_def_quality_pre',100),
            'v3_home_schedule_corrected_offense':r.get('home_schedule_corrected_offense',100),'v3_away_schedule_corrected_offense':r.get('away_schedule_corrected_offense',100),
            'v3_margin_p10':sim['margin_p10'],'v3_margin_p90':sim['margin_p90'],'v3_total_p10':sim['total_p10'],'v3_total_p90':sim['total_p90'],'v3_blowout_28_plus_probability':sim['blowout_28_plus_probability'],
            'v2_home_win_probability':r.get('home_win_probability'),'v2_projected_home_points':r.get('projected_home_score'),'v2_projected_away_points':r.get('projected_away_score'),'completed':r.get('completed'),'actual_home_points':r.get('actual_home_points'),'actual_away_points':r.get('actual_away_points'),
        })
    out=pd.DataFrame(rows).sort_values(['week','start_date','game_id']); path=D/'2026_v3_predictions.csv'; out.to_csv(path,index=False)
    print('='*78); print('CFB-V3 2026 CHALLENGER RUN COMPLETE'); print('='*78)
    print(f'Games: {len(out):,}')
    print(f'Average projected total: {out.v3_projected_total.mean():.2f}')
    lo=min(out.v3_projected_home_points.min(),out.v3_projected_away_points.min()); hi=max(out.v3_projected_home_points.max(),out.v3_projected_away_points.max())
    print(f'Projected team-score range: {lo:.1f} to {hi:.1f}')
    print(f'✓ {path}')
    print('V2 outputs were read-only and remain unchanged.')
if __name__=='__main__': main()
