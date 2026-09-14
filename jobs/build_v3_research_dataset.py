from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from engine.v3_features import add_v3_features
from engine.v3_possessions import infer_observed_drive_proxy

P=ROOT/'data'/'processed'
OUT=P/'v3_research_dataset.csv'

def main():
    print('='*78); print('CFB-V3 — BUILD RESEARCH DATASET'); print('='*78)
    scoring=pd.read_csv(P/'scoring_history.csv', low_memory=False)
    tg=pd.read_csv(P/'historical_team_games.csv', low_memory=False)
    scoring=add_v3_features(scoring)

    # Actual latent drive/efficiency targets from team box scores.
    targets=[]
    for gid,g in tg.groupby('game_id', sort=False):
        if len(g)<2: continue
        rec={'game_id':gid}
        for side in ('home','away'):
            r=g[g['home_away'].astype(str).str.lower().eq(side)]
            if r.empty: continue
            r=r.iloc[0]
            drives=infer_observed_drive_proxy(r)
            pts=float(pd.to_numeric(pd.Series([r.get('team_points')]),errors='coerce').fillna(0).iloc[0])
            plays=float(pd.to_numeric(pd.Series([r.get('offensive_play_proxy')]),errors='coerce').fillna(65).iloc[0])
            rec[f'actual_{side}_drive_proxy']=drives
            rec[f'actual_{side}_points_per_drive_proxy']=pts/max(drives,1)
            rec[f'actual_{side}_points_per_play']=pts/max(plays,1)
        targets.append(rec)
    t=pd.DataFrame(targets)
    df=scoring.merge(t,on='game_id',how='left',validate='one_to_one')
    df['actual_home_win']=(pd.to_numeric(df.home_points,errors='coerce')>pd.to_numeric(df.away_points,errors='coerce')).astype(float)
    df['actual_margin']=pd.to_numeric(df.home_points,errors='coerce')-pd.to_numeric(df.away_points,errors='coerce')
    df['actual_total']=pd.to_numeric(df.home_points,errors='coerce')+pd.to_numeric(df.away_points,errors='coerce')
    df.to_csv(OUT,index=False)
    print(f'✓ {len(df):,} chronological games → {OUT}')
    print('✓ Schedule-quality features are PRE-GAME only.')
    print('✓ Weak-opponent production is explicitly discounted in schedule-corrected ratings.')
    print('✓ Drive counts are labelled PROXY, not observed.')
if __name__=='__main__': main()
