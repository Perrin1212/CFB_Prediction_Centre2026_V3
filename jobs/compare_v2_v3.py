from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'data'/'processed'

def mae(a,b):
 a=pd.to_numeric(a,errors='coerce'); b=pd.to_numeric(b,errors='coerce'); m=a.notna()&b.notna(); return float(np.mean(np.abs(a[m]-b[m])))

def main():
 v3=pd.read_csv(P/'v3_2025_holdout_predictions.csv')
 v2=pd.read_csv(P/'score_model_2025_predictions.csv',low_memory=False)
 # normalize known V2 column names
 key='game_id'
 candidates={
  'v2_home':['final_expected_home_points','expected_home_points','projected_home_score'],
  'v2_away':['final_expected_away_points','expected_away_points','projected_away_score'],
  'v2_margin':['final_expected_home_margin','expected_home_margin','projected_margin'],
  'v2_total':['final_expected_total_points','expected_total_points','projected_total'],
 }
 keep=[key]
 for out,opts in candidates.items():
  found=next((c for c in opts if c in v2.columns),None)
  if found: v2[out]=v2[found]; keep.append(out)
 x=v3.merge(v2[keep],on=key,how='left')
 print('='*78); print('CFB-V2 vs V3 — 2025 HOLDOUT / SCORE COMPRESSION AUDIT'); print('='*78)
 for label,prefix in [('V3','v3'),('V2','v2')]:
  if f'{prefix}_margin' not in x: continue
  print(f'\n{label}')
  print(f"Margin MAE: {mae(x.actual_margin,x[f'{prefix}_margin']):.3f}")
  print(f"Total MAE:  {mae(x.actual_total,x[f'{prefix}_total']):.3f}")
  ph=x[f'{prefix}_home']; pa=x[f'{prefix}_away']
  print(f"Projected team-score range: {min(ph.min(),pa.min()):.1f} to {max(ph.max(),pa.max()):.1f}")
  print(f"Projected 40+ team rate: {pd.concat([ph,pa]).ge(40).mean():.2%}")
 actual_team=pd.concat([x.actual_home,x.actual_away],ignore_index=True)
 print(f"\nActual 40+ team rate: {actual_team.ge(40).mean():.2%}")
 print(f"Actual <=10 team rate: {actual_team.le(10).mean():.2%}")
 for label,prefix in [('V3','v3'),('V2','v2')]:
  if f'{prefix}_home' in x:
   pred=pd.concat([x[f'{prefix}_home'],x[f'{prefix}_away']],ignore_index=True)
   print(f"{label} projected <=10 rate: {pred.le(10).mean():.2%}")
if __name__=='__main__': main()
