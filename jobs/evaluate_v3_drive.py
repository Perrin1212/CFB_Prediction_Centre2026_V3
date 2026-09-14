from pathlib import Path
import joblib, json, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'data'/'v3'
def main():
 d=pd.read_csv(D/'v3_drive_training_rows.csv',low_memory=False);pack=joblib.load(D/'v3_drive_models.joblib');f=pack['features'];x=d.reindex(columns=f).apply(pd.to_numeric,errors='coerce').fillna(0);m=d.season.eq(2025)&d.home_points.notna()&d.away_points.notna();z=d[m].copy();hp=pack['models']['home_points'].predict(x[m]);ap=pack['models']['away_points'].predict(x[m]);am=z.home_points-z.away_points;at=z.home_points+z.away_points
 # smooth win probability from projected margin, evaluated separately from V2 official probability
 wp=1/(1+np.exp(-(hp-ap)/13.5)); actual=(am>0).astype(float)
 r={'games':int(m.sum()),'winner_accuracy':float(((hp>ap)==(am>0)).mean()),'brier':float(brier_score_loss(actual,wp)),'margin_mae':float(mean_absolute_error(am,hp-ap)),'total_mae':float(mean_absolute_error(at,hp+ap)),'home_mae':float(mean_absolute_error(z.home_points,hp)),'away_mae':float(mean_absolute_error(z.away_points,ap)),'pred_40_plus_rate':float(np.mean(np.r_[hp,ap]>=40)),'actual_40_plus_rate':float(np.mean(np.r_[z.home_points,z.away_points]>=40)),'pred_10_or_less_rate':float(np.mean(np.r_[hp,ap]<=10)),'actual_10_or_less_rate':float(np.mean(np.r_[z.home_points,z.away_points]<=10))}
 (D/'v3_drive_2025_evaluation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
