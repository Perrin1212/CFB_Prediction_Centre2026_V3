from __future__ import annotations
from .matchup import MatchupEngine
from .environment import GameEnvironmentEngine
from .types import SimulationResult
from .v3_game_simulator import simulate_game

class CFBPredictionEngine:
    """Interactive compatibility facade.

    Production V3 predictions are built from chronological states. This facade is
    retained for existing callers that pass TeamRatings directly.
    """
    def __init__(self, simulations=10_000):
        self.simulations=int(simulations);self.matchup_engine=MatchupEngine();self.environment_engine=GameEnvironmentEngine()

    def predict(self,home_team,away_team):
        matchup=self.matchup_engine.compare(home_team,away_team)
        try: environment=self.environment_engine.predict(home_team,away_team)
        except Exception: environment=None
        hp=getattr(home_team,"points_per_drive",2.0);ap=getattr(away_team,"points_per_drive",2.0)
        hpa=getattr(away_team,"points_per_drive_allowed",2.0);apa=getattr(home_team,"points_per_drive_allowed",2.0)
        hd=(getattr(home_team,"pace",70)/6.1);ad=(getattr(away_team,"pace",70)/6.1)
        m={"home_drives":hd,"away_drives":ad,"home_ppd":(hp+hpa)/2,"away_ppd":(ap+apa)/2,
           "home_td_rate":.30,"away_td_rate":.30,"home_fg_rate":.12,"away_fg_rate":.12,
           "home_turnover_rate":max(.02,getattr(home_team,"turnover_rate",.02)),"away_turnover_rate":max(.02,getattr(away_team,"turnover_rate",.02)),
           "home_start_field":getattr(home_team,"starting_field_position",28),"away_start_field":getattr(away_team,"starting_field_position",28),
           "home_success_rate":getattr(home_team,"third_down_pct",.4),"away_success_rate":getattr(away_team,"third_down_pct",.4),
           "home_explosive_rate":getattr(home_team,"explosive_play_rate",.1),"away_explosive_rate":getattr(away_team,"explosive_play_rate",.1),"home_ppa":0,"away_ppa":0}
        s=simulate_game(m,self.simulations,20260908)
        # Existing UI/tests expect the historical SimulationResult dataclass.
        sim=SimulationResult(home_team=home_team.team_name,away_team=away_team.team_name,simulations=self.simulations,home_scores=[],away_scores=[],
          home_win_probability=s["home_win_probability"],away_win_probability=s["away_win_probability"],tie_probability=0,
          expected_home_score=s["projected_home_points"],expected_away_score=s["projected_away_points"],median_home_score=s["projected_home_points"],median_away_score=s["projected_away_points"],
          home_score_p10=s["home_p10"],home_score_p90=s["home_p90"],away_score_p10=s["away_p10"],away_score_p90=s["away_p90"],
          expected_margin=s["projected_margin"],margin_std=(s["margin_p90"]-s["margin_p10"])/2.563,expected_total=s["projected_total"],total_std=(s["total_p90"]-s["total_p10"])/2.563,
          margin_p10=s["margin_p10"],margin_p90=s["margin_p90"],total_p10=s["total_p10"],total_p90=s["total_p90"])
        return {"matchup":matchup,"environment":environment,"simulation":sim,"explanation":"V3 possession-level simulation","bullets":[]}
