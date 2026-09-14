import pandas as pd
from engine.v3_possession_engine import reconstruct_drives,aggregate_team_games
from engine.v3_team_state import ChronologicalStateEngine
from engine.v3_matchup_engine import build_matchup
from engine.v3_game_simulator import simulate_game

def test_drive_reconstruction_and_aggregation():
    p=pd.DataFrame([
      {"gameId":1,"driveId":"a","playNumber":1,"period":1,"offense":"A","defense":"B","yardsToGoal":75,"yardsGained":5,"ppa":.1,"down":1,"distance":10,"offenseScore":0,"defenseScore":0,"playText":"run","playType":"Rush"},
      {"gameId":1,"driveId":"a","playNumber":2,"period":1,"offense":"A","defense":"B","yardsToGoal":70,"yardsGained":70,"ppa":2,"down":2,"distance":5,"offenseScore":7,"defenseScore":0,"playText":"Touchdown","playType":"Passing Touchdown"},
      {"gameId":1,"driveId":"b","playNumber":1,"period":1,"offense":"B","defense":"A","yardsToGoal":80,"yardsGained":0,"ppa":-.2,"down":1,"distance":10,"offenseScore":0,"defenseScore":7,"playText":"punt","playType":"Punt"},])
    d=reconstruct_drives(p);assert len(d)==2;assert set(d.result)=={"TD","PUNT"};t=aggregate_team_games(d);assert len(t)==2

def test_opponent_adjustment_rewards_strong_defense():
    e1=ChronologicalStateEngine();e2=ChronologicalStateEngine();obs={"drives":11,"points_per_drive":3,"td_rate":.4,"fg_rate":.1,"turnover_rate":.1,"punt_rate":.3,"downs_rate":.05,"avg_start_field":30,"plays_per_drive":6.5,"yards_per_drive":42,"success_rate":.48,"ppa_per_play":.2,"explosive_rate":.12,"positive_ppa":1,"third_down_rate":.45,"fourth_down_rate":.5}
    # Make opponent in e1 strong defensively before the observed game.
    b=e1.get("B");b.defense.values["points_per_drive"]=1.0;b.defense.values["success_rate"]=.28;b.defense.values["yards_per_drive"]=20
    e1.update_game("A","B",obs,obs,30,20);e2.update_game("A","C",obs,obs,30,20)
    assert e1.get("A").offense.values["points_per_drive"] > e2.get("A").offense.values["points_per_drive"]

def test_simulator_has_tails_and_probability():
    e=ChronologicalStateEngine();m=build_matchup(e.snapshot("A"),e.snapshot("B"));s=simulate_game(m,2000,7)
    assert 0<=s["home_win_probability"]<=1;assert s["home_p90"]>s["home_p10"];assert s["total_p90"]>s["total_p10"]


def test_current_season_weight_curve_is_explicit():
    engine = ChronologicalStateEngine()
    assert [engine.current_season_weight(n) for n in range(7)] == [
        0.0, 0.35, 0.55, 0.70, 0.80, 0.90, 0.92
    ]


def test_turnovers_are_sampled_as_actual_drive_outcomes():
    engine = ChronologicalStateEngine()
    matchup = build_matchup(engine.snapshot("A"), engine.snapshot("B"))
    low = dict(matchup, home_turnover_rate=.03)
    high = dict(matchup, home_turnover_rate=.28)
    low_result = simulate_game(low, 5000, 99)
    high_result = simulate_game(high, 5000, 99)
    assert high_result["expected_home_turnovers"] > low_result["expected_home_turnovers"]
    assert high_result["home_win_probability"] < low_result["home_win_probability"]
