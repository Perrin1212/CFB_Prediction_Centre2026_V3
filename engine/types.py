from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TeamRatings:
    team_id: int
    team_name: str

    elo: float = 1500.0
    offensive_rating: float = 0.0
    defensive_rating: float = 0.0
    special_teams_rating: float = 0.0

    pace: float = 70.0

    points_per_drive: float = 2.0
    points_per_drive_allowed: float = 2.0

    yards_per_play: float = 5.5
    yards_per_play_allowed: float = 5.5

    pass_efficiency: float = 1.0
    pass_efficiency_allowed: float = 1.0

    rush_efficiency: float = 5.0
    rush_efficiency_allowed: float = 5.0

    third_down_pct: float = 0.40
    third_down_def_pct: float = 0.40

    red_zone_td_pct: float = 0.60
    red_zone_td_def_pct: float = 0.60

    explosive_play_rate: float = 0.10
    explosive_play_rate_allowed: float = 0.10

    turnover_rate: float = 0.02
    turnover_rate_forced: float = 0.02

    starting_field_position: float = 28.0

    strength_of_schedule: float = 1500.0

    games_played: int = 0


@dataclass
class MatchupResult:
    home_team: str
    away_team: str

    home_elo: float
    away_elo: float

    home_offensive_edge: float
    away_offensive_edge: float

    home_defensive_edge: float
    away_defensive_edge: float

    home_special_teams_edge: float
    away_special_teams_edge: float

    home_overall_edge: float
    away_overall_edge: float

    home_matchup_score: float
    away_matchup_score: float

    advantages: Dict[str, str] = field(default_factory=dict)
    risks: Dict[str, str] = field(default_factory=dict)


@dataclass
class GameEnvironment:
    expected_pace: float

    home_expected_possessions: float
    away_expected_possessions: float

    home_possession_low: float
    home_possession_high: float

    away_possession_low: float
    away_possession_high: float

    expected_total_possessions: float

    home_starting_field_position: float
    away_starting_field_position: float

    home_time_of_possession: float
    away_time_of_possession: float


@dataclass
class TeamScoringModel:
    team_name: str

    expected_points_per_drive: float
    p10_points_per_drive: float
    p50_points_per_drive: float
    p90_points_per_drive: float

    touchdown_probability: float
    field_goal_probability: float
    turnover_probability: float
    punt_probability: float

    expected_points: float


@dataclass
class SimulationResult:
    home_team: str
    away_team: str

    simulations: int

    home_scores: List[float]
    away_scores: List[float]

    home_win_probability: float
    away_win_probability: float
    tie_probability: float

    expected_home_score: float
    expected_away_score: float

    median_home_score: float
    median_away_score: float

    home_score_p10: float
    home_score_p90: float

    away_score_p10: float
    away_score_p90: float

    expected_margin: float
    margin_std: float

    expected_total: float
    total_std: float

    margin_p10: float
    margin_p90: float

    total_p10: float
    total_p90: float