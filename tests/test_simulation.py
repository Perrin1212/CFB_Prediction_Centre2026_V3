from engine.predictor import CFBPredictionEngine
from engine.types import TeamRatings


def test_simulation_runs():

    home = TeamRatings(
        team_id=1,
        team_name="Home",
        elo=1700,
        offensive_rating=1.2,
        defensive_rating=1.2
    )

    away = TeamRatings(
        team_id=2,
        team_name="Away",
        elo=1500,
        offensive_rating=1.0,
        defensive_rating=1.0
    )

    engine = CFBPredictionEngine(
        simulations=1000
    )

    result = engine.predict(
        home,
        away
    )

    simulation = result["simulation"]

    assert simulation.simulations == 1000

    assert (
        0
        <= simulation.home_win_probability
        <= 1
    )

    assert (
        0
        <= simulation.away_win_probability
        <= 1
    )