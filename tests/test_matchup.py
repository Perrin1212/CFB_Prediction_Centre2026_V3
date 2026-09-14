from engine.types import TeamRatings
from engine.matchup import MatchupEngine


def test_matchup():

    home = TeamRatings(
        team_id=1,
        team_name="Home",
        elo=1600,
        offensive_rating=1.2,
        defensive_rating=1.3
    )

    away = TeamRatings(
        team_id=2,
        team_name="Away",
        elo=1500,
        offensive_rating=1.0,
        defensive_rating=1.0
    )

    engine = MatchupEngine()

    result = engine.compare(
        home,
        away
    )

    assert result.home_elo == 1600
    assert result.away_elo == 1500