from engine.types import TeamRatings
from engine.predictor import CFBPredictionEngine


def main():

    georgia = TeamRatings(

        team_id=1,

        team_name="Georgia",

        elo=2100,

        offensive_rating=1.20,

        defensive_rating=1.35,

        special_teams_rating=0.20,

        pace=68,

        points_per_drive=2.6,

        points_per_drive_allowed=1.5,

        yards_per_play=6.5,

        yards_per_play_allowed=4.5,

        pass_efficiency=1.15,

        pass_efficiency_allowed=0.85,

        rush_efficiency=6.0,

        rush_efficiency_allowed=3.8,

        third_down_pct=0.48,

        third_down_def_pct=0.34,

        red_zone_td_pct=0.68,

        red_zone_td_def_pct=0.50,

        explosive_play_rate=0.13,

        explosive_play_rate_allowed=0.08,

        turnover_rate=0.015,

        turnover_rate_forced=0.025,

        starting_field_position=31
    )

    clemson = TeamRatings(

        team_id=2,

        team_name="Clemson",

        elo=1950,

        offensive_rating=1.00,

        defensive_rating=1.10,

        special_teams_rating=0.10,

        pace=70,

        points_per_drive=2.2,

        points_per_drive_allowed=1.9,

        yards_per_play=5.9,

        yards_per_play_allowed=5.0,

        pass_efficiency=1.05,

        pass_efficiency_allowed=0.98,

        rush_efficiency=5.3,

        rush_efficiency_allowed=4.5,

        third_down_pct=0.41,

        third_down_def_pct=0.39,

        red_zone_td_pct=0.60,

        red_zone_td_def_pct=0.57,

        explosive_play_rate=0.11,

        explosive_play_rate_allowed=0.10,

        turnover_rate=0.022,

        turnover_rate_forced=0.018,

        starting_field_position=28
    )

    engine = CFBPredictionEngine(
        simulations=10_000
    )

    result = engine.predict(
        home_team=georgia,
        away_team=clemson
    )

    simulation = result["simulation"]

    print()
    print("=" * 60)
    print("CFB PREDICTION")
    print("=" * 60)

    print()

    print(
        f"{georgia.team_name}: "
        f"{simulation.expected_home_score:.1f}"
    )

    print(
        f"{clemson.team_name}: "
        f"{simulation.expected_away_score:.1f}"
    )

    print()

    print(
        f"Georgia win: "
        f"{simulation.home_win_probability * 100:.2f}%"
    )

    print(
        f"Clemson win: "
        f"{simulation.away_win_probability * 100:.2f}%"
    )

    print()

    print(
        f"Expected margin: "
        f"{simulation.expected_margin:+.1f}"
    )

    print(
        f"Expected total: "
        f"{simulation.expected_total:.1f}"
    )

    print()

    print("MODEL EXPLANATION")
    print("-" * 60)

    print(
        result["explanation"]
    )

    print()

    print("KEY MODEL OUTPUTS")
    print("-" * 60)

    for bullet in result["bullets"]:
        print(f"• {bullet}")


if __name__ == "__main__":
    main()