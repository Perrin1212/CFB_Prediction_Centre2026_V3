from engine.elo import (
    EloConfig,
    EloEngine,
)


def test_default_rating():

    engine = EloEngine()

    assert (
        engine.get_rating(
            "Georgia"
        )
        ==
        1500.0
    )


def test_home_field_advantage():

    engine = EloEngine()

    probability = (
        engine.expected_home_win(
            home_rating=1500,
            away_rating=1500,
            neutral_site=False,
        )
    )

    assert probability > 0.50


def test_neutral_equal_teams():

    engine = EloEngine()

    probability = (
        engine.expected_home_win(
            home_rating=1500,
            away_rating=1500,
            neutral_site=True,
        )
    )

    assert abs(
        probability
        -
        0.50
    ) < 0.0001


def test_winner_gains_rating():

    engine = EloEngine()

    result = engine.update_game(
        home_team="Georgia",
        away_team="Clemson",
        home_points=31,
        away_points=21,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fbs",
    )

    assert (
        result["home_elo_post"]
        >
        result["home_elo_pre"]
    )

    assert (
        result["away_elo_post"]
        <
        result["away_elo_pre"]
    )


def test_ratings_remain_zero_sum():

    engine = EloEngine()

    result = engine.update_game(
        home_team="Georgia",
        away_team="Clemson",
        home_points=31,
        away_points=21,
        neutral_site=False,
        home_classification="fbs",
        away_classification="fbs",
    )

    before = (
        result["home_elo_pre"]
        +
        result["away_elo_pre"]
    )

    after = (
        result["home_elo_post"]
        +
        result["away_elo_post"]
    )

    assert abs(
        before
        -
        after
    ) < 0.0001


def test_season_regression():

    config = EloConfig(
        preseason_regression=0.35
    )

    engine = EloEngine(
        config
    )

    engine.set_rating(
        "Georgia",
        1800,
    )

    engine.regress_for_new_season()

    assert (
        engine.get_rating(
            "Georgia"
        )
        <
        1800
    )

    assert (
        engine.get_rating(
            "Georgia"
        )
        >
        1500
    )


def test_tie_moves_toward_expectation():

    engine = EloEngine()

    engine.set_rating(
        "Georgia",
        1700,
    )

    engine.set_rating(
        "Clemson",
        1500,
    )

    result = engine.update_game(
        home_team="Georgia",
        away_team="Clemson",
        home_points=24,
        away_points=24,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fbs",
    )

    assert (
        result["home_elo_post"]
        <
        result["home_elo_pre"]
    )

    assert (
        result["away_elo_post"]
        >
        result["away_elo_pre"]
    )


def test_fbs_win_over_fcs_is_dampened():

    normal_engine = EloEngine()

    cross_engine = EloEngine()

    normal = normal_engine.update_game(
        home_team="Georgia",
        away_team="Clemson",
        home_points=42,
        away_points=14,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fbs",
    )

    cross = cross_engine.update_game(
        home_team="Georgia",
        away_team="Mercer",
        home_points=42,
        away_points=14,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fcs",
    )

    assert abs(
        cross["elo_change"]
    ) < abs(
        normal["elo_change"]
    )

    assert (
        cross["game_weight"]
        ==
        cross_engine.config.fbs_over_fcs_weight
    )


def test_fcs_upset_has_more_weight_than_routine_fbs_win():

    config = EloConfig(
        fbs_over_fcs_weight=0.25,
        fcs_over_fbs_weight=0.70,
    )

    fbs_win_engine = EloEngine(
        config
    )

    upset_engine = EloEngine(
        config
    )

    routine = (
        fbs_win_engine.update_game(
            home_team="Georgia",
            away_team="Mercer",
            home_points=35,
            away_points=14,
            neutral_site=True,
            home_classification="fbs",
            away_classification="fcs",
        )
    )

    upset = (
        upset_engine.update_game(
            home_team="Georgia",
            away_team="Mercer",
            home_points=21,
            away_points=24,
            neutral_site=True,
            home_classification="fbs",
            away_classification="fcs",
        )
    )

    assert (
        upset["game_weight"]
        >
        routine["game_weight"]
    )


def test_fbs_fcs_mov_is_capped():

    config = EloConfig(
        fbs_fcs_mov_cap=1.35
    )

    engine = EloEngine(
        config
    )

    result = engine.update_game(
        home_team="Georgia",
        away_team="Mercer",
        home_points=70,
        away_points=0,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fcs",
    )

    assert (
        result["mov_multiplier"]
        <=
        1.35
    )


def test_fbs_vs_fbs_gets_full_weight():

    engine = EloEngine()

    result = engine.update_game(
        home_team="Georgia",
        away_team="Alabama",
        home_points=31,
        away_points=28,
        neutral_site=True,
        home_classification="fbs",
        away_classification="fbs",
    )

    assert (
        result["game_weight"]
        ==
        1.0
    )

    assert (
        result["effective_k"]
        ==
        engine.config.k_factor
    )