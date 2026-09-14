from __future__ import annotations

from pathlib import Path
import inspect
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from engine.elo import (
    EloConfig,
    EloEngine,
)

from engine.ratings import (
    RatingConfig,
    TeamRatingsEngine,
)

from engine.opponent_adjustment import (
    OpponentAdjustmentConfig,
    OpponentAdjustmentEngine,
)

from engine.matchup import (
    MatchupConfig,
    MatchupEngine,
)

from engine.production_model import (
    ProductionProbabilityModel,
)


# ============================================================
# HELPERS
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_signature(
    cls,
    method_name: str,
) -> None:

    method = getattr(
        cls,
        method_name,
        None,
    )

    if method is None:

        print(
            f"{method_name:<32} "
            "NOT PRESENT"
        )

        return

    try:

        signature = inspect.signature(
            method
        )

        print(
            f"{method_name:<32} "
            f"{signature}"
        )

    except Exception as error:

        print(
            f"{method_name:<32} "
            f"ERROR: {error}"
        )


def print_attributes(
    instance,
) -> None:

    attributes = sorted(
        key
        for key in vars(
            instance
        ).keys()
        if not key.startswith(
            "__"
        )
    )

    if not attributes:

        print(
            "No instance attributes visible."
        )

        return

    for attribute in attributes:

        value = getattr(
            instance,
            attribute,
            None,
        )

        print(
            f"  {attribute:<30} "
            f"{type(value).__name__}"
        )


def inspect_engine(
    title: str,
    cls,
    config,
    methods: list[str],
) -> None:

    section(
        title
    )

    print(
        f"Class: {cls.__name__}"
    )

    print(
        f"Constructor: "
        f"{inspect.signature(cls)}"
    )

    print()
    print(
        "Methods:"
    )

    for method_name in methods:

        print_signature(
            cls,
            method_name,
        )

    print()
    print(
        "Fresh instance attributes:"
    )

    try:

        instance = cls(
            config=config
        )

    except TypeError:

        try:

            instance = cls(
                config
            )

        except TypeError:

            instance = cls()

    print_attributes(
        instance
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "CFB PREDICTION CENTRE 2026 V2 "
        "— FORWARD ENGINE INTERFACE AUDIT"
    )

    # ========================================================
    # ELO
    # ========================================================

    inspect_engine(
        title="ELO ENGINE",

        cls=EloEngine,

        config=EloConfig(),

        methods=[
            "process_games",
            "current_ratings",
            "expected_score",
            "expected_probability",
            "win_probability",
            "predict_game",
            "rate_game",
            "update",
            "update_rating",
            "regress_ratings",
            "regress_to_mean",
        ],
    )

    # ========================================================
    # RAW RATINGS
    # ========================================================

    inspect_engine(
        title="RAW TEAM RATINGS ENGINE",

        cls=TeamRatingsEngine,

        config=RatingConfig(),

        methods=[
            "process_games",
            "current_ratings",
            "team_snapshot",
            "snapshot",
            "update_team",
            "update_national",
            "update",
            "regress_team",
            "regress_states",
        ],
    )

    # ========================================================
    # OPPONENT ADJUSTMENT
    # ========================================================

    inspect_engine(
        title="OPPONENT ADJUSTMENT ENGINE",

        cls=OpponentAdjustmentEngine,

        config=OpponentAdjustmentConfig(),

        methods=[
            "process_games",
            "current_ratings",
            "team_snapshot",
            "snapshot",
            "update_team",
            "update",
            "regress_team",
            "regress_states",
        ],
    )

    # ========================================================
    # MATCHUP
    # ========================================================

    inspect_engine(
        title="MATCHUP ENGINE",

        cls=MatchupEngine,

        config=MatchupConfig(),

        methods=[
            "build_game_matchup",
            "build_matchup",
            "process_games",
            "calculate_offensive_matchup",
            "offensive_matchup_index",
            "overall_strength",
            "team_strength",
        ],
    )

    # ========================================================
    # PRODUCTION MODEL
    # ========================================================

    section(
        "FROZEN PRODUCTION PROBABILITY MODEL"
    )

    print(
        f"Class: "
        f"{ProductionProbabilityModel.__name__}"
    )

    print()

    for method_name in [
        "load",
        "prepare_features",
        "predict_home_probability",
        "predict",
    ]:

        print_signature(
            ProductionProbabilityModel,
            method_name,
        )

    # ========================================================
    # CONFIG FIELDS
    # ========================================================

    section(
        "CONFIGURATION FIELDS"
    )

    configs = [
        (
            "EloConfig",
            EloConfig(),
        ),

        (
            "RatingConfig",
            RatingConfig(),
        ),

        (
            "OpponentAdjustmentConfig",
            OpponentAdjustmentConfig(),
        ),

        (
            "MatchupConfig",
            MatchupConfig(),
        ),
    ]

    for (
        name,
        config,
    ) in configs:

        print()
        print(
            name
        )

        for (
            key,
            value,
        ) in vars(
            config
        ).items():

            print(
                f"  {key:<35} "
                f"{value}"
            )

    # ========================================================
    # SOURCE FILE LOCATIONS
    # ========================================================

    section(
        "ENGINE SOURCE FILES"
    )

    classes = [
        EloEngine,
        TeamRatingsEngine,
        OpponentAdjustmentEngine,
        MatchupEngine,
        ProductionProbabilityModel,
    ]

    for cls in classes:

        print(
            f"{cls.__name__:<32} "
            f"{inspect.getfile(cls)}"
        )

    # ========================================================
    # DONE
    # ========================================================

    section(
        "FORWARD ENGINE INTERFACE AUDIT COMPLETE"
    )

    print(
        "No model state was changed."
    )

    print(
        "No CSV or model artifact was changed."
    )

    print()
    print(
        "This output gives us the exact interfaces needed "
        "for the leakage-safe 2026 forward runner."
    )


if __name__ == "__main__":

    main()