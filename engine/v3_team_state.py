from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


# =============================================================================
# V3 CHRONOLOGICAL TEAM STATE
# =============================================================================
#
# The state engine deliberately separates:
#
#   1. possession-derived performance state
#   2. opponent/schedule context
#   3. authoritative game-result ELO
#
# A game with malformed or incomplete PBP may therefore still contribute its
# trustworthy final result to ELO without contaminating possession metrics.
# =============================================================================


METRICS = {
    "drives": 11.5,
    "points_per_drive": 2.25,
    "td_rate": 0.30,
    "fg_rate": 0.12,
    "turnover_rate": 0.12,
    "punt_rate": 0.36,
    "downs_rate": 0.05,
    "avg_start_field": 27.5,
    "plays_per_drive": 6.0,
    "yards_per_drive": 35.0,
    "success_rate": 0.42,
    "ppa_per_play": 0.0,
    "explosive_rate": 0.10,
    "positive_ppa": 0.90,
    "third_down_rate": 0.40,
    "fourth_down_rate": 0.52,
}


@dataclass
class SideState:
    n: int = 0
    values: dict[str, float] = field(
        default_factory=lambda: dict(METRICS)
    )
    # ``prior_values`` is the regressed entering-season identity.  Current
    # observations are accumulated separately, then blended using the explicit
    # maturity curve below.  This prevents the fifth game from receiving an
    # implausibly large one-game EWMA weight merely to reach 90% current season.
    prior_values: dict[str, float] = field(default_factory=dict)
    current_values: dict[str, float] = field(default_factory=dict)


@dataclass
class TeamState:
    # Number of games that supplied VALID possession observations.
    #
    # This intentionally does not count games whose PBP was quarantined,
    # because this value feeds possession-state maturity.
    games: int = 0

    offense: SideState = field(
        default_factory=SideState
    )

    defense: SideState = field(
        default_factory=SideState
    )

    elo: float = 1500.0

    schedule_off: float = 100.0
    schedule_def: float = 100.0


@dataclass(frozen=True)
class StateConfig:
    current_season_weights: tuple[float, ...] = (
        0.00,
        0.35,
        0.55,
        0.70,
        0.80,
        0.90,
        0.92,
    )
    current_recency_floor: float = 0.25

    opponent_power: float = 0.55
    opponent_floor: float = 0.72
    opponent_ceiling: float = 1.32

    preseason_regression: float = 0.42

    elo_k: float = 28.0
    elo_hfa: float = 75.0


class ChronologicalStateEngine:
    """
    Chronological pregame team state.

    Core principles
    ---------------
    - No future information.
    - Opponent-adjust every valid possession observation.
    - Equal kickoff timestamps must be handled externally as batches.
    - PBP corruption must never enter possession state.
    - Authoritative game results may still update ELO when PBP is quarantined.
    """

    def __init__(
        self,
        cfg: StateConfig | None = None,
    ):
        self.cfg = cfg or StateConfig()
        self.states: dict[str, TeamState] = {}

    # =========================================================================
    # BASIC STATE ACCESS
    # =========================================================================

    def get(
        self,
        team,
    ) -> TeamState:
        return self.states.setdefault(
            str(team).strip(),
            TeamState(),
        )

    def regress_season(
        self,
    ) -> None:
        """
        Regress all stored state toward neutral priors between seasons.
        """

        r = self.cfg.preseason_regression

        for s in self.states.values():

            for side in (
                s.offense,
                s.defense,
            ):
                side.values = {
                    k: (
                        (1 - r)
                        * side.values.get(k, v)
                        + r * v
                    )
                    for k, v in METRICS.items()
                }
                side.prior_values = dict(side.values)
                side.current_values = {}
                side.n = 0

            s.elo = (
                (1 - r) * s.elo
                + r * 1500.0
            )

            s.schedule_off = (
                (1 - r) * s.schedule_off
                + r * 100.0
            )

            s.schedule_def = (
                (1 - r) * s.schedule_def
                + r * 100.0
            )

            # Possession maturity resets for the new season.
            s.games = 0

    # =========================================================================
    # TEAM QUALITY RATINGS
    # =========================================================================

    @staticmethod
    def offense_rating(
        s: TeamState,
    ) -> float:
        """
        Composite offensive quality index.

        100 = approximately neutral/reference quality.
        """

        v = s.offense.values

        score = (
            100
            * (
                max(
                    v["points_per_drive"],
                    0.15,
                )
                / 2.25
            )
            ** 0.34
            * (
                max(
                    v["success_rate"],
                    0.08,
                )
                / 0.42
            )
            ** 0.19
            * (
                max(
                    v["yards_per_drive"],
                    5.0,
                )
                / 35.0
            )
            ** 0.16
            * (
                max(
                    v["plays_per_drive"],
                    2.0,
                )
                / 6.0
            )
            ** 0.08
            * (
                max(
                    v["avg_start_field"],
                    10.0,
                )
                / 27.5
            )
            ** 0.08
            * (
                (1 - v["turnover_rate"])
                / 0.88
            )
            ** 0.15
        )

        return float(
            np.clip(
                score,
                50.0,
                160.0,
            )
        )

    @staticmethod
    def defense_rating(
        s: TeamState,
    ) -> float:
        """
        Composite defensive quality index.

        Defensive state stores what the defense ALLOWS, therefore lower
        allowed PPD/success/yards produces a stronger rating.
        """

        v = s.defense.values

        score = (
            100
            * (
                2.25
                / max(
                    v["points_per_drive"],
                    0.15,
                )
            )
            ** 0.34
            * (
                0.42
                / max(
                    v["success_rate"],
                    0.08,
                )
            )
            ** 0.19
            * (
                35.0
                / max(
                    v["yards_per_drive"],
                    5.0,
                )
            )
            ** 0.16
            * (
                6.0
                / max(
                    v["plays_per_drive"],
                    2.0,
                )
            )
            ** 0.08
            * (
                27.5
                / max(
                    v["avg_start_field"],
                    10.0,
                )
            )
            ** 0.08
            * (
                (1 - v["turnover_rate"])
                / 0.88
            )
            ** -0.15
        )

        return float(
            np.clip(
                score,
                50.0,
                160.0,
            )
        )

    # =========================================================================
    # SNAPSHOT
    # =========================================================================

    def snapshot(
        self,
        team,
    ) -> dict:
        """
        Return the complete PRE-GAME state used by matchup construction.
        """

        s = self.get(team)

        maturity = self.current_season_weight(s.games)

        out = {
            "games": s.games,
            "elo": s.elo,
            "off_rating": self.offense_rating(s),
            "def_rating": self.defense_rating(s),
            "schedule_off": s.schedule_off,
            "schedule_def": s.schedule_def,
            "maturity": maturity,
        }

        out.update(
            {
                "off_" + k: v
                for k, v
                in s.offense.values.items()
            }
        )

        out.update(
            {
                "def_" + k: v
                for k, v
                in s.defense.values.items()
            }
        )

        return out

    def current_season_weight(self, games: int) -> float:
        """Return the explicit share assigned to current-season identity."""
        n = max(0, int(games))
        curve = self.cfg.current_season_weights
        if n < len(curve):
            return float(curve[n])
        return float(curve[-1])

    # =========================================================================
    # OPPONENT ADJUSTMENT
    # =========================================================================

    def _factor(
        self,
        rating,
    ) -> float:
        """
        Convert opponent quality to a bounded multiplicative adjustment.
        """

        return float(
            np.clip(
                (
                    rating / 100.0
                )
                ** self.cfg.opponent_power,
                self.cfg.opponent_floor,
                self.cfg.opponent_ceiling,
            )
        )

    def _ew(
        self,
        side: SideState,
        observed: dict,
        factor: float,
        offense: bool,
    ) -> None:
        """
        Exponentially weight one valid possession-derived team-game
        observation after opponent adjustment.
        """

        if not side.prior_values:
            side.prior_values = dict(side.values)

        observation_number = side.n + 1
        recency_alpha = (
            1.0
            if side.n == 0
            else max(
                self.cfg.current_recency_floor,
                1.0 / observation_number,
            )
        )
        maturity = self.current_season_weight(observation_number)

        higher_good = {
            "points_per_drive",
            "td_rate",
            "avg_start_field",
            "plays_per_drive",
            "yards_per_drive",
            "success_rate",
            "ppa_per_play",
            "explosive_rate",
            "positive_ppa",
            "third_down_rate",
            "fourth_down_rate",
        }

        lower_good = {
            "turnover_rate",
            "punt_rate",
            "downs_rate",
        }

        for k, prior in METRICS.items():

            raw = observed.get(
                k,
                prior,
            )

            try:
                raw_float = float(raw)
            except (
                TypeError,
                ValueError,
            ):
                raw_float = prior

            x = (
                raw_float
                if np.isfinite(raw_float)
                else prior
            )

            if offense:

                if k in higher_good:
                    x *= factor

                elif k in lower_good:
                    x /= factor

            else:
                # Defensive state stores values ALLOWED.
                #
                # If the offense faced was strong, discount what the defense
                # allowed. If the offense was weak, penalise the defense more.
                if k in higher_good:
                    x /= factor

                elif k in lower_good:
                    x *= factor

            previous_current = side.current_values.get(k, x)
            current = (
                x
                if side.n == 0
                else (1 - recency_alpha) * previous_current + recency_alpha * x
            )
            side.current_values[k] = current
            entering = side.prior_values.get(k, prior)
            side.values[k] = (1 - maturity) * entering + maturity * current

        side.n += 1

    # =========================================================================
    # RESULT / CONTEXT UPDATES
    # =========================================================================

    def _update_schedule_context(
        self,
        hs: TeamState,
        as_: TeamState,
        home_off_rating: float,
        away_off_rating: float,
        home_def_rating: float,
        away_def_rating: float,
    ) -> None:
        """
        Update schedule-quality context using PRE-GAME opponent ratings.

        This does not require trustworthy PBP from the current game.
        """

        hs.schedule_def = (
            0.8 * hs.schedule_def
            + 0.2 * away_def_rating
        )

        as_.schedule_def = (
            0.8 * as_.schedule_def
            + 0.2 * home_def_rating
        )

        hs.schedule_off = (
            0.8 * hs.schedule_off
            + 0.2 * away_off_rating
        )

        as_.schedule_off = (
            0.8 * as_.schedule_off
            + 0.2 * home_off_rating
        )

    def _update_elo(
        self,
        hs: TeamState,
        as_: TeamState,
        home_points: float,
        away_points: float,
        neutral: bool,
        home_classification: str | None = None,
        away_classification: str | None = None,
    ) -> None:
        """
        Update ELO using only authoritative final score/result information.
        """

        hfa = (
            0.0
            if neutral
            else self.cfg.elo_hfa
        )

        expected_home = (
            1
            / (
                1
                + 10
                ** (
                    -(
                        (
                            hs.elo
                            + hfa
                        )
                        - as_.elo
                    )
                    / 400.0
                )
            )
        )

        if home_points > away_points:
            actual_home = 1.0

        elif home_points < away_points:
            actual_home = 0.0

        else:
            actual_home = 0.5

        margin = abs(
            home_points
            - away_points
        )

        mov = math.log1p(
            margin
        )

        multiplier = min(
            1.8,
            max(
                0.7,
                mov / 2.0,
            ),
        )

        # ELO learns faster during the first few games, when the regressed
        # preseason rating is most uncertain.
        early_games = min(hs.games, as_.games)
        early_multiplier = 1.25 if early_games <= 1 else 1.10 if early_games <= 3 else 1.0

        # Routine FBS wins over FCS opponents move ratings very little.  An FCS
        # upset retains full impact and remains meaningful internally.
        hc = str(home_classification or "").strip().casefold()
        ac = str(away_classification or "").strip().casefold()
        cross_division_multiplier = 1.0
        if {hc, ac} == {"fbs", "fcs"}:
            fbs_won = (
                (hc == "fbs" and home_points > away_points)
                or (ac == "fbs" and away_points > home_points)
            )
            cross_division_multiplier = 0.35 if fbs_won else 1.0

        delta = (
            self.cfg.elo_k
            * early_multiplier
            * cross_division_multiplier
            * multiplier
            * (
                actual_home
                - expected_home
            )
        )

        hs.elo += delta
        as_.elo -= delta

    # =========================================================================
    # HEALTHY GAME UPDATE
    # =========================================================================

    def update_game(
        self,
        home,
        away,
        h_obs,
        a_obs,
        home_points,
        away_points,
        neutral=False,
        home_classification: str | None = None,
        away_classification: str | None = None,
    ) -> None:
        """
        Full update for a game with trustworthy possession data.

        Updates:
        - offense possession state
        - defense possession state
        - schedule context
        - ELO
        - possession-state maturity
        """

        hs = self.get(home)
        as_ = self.get(away)

        # Capture PRE-GAME quality ratings.
        home_off_rating = self.offense_rating(
            hs
        )

        away_off_rating = self.offense_rating(
            as_
        )

        home_def_rating = self.defense_rating(
            hs
        )

        away_def_rating = self.defense_rating(
            as_
        )

        # ---------------------------------------------------------------------
        # Possession state
        # ---------------------------------------------------------------------

        self._ew(
            hs.offense,
            h_obs,
            self._factor(
                away_def_rating
            ),
            True,
        )

        self._ew(
            as_.offense,
            a_obs,
            self._factor(
                home_def_rating
            ),
            True,
        )

        self._ew(
            hs.defense,
            a_obs,
            self._factor(
                away_off_rating
            ),
            False,
        )

        self._ew(
            as_.defense,
            h_obs,
            self._factor(
                home_off_rating
            ),
            False,
        )

        # ---------------------------------------------------------------------
        # Opponent/schedule context
        # ---------------------------------------------------------------------

        self._update_schedule_context(
            hs=hs,
            as_=as_,
            home_off_rating=home_off_rating,
            away_off_rating=away_off_rating,
            home_def_rating=home_def_rating,
            away_def_rating=away_def_rating,
        )

        # ---------------------------------------------------------------------
        # Authoritative result
        # ---------------------------------------------------------------------

        self._update_elo(
            hs=hs,
            as_=as_,
            home_points=float(
                home_points
            ),
            away_points=float(
                away_points
            ),
            neutral=bool(
                neutral
            ),
            home_classification=home_classification,
            away_classification=away_classification,
        )

        # Number of VALID possession observations.
        hs.games += 1
        as_.games += 1

    # =========================================================================
    # QUARANTINED GAME UPDATE
    # =========================================================================

    def update_result_only(
        self,
        home,
        away,
        home_points,
        away_points,
        neutral=False,
        home_classification: str | None = None,
        away_classification: str | None = None,
    ) -> None:
        """
        Update trustworthy result/context information for a game whose PBP
        has been quarantined.

        Updates:
        - schedule context
        - ELO

        Does NOT update:
        - offensive possession metrics
        - defensive possession metrics
        - possession maturity (`games`)
        - SideState observation counts

        This allows malformed CFBD play-by-play to be ignored without
        pretending the actual game result never happened.
        """

        hs = self.get(home)
        as_ = self.get(away)

        # Capture PRE-GAME opponent quality.
        home_off_rating = self.offense_rating(
            hs
        )

        away_off_rating = self.offense_rating(
            as_
        )

        home_def_rating = self.defense_rating(
            hs
        )

        away_def_rating = self.defense_rating(
            as_
        )

        self._update_schedule_context(
            hs=hs,
            as_=as_,
            home_off_rating=home_off_rating,
            away_off_rating=away_off_rating,
            home_def_rating=home_def_rating,
            away_def_rating=away_def_rating,
        )

        self._update_elo(
            hs=hs,
            as_=as_,
            home_points=float(
                home_points
            ),
            away_points=float(
                away_points
            ),
            neutral=bool(
                neutral
            ),
            home_classification=home_classification,
            away_classification=away_classification,
        )
