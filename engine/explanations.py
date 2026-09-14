from .types import MatchupResult, GameEnvironment, SimulationResult


class ExplanationEngine:

    def generate(
        self,
        matchup: MatchupResult,
        environment: GameEnvironment,
        simulation: SimulationResult
    ):

        home = matchup.home_team
        away = matchup.away_team

        paragraphs = []

        # --------------------------------------------------
        # Overall strength
        # --------------------------------------------------

        if matchup.home_elo > matchup.away_elo:

            elo_difference = (
                matchup.home_elo
                - matchup.away_elo
            )

            paragraphs.append(
                f"{home} enters the matchup with an "
                f"ELO rating of {matchup.home_elo:.0f}, "
                f"{elo_difference:.0f} points higher than "
                f"{away}."
            )

        else:

            elo_difference = (
                matchup.away_elo
                - matchup.home_elo
            )

            paragraphs.append(
                f"{away} enters the matchup with an "
                f"ELO rating of {matchup.away_elo:.0f}, "
                f"{elo_difference:.0f} points higher than "
                f"{home}."
            )

        # --------------------------------------------------
        # Matchup
        # --------------------------------------------------

        if (
            matchup.home_defensive_edge
            > matchup.away_defensive_edge
        ):

            paragraphs.append(
                f"The defensive matchup favours {home}, "
                f"with the model identifying a stronger "
                f"defensive profile against the opposing "
                f"offence."
            )

        else:

            paragraphs.append(
                f"The defensive matchup favours {away}, "
                f"with the model identifying a stronger "
                f"defensive profile against the opposing "
                f"offence."
            )

        # --------------------------------------------------
        # Pace
        # --------------------------------------------------

        paragraphs.append(
            f"We expect approximately "
            f"{environment.expected_total_possessions:.1f} "
            f"combined possessions, with an estimated "
            f"game pace of {environment.expected_pace:.1f} "
            f"plays per team."
        )

        # --------------------------------------------------
        # Scoring
        # --------------------------------------------------

        paragraphs.append(
            f"The simulation projects "
            f"{home} for an average of "
            f"{simulation.expected_home_score:.1f} points "
            f"and {away} for "
            f"{simulation.expected_away_score:.1f} points."
        )

        # --------------------------------------------------
        # Distribution
        # --------------------------------------------------

        paragraphs.append(
            f"Most simulations place {home} between "
            f"{simulation.home_score_p10:.0f} and "
            f"{simulation.home_score_p90:.0f} points, "
            f"while {away} falls between "
            f"{simulation.away_score_p10:.0f} and "
            f"{simulation.away_score_p90:.0f}."
        )

        # --------------------------------------------------
        # Winner
        # --------------------------------------------------

        if (
            simulation.home_win_probability
            >
            simulation.away_win_probability
        ):

            winner = home
            probability = (
                simulation.home_win_probability
            )

        else:

            winner = away
            probability = (
                simulation.away_win_probability
            )

        paragraphs.append(
            f"The model favours {winner} with a "
            f"{probability * 100:.1f}% simulated win "
            f"probability."
        )

        return " ".join(paragraphs)

    def generate_bullets(
        self,
        matchup: MatchupResult,
        environment: GameEnvironment,
        simulation: SimulationResult
    ):

        home = matchup.home_team
        away = matchup.away_team

        return [
            f"{home} ELO: {matchup.home_elo:.0f}",
            f"{away} ELO: {matchup.away_elo:.0f}",

            (
                f"Expected possessions: "
                f"{environment.expected_total_possessions:.1f}"
            ),

            (
                f"Expected score: "
                f"{simulation.expected_home_score:.1f} "
                f"- "
                f"{simulation.expected_away_score:.1f}"
            ),

            (
                f"Expected margin: "
                f"{simulation.expected_margin:+.1f}"
            ),

            (
                f"Win probability: "
                f"{max(
                    simulation.home_win_probability,
                    simulation.away_win_probability
                ) * 100:.1f}%"
            ),

            (
                f"Score volatility: "
                f"{simulation.margin_std:.1f} points"
            )
        ]