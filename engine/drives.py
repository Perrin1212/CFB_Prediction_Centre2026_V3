import numpy as np


class DriveModel:

    def __init__(self):
        pass

    def expected_drive_outcomes(
        self,
        offensive_rating,
        defensive_rating,
        third_down_pct,
        red_zone_td_pct,
        turnover_rate,
        explosive_rate,
        starting_field_position
    ):

        efficiency = (
            offensive_rating
            - defensive_rating
        )

        touchdown_probability = (
            0.18
            + efficiency * 0.015
            + (red_zone_td_pct - 0.60) * 0.20
        )

        touchdown_probability += (
            explosive_rate - 0.10
        ) * 0.30

        touchdown_probability += (
            third_down_pct - 0.40
        ) * 0.15

        touchdown_probability += (
            starting_field_position - 28.0
        ) * 0.005

        touchdown_probability = np.clip(
            touchdown_probability,
            0.05,
            0.50
        )

        turnover_probability = np.clip(
            turnover_rate,
            0.005,
            0.15
        )

        field_goal_probability = np.clip(
            0.20
            + (
                starting_field_position - 28.0
            ) * 0.01,
            0.05,
            0.50
        )

        punt_probability = max(
            0.0,
            1.0
            - touchdown_probability
            - turnover_probability
            - field_goal_probability
        )

        return {
            "touchdown_probability":
                touchdown_probability,

            "field_goal_probability":
                field_goal_probability,

            "turnover_probability":
                turnover_probability,

            "punt_probability":
                punt_probability
        }

    def simulate_drives(
        self,
        possessions,
        drive_parameters,
        rng=None
    ):

        if rng is None:
            rng = np.random.default_rng()

        scores = []

        for count in possessions:

            total_points = 0

            for _ in range(int(count)):

                random_value = rng.random()

                td = drive_parameters[
                    "touchdown_probability"
                ]

                fg = drive_parameters[
                    "field_goal_probability"
                ]

                turnover = drive_parameters[
                    "turnover_probability"
                ]

                if random_value < td:
                    total_points += 7

                elif random_value < td + fg:
                    total_points += 3

                elif random_value < td + fg + turnover:
                    total_points += 0

                else:
                    total_points += 0

            scores.append(total_points)

        return np.array(scores)