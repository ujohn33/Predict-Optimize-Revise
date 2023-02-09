import numpy as np
from gym.spaces import Box
import pandas as pd
import math

low_obs = np.array(
    [
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        4.5999999e00,
        4.5999999e00,
        4.5999999e00,
        4.5999999e00,
        9.0000000e00,
        9.0000000e00,
        9.0000000e00,
        9.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        -1.0000000e00,
        0.07,
        -10,
        -1.0000000e00,
        -1.0000000e00,
        -10,
        0.21,
        0.21,
        0.21,
        0.21,
    ],
    dtype=np.float32,
)
high_obs = np.array(
    [
        13.0,
        8.0,
        25.0,
        33.2,
        33.2,
        33.2,
        33.2,
        101.0,
        101.0,
        101.0,
        101.0,
        1018.0,
        1018.0,
        1018.0,
        1018.0,
        954.0,
        954.0,
        954.0,
        954.0,
        0.2817962,
        10,
        7.0,
        2.0,
        10,
        0.54,
        0.54,
        0.54,
        0.54,
    ],
    dtype=np.float32,
)

active_observations = [
    "month",
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_6h",
    "outdoor_dry_bulb_temperature_predicted_12h",
    "outdoor_dry_bulb_temperature_predicted_24h",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_6h",
    "outdoor_relative_humidity_predicted_12h",
    "outdoor_relative_humidity_predicted_24h",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_6h",
    "diffuse_solar_irradiance_predicted_12h",
    "diffuse_solar_irradiance_predicted_24h",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_6h",
    "direct_solar_irradiance_predicted_12h",
    "direct_solar_irradiance_predicted_24h",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_6h",
    "electricity_pricing_predicted_12h",
    "electricity_pricing_predicted_24h",
]

observation_space_const = [
    Box(low=low_obs, high=high_obs, shape=(28,)) for _ in range(17)
]

carbon_file = "data/citylearn_challenge_2022_phase_1/carbon_intensity_full.csv"

def mask_observations(
    observations,
    num_buildings,
    mask_type="no_weather_no_forecast_one_ems",
):
    new_obs = list()
    if mask_type == "all":
        mask = [1] * 28 + ([0] * 20 + [1] * 4 + [0] * 4) * (num_buildings - 1)
    elif mask_type == "no_weather_no_forecast":
        mask = ([1] * 3 + [0] * 16 + [1] * 6 + [0] * 3) + (
            [0] * 20 + [1] * 4 + [0] * 4
        ) * (num_buildings - 1)
    elif mask_type == "no_weather_no_forecast_one_ems":
        mask = (
            [1] * 3
            + [0] * 16
            + [1] * 1
            + [0] * 4
            + [1] * 1
            + [0] * 3
            + ([0] * 28 * (num_buildings - 1))
        )

    for i, obs in enumerate(observations):
        if mask[i]:
            new_obs.append(obs)

    if mask_type == "no_weather_no_forecast_one_ems":
        if isinstance(observations[0], str):
            building_values = [f"average_{i[:-2]}" for i in observations[20:24]]
        else:
            building_values = [0, 0, 0, 0]
            for i in range(num_buildings):
                for j in range(4):
                    building_values[j] += observations[28 * i + 20 + j] / num_buildings
        new_obs += building_values

    return new_obs

def unrol_observation_space(
    observation, mask_type="no_weather_no_forecast_one_ems", low=True
):
    new_obs = list()
    for obs_box in observation:
        if low:
            new_obs += list(obs_box.low)
        else:
            new_obs += list(obs_box.high)
    new_obs = mask_observations(new_obs, len(observation), mask_type)

    return new_obs


def observation_to_input_function(
    agent, obs, agent_id=0, mask_type="no_weather_no_forecast_one_ems", no_forec=False
):

    inputs = list()
    input_info = list()

    num_buildings = len(obs)

    observation_space = observation_space_const[:num_buildings]
    min_space = unrol_observation_space(observation_space, mask_type, low=True)
    max_space = unrol_observation_space(observation_space, mask_type, low=False)

    names = list()
    values = list()

    for i in range(num_buildings):
        for j, input_name in enumerate(active_observations):
            indiv_obs = [
                "non_shiftable_load",
                "solar_generation",
                "electrical_storage_soc",
                "net_electricity_consumption",
            ]
            if input_name in indiv_obs:
                names.append(f"{input_name}_{i}")
            else:
                names.append(input_name)
            values.append(obs[i][j])

    names = mask_observations(names, num_buildings, mask_type)
    values = mask_observations(values, num_buildings, mask_type)

    for i, name in enumerate(names):
        input_info.append([name, [min_space[i], max_space[i]]])
        inputs.append(values[i])

    month = obs[0][0]
    day_type = obs[0][1]
    hour = obs[0][2]
    co2_obs = obs[0][19]

    if not no_forec:
        if agent is not None:
            row_index = agent.time_step - 1

            if month == hour == day_type == co2_obs == 0:
                diff_carb_1h = 0
                diff_carb_6h = 0
                diff_carb_12h = 0
                co2_next = 0
            else:
                len_carb = len(agent.carbon_df)
                diff_carb_1h = (
                    agent.carbon_df["kg_CO2/kWh"][(row_index + 2) % len_carb]
                    - agent.carbon_df["kg_CO2/kWh"][(row_index + 1) % len_carb]
                )
                diff_carb_6h = (
                    agent.carbon_df["kg_CO2/kWh"][(row_index + 7) % len_carb]
                    - agent.carbon_df["kg_CO2/kWh"][(row_index + 1) % len_carb]
                )
                diff_carb_12h = (
                    agent.carbon_df["kg_CO2/kWh"][(row_index + 13) % len_carb]
                    - agent.carbon_df["kg_CO2/kWh"][(row_index + 1) % len_carb]
                )
                co2_next = agent.carbon_df["kg_CO2/kWh"][(row_index + 1) % len_carb]
        else:
            diff_carb_1h = 0
            diff_carb_6h = 0
            diff_carb_12h = 0
            co2_next = 0

        input_info.append(["diff_carb_1h", [-0.1, 0.1]])
        inputs.append(diff_carb_1h)
        input_info.append(["diff_carb_6h", [-0.1, 0.1]])
        inputs.append(diff_carb_6h)
        input_info.append(["diff_carb_12h", [-0.15, 0.15]])
        inputs.append(diff_carb_12h)

        input_info.append(["CO2 next step", [0.070, 0.29]])
        inputs.append(co2_next)

    return inputs, input_info

    
        # if agent is not None:
        #     if agent.mode == "realistic":
        #         forec_consum = agent.get_real_forecast_consum()
        #     elif agent.mode == "perfect":
        #         forec_consum = agent.get_perf_forecast_consum()
        #     elif agent.mode == "real_time":
        #         forec_consum = agent.get_forecast_consum()

        #     forec_1h = sum(forec_consum) / num_buildings

        #     forec_3h = sum(agent.get_perf_sum_neg_forec(3)) / num_buildings
        #     forec_6h = sum(agent.get_perf_sum_neg_forec(6)) / num_buildings
        #     forec_12h = sum(agent.get_perf_sum_neg_forec(12)) / num_buildings
        #     forec_18h = sum(agent.get_perf_sum_neg_forec(18)) / num_buildings
        # else:
        #     forec_1h = 0
        #     forec_3h = 0
        #     forec_6h = 0
        #     forec_12h = 0
        #     forec_18h = 0

        # # 1h forec
        # input_info.append(["Average net forecast 1h", [-10, 10]])
        # inputs.append(forec_1h)

        # cur_power = inputs[8]
        # # Difference next_cur_step
        # input_info.append(["Average power difference curent and 1h", [-15, 15]])
        # inputs.append(forec_1h - cur_power)

        # # 3h forec

        # input_info.append(["Average forecast production 3h", [-25, 0]])
        # inputs.append(forec_3h)

        # # 6h forec

        # input_info.append(["Average forecast production 6h", [-25, 0]])
        # inputs.append(forec_6h)

        # # 12h forec

        # input_info.append(["Average forecast production 12h", [-25, 0]])
        # inputs.append(forec_12h)

        # # 18h forec

        # input_info.append(["Average forecast production 18h", [-25, 0]])
        # inputs.append(forec_18h)

def observation_mul_trees_no_forecast(
    agent, obs, agent_id=0, mask_type="no_weather_no_forecast"
):
    inputs, input_info = observation_to_input_function(agent, obs, no_forec=True)
    inputs_agent, input_info_agent = observation_to_input_function(
        agent, obs, mask_type="no_weather_no_forecast", no_forec=True
    )
    for i, info_input_ag in enumerate(input_info_agent):
        if info_input_ag[0].endswith(f"_{agent_id}"):
            inputs.append(inputs_agent[i])
            input_info.append(info_input_ag)
    return inputs, input_info


def observation_no_forecast_added_hour_range(
    agent, obs, agent_id=0, mask_type="no_weather_no_forecast"
):
    inputs, input_info = observation_mul_trees_no_forecast(agent, obs)

    hour = inputs[2]

    inputs.append(hour)
    input_info.append(["Hour range", [0.0, 24.0]])

    return inputs, input_info
