from ems.mpc import MPC
from forecast.perfect import PerfectForecast, PerfectRealForecast, RealForecast
from input_function import observation_no_forecast_added_hour_range as input_func
from input_function import active_observations
from forecast.scenarios import Scenario_Generator
import numpy as np

class MPCAgent:
    def __init__(self):
        self.action_space = {}
        self.prev_steps = {}
        self.manager = MPC(0)
        self.scenario_gen = Scenario_Generator(type = 'point_and_variance', n_scenarios =5)
        # self.scenario_gen = PerfectForecast(12)
        self.time_step = 0
        self.num_buildings = None
        self.pv_capacity = None

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation):
        """Get observation return action"""
        self.populate_prev_steps(observation)

        forec_scenarios = self.scenario_gen.generate_scenarios(self.prev_steps, self.time_step, 24)

        actions = self.manager.calculate_powers(
           observation, forec_scenarios, self.time_step
        )
        # actions = [[0], [0], [0], [0], [0]]
        self.time_step += 1
        return actions

    def set_pv_capacity_and_num_buildings(self, building_info):
        self.num_buildings = len(building_info)
        self.pv_capacity = [i["solar_power"] for i in building_info]
        # self.init_forecast()

    def populate_prev_steps(self, observations):
        done_keys = list()
        new_obs, obs_info = input_func(self, observations)
        num_buildings = len(observations)

        # Add history of observations
        for i, obs in enumerate(observations[0]):
            if not 20 <= i < 23:
                key = active_observations[i]
                done_keys.append(key)
                if key not in self.prev_steps.keys():
                    self.prev_steps[key] = [obs]
                else:
                    self.prev_steps[key].append(obs)

        for i in range(num_buildings):
            for j in range(20, 24):
                key = f"{active_observations[j]}_{i}"
                done_keys.append(key)
                obs = observations[i][j]
                if key not in self.prev_steps.keys():
                    self.prev_steps[key] = [obs]
                else:
                    self.prev_steps[key].append(obs)

        for i, obs in enumerate(new_obs):
            key = obs_info[i][0]
            if key not in done_keys:
                if key not in self.prev_steps.keys():
                    self.prev_steps[key] = [obs]
                else:
                    self.prev_steps[key].append(obs)
