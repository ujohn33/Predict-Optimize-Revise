import numpy as np
import pandas as pd
import pickle as pk
from copy import deepcopy
from utils.lgb_feature_engineering_test import init_test_data
from ngboost import NGBRegressor
from ngboost.distns import Normal


class Forecast:
    def __init__(self, n_build, model_dir="models/ngboost/"):
        self.num_buildings = n_build
        self.prev_steps = {}
        self.model_dict = {}
        self.timestep = int(-1)
        path_to_model_load = "models/ngboost/ngboost_direct24.pkl"
        with open(path_to_model_load, 'rb') as file:
            self.ngboost_direct = pk.load(file)
        self.df = init_test_data()
        # pred + optimization
        self.pv_capacity_dict = {}
        self.episode_id = -1
        self.index = {}
        self.max_net_loads_len = int(24 * 1.25)
        self.net_loads_past = []
        self.net_loads_mean = [1.11043, 1.0234, 0.83186, 0.75948, 0.70178, 0.73005, 0.73634, 0.76259, 0.54635, 0.13681,
                               -0.27663, -0.49083, -0.76686, -0.90669, -1.04496, -0.97857, -0.58167, 0.05827, 0.64439,
                               1.06593, 1.31566, 1.14498, 1.12778, 1.1553]

    # pass building info to read pv capacity
    def set_building_info(self, agent_id, building_info):
        self.pv_capacity_dict[agent_id] = building_info['solar_power']
        print('building info {} - pv number {}.'.format(agent_id, 
              self.pv_capacity_dict[agent_id]))

    def update_prev_steps(self, prev_steps):
        self.prev_steps = prev_steps

    def update_current_step(self, current_step):
        self.timestep = current_step 

    def lgb_memo_data(self):
        agent_num = self.n_buildings
        input_ngboost = pd.DataFrame([deepcopy(self.df.iloc[self.timestep, :]) for _ in range(agent_num)])
        cur_net_load = [
            self.prev_steps[f"non_shiftable_load_{id}"][-1] - (self.prev_steps[f"solar_generation_{id}"][-1]) for id in range(agent_num)
        ]
        self.solar_net_loads.append(cur_net_load)
        end_index = len(self.net_loads_past) - self.max_net_loads_len
        end_index = end_index if end_index < 0 else None
        input_ngboost.iloc[:, -self.max_net_loads_len:end_index] = np.array(self.solar_loads_past[::-1]).T
        if len(self.net_loads_past) >= self.max_net_loads_len:
            self.net_loads_past.pop(0)
        self.net_loads_past.append(cur_net_load)

    def lgb_memo_data_and_predict(self):
        hour = self.prev_steps["hour"][-1]
        agent_num = self.num_buildings
        input_ngboost = pd.DataFrame([deepcopy(self.df.iloc[self.timestep, :]) for _ in range(agent_num)])
        cur_net_load = [
            self.prev_steps[f"non_shiftable_load_{id}"][-1] - (self.prev_steps[f"solar_generation_{id}"][-1]) for id in range(agent_num)
        ]
        self.net_loads_past.append(cur_net_load)
        end_index = len(self.net_loads_past) - self.max_net_loads_len
        end_index = end_index if end_index < 0 else None
        input_ngboost.iloc[:, -self.max_net_loads_len:end_index] = np.array(self.net_loads_past[::-1]).T
        if len(self.net_loads_past) >= self.max_net_loads_len:
            self.net_loads_past.pop(0)
        self.net_loads_past.append(cur_net_load)

        # imputation
        if input_ngboost.isnull().values.any():
            for i in range(self.max_net_loads_len):
                if input_ngboost['Net_Past_{}'.format(i)].isnull().sum():
                    input_ngboost['Net_Past_{}'.format(i)] = (
                        self.net_loads_mean[(hour - i - 1) % 24]
                    )
        if self.timestep > self.max_net_loads_len:
            input_ngboost = input_ngboost.fillna(0)

        next_loads24 = self.ngboost_direct.predict(input_ngboost)  # (5, 24)
        next_load24_dists = [[
            self.ngboost_direct.estimators_[i].pred_dist(input_ngboost.iloc[j].values.reshape(1,-1)) for i in range(24)] for j in range(agent_num)
            ]
        return next_loads24, next_load24_dists
