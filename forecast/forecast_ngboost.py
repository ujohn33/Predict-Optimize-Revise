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
        path_to_model_load = "models/ngboost/ngboost_direct24_load_num_leaves_16.pkl"
        with open(path_to_model_load, 'rb') as file:
            self.ngboost_direct = pk.load(file)
            # print the input shape of the ngboost model
        path_to_model_solar = 'models/lgbm_direct24_solar_num_leaves_32.pkl'
        with open(path_to_model_solar, 'rb') as file:
            self.lgbm_direct = pk.load(file)
        self.df = init_test_data('load')
        self.df_solar = init_test_data('solar')
        # pred + optimization
        self.pv_capacity_dict = {}
        self.episode_id = -1
        self.index = {}
        self.max_loads_len = int(24 * 1.25)
        self.loads_past = []
        self.solar_past = []
        self.solar_train_mean = [0., 0., 0., 0., 0., 0.,
                                          0.00772142, 0.06758006, 0.17815801, 0.3080483, 0.42879198,
                                          0.50559368, 0.54869532, 0.54497649, 0.50235824, 0.41020424,
                                          0.28657643, 0.16933703, 0.0677388, 0.0029458, 0.,
                                          0., 0., 0.]
        self.loads_train_mean = [1.02500752, 0.83214035, 0.75899703, 0.70115535, 0.72968285, 0.73625935,
                                         0.79389332, 0.82114927, 0.86693869, 0.99444716, 1.28633053,
                                         1.33460672, 1.37912689, 1.23035236, 1.12216157, 1.13794405,
                                         1.26107395, 1.35709849, 1.3525594, 1.32629516, 1.14569356,
                                         1.13128801, 1.15672385, 1.11261601]


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
        self.solar_loads.append(cur_net_load)
        end_index = len(self.loads_past) - self.max_loadss_len
        end_index = end_index if end_index < 0 else None
        input_ngboost.iloc[:, -self.max_loadss_len:end_index] = np.array(self.solar_loads_past[::-1]).T
        if len(self.loads_past) >= self.max_loadss_len:
            self.loads_past.pop(0)
        #self.loads_past.append(cur_net_load)

    def ngboost_memo_data_and_predict_load(self):
        hour = self.prev_steps["hour"][-1]
        agent_num = self.num_buildings
        input_ngboost = pd.DataFrame([deepcopy(self.df.iloc[self.timestep, :]) for _ in range(agent_num)])
        cur_load = [
            self.prev_steps[f"non_shiftable_load_{id}"][-1] for id in range(agent_num)
        ]
        self.loads_past.append(cur_load)
        end_index = len(self.loads_past) - self.max_loads_len
        end_index = end_index if end_index < 0 else None
        input_ngboost.iloc[:, -self.max_loads_len:end_index] = np.array(self.loads_past[::-1]).T
        if len(self.loads_past) >= self.max_loads_len:
            self.loads_past.pop(0)
        #self.loads_past.append(cur_load)

        # imputation
        if input_ngboost.isnull().values.any():
            for i in range(self.max_loads_len):
                if input_ngboost['Load_Past_{}'.format(i)].isnull().sum():
                    input_ngboost['Load_Past_{}'.format(i)] = (
                        self.loads_train_mean[(hour - i - 1) % 24]
                    )
        if self.timestep > self.max_loads_len:
            input_ngboost = input_ngboost.fillna(0)

        next_loads24 = self.ngboost_direct.predict(input_ngboost)  # (5, 24)
        next_load24_dists = [[
            self.ngboost_direct.estimators_[i].pred_dist(input_ngboost.iloc[j].values.reshape(1,-1)) for i in range(24)] for j in range(agent_num)
            ]
        # clip at 0
        next_loads24 = np.clip(next_loads24, 0, None)
        return next_loads24, next_load24_dists

    def lgb_memo_data_and_predict_solar(self):
        hour = self.prev_steps["hour"][-1]
        agent_num = self.num_buildings
        input_lgbm = pd.DataFrame([deepcopy(self.df_solar.iloc[self.timestep, :]) for _ in range(agent_num)])
        pv_capacity = list(self.pv_capacity_dict.values())
        cur_solar = [
            self.prev_steps[f"solar_generation_{id}"][-1] / pv_capacity[id] for id in range(agent_num)
        ] 
        self.solar_past.append(cur_solar)
        end_index = len(self.solar_past) - self.max_loads_len
        end_index = end_index if end_index < 0 else None
        input_lgbm.iloc[:, -self.max_loads_len:end_index] = np.array(self.solar_past[::-1]).T
        if len(self.solar_past) >= self.max_loads_len:
            self.solar_past.pop(0)
        #self.solar_past.append(cur_solar)

        # imputation
        if input_lgbm.isnull().values.any():
            for i in range(self.max_loads_len):
                if input_lgbm['Solar_Past_{}'.format(i)].isnull().sum():
                    input_lgbm['Solar_Past_{}'.format(i)] = (
                        self.solar_train_mean[(hour - i - 1) % 24]
                    )
        if self.timestep > self.max_loads_len:
            input_lgbm = input_lgbm.fillna(0)
        solar_capacity_scaling = np.expand_dims(pv_capacity, -1).repeat(24, 1)
        next_solar24 = self.lgbm_direct.predict(input_lgbm) * solar_capacity_scaling  # (5, 24)
        # divide by 1000
        next_solar24 /= 1000
        # clip at 0
        next_solar24 = np.clip(next_solar24, 0, None)
        return next_solar24

    
