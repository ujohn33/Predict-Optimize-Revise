import numpy as np
import joblib
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm, multivariate_normal


class Forecast:
    def __init__(self, n_build, model_dir="models/lag_minus_1/", point_forecast=False):
        self.num_buildings = n_build
        self.prev_steps = {}
        self.model_dict = {}
        self.pv_capacity = [4.0, 4.0, 4.0, 5.0, 4.0]

        # start dicts for min and max values for each building
        self.cons_min_dict = {}
        self.cons_max_dict = {}
        self.diffuse_solar_init = {}
        self.drybulb_temp_init = {}
        self.relative_humidity_init = {}
        self.diffuse_solar_pred = {}
        self.X_old = np.zeros((8760, self.num_buildings, 17))
        self.y_old = np.zeros((8760, self.num_buildings, 1))
        self.init_forecast()
        self.model_pt = joblib.load(model_dir+"lgb_cons_nodiff_next_step.pkl")
        self.model_pt_next = joblib.load(model_dir+'lgb_cons_nodiff_next_step.pkl')
        self.solar_model = joblib.load(model_dir+'lgb_solar_next_step.pkl')


    def init_forecast(self):

        # make a conservative estimate for max and min consumption
        self.cons_min = 0
        self.cons_max = 5

        # initialize dicts with conservative estimates
        for i in range(self.num_buildings):
            self.cons_min_dict[i] = self.cons_min
            self.cons_max_dict[i] = self.cons_max

        self.cons_min_dict[0] = 0.057
        self.cons_min_dict[1] = 9.752909342447929e-08
        self.cons_min_dict[2] = 9.155273437500004e-08
        self.cons_min_dict[3] = 9.833333084922426e-08
        self.cons_min_dict[4] = 0.0
        self.cons_max_dict[0] = 7.98748331705729
        self.cons_max_dict[1] = 6.843133333333333
        self.cons_max_dict[2] = 6.101333333333332
        self.cons_max_dict[3] = 6.749616666666667
        self.cons_max_dict[4] = 4.938766666666667
        self.diffuse_solar_init[0] = 0.0
        self.diffuse_solar_init[1] = 0.0
        self.diffuse_solar_init[2] = 0.0
        self.diffuse_solar_init[3] = 0.0
        self.diffuse_solar_init[4] = 0.0
        self.drybulb_temp_init[0] = 20.1
        self.drybulb_temp_init[1] = 19.7
        self.drybulb_temp_init[2] = 19.3
        self.drybulb_temp_init[3] = 18.9
        self.drybulb_temp_init[4] = 18.3
        self.relative_humidity_init[0] = 79.0
        self.relative_humidity_init[1] = 79.0
        self.relative_humidity_init[2] = 78.0
        self.relative_humidity_init[3] = 78.0
        self.relative_humidity_init[4] = 81.0

    def min_max_normalize(self, vector, min_value, max_value):
        return (vector - min_value) / (max_value - min_value)

    def min_max_denormalize(self, vector, min_value, max_value):
        return vector * (max_value - min_value) + min_value

    def update_prev_steps(self, prev_steps):
        self.prev_steps = prev_steps

    def update_current_step(self, current_step):
        self.time_step = current_step 

    def update_min_max_scaler(self, id):
        last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] 
        self.cons_min_dict[id] = min(self.cons_min_dict[id], last_val)
        self.cons_max_dict[id] = max(self.cons_max_dict[id], last_val)

    def get_cons_forecast_step(self, step: int, id: int, last_param=False):
        # ['Month', 'Hour', 'hour_x', 'hour_y', 'month_x', 'month_y',
        #  'net_target-23', 'diffuse_solar_radiation+1', 'direct_solar_radiation+1',
        #   'relative_humidity+1', 'drybulb_temp+1']
        if step == 1:
            X_order = self.model_pt_next.feature_name()
        else:
            X_order = self.model_pt.feature_name()
        # make a vector of last values from prev steps using keys from X_order
        X = np.zeros(len(X_order))
        # print(self.time_step)
        for i, key in enumerate(X_order):
            # if key starts with net_target
            if key == "cons_target-23":
                lag_step = 24 - step
                if self.time_step > lag_step:
                    lag_step = -lag_step - 1
                    last_val = self.prev_steps[f"non_shiftable_load_{id}"][lag_step] 
                    norm_val = self.min_max_normalize(
                        last_val, self.cons_min_dict[id], self.cons_max_dict[id]
                    )
                else:
                    norm_val = np.nan
                X[i] = norm_val
            elif key == "cons_target_diff":
                if self.time_step > 2:
                    last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] 
                    last_last_val = self.prev_steps[f"non_shiftable_load_{id}"][-2] 
                    norm_val = self.min_max_normalize(
                        last_val, self.cons_min_dict[id], self.cons_max_dict[id]
                    )
                    norm_last_val = self.min_max_normalize(
                        last_last_val, self.cons_min_dict[id], self.cons_max_dict[id]
                    )
                    X[i] = norm_val - norm_last_val
                else:
                    X[i] = np.nan
            elif key == "cons_target":
                if last_param is False:
                    last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] 
                    norm_val = self.min_max_normalize(
                        last_val, self.cons_min_dict[id], self.cons_max_dict[id]
                    )
                else:
                    norm_val = self.min_max_normalize(
                        last_param, self.cons_min_dict[id], self.cons_max_dict[id]
                    )
                X[i] = norm_val
            elif key.startswith("diffuse_solar_radiation+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "diffuse_solar_irradiance_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = np.log1p(last_val)
            elif key.startswith("direct_solar_radiation+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "direct_solar_irradiance_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = np.log1p(last_val)
            elif key.startswith("relative_humidity+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "outdoor_relative_humidity_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = last_val
            elif key.startswith("drybulb_temp+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "outdoor_dry_bulb_temperature_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = last_val
            elif key == "hour":
                X[i] = (self.prev_steps["hour"][-1] + step) % 24
            elif key == "hour_x":
                X[i] = np.cos(
                    2 * np.pi * ((self.prev_steps["hour"][-1] + step) % 24) / 24
                )
            elif key == "hour_y":
                X[i] = np.sin(
                    2 * np.pi * ((self.prev_steps["hour"][-1] + step) % 24) / 24
                )
            elif key == "month_x":
                X[i] = np.cos(2 * np.pi * self.prev_steps["month"][-1] / 12)
            elif key == "month_y":
                X[i] = np.sin(2 * np.pi * self.prev_steps["month"][-1] / 12)
            elif key in self.prev_steps.keys():
                X[i] = self.prev_steps[key][-1]
        #print(X)
        if step == 1:
            # add a value to a prediction vector
            forec = self.model_pt_next.predict(X.reshape(1, -1))
        else:
            # add a value to a prediction vector
            forec = self.model_pt.predict(X.reshape(1, -1))
        # denormalize the values
        forec = self.min_max_denormalize(
            forec, self.cons_min_dict[id], self.cons_max_dict[id]
        )
        forec = forec[0]
        return forec

    def get_solar_forecast_step(self, step: int, id: int):
        # ['Month', 'Hour', 'hour_x', 'hour_y', 'month_x', 'month_y',
        #  'net_target-23', 'diffuse_solar_radiation+1', 'direct_solar_radiation+1',
        #   'relative_humidity+1', 'drybulb_temp+1']
        X_order = self.solar_model.feature_name()
        # make a vector of last values from prev steps using keys from X_order
        X = np.zeros(len(X_order))
        # print(self.time_step)
        for i, key in enumerate(X_order):
            if key.startswith("diffuse_solar_radiation+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "diffuse_solar_irradiance_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = np.log1p(last_val)
            elif key.startswith("direct_solar_radiation+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "direct_solar_irradiance_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = np.log1p(last_val)
            elif key.startswith("relative_humidity+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "outdoor_relative_humidity_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = last_val
            elif key.startswith("drybulb_temp+"):
                step_back = -(25 - step)
                if self.time_step > 23:
                    last_val = self.prev_steps[
                        "outdoor_dry_bulb_temperature_predicted_24h"
                    ][step_back]
                else:
                    last_val = np.nan
                X[i] = last_val
            elif key in self.prev_steps.keys():
                X[i] = self.prev_steps[key][-1]
        forec = self.solar_model.predict(X.reshape(1, -1))
        forec = forec[0] * self.pv_capacity[id] * 0.001
        if self.time_step < 23:
            forec = 0
        return forec

    def get_net_forecast_step(self, step: int, id: int, last_param=False):
        # if self.time_step > 23:
        #     print('here')
        cons = self.get_cons_forecast_step(step, id, last_param)
        pv = self.get_solar_forecast_step(step, id)
        net = (cons - pv) 
        return net