import numpy as np
import joblib
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm, multivariate_normal


class Forecast:
    def __init__(self, n_build, model_dir="models/lag_minus_1/", point_forecast=False):
        self.num_buildings = n_build
        self.pv_capacity = [4.0, 4.0, 4.0, 5.0, 4.0]
        self.quantiles = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
        self.quantiles = np.round(self.quantiles, 3)
        self.prev_steps = {}
        self.model_dict = {}
        self.renamer = {"Month": "month", "Hour": "hour"}

        # start dicts for min and max values for each building
        self.cons_min_dict = {}
        self.cons_max_dict = {}
        self.gen_min_dict = {}
        self.gen_max_dict = {}
        self.net_min_dict = {}
        self.net_max_dict = {}
        self.diffuse_solar_init = {}
        self.drybulb_temp_init = {}
        self.relative_humidity_init = {}
        self.diffuse_solar_pred = {}
        self.X_old = np.zeros((8760, self.num_buildings, 17))
        self.y_old = np.zeros((8760, self.num_buildings, 1))
        self.init_forecast()
        if point_forecast:
            self.model_pt = joblib.load(model_dir+"lgb_point_step_24.pkl")
            #self.model_pt_next = joblib.load(model_dir+"lgb_next_step_1.pkl")
            self.model_pt_next = lgb.Booster(model_file=model_dir+"lgb_next_step_1.txt")
        else:
            for qt in self.quantiles:
                # read an lgb model in a txt file
                self.model_dict[qt] = lgb.Booster(model_file=model_dir+"lgb_{}.txt".format(qt))

    def init_forecast(self):
        # look-up varinance for each hour and month and save it to a dict
        # save the pandas dataframe with hour in index and month in columnds to a dictionary 
        self.variance_dict = pd.read_csv("data/variance_hour_month.csv", index_col=0).T
        self.variance_dict = self.variance_dict.to_dict(orient="dict")
        # print("variance dict: ", self.variance)


        # make a conservative estimate for max and min consumption
        self.cons_min = 0
        self.cons_max = 5
        # make a conservative esimate for max and min generation
        self.gen_min = 0
        self.gen_max = 700
        # make a conservative estimate for max and min net consumption
        self.net_min = -3
        self.net_max = 4.6
        # make a conservative estimate for aggregate max and min consumption
        self.cons_agg_min = 0.850
        self.cons_agg_max = 20.447
        # make a conservative esimate for aggregate max and min generation
        self.gen_agg_min = 0
        self.gen_agg_max = 4018.496
        # make a conservative estimate for aggregate max and min net consumption
        self.net_agg_min = -14.754
        self.net_agg_max = 16.718

        # initialize dicts with conservative estimates
        for i in range(self.num_buildings):
            self.cons_min_dict[i] = self.cons_min
            self.cons_max_dict[i] = self.cons_max
            self.gen_min_dict[i] = self.gen_min
            self.gen_max_dict[i] = self.gen_max
            self.net_min_dict[i] = self.net_min
            self.net_max_dict[i] = self.net_max

        self.net_min_dict[0] = -3.5817333333333323
        self.net_min_dict[1] = -2.8163
        self.net_min_dict[2] = -3.1906499999999998
        self.net_min_dict[3] = -3.0923333414713543
        self.net_min_dict[4] = -3.03125
        self.net_max_dict[0] = 7.98043331705729
        self.net_max_dict[1] = 5.847749999999999
        self.net_max_dict[2] = 4.606916666666666
        self.net_max_dict[3] = 4.86045
        self.net_max_dict[4] = 4.938766666666667
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
        self.gen_min_dict[0] = 0
        self.gen_min_dict[1] = 0
        self.gen_min_dict[2] = 0
        self.gen_min_dict[3] = 0
        self.gen_min_dict[4] = 0
        self.gen_max_dict[0] = 976.25
        self.gen_max_dict[1] = 786.0833333333334
        self.gen_max_dict[2] = 843.7125
        self.gen_max_dict[3] = 706.0366673787435
        self.gen_max_dict[4] = 818.2875
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
        last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] - (
            self.prev_steps[f"solar_generation_{id}"][-1]
        )
        self.net_min_dict[id] = min(self.net_min_dict[id], last_val)
        self.net_max_dict[id] = max(self.net_max_dict[id], last_val)


    def forecast_next_step_for_B(self, id: int, step, last_param=False):
        # ['Month', 'Hour', 'hour_x', 'hour_y', 'month_x', 'month_y',
        #  'net_target-1', 'diffuse_solar_radiation+1', 'direct_solar_radiation+1',
        #   'relative_humidity+1', 'drybulb_temp+1']
        columnames = self.model_dict[0.001].feature_name()
        # rename items in the list according to a dict
        X_order = [self.renamer.get(x, x) for x in columnames]
        # make a vector of last values from prev steps using keys from X_order
        X = np.zeros(len(X_order))
        # create a vector of 5 for forecast
        forec = np.zeros(len(self.quantiles))
        forec_denorm = np.zeros(len(self.quantiles))
        for qt_cnt, qt in enumerate(self.quantiles):
            for i, key in enumerate(X_order):
                # if key starts with net_target
                if key == "net_target":
                    if last_param == False:
                        last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] - (
                            self.prev_steps[f"solar_generation_{id}"][-1]
                        )
                        norm_val = self.min_max_normalize(
                            last_val, self.net_min_dict[id], self.net_max_dict[id]
                        )
                    else:
                        norm_val = self.min_max_normalize(
                            last_param, self.net_min_dict[id], self.net_max_dict[id]
                        )
                    X[i] = norm_val
                elif key == "net_target-23":
                    lag_step = 24 - step
                    if self.time_step > lag_step:
                        lag_step = -lag_step - 1
                        last_val = self.prev_steps[f"non_shiftable_load_{id}"][
                            lag_step
                        ] - (self.prev_steps[f"solar_generation_{id}"][lag_step])
                        norm_val = self.min_max_normalize(
                            last_val, self.net_min_dict[id], self.net_max_dict[id]
                        )
                    else:
                        norm_val = np.nan
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
            # add a value to a prediction vector
            forec[qt_cnt] = self.model_dict[qt].predict(X.reshape(1, -1))
            # denormalize the values
            forec_denorm[qt_cnt] = self.min_max_denormalize(
                forec[qt_cnt], self.net_min_dict[id], self.net_max_dict[id]
            )
        # sort quantile values in ascending order to prevent quantile crossings
        forec_denorm = np.sort(forec_denorm)
        """
        if self.time_step > 23:
            if step == 1:
                print(self.time_step)
                print(id)
                print('normalized:')
                print(forec)
                print('denormalized:')
                print(forec_denorm)
        """
        #forec = [i[0] for i in forec]
        return forec_denorm


    def get_point_forecast_step(self, step: int, id: int):
            # ['Month', 'Hour', 'hour_x', 'hour_y', 'month_x', 'month_y',
            #  'net_target-23', 'diffuse_solar_radiation+1', 'direct_solar_radiation+1',
            #   'relative_humidity+1', 'drybulb_temp+1']
            if step == 1:
                columnames = self.model_pt_next.feature_name()
            else:
                columnames = self.model_pt.feature_name()
            # rename items in the list according to a dict
            X_order = [self.renamer.get(x, x) for x in columnames]
            # make a vector of last values from prev steps using keys from X_order
            X = np.zeros(len(X_order))
            # print(self.time_step)
            for i, key in enumerate(X_order):
                # if key starts with net_target
                if key == "net_target-23":
                    lag_step = 24 - step
                    if self.time_step > lag_step:
                        lag_step = -lag_step - 1
                        lag_val = self.prev_steps[f"non_shiftable_load_{id}"][
                            lag_step
                        ] - (self.prev_steps[f"solar_generation_{id}"][lag_step])
                        norm_val = self.min_max_normalize(
                            lag_val, self.net_min_dict[id], self.net_max_dict[id]
                        )
                    else:
                        norm_val = np.nan
                    X[i] = norm_val
                elif key == "net_target_diff":
                    if self.time_step > 2:
                        last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] - (
                            self.prev_steps[f"solar_generation_{id}"][-1]
                        ) 
                        last_last_val = self.prev_steps[f"non_shiftable_load_{id}"][-2] - (
                            self.prev_steps[f"solar_generation_{id}"][-2]
                        )
                        norm_val = self.min_max_normalize(
                            last_val, self.net_min_dict[id], self.net_max_dict[id]
                        )
                        norm_last_val = self.min_max_normalize(
                            last_last_val, self.net_min_dict[id], self.net_max_dict[id]
                        )
                        X[i] = norm_val - norm_last_val
                    else:
                        X[i] = np.nan
                elif key == "net_target":
                    last_val = self.prev_steps[f"non_shiftable_load_{id}"][-1] - (
                        self.prev_steps[f"solar_generation_{id}"][-1]
                    )
                    norm_val = self.min_max_normalize(
                        last_val, self.net_min_dict[id], self.net_max_dict[id]
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
                    X[i] = (self.prev_steps["hour"][-1] + step - 1) % 24
                elif key == "hour_x":
                    X[i] = np.cos(
                        2 * np.pi * ((self.prev_steps["hour"][-1] + step -1) % 24) / 24
                    )
                elif key == "hour_y":
                    X[i] = np.sin(
                        2 * np.pi * ((self.prev_steps["hour"][-1] + step -1) % 24) / 24
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
            #forec = self.model_pt.predict(X.reshape(1, -1))
            # denormalize the values
            forec = self.min_max_denormalize(
                forec, self.net_min_dict[id], self.net_max_dict[id]
            )
            forec = forec[0]
            #print(forec)
            return forec

    def get_point_and_variance(self, step: int, id: int):
        forec = self.get_point_forecast_step(step, id)
        # read the gmm model from models/gmm/ folder with joblib
        gmm = joblib.load(f"models/gmm/gmm_residual_hour_{id}.joblib")
        # sample from the gmm model
        var = gmm.sample(1)[0][0]
        # look-up the variance in a variance dict
        # var = self.variance_dict[self.prev_steps["month"][0]][str(self.prev_steps["hour"][0]-1)]
        # min max normalize the variance
        var = self.min_max_normalize(var, self.net_min_dict[id], self.net_max_dict[id])
        return forec, var
    