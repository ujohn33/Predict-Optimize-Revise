import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast.forecast_functions import Forecast
from forecast.file import PerfectFile, RealForecast, ScenarioFile
from forecast.metrics import crps
import random

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

class Scenario_Generator:
    def __init__(self,  forec_file: str, type='norm_noise', n_scenarios=10, n_buildings=5, steps_ahead=24):
        self.type = type
        self.n_scenarios = n_scenarios
        self.n_buildings = n_buildings
        self.steps_ahead = steps_ahead
        self.gen_min_dict = {}
        self.gen_max_dict = {}
        self.net_min_dict = {}
        self.net_max_dict = {}
        self.stds_hourly = {}
        # make a conservative estimate for max and min net consumption
        self.net_min = -3
        self.net_max = 4.6
        # initialize dicts with conservative estimates
        for i in range(n_buildings):
            self.net_min_dict[i] = self.net_min
            self.net_max_dict[i] = self.net_max
        self.debugger_is_active = debugger_is_active()
        self.forecast_gen = ScenarioFile(forec_file, n_scenarios=1)
        if type == 'full_covariance':
            # read multivariate normal distribution from pickle file
            self.mv_norm = joblib.load('models/residuals_norm/mvn_hour.pkl')
        elif type == 'norm_noise':
            # change model for steps 2 to 24 to recurrent next step model 
            self.variance_dict = joblib.load('models/residuals_norm/std_residuals_norm.pkl')
        elif type == 'norm_noise_online':
            # change model for steps 2 to 24 to recurrent next step model 
            self.variance_dict = joblib.load('models/residuals_norm/std_residuals_norm.pkl')
        self.scenarios = []
        self.logger = None 

    def swap_levels(self, lst):
        return [[sublist[i] for sublist in lst] for i in range(len(lst[0]))]
    
    def update_min_max_scaler(self, prev_steps, id):
        last_val = prev_steps[f"non_shiftable_load_{id}"][-1] - (
            prev_steps[f"solar_generation_{id}"][-1]
        )
        self.net_min_dict[id] = min(self.net_min_dict[id], last_val)
        self.net_max_dict[id] = max(self.net_max_dict[id], last_val)

    def generate_scenarios(self, prev_steps, current_step):
        horizon = self.steps_ahead
        scenarios = []
        for b in range(self.n_buildings):
            self.update_min_max_scaler(prev_steps, b)
            scens_B_temp = self.generate_scenarios_for_B(self.type, b, prev_steps, current_step, horizon)
            scenarios.append(scens_B_temp)
            #if current_step > 168:
                # plot a list of lists with the same length and range on the x-axis
                # for scen in scenarios:
                #     for i in range(len(scen)):
                #         if self.debugger_is_active:
                #             plt.title("")
                #             plt.plot(scen[i])
                #             plt.show()
                #             # clean up
                #             plt.close()
        # if self.debugger_is_active:
        #     plt.show()
        #     # clean up
        #     plt.clf()
        scenarios = self.swap_levels(scenarios)
        return scenarios

    def generate_scenarios_for_B(self, type, id_param, prev_steps, current_step, horizon=24):
        scenarios_B = []
        # if type == 'full_covariance':
        #     scenarios_B = self.full_covariance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        if type == 'norm_noise':
            scenarios_B = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon, dist_type='norm')
        elif type == 'gmm_noise':
            scenarios_B = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'norm_noise_online':
            scenarios_B = self.point_and_variance_online(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon, dist_type='norm')
        return scenarios_B

    def point_and_variance(self, prev_steps, current_step, id_param, horizon=24, dist_type='norm'):
        scenario_B = [0] * horizon
        current_hour = prev_steps['hour'][-1] % 24
        base = self.forecast_gen.scen_dict[0][id_param][current_step][:horizon].tolist()
        for i in range(horizon):
            lead_hour = (current_hour + i + 1) % 24
            if dist_type == 'norm':
                # get the variance of the error for the current hour
                std = self.variance_dict[current_hour][lead_hour]
                # set a normal distribution with mean = 0 and variance = variance
                dist = norm(loc=0, scale=std)
                # sample from a normal distribution with variance = variance
                resids = dist.rvs(self.n_scenarios)
            else:
                raise ValueError('dist_type must be either norm or gmm')
            resids = np.array(np.array(resids).flatten())
            resids = resids * (self.net_max_dict[id_param] - self.net_min_dict[id_param]) 
            scenario_B[i] = (base[i] + resids).tolist()
        scenario_B = self.swap_levels(scenario_B)
        return scenario_B
    
    def point_and_variance_online(self, prev_steps, current_step, id_param, horizon=24, dist_type='norm'):
        scenario_B = [0] * horizon
        current_hour = prev_steps['hour'][-1] % 24
        base = self.forecast_gen.scen_dict[0][id_param][current_step][:horizon].tolist()
        if current_step > 168:
            means_last_week, stds_last_week = self.update_variance_last_week(prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
            for h in range(horizon):
                if dist_type == 'norm':
                    # get the variance of the error for the current hour
                    std = stds_last_week[h]
                    mean = means_last_week[h]
                    # set a normal distribution with mean = 0 and variance = variance   
                    dist = norm(loc=mean, scale=std)
                    # sample from a normal distribution with variance = variance
                    resids = dist.rvs(self.n_scenarios)
                else:
                    raise ValueError('dist_type must be either norm or gmm')
                resids = np.array(np.array(resids).flatten())
                # if self.debugger_is_active:
                #     plt.plot(resids)
                    #print(resids)
                scenario_B[h] = (base[h] + resids).tolist()
        else:
            for i in range(horizon):
                lead_hour = (current_hour + i + 1) % 24
                if dist_type == 'norm':
                    # get the variance of the error for the current hour
                    std = self.variance_dict[current_hour][lead_hour]
                    # set a normal distribution with mean = 0 and variance = variance
                    dist = norm(loc=0, scale=std)
                    # sample from a normal distribution with variance = variance
                    resids = dist.rvs(self.n_scenarios)
                else:
                    raise ValueError('dist_type must be either norm or gmm')
                resids = np.array(np.array(resids).flatten())
                resids = resids * (self.net_max_dict[id_param] - self.net_min_dict[id_param]) 
                scenario_B[i] = (base[i] + resids).tolist()
        # if self.debugger_is_active:
        #     if current_step > 168:
        #         temp_scen = np.array(np.array(scenario_B))
        #         temp_base = np.array(base)
        #         plot_noise = temp_scen - temp_base[:,np.newaxis]
        #         plt.plot(plot_noise)
        scenario_B = self.swap_levels(scenario_B)
        return scenario_B


    # make a function that finds the variance of error for each hour of the day
    def update_variance_last_week(self, prev_steps, current_step, id_param, horizon=24):
        #current_hour = prev_steps['hour'][-1] % 24
        # sample_preds is a dict of dicts
        sample_preds = {}
        sample_actual = {}
        resid_stds = {}
        resid_means = {}
        # actuals sample at certain hour
        #sample_actual = prev_steps[f'non_shiftable_load_{id_param}'][-168::-24]
        for h in range(horizon):
            # forecasts sample at certain hour for certain horizon
            selected_steps = np.array([])
            for t in range(current_step, current_step-168, -24):
                selected_steps = np.append(selected_steps, self.forecast_gen.scen_dict[0][id_param][t][h])
            sample_preds[h] = selected_steps
            load = np.array(prev_steps[f'non_shiftable_load_{id_param}'][-24+h:-169+h:-24])
            solar = np.array(prev_steps[f'solar_generation_{id_param}'][-24+h:-169+h:-24])
            sample_actual[h] = load - solar
            # get the residuals std for each horizon
            resid_stds[h] = np.std(sample_actual[h] - sample_preds[h])
            resid_means[h] = np.mean(sample_actual[h] - sample_preds[h])
        if self.debugger_is_active:
            # plot sample preds and actuals
            plt.plot(sample_preds.keys(), sample_preds.values(), label='preds', linestyle='--')
            plt.plot(sample_actual.keys(), sample_actual.values(), label='actuals')
            # plt.plot(resid_stds.keys(), resid_stds.values(), label='stds')
            # plt.plot(resid_means.keys(), resid_means.values(), label='means')
            plt.legend()
            plt.show()
            # close canvas
            plt.close()
        return resid_means, resid_stds