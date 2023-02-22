import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast.forecast_functions import Forecast
import random

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

class Scenario_Generator:
    def __init__(self, type='norm', n_scenarios=10, n_buildings=5, steps_ahead=24):
        self.type = type
        self.n_scenarios = n_scenarios
        self.n_buildings = n_buildings
        self.steps_ahead = steps_ahead
        self.debugger_is_active = debugger_is_active()
        if type == 'recurrent_gaussian_qts':
            self.qts_model = Forecast(n_buildings, model_dir='models/lag_minus_1/')
        elif type == 'quantiles':
            self.qts_model = Forecast(n_buildings, model_dir='models/lag_minus_24/')
        elif type == 'point':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
        elif type == 'point_and_variance':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)    
        self.scenarios = []
        self.base_quantiles = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
        # round to 3 decimals
        self.base_quantiles = np.round(self.base_quantiles, 3)
        self.logger = None
    
    def sample_from_quantiles(self, quantile_val, quantile_bounds):
        sample_point = random.random()
        for i, quant in enumerate(quantile_bounds):
            if sample_point<quant:
                if i == 0:
                    return quantile_val[0]
                else:
                    diff_val = quantile_val[i]-quantile_val[i-1]
                    diff_quant = quantile_bounds[i]-quantile_bounds[i-1]
                    diff_sample = sample_point-quantile_bounds[i-1]
                    value_point = quantile_val[i-1]+diff_sample*diff_val/diff_quant
                    return value_point
    
        return quantile_val[-1]

    def generate_next_step(self, id_param, last_param = False):
        quantile_bounds = self.qts_model.quantiles
        if last_param == False:
            quantile_values = self.qts_model.forecast_next_step_for_B(id_param)
            if self.logger is not None:
                time_step = self.qts_model.time_step
                build_num = id_param
                self.logger.log_quantiles(quantile_bounds, quantile_values, time_step, build_num)
        else:
            quantile_values = self.qts_model.forecast_next_step_for_B(id_param, last_param)
            
        sample_temp = self.sample_from_quantiles(quantile_values, quantile_bounds)
        return sample_temp

    def swap_levels(self, lst):
        return [[sublist[i] for sublist in lst] for i in range(len(lst[0]))]

    def sample_quantiles(self, num_quantiles):
        start = 0.05
        end = 0.95
        quantile_samples = np.linspace(start, end, num=num_quantiles)
        return quantile_samples

    def generate_scenarios(self, prev_steps, current_step):
        horizon = self.steps_ahead
        scenarios = []
        for b in range(self.n_buildings):
            scens_B_temp = self.generate_scenarios_for_B(self.type, b, prev_steps, current_step, horizon)
            scenarios.append(scens_B_temp)
            # plot a list of lists with the same length and range on the x-axis
            for scen in scenarios:
                for i in range(len(scen)):
                    if self.debugger_is_active:
                        plt.title("")
                        plt.plot(scen[i])
                        #print(scen[i])
        if self.debugger_is_active:
            plt.show()
        scenarios = self.swap_levels(scenarios)
        return scenarios

    def generate_scenarios_for_B(self, type, id_param, prev_steps, current_step, horizon=24):
        scenarios_B = []
        if type == 'recurrent_gaussian_qts':
            for i in range(self.n_scenarios):
                scenarios_B.append(self.recurrent_gaussian(prev_steps, current_step, id_param, horizon))
        elif type == 'quantiles':
            scenarios_B = self.quantiles(prev_steps, current_step, id_param, horizon)
        elif type == 'point':
            scenarios_B = [self.point_forecast(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)]
        elif type == 'point_and_variance':
            sceni, vari = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
            for i in range(self.n_scenarios):
                scenarios_B.append(sceni + np.random.normal(0, vari, horizon))
        return scenarios_B


    def recurrent_gaussian(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        sample = False
        for i in range(horizon):
            sample = self.generate_next_step(id_param= id_param, last_param = sample)
            scenario[i] = sample
        return scenario

    def quantiles(self, prev_steps, current_step, id_param, horizon=24):
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        if self.n_scenarios <= 20:
            self.qts_model.update_prev_steps(prev_steps)
            # if number of scenarios is even, take equally spaced quantiles from the list of base quantiles
            quantiles = self.sample_quantiles(self.n_scenarios)
            # round to 3 decimals
            quantiles = np.round(quantiles, 3)
            qts_final = np.zeros((horizon, self.n_scenarios))
            # find the indexes of the quantiles in the list of base quantiles
            quantile_indexes = np.where(np.isin(self.base_quantiles, quantiles))[0]
            for i_step in range(horizon):
                qts_temp = self.qts_model.forecast_next_step_for_B(id_param, step=i_step+1)
                qts_final[i_step, :] = qts_temp[quantile_indexes]
        else:
            print('Number of scenarios should be less than 20')
        return qts_final

    def point_forecast(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        for i in range(horizon):
            scenario[i] = self.qts_model.get_point_forecast_step(step=i+1, id=id_param)
        #self.plot_scenario(scenario)
        return list(scenario)

    def point_and_variance(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        variances = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        for i in range(horizon):
            scenario[i], variances[i]  = self.qts_model.get_point_and_variance(step=i+1, id=id_param)
        # add uncertainty to the point forecast
        return scenario, variances

    def plot_scenario(self, scenario: list):
        plt.plot(range(len(scenario)), scenario)
        #plt.show()