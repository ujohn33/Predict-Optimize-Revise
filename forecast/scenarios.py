import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast.forecast_functions import Forecast

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

class Scenario_Generator:
    def __init__(self, type='norm', n_scenarios=10, n_buildings=5):
        self.type = type
        self.n_scenarios = n_scenarios
        self.n_buildings = n_buildings
        self.debugger_is_active = debugger_is_active()
        if type == 'recurrent_gaussian_qts':
            self.qts_model = Forecast(5, model_dir='models/lag_minus_1/')
        elif type == 'quantiles':
            self.qts_model = Forecast(5, model_dir='models/lag_minus_24/')
        elif type == 'point':
            self.qts_model = Forecast(5, model_dir='models/point/', point_forecast=True)
        self.scenarios = []
        self.base_quantiles = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])

    def estimate_pdf(self, qts_temp):
        mu, std = norm.fit(qts_temp)
        pdf = norm(mu, std)
        return pdf

    def generate_next_step(self, id_param, last_param = False):
        if last_param == False:
            pdf = self.estimate_pdf(self.qts_model.forecast_next_step_for_B(id_param))
        else:
            pdf = self.estimate_pdf(self.qts_model.forecast_next_step_for_B(id_param, last_param))
        sample_temp = pdf.rvs(1)
        return sample_temp

    def swap_levels(self, lst):
        return [[sublist[i] for sublist in lst] for i in range(len(lst[0]))]

    def generate_scenarios(self, prev_steps, current_step, horizon=24):
        scenarios = []
        for i in range(self.n_buildings):
            scens_B_temp = self.generate_scenarios_for_B(self.type, i, prev_steps, current_step, horizon)
            scenarios.append(scens_B_temp)
            # plot a list of lists with the same length and range on the x-axis
            for scen in scenarios:
                for i in range(len(scen)):
                    if self.debugger_is_active:
                        plt.plot(scen[i])
                if self.debugger_is_active:
                    plt.show()
        scenarios = self.swap_levels(scenarios)
        return scenarios

    def generate_scenarios_for_B(self, type, id_param, prev_steps, current_step, horizon=24):
        scenarios_B = []
        if type == 'recurrent_gaussian_qts':
            for i in range(self.n_scenarios):
                scenarios_B.append(self.recurrent_gaussian(prev_steps, current_step, id_param, horizon))
                print('Scenario {} generated'.format(i+1))
            print('All scenarios generated')
        elif type == 'quantiles':
            scenarios_B = self.quantiles()
        elif type == 'point':
            scenarios_B = [self.point_forecast(prev_steps=prev_steps, current_step=current_step, id_param=id_param)]
        return scenarios_B


    def recurrent_gaussian(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        sample = False
        for i in range(1, horizon):
            sample = self.generate_next_step(id_param= id_param, last_param = sample)
            scenario[i] = sample
        return scenario

    def quantiles(self, prev_steps, id_param, horizon=24):
        if self.n_scenarios <= 20:
            self.qts_model.update_prev_steps(prev_steps)
            # if number of scenarios is even, take equally spaced quantiles from the list of base quantiles
            if self.n_scenarios % 2 == 0:
                quantiles = self.base_quantiles[::int(len(self.base_quantiles)/self.n_scenarios)]
            else:
                # if number of scenarios is odd, take the 0.5 quantile and even number of quantiles from both sides of the sorted list of base quantiles
                quantiles = self.base_quantiles[::int(len(self.base_quantiles)/(self.n_scenarios-1))]
                quantiles = np.concatenate([[0.5], quantiles])
            qts_final = np.zeros((horizon, self.n_scenarios))
            # find the indexes of the quantiles in the list of base quantiles
            quantile_indexes = np.where(np.isin(self.base_quantiles, quantiles))[0]
            for i_step in range(horizon):
                qts_temp = self.qts_model.forecast_next_step_for_B(id_param, step=i_step)
                qts_final[i_step, :] = qts_temp[quantile_indexes]
        else:
            print('Number of scenarios should be less than 20')
        return qts_final

    def point_forecast(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        for i in range(1, horizon):
            scenario[i] = self.qts_model.get_point_forecast_step(step=i, id=id_param)
        #self.plot_scenario(scenario)
        return list(scenario)

    def plot_scenario(self, scenario: list):
        plt.plot(range(len(scenario)), scenario)
        #plt.show()