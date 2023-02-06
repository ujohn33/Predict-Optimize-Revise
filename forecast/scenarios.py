import numpy as np
import joblib
import pandas as pd
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast_functions import Forecast


class Scenario_Generator:
    def __init(self, type='norm', n_scenarios=10):
        self.type = type
        self.n_scenarios = n_scenarios
        self.qts_model = Forecast(5)
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

    def generate_scenarios(self, prev_steps, id_param, type, horizon=24):
        if type == 'recurrent_gaussian_qts':
            for i in range(self.n_scenarios):
                self.scenarios.append(self.recurrent_gaussian(prev_steps, id_param, horizon))
                print('Scenario {} generated'.format(i+1))
            print('All scenarios generated')
        elif type == 'quantiles':
            self.scenarios = self.quantiles()


    def recurrent_gaussian(self, prev_steps, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        sample = False
        for i in range(1, horizon):
            sample = self.generate_next_step(id = id_param, last_param = sample)
            scenario[i] = sample
        self.scenario = scenario

    def quantiles(self):
        # if number of scenarios is even, take equally spaced quantiles from the sorted list of base quantiles
        if self.n_scenarios % 2 == 0:
            quantiles = np.sort(self.base_quantiles)[::int(len(self.base_quantiles)/self.n_scenarios)]
        else:
            # if number of scenarios is odd, take the 0.5 quantile and even number of quantiles from both sides of the sorted list of base quantiles
            quantiles = np.sort(self.base_quantiles)[::int(len(self.base_quantiles)/(self.n_scenarios-1))]
            quantiles = np.concatenate([[0.5], quantiles])
        return quantiles

    def 