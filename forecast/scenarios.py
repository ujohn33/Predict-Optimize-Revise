import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast.forecast_functions import Forecast
from forecast.metrics import crps
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
        elif type == 'point_recurrent':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            self.qts_model.model_pt = joblib.load('models/point/lgb_next_step_diff_12march.pkl')
        elif type == 'full_covariance':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            # read cov matrix from pickle file
            self.cov_matrix = joblib.load('data/residuals_corr/cov_hour.pkl')
        elif type == 'point_and_variance':
            self.variance_dict = pd.read_csv("data/variance_hour_month.csv", index_col=0).T
            self.variance_dict = self.variance_dict.to_dict(orient="dict")
            self.gmm_dict = {}
            for hour in range(24):
                self.gmm_dict[hour] = joblib.load(f"models/gmm/gmm_residual_hour_{hour}.joblib")
            # intialize the point forecast
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            self.qts_model.model_pt = joblib.load('models/point/lgb_next_step_diff_12march.pkl') 
        elif type == 'point_and_variance_gmm':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            self.qts_model.model_pt = joblib.load('models/point/lgb_next_step_diff_12march.pkl')    
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

    def generate_next_step(self, id_param, p_step, last_param = False):
        quantile_bounds = self.qts_model.quantiles
        if last_param == False:
            quantile_values = self.qts_model.forecast_next_step_for_B(id_param, p_step)
            if self.logger is not None:
                time_step = self.qts_model.time_step
                build_num = id_param
                self.logger.log_quantiles(quantile_bounds, quantile_values, time_step, build_num)
        else:
            quantile_values = self.qts_model.forecast_next_step_for_B(id_param, p_step, last_param)
            
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
            scenarios_B = self.swap_levels(scenarios_B)
        elif type == 'point':
            scenarios_B = [self.point_forecast(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)]
        elif type == 'point_recurrent':
            scenarios_B = [self.point_recurrent_forecast(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)]
        elif type == 'full_covariance':
            scenarios_B = self.full_covariance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'point_and_variance':
            scenarios_B = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
            # for i in range(self.n_scenarios):
            #     if self.logger is not None:
            #         quantile_bounds = self.base_quantiles
            #         quantile_values = norm.ppf(quantile_bounds,sceni[0], vari[0])
            #         time_step = self.qts_model.time_step
            #         build_num = id_param
            #         self.logger.log_quantiles(quantile_bounds, quantile_values, time_step, build_num)
        elif type == 'point_and_variance_gmm':
            sceni, vari = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        return scenarios_B


    def recurrent_gaussian(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        sample = False
        for i in range(horizon):
            sample = self.generate_next_step(id_param= id_param, p_step=i+1, last_param = sample)
            scenario[i] = sample
        return scenario

    def quantiles(self, prev_steps, current_step, id_param, horizon=24):
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        if self.n_scenarios < 21:
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
        elif self.n_scenarios == 21:
            qts_final = np.zeros((horizon, self.n_scenarios))
            for i_step in range(horizon):
                qts_temp = self.qts_model.forecast_next_step_for_B(id_param, step=i_step+1)
                qts_final[i_step, :] = qts_temp
        else:
            print('Number of scenarios should be less than 20')
        return qts_final

    def full_covariance(self, prev_steps, current_step, id_param, horizon=24):
        point_scen = self.point_recurrent_forecast(prev_steps, current_step, id_param, horizon)
        cov_matrix = self.cov_matrix[prev_steps['hour'][-1] % 24]
        rv_mvnorm = multivariate_normal([0]*24, cov_matrix)
        samples = rv_mvnorm.rvs(self.n_scenarios)
        # denormalize samples
        samples = samples * (self.qts_model.net_max_dict[id_param] - self.qts_model.net_min_dict[id_param]) 
        scenarios = []
        for i in range(self.n_scenarios):
            scenarios.append(point_scen + samples[i])
        return scenarios
    
    def full_covariance_bymonth(self, prev_steps, current_step, id_param, horizon=24):
        point_scen = self.point_recurrent_forecast(prev_steps, current_step, id_param, horizon)
        cov_matrix = self.cov_matrix[prev_steps['hour'][-1] % 24][prev_steps['month'][-1]]
        rv_mvnorm = multivariate_normal([0]*24, cov_matrix)
        samples = rv_mvnorm.rvs(self.n_scenarios)
        scenarios = []
        for i in range(self.n_scenarios):
            scenarios.append(point_scen + samples[i])
        return scenarios

    def point_forecast(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        sample = False
        for i in range(horizon):
           scenario[i] = self.qts_model.get_point_forecast_step(step=i+1, id=id_param)
        #self.plot_scenario(scenario)
        return list(scenario)
    
    def point_recurrent_forecast(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        sample = False
        for i in range(horizon):
            sample = self.qts_model.get_point_forecast_step(step=i+1, id=id_param, last_param = sample)
            scenario[i] = sample
        #self.plot_scenario(scenario)
        return list(scenario)

    def point_and_variance(self, prev_steps, current_step, id_param, horizon=24, dist_type='gmm'):
        scenario_B = [0 * 10] * 24
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        sample = False
        for i in range(horizon):
            step_temp = i+1
            sample = self.qts_model.get_point_forecast_step(step=step_temp, id=id_param, last_param = sample)
            hour = (prev_steps["hour"][-1] + step_temp) % 24
            if dist_type == 'gmm':
                dist = self.gmm_dict[hour]
            elif dist_type == 'norm':
                # gaussian with a variance looked up in a variance dict
                dist = norm(loc=0, scale=self.variance_dict[str(self.prev_steps["month"][-1])][hour])
            # sample from the distribution with n_scen scenarios
            resids = dist.sample(self.n_scenarios)[0]
            resids = np.array(np.array(resids).flatten())
            #min max denormalize the resids_list
            resids = resids * (self.qts_model.net_max_dict[id_param] - self.qts_model.net_min_dict[id_param]) 
            resids_list = resids.tolist()
            scenario_B[i] = sample + resids_list
        scenario_B = self.swap_levels(scenario_B)
        return scenario_B
    

    def plot_scenario(self, scenario: list):
        plt.plot(range(len(scenario)), scenario)
        #plt.show()

    