import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
# import Forecast class from forecast-function.py
from forecast.forecast_functions_dc import Forecast
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
        if type == 'point_recurrent':
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
        elif type == 'full_covariance':
            # change model for steps 2 to 24 to recurrent next step model 
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            # read multivariate normal distribution from pickle file
            self.mv_norm = joblib.load('models/dc_cov/mvn_hour.pkl')
        elif type == 'full_covariance_monthly':
            # change model for steps 2 to 24 to recurrent next step model 
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            # read cov matrix from pickle file
            self.cov_matrix = joblib.load('models/dc_cov/cov_hour.pkl')
        elif type == 'full_covariance_differences':
            # change model for steps 2 to 24 to recurrent next step model 
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)
            # read multivariate normal distribution from pickle file
            self.mv_norm = joblib.load('models/differences_cov_matrix/mvn_hour.pkl')
        elif type == 'point_and_variance':
            # change model for steps 2 to 24 to recurrent next step model 
            self.variance_dict = pd.read_csv("data/cons_variance_hour_month.csv", index_col=0).T
            self.variance_dict = self.variance_dict.to_dict(orient="dict")
            self.gmm_dict = {}
            for hour in range(24):
                self.gmm_dict[hour] = joblib.load(f"models/gmm_cons/gmm_residual_hour_{hour}.joblib")
            # intialize the point forecast
            self.qts_model = Forecast(n_buildings, model_dir='models/point/', point_forecast=True)   
        self.scenarios = []
        self.base_quantiles = np.concatenate([[0.001],np.arange(0.05,0.951,0.05),[0.999]])
        # round to 3 decimals
        self.base_quantiles = np.round(self.base_quantiles, 3)
        self.logger = None

    def swap_levels(self, lst):
        return [[sublist[i] for sublist in lst] for i in range(len(lst[0]))]

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
                        plt.plot(scen[i], label=f"Scenario {i}")
                        #print(scen[i])
            if self.debugger_is_active:
                plt.show()
                # clean up the plot
                plt.clf()
                plt.close()

        scenarios = self.swap_levels(scenarios)
        return scenarios

    def generate_scenarios_for_B(self, type, id_param, prev_steps, current_step, horizon=24):
        scenarios_B = []
        if type == 'point_recurrent':
            scenarios_B = [self.point_recurrent_forecast(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)]
        elif type == 'full_covariance':
            scenarios_B = self.full_covariance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'full_covariance_monthly':
            scenarios_B = self.full_covariance_monthly(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'full_covariance_differences':
            scenarios_B = self.full_covariance_differences(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'point_and_variance':
            scenarios_B = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        elif type == 'point_and_variance_gmm':
            sceni, vari = self.point_and_variance(prev_steps=prev_steps, current_step=current_step, id_param=id_param, horizon=horizon)
        else:
            raise ValueError('Scenario type not implemented')
        return scenarios_B

    def full_covariance(self, prev_steps, current_step, id_param, horizon=24):
        point_scen = self.point_recurrent_forecast(prev_steps, current_step, id_param, horizon)
        current_hour = prev_steps['hour'][-1] % 24
        rv_mvnorm = self.mv_norm[current_hour]
        samples = rv_mvnorm.rvs(self.n_scenarios)
        # denormalize samples
        samples = samples * (self.qts_model.cons_max_dict[id_param] - self.qts_model.cons_min_dict[id_param]) 
        scenarios = []
        for i in range(self.n_scenarios):
            scenarios.append(point_scen + samples[i])
        return scenarios
    
    def full_covariance_monthly(self, prev_steps, current_step, id_param, horizon=24):
        point_scen = self.point_recurrent_forecast(prev_steps, current_step, id_param, horizon)
        cov_matrix = self.cov_matrix[prev_steps['hour'][-1] % 24][prev_steps['month'][-1]]
        rv_mvnorm = multivariate_normal([0]*24, cov_matrix)
        samples = rv_mvnorm.rvs(self.n_scenarios)
        # denormalize samples
        samples = samples * (self.qts_model.cons_max_dict[id_param] - self.qts_model.cons_min_dict[id_param]) 
        scenarios = []
        for i in range(self.n_scenarios):
            scenarios.append(point_scen + samples[i])
        return scenarios
    
    def full_covariance_differences(self, prev_steps, current_step, id_param, horizon=24):
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        start = prev_steps[f"non_shiftable_load_{id_param}"][-1] 
        current_hour = prev_steps['hour'][-1] % 24
        rv_mvnorm = self.mv_norm[current_hour]
        # sample scenarios of residuals from previous steps
        residuals = rv_mvnorm.rvs(self.n_scenarios)
        # denormalize samples
        residuals = residuals * (self.qts_model.cons_max_dict[id_param] - self.qts_model.cons_min_dict[id_param])       
        # Initialize the scenario matrix
        scenarios = np.zeros((self.n_scenarios, 24))
        scenarios[:, 0] = start
        # Generate scenarios for each time step
        for i in range(horizon):
            # Add the residuals to the previous value to generate a new scenario
            scenarios[:, i] = scenarios[:, i-1] + residuals[:, i-i]
        # Convert the scenario matrix to a list of lists
        scenario_list = scenarios.tolist()
        return scenario_list    
    
    def point_recurrent_forecast(self, prev_steps, current_step, id_param, horizon=24):
        scenario = np.zeros(horizon)
        self.qts_model.update_prev_steps(prev_steps)
        self.qts_model.update_current_step(current_step)
        self.qts_model.update_min_max_scaler(id_param)
        sample = False
        for i in range(horizon):
            sample = self.qts_model.get_net_forecast_step(step=i+1, id=id_param, last_param = sample)
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
            sample = self.qts_model.get_net_forecast_step(step=step_temp, id=id_param, last_param = sample)
            hour = (prev_steps["hour"][-1] + step_temp) % 24
            if dist_type == 'gmm':
                dist = self.gmm_dict[hour]
                # sample from the distribution with n_scen scenarios
                resids = dist.sample(self.n_scenarios)[0]
            elif dist_type == 'norm':
                # gaussian with a variance looked up in a variance dict
                dist = norm(loc=0, scale=self.variance_dict[prev_steps["month"][-1]][str(hour)])
                # sample from the distribution with n_scen scenarios
                resids = dist.rvs(self.n_scenarios)
            resids = np.array(np.array(resids).flatten())
            #min max denormalize the resids_list
            resids = resids * (self.qts_model.cons_max_dict[id_param] - self.qts_model.cons_min_dict[id_param]) 
            resids_list = resids.tolist()
            scenario_B[i] = sample + resids_list
        scenario_B = self.swap_levels(scenario_B)
        return scenario_B
    

    def plot_scenario(self, scenario: list):
        plt.plot(range(len(scenario)), scenario)
        #plt.show()

    