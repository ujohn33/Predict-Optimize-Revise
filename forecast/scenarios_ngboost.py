import sys
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import norm, multivariate_normal
from forecast.forecast_ngboost import Forecast


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def swap_levels(lst):
    return [[sublist[i] for sublist in lst] for i in range(len(lst[0]))]


class Scenario_Generator:
    def __init__(self, n_scenarios=10, n_buildings=5, steps_ahead=24):
        self.n_scenarios = n_scenarios
        self.n_buildings = n_buildings
        self.steps_ahead = steps_ahead
        self.debugger_is_active = debugger_is_active()
        self.model_direct24 = Forecast(n_buildings, model_dir="models/ngboost/")
        
    def generate_scenarios(self, prev_steps, current_step):
        self.model_direct24.update_prev_steps(prev_steps)
        self.model_direct24.update_current_step(current_step)
        points, dists = self.model_direct24.lgb_memo_data_and_predict()
        scenarios = []
        # dists has shape (n_buildings, steps_ahead)
        for b in range(self.n_buildings):
            # get mean and std of the distribution for 24 steps ahead
            mean = [dist.params['loc'][0] for dist in dists[b]]            
            std = [dist.params['scale'][0] for dist in dists[b]]   
            # generate n_scenarios samples from the distribution
            samples = np.random.normal(mean, std, size=(self.n_scenarios, self.steps_ahead))
            scenarios.append(samples)
            if self.debugger_is_active:
                for scen in scenarios:
                    for i in range(len(scen)):
                        if self.debugger_is_active:
                            plt.title("")
                            plt.plot(scen[i])
                    plt.show()
            if self.debugger_is_active:
                # clean up
                plt.clf()
        scenarios = swap_levels(scenarios)
        return scenarios
