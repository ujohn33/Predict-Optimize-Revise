import numpy as np
import time
import pandas as pd
import itertools
from agents.general_agent import GeneralAgent
from ems.logger_manager import LoggerManager
from ems.mpc import MPC
from forecast.scenarios_lean import Scenario_Generator
from forecast.file import PerfectFile, RealForecast, ScenarioFile, ScenarioFileAndNaive
from utils.logger import log_usefull
from ems.pyo_mpc import PyoMPC
from ems.gurobi_mpc import GurobiMPC
from ems.gurobi_matrix_mpc import GurobiMatrixMPC
from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv
from local_evaluation import action_space_to_dict, env_reset, evaluate

case_study = "logging"
phase_num = 2
total_steps = 400
if phase_num == 3:
    n_buildings = 7
else:
    n_buildings = 5
scen_file = "scenarios_together+naive_9000_2.csv"

def run_ensemble(combo: list):
    n_scen = len(combo)
    combo_id = ''.join(str(e) for e in combo)
    file_name = f"debug_logs/ensembles/{combo_id}.csv"
    # file_name = f"debug_logs/scenarios_point_and_variance_9000_3.csv"
    scenario_gen = ScenarioFile(file_name, n_scenarios=n_scen)
    manager = GurobiMPC(0)
    agent_used = GeneralAgent(scenario_gen, manager)
    tc, apc, aec, agc = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num)
    return tc, apc, aec, agc

def make_ensembles(file_name):
    # open csv file for scenarios
    scens = pd.read_csv('debug_logs/'+file_name)
    # keep first 400 rows
    scens = scens.loc[scens['time_step'] <= 400]
    scens = scens.set_index(['time_step', 'building'], drop=True)
    # take unique scenario numbers if scens and get a list of lists with unique combinations of 5 that exist
    scens_unique = scens['scenario'].unique()
    scens_unique = scens_unique.tolist()
    # get combinations of 5
    scens_combinations = list(itertools.combinations(scens_unique, 5))
    for combo in scens_combinations:
        combo_id = ''.join(str(e) for e in combo)
        temp_scens = scens[scens['scenario'].isin(combo)]
        # save temp_scens to csv into debug_logs/ensembles/
        temp_scens.to_csv(f"debug_logs/ensembles/{combo_id}.csv")


# if main
if __name__ == "__main__":
    # make ensembles and save to csv
    #make_ensembles(scen_file)
    scens = pd.read_csv('debug_logs/'+scen_file)
    scens_unique = scens['scenario'].unique()
    # metrics as columns
    index = ['unique_id']
    line = ['total_cost', 'price_cost', 'emission_cost', 'grid_cost']
    scens_unique = scens_unique.tolist()
    # get combinations of 5
    scens_combinations = list(itertools.combinations(scens_unique, 5))
    metric_file = open('opt_metrics_200_fix.csv', 'a')
    #metric_file.write(','.join(index + line) + '\n')
    for combo in scens_combinations[223:]:
        total_cost, price_cost, emission_cost, grid_cost = run_ensemble(combo)
        line_start = ''.join(str(e) for e in combo)
        metric_file.write(','.join([line_start, str(total_cost), str(price_cost), str(emission_cost), str(grid_cost)]) + '\n')
        metric_file.flush()
    metric_file.close()

