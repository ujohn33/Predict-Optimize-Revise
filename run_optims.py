import numpy as np
import time
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
total_steps = 9000

def run_ensemble(combo: list):
    n_scen = len(combo)
    file_name = f"debug_logs/scenarios_recurrent_quant_s10_p{phase_num}_24h.csv"
    # file_name = f"debug_logs/scenarios_point_and_variance_9000_3.csv"
    scenario_gen = ScenarioFile(file_name, n_scenarios=n_scen)
    manager = GurobiMPC(0)
    agent_used = GeneralAgent(scenario_gen, manager)
    evaluate(agent_used, total_steps=total_steps, phase_num=phase_num)