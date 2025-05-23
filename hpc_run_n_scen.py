from agents.general_agent import GeneralAgent
from forecast.scenarios_lean import Scenario_Generator
from ems.gurobi_mpc import GurobiMPC
from forecast.file import ScenarioFile_sliding
from local_evaluation import evaluate
import sys


def hpc_evaluate(phase_num, n_scen, steps_skip, steps_skip_forecast):
    
    phase_num = int(phase_num)
    n_scen = int(n_scen)
    steps_skip = int(steps_skip) 

    total_steps = 9000
    if phase_num == 3:
        n_buildings = 7
    else:
        n_buildings = 5
    
    file_name = f"debug_logs/scen_gurobi_phase_{phase_num}_step_leap_1_forecast_step_1.csv"
    scenario_gen = ScenarioFile_sliding(file_name, n_scenarios=n_scen, steps_ahead=24, steps_skip=steps_skip_forecast)
    log_exten = f"debug_logs/scen_gurobi_step_leap_{steps_skip}_forecast_step_{steps_skip_forecast}.csv"
    manager = GurobiMPC(0, steps_skip=steps_skip, file_name=log_exten)

    agent_used = GeneralAgent(scenario_gen, manager)
    tc, apc, aec, agc, agent_time_elapsed = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num)
    file = open(f"opt_and_forecast_revision_study_n_scenarios_phase{phase_num}.csv", "a+")
    
    file.write(f"\n{phase_num},{steps_skip},{steps_skip_forecast},{tc},{apc},{aec},{agc},{agent_time_elapsed}")
    
    file.close()

def hpc_single_argument(run_seed):
    run_seed = int(run_seed)
    
    # steps_ahead = [1, 2, 4, 6, 9, 12, 16, 24]
    n_scen = [1, 5, 10, 20, 30, 40, 50, 75]
    steps_opt_revision = [1]
    steps_forecast_revision = [1]
    phases = [3]
    total_runs = len(n_scen) * len(steps_opt_revision) * len(steps_forecast_revision) * len(phases)
    
    # Calculate indices for each parameter based on run_seed
    phase_num = phases[(run_seed // (total_runs // len(phases))) % len(phases)]
    n_scen_ind = (run_seed // (total_runs // (len(phases) * len(n_scen)))) % len(n_scen)
    steps_skip_forecast = steps_forecast_revision[(run_seed // len(steps_opt_revision)) % len(steps_forecast_revision)]
    steps_skip = steps_opt_revision[run_seed % len(steps_opt_revision)]
    
    n_scen = n_scen[n_scen_ind]

    print('CONFIGURATION: phase number, scenario number, skip optimizaiton steps, skip forecast steps')
    print(phase_num, n_scen, steps_skip, steps_skip_forecast)
    #hpc_evaluate(phase_num, n_scen, 1, steps_skip_forecast)
    if steps_skip <= steps_skip_forecast:
        hpc_evaluate(phase_num, n_scen, steps_skip, steps_skip_forecast)

if __name__ == "__main__":
    hpc_single_argument(sys.argv[1])