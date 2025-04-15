from agents.general_agent import GeneralAgent
from forecast.scenarios_lean import Scenario_Generator
from ems.gurobi_mpc import GurobiMPC
from local_evaluation import evaluate
import sys


def hpc_evaluate(phase_num, n_scen, steps_skip, steps_skip_forecast):
    
    phase_num = int(phase_num)
    n_scen = int(n_scen)
    steps_skip = int(steps_skip) 
    grid_cost_bool = False

    total_steps = 9000
    if phase_num == 3:
        n_buildings = 7
    else:
        n_buildings = 5
    file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
    scenario_gen = Scenario_Generator(
            forec_file=file_name,
            type="norm_noise",
            n_scenarios=n_scen,
            steps_ahead=24,
            revision_forec_freq=steps_skip_forecast,
            n_buildings=n_buildings,
    )
    log_exten = f"/data/brussel/vo/000/bvo00037/vsc10528/debug_logs/gurobi_phase_{phase_num}_step_leap_{steps_skip}_forecast_step_{steps_skip_forecast}.csv"
    manager = GurobiMPC(0, steps_skip=steps_skip, grid_include=grid_cost_bool, file_name=log_exten)
    #manager = GurobiMPC(0, steps_skip=steps_skip)

    agent_used = GeneralAgent(scenario_gen, manager)
    tc, apc, aec, agc, agent_time_elapsed = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num, grid_include=grid_cost_bool)
    if grid_cost_bool:
        print("Grid cost included")
        file = open(f"opt_and_forecast_revision_study_12x12_phase{phase_num}.csv", "a+")
    else:
        print("Grid cost NOT included")
        file = open(f"opt_and_forecast_revision_study_12x12_nogridscore_phase{phase_num}.csv", "a+")
    
    file.write(f"\n{phase_num},{steps_skip},{steps_skip_forecast},{tc},{apc},{aec},{agc},{agent_time_elapsed}")
    
    file.close()

def hpc_single_argument(run_seed):
    run_seed = int(run_seed)
    
    n_scen = 75
    steps = list(range(1, 13))  # Use same list for both skip parameters
    phases = [1]
    
    # Total runs is now just phases × steps since we're only doing diagonal combinations
    total_runs = len(phases) * len(steps)
    
    # Set parameters from run_seed
    phase_num = phases[run_seed // len(steps)]
    steps_value = steps[run_seed % len(steps)]
    
    # Set both step parameters to the same value
    steps_skip = steps_value
    steps_skip_forecast = 1
    
    print('CONFIGURATION: phase number, scenario number, skip optimizaiton steps, skip forecast steps')
    print(phase_num, n_scen, steps_skip, steps_skip_forecast)
    
    hpc_evaluate(phase_num, n_scen, steps_skip, steps_skip_forecast)
if __name__ == "__main__":
    hpc_single_argument(sys.argv[1])