from agents.general_agent import GeneralAgent
from forecast.scenarios_lean import Scenario_Generator
from ems.gurobi_mpc import GurobiMPC
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
    file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
    scenario_gen = Scenario_Generator(
            forec_file=file_name,
            type="norm_noise",
            n_scenarios=n_scen,
            steps_ahead=24,
            revision_forec_freq=steps_skip_forecast,
            n_buildings=n_buildings,
    )
    manager = GurobiMPC(0, steps_skip=steps_skip)

    agent_used = GeneralAgent(scenario_gen, manager)
    tc, apc, aec, agc, agent_time_elapsed = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num)
    file = open("scen_skipped_steps_res.csv", "a+")
    
    file.write(f"\n{phase_num},{n_scen},{steps_skip},{tc},{apc},{aec},{agc},{agent_time_elapsed}")
    
    file.close()

def hpc_single_argument(run_seed):
    run_seed = int(run_seed)
    
    # steps_ahead = [1, 2, 4, 6, 9, 12, 16, 24]
    steps_ahead = [1,2,3,4,5,6,7,8]
    phases = [1, 2, 3]
    num_scenarios = [102+(i) * 2 for i in range(50)]
    total_runs = len(steps_ahead)*len(phases)*len(num_scenarios)
    # Set three parameters from run_seed and the rest from the above lists
    phase_num = phases[run_seed // (total_runs // len(phases))]
    n_scen = num_scenarios[run_seed % len(num_scenarios)]
    steps_skip_ind = (run_seed // len(num_scenarios)) % len(steps_ahead)
    steps_skip = steps_ahead[steps_skip_ind]

    hpc_evaluate(phase_num, n_scen, steps_skip)

if __name__ == "__main__":
    hpc_single_argument(sys.argv[1])