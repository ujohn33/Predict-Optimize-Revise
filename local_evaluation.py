import numpy as np
import time
from agents.general_agent import GeneralAgent
from ems.logger_manager import LoggerManager
from ems.mpc import MPC
from forecast.scenarios_lean import Scenario_Generator
from forecast.file import PerfectFile, RealForecast, ScenarioFile, ScenarioFileAndNaive, ScenarioFile_sliding
from utils.logger import log_usefull
from ems.pyo_mpc import PyoMPC
from ems.gurobi_mpc import GurobiMPC
from ems.gurobi_matrix_mpc import GurobiMatrixMPC
from ems.multi_stage_mpc import MultiStageMPC

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv


def action_space_to_dict(aspace):
    """Only for box space"""
    return {
        "high": aspace.high,
        "low": aspace.low,
        "shape": aspace.shape,
        "dtype": str(aspace.dtype),
    }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {
        "action_space": action_space_dicts,
        "observation_space": observation_space_dicts,
        "building_info": building_info,
        "observation": observations,
    }
    return obs_dict


def evaluate(agent_used, total_steps=9000, phase_num=1, grid_include=True):
    print("Starting local evaluation")

    schema_path = f"./data/citylearn_challenge_2022_phase_{phase_num}/schema.json"
    env = CityLearnEnv(schema=schema_path)
    agent = OrderEnforcingAgent(agent_used)

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while True:
            ### This is only a reference script provided to allow you
            ### to do local evaluation. The evaluator **DOES NOT**
            ### use this script for orchestrating the evaluations.

            observations, _, done, _ = env.step(actions)
            if done or (num_steps + 1) == total_steps:
                # Log run
                filename = f"debug_logs/run_logs.csv"
                log_usefull(env, filename)

                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {
                    "price_cost": metrics_t[0],
                    "emmision_cost": metrics_t[1],
                    "grid_cost": metrics_t[2],
                }
                if np.any(np.isnan(metrics_t)):
                    raise ValueError(
                        "Episode metrics are nan, please contant organizers"
                    )
                episode_metrics.append(metrics)
                print(
                    f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}",
                )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 100 == 0:
                # filename = f"debug_logs/run_logs_{episodes_completed}.csv"
                # log_usefull(env, filename)
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= 1:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print(
            "Average Price Cost:", np.mean([e["price_cost"] for e in episode_metrics])
        )
        print(
            "Average Emmision Cost:",
            np.mean([e["emmision_cost"] for e in episode_metrics]),
        )
        print("Average Grid Cost:", np.mean([e["grid_cost"] for e in episode_metrics]))
        if grid_include == True:
            total_cost = np.mean(
                [
                    e["price_cost"] + e["emmision_cost"] + e["grid_cost"]
                    for e in episode_metrics
                ]
            )
            print("Average Total Cost:", total_cost / 3)
            tc = total_cost / 3
        else:
            total_cost = np.mean([e["price_cost"] + e["emmision_cost"] for e in episode_metrics])
            print("Average Total Cost:", total_cost / 2)
            tc = total_cost / 2
        apc = np.mean([e["price_cost"] for e in episode_metrics])
        aec = np.mean([e["emmision_cost"] for e in episode_metrics])
        agc = np.mean([e["grid_cost"] for e in episode_metrics])
        
    print(f"Total time taken by agent: {agent_time_elapsed}s")

    return tc, apc, aec, agc, agent_time_elapsed


if __name__ == "__main__":
    case_study = "together_live"
    #case_study = "multi_stage_mpc"
    #case_study = "comp_multi_stage_mpc"
    phase_num = 3
    total_steps = 9000
    n_scen = 75
    steps_skip = 3
    steps_skip_forecast = 3
    if phase_num == 3:
        n_buildings = 7
    else:
        n_buildings = 5
    if case_study == "realistic_file_forec":
        scenario_gen = RealForecast()
        manager = MPC(0, weight_step="equal")
    elif case_study == "perfect_file_forec":
        scenario_gen = PerfectFile(24)
        manager = GurobiMatrixMPC(0)
    elif case_study == "logging":
        type_forec = "tree_scenario"
        param = f"{type_forec}_{total_steps}_{phase_num}"
        scenario_gen = Scenario_Generator(
            type=type_forec, n_scenarios=n_scen, steps_ahead=24, n_buildings=n_buildings
        )
        logger = LoggerManager(param)
        manager = logger
        scenario_gen.logger = logger
    elif case_study == "read_scenarios_files":
        file_name = f"debug_logs/scenarios_recurrent_quant_s10_p{phase_num}_24h.csv"
        n_scen = 10
        scenario_gen = ScenarioFile(file_name, n_scenarios=n_scen)
        manager = PyoMPC(0)
    elif case_study == "read_log_mpc":
        method = "recurrent_quant"
        file_name = f"debug_logs/scenarios_{method}_s{n_scen}_p{phase_num}_24h.csv"

        scenario_gen = ScenarioFile(file_name, n_scenarios=n_scen)
        mpc_log = f"debug_logs/mpc_{method}_s{n_scen}_p{phase_num}_t{total_steps}.csv"
        manager = MPC(0, file_name=mpc_log)

    elif case_study == "debug_pyo_mpc":
        scenario_gen = PerfectFile()
        manager = PyoMPC(0)
    elif case_study == "together":
        file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
        scenario_gen = ScenarioFile_sliding(file_name, n_scenarios=n_scen, steps_ahead=24, steps_skip=steps_skip_forecast)
        log_exten = f"debug_logs/gurobi_step_leap_{steps_skip}_forecast_step_{steps_skip_forecast}.csv"
        manager = GurobiMPC(0, steps_skip=steps_skip, file_name=log_exten)
        #manager = MPC(0)
    elif case_study == "together+naive":
        file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
        scenario_gen = ScenarioFileAndNaive(file_name, n_scenarios=1)
    elif case_study == "together_live":
        file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
        scenario_gen = Scenario_Generator(
            forec_file=file_name,
            type="norm_noise",
            n_scenarios=n_scen,
            steps_ahead=24,
            revision_forec_freq=steps_skip_forecast,
            n_buildings=n_buildings,
        )
        manager = GurobiMPC(0, steps_skip=steps_skip, grid_include=True)
    elif case_study == "multi_stage_mpc":
        type_forec = "tree_scenario"

        file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"

        num_child = 2
        robust_horizon = 2
        scenario_gen = Scenario_Generator(
            type=type_forec,
            n_scenarios=n_scen,
            steps_ahead=24,
            n_buildings=n_buildings,
            forec_file=file_name,
        )
        scenario_gen.num_child = num_child
        scenario_gen.robust_horizon = robust_horizon
        scenario_gen.steps_skip = 1
        log_exten = f"debug_logs/multi_stage_mpc_{scenario_gen.steps_skip}.csv"
        manager = MultiStageMPC(file_name=log_exten)
        # manager = GurobiMPC(0)
    elif case_study == "comp_multi_stage_mpc":
        # type_forec = "norm_noise"
        # n_scen = 9
        # file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
        # scenario_gen = Scenario_Generator(
        #     type=type_forec,
        #     n_scenarios=n_scen,
        #     steps_ahead=24,
        #     n_buildings=n_buildings,
        #     forec_file=file_name,
        # )

        # manager = GurobiMPC(0, steps_skip=3)
        n_scen = 10
        scenario_gen = Scenario_Generator(
            n_scenarios=n_scen, 
            n_buildings=n_buildings, 
            steps_ahead=24
        )
        manager = GurobiMPC(0, steps_skip=3)

    agent_used = GeneralAgent(scenario_gen, manager)
    evaluate(agent_used, total_steps=total_steps, phase_num=phase_num)
