from copy import deepcopy
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from collections import deque


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


class PerfectFile:
    def __init__(self, steps_ahead=24):
        self.steps_ahead = steps_ahead
        consum_file = "data/citylearn_challenge_2022_phase_3/perfect_forecast.csv"
        self.df_forec = pd.read_csv(consum_file).to_dict("list")

    def generate_scenarios(self, prev_steps, time_step):
        init_ind = time_step
        num_buildings = 7
        forec = list()
        for i in range(num_buildings):
            df_slice = self.df_forec[f"building_{i}"][
                init_ind:(init_ind + self.steps_ahead)
            ]
            forec.append(list(df_slice))
        return [forec]


class PerfectRealForecast:
    def __init__(self, steps_ahead=24):
        self.steps_ahead = steps_ahead
        consum_file = "data/citylearn_challenge_2022_phase_1/net_minmax_preds.csv"
        self.df_forec = pd.read_csv(consum_file).to_dict("list")

    def generate_scenarios(self, prev_steps, time_step):
        num_buildings = 5

        init_ind = time_step
        forec_perf = list()
        forec_real = list()
        for i in range(num_buildings):
            df_perf = self.df_forec[f"Net+1_{i+1}"][
                init_ind : init_ind + self.steps_ahead
            ]

            forec_perf.append(list(df_perf))

            df_real = self.df_forec[f"pred_ds_{i+1}"][
                init_ind : init_ind + self.steps_ahead
            ]
            forec_real.append(list(df_real))

        return [forec_perf, forec_real]


class RealForecast:
    def __init__(self, steps_ahead=24):
        consum_file = "data/citylearn_challenge_2022_phase_1/net_minmax_preds.csv"
        preds_file = "data/citylearn_challenge_2022_phase_1/24h_pred.csv"
        self.df_forec = pd.read_csv(consum_file).to_dict("list")
        self.steps_ahead = steps_ahead

    def generate_scenarios(self, prev_steps, time_step, steps_ahead=24):
        steps_ahead = self.steps_ahead
        num_buildings = 5

        init_ind = time_step
        forec_real = list()
        for i in range(num_buildings):
            df_real = self.df_forec[f"pred_ds_{i+1}"][init_ind : init_ind + steps_ahead]
            forec_real.append(list(df_real))

        return [forec_real]


class ScenarioFile:
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24):
        self.n_scenarios = n_scenarios
        self.steps_ahead = steps_ahead
        self.scen_dict = {}
        file_df = pd.read_csv(scen_file)

        # Speed update for file reading
        # for _, row in file_df.iterrows():
        for row in file_df.values:
            # time_step = int(row["time_step"])
            time_step = int(row[0])
            # scen_num = int(row["scenario"])
            scen_num = int(row[1])
            if scen_num >= n_scenarios:
                continue
            # build_num = int(row["building"])
            build_num = int(row[2])
            # scen_val = row.values[3:]
            scen_val = row[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val

    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        # get the value of the first key in self.scen_dict
        # ind = list(self.scen_dict.keys())[0]
        num_buildings = len(self.scen_dict[list(self.scen_dict.keys())[0]].keys())
        for num_scen in self.scen_dict.keys():
            forec_real = list()
            for num_buil in range(num_buildings):
                scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]

                forec_real.append(list(scenario))
            scenarios.append(forec_real)

        return scenarios


class ScenarioFile_sliding:
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24, steps_skip=1):
        self.n_scenarios = n_scenarios
        self.steps_ahead = steps_ahead
        self.steps_skip = steps_skip
        self.scen_dict = {}
        self.forec_cache = list()
        file_df = pd.read_csv(scen_file)
        self.debugger_is_active = debugger_is_active()

        # Speed update for file reading
        # for _, row in file_df.iterrows():
        for row in file_df.values:
            # time_step = int(row["time_step"])
            time_step = int(row[0])
            # scen_num = int(row["scenario"])
            scen_num = int(row[1])
            if scen_num >= n_scenarios:
                continue
            # build_num = int(row["building"])
            build_num = int(row[2])
            # scen_val = row.values[3:]
            scen_val = row[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val

    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        index_cache = time_step % self.steps_skip
        # get the value of the first key in self.scen_dict
        # ind = list(self.scen_dict.keys())[0]
        num_buildings = len(self.scen_dict[list(self.scen_dict.keys())[0]].keys())
        for num_scen in self.scen_dict.keys():
            forec_real = list()
            for num_buil in range(num_buildings):
                scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]
                forec_real.append(list(scenario))
            if index_cache == 0:
                scenarios.append(forec_real)
                self.forec_cache = forec_real
                if self.debugger_is_active:
                    for data in forec_real:
                        plt.plot(data)
                    plt.show()
            else:
                # shift forec_real by index_cache
                shifted_forec = list()
                for i in range(num_buildings):
                    shifted_forec.append(
                        self.forec_cache[i][index_cache:] + self.forec_cache[i][:index_cache]
                    )
                    # change the last elements to the values from forec real
                    shifted_forec[i][-index_cache:] = forec_real[i][-index_cache:]
                scenarios.append(shifted_forec)
                if self.debugger_is_active:
                    for data in shifted_forec:
                        plt.plot(data)
                    plt.show()
        return scenarios
    

class ScenarioFile_averaging:
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24, average_window=5):
        self.n_scenarios = n_scenarios
        self.steps_ahead = steps_ahead
        self.average_window = average_window
        self.scen_dict = {}
        file_df = pd.read_csv(scen_file)
        self.debugger_is_active = debugger_is_active()
        self.forec_cache = [[deque() for _ in range(7)] for _ in range(n_scenarios)]

        for row in file_df.values:
            time_step = int(row[0])
            scen_num = int(row[1])
            if scen_num >= n_scenarios:
                continue
            build_num = int(row[2])
            scen_val = row[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val

    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        num_buildings = len(self.scen_dict[list(self.scen_dict.keys())[0]].keys())
        if len(self.forec_cache[0][0]) >= self.average_window:
            # popleft at every scenario at every building
            for i in range(self.n_scenarios):
                for j in range(num_buildings):
                    self.forec_cache[i][j].popleft()
        for num_scen in self.scen_dict:
            forec_real = list()
            for num_buil in range(num_buildings):
                scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]
                self.forec_cache[num_scen][num_buil].append(scenario)
                # Average the forecasts across sliding windows
                # Initialize a list to store the sums of the actions
                forec_sum = [0] * steps_ahead
                for j in range(steps_ahead):
                    count = 0
                    iteration = min(time_step,  (self.average_window - 1)) 
                    for i in range(min(iteration + 1, len(self.forec_cache[num_scen][num_buil]))):
                        if iteration - i + j < 24:
                            forec_sum[j] += self.forec_cache[num_scen][num_buil][i][iteration - i + j]
                            count += 1
                        else:
                            pass
                    forec_sum[j] = forec_sum[j] / count if count > 0 else scenario[j]
                forec_real.append(forec_sum)
            scenarios.append(forec_real)
        return scenarios
    

class ScenarioFileDailyNaive(ScenarioFile):
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24, steps_skip=1):
        self.n_scenarios = n_scenarios
        self.steps_ahead = steps_ahead
        self.steps_skip = steps_skip
        self.scen_dict = {}
        self.forec_cache = list()
        file_df = pd.read_csv(scen_file)
        self.debugger_is_active = debugger_is_active()
        # Speed update for file reading
        # for _, row in file_df.iterrows():
        for row in file_df.values:
            # time_step = int(row["time_step"])
            time_step = int(row[0])
            # scen_num = int(row["scenario"])
            scen_num = int(row[1])
            if scen_num >= n_scenarios:
                continue
            # build_num = int(row["building"])
            build_num = int(row[2])
            # scen_val = row.values[3:]
            scen_val = row[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val
    
        # make a function that finds the variance of error for each hour of the day
    def daily_persistence(self, prev_steps, id_param, horizon=24):
        # current_hour = prev_steps['hour'][-1] % 24
        # sample_preds is a dict of dicts
        load = np.array(prev_steps[f"non_shiftable_load_{id_param}"][-horizon:])
        pv = np.array(prev_steps[f"solar_generation_{id_param}"][-horizon:])
        net_load = load - pv
        return net_load
    
    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        index_cache = time_step % self.steps_skip
        # get the value of the first key in self.scen_dict
        # ind = list(self.scen_dict.keys())[0]
        num_buildings = len(self.scen_dict[list(self.scen_dict.keys())[0]].keys())
        for num_scen in self.scen_dict.keys():
            forec_real = list()
            for num_buil in range(num_buildings):
                if time_step < 24:
                    scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]
                else:
                    scenario = self.daily_persistence(prev_steps, num_buil)
                forec_real.append(list(scenario))
            if index_cache == 0:
                scenarios.append(forec_real)
                self.forec_cache = forec_real
                if self.debugger_is_active:
                    for data in forec_real:
                        plt.plot(data)
                    plt.show()
            else:
                # shift forec_real by index_cache
                shifted_forec = list()
                for i in range(num_buildings):
                    shifted_forec.append(
                        self.forec_cache[i][index_cache:] + self.forec_cache[i][:index_cache]
                    )
                    # change the last elements to the values from forec real
                    shifted_forec[i][-index_cache:] = forec_real[i][-index_cache:]
                scenarios.append(shifted_forec)
                if self.debugger_is_active:
                    for data in shifted_forec:
                        plt.plot(data)
                    plt.show()
        return scenarios


class ScenarioFileWeeklyMeans(ScenarioFile):
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24, steps_skip=1):
        self.n_scenarios = n_scenarios
        self.steps_ahead = steps_ahead
        self.steps_skip = steps_skip
        self.scen_dict = {}
        self.forec_cache = list()
        file_df = pd.read_csv(scen_file)
        self.debugger_is_active = debugger_is_active()
        # Speed update for file reading
        # for _, row in file_df.iterrows():
        for row in file_df.values:
            # time_step = int(row["time_step"])
            time_step = int(row[0])
            # scen_num = int(row["scenario"])
            scen_num = int(row[1])
            if scen_num >= n_scenarios:
                continue
            # build_num = int(row["building"])
            build_num = int(row[2])
            # scen_val = row.values[3:]
            scen_val = row[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val
    
        # make a function that finds the variance of error for each hour of the day
    def update_means_last_week(self, prev_steps, id_param, horizon=24):
        # sample_preds is a dict of dicts
        sample_actual = {}
        resid_means = np.empty(horizon)
        # actuals sample at certain hour
        for h in range(horizon):
            # forecasts sample at certain hour for certain horizon
            load = np.array(
                prev_steps[f"non_shiftable_load_{id_param}"][-24 + h : -169 + h : -24]
            )
            solar = np.array(
                prev_steps[f"solar_generation_{id_param}"][-24 + h : -169 + h : -24]
            )
            sample_actual[h] = load - solar
            # get the residuals std for each horizon
            resid_means[h] = np.mean(sample_actual[h])
        return resid_means
    
    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        index_cache = time_step % self.steps_skip
        # get the value of the first key in self.scen_dict
        # ind = list(self.scen_dict.keys())[0]
        num_buildings = len(self.scen_dict[list(self.scen_dict.keys())[0]].keys())
        for num_scen in self.scen_dict.keys():
            forec_real = list()
            for num_buil in range(num_buildings):
                if time_step < 168:
                    scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]
                else:
                    scenario = self.update_means_last_week(prev_steps, num_buil)
                forec_real.append(list(scenario))
            if index_cache == 0:
                scenarios.append(forec_real)
                self.forec_cache = forec_real
                if self.debugger_is_active:
                    for data in forec_real:
                        plt.plot(data)
                    plt.show()
            else:
                # shift forec_real by index_cache
                shifted_forec = list()
                for i in range(num_buildings):
                    shifted_forec.append(
                        self.forec_cache[i][index_cache:] + self.forec_cache[i][:index_cache]
                    )
                    # change the last elements to the values from forec real
                    shifted_forec[i][-index_cache:] = forec_real[i][-index_cache:]
                scenarios.append(shifted_forec)
                if self.debugger_is_active:
                    for data in shifted_forec:
                        plt.plot(data)
                    plt.show()
        return scenarios


if __name__ == "__main__":
    file_name = "debug_logs/scenarios_recurrent_gaussian_qts_9000_1.csv"
    a = ScenarioFile(file_name)
    for i in range(10):
        print(a.generate_scenarios({}, 0)[0])
