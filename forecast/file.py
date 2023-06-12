from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


class PerfectFile:
    def __init__(self, steps_ahead=24):
        self.steps_ahead = steps_ahead
        consum_file = "data/citylearn_challenge_2022_phase_1/net_minmax_preds.csv"
        self.df_forec = pd.read_csv(consum_file).to_dict("list")

    def generate_scenarios(self, prev_steps, time_step):
        init_ind = time_step
        num_buildings = 5
        forec = list()
        for i in range(num_buildings):
            df_slice = self.df_forec[f"Net+1_{i+1}"][
                init_ind : init_ind + self.steps_ahead
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
        for _, row in file_df.iterrows():
            time_step = int(row["time_step"])
            scen_num = int(row["scenario"])
            build_num = int(row["building"])
            scen_val = row.values[3:]
            if scen_num not in self.scen_dict.keys():
                self.scen_dict[scen_num] = dict()
            if build_num not in self.scen_dict[scen_num].keys():
                self.scen_dict[scen_num][build_num] = dict()
            self.scen_dict[scen_num][build_num][time_step] = scen_val

    def generate_scenarios(self, prev_steps, time_step):
        scenarios = list()
        steps_ahead = self.steps_ahead
        num_buildings = len(self.scen_dict[0].keys())
        for num_scen in range(self.n_scenarios):
            forec_real = list()
            for num_buil in range(num_buildings):
                scenario = self.scen_dict[num_scen][num_buil][time_step][:steps_ahead]

                forec_real.append(list(scenario))
            scenarios.append(forec_real)

        return scenarios


class ScenarioFileAndNaive(ScenarioFile):
    def __init__(self, scen_file, n_scenarios=10, steps_ahead=24):
        super().__init__(scen_file, n_scenarios, steps_ahead)
        self.forecast_hist = list()
        self.num_buildings = len(self.scen_dict[0].keys())

        self.forec_func = {
            "day": single_day_forecasting,
            "week": single_week_forecasting,
            "prev": single_prev_hour_forecasting,
            "grad": single_gradient_forecasting,
        }

        self.save_naive_forec = {
            key: [[] for _ in range(self.num_buildings)]
            for key in self.forec_func.keys()
        }

        self.save_scenarios = list()

    def generate_scenarios(self, prev_steps, time_step):
        scenarios = super().generate_scenarios(prev_steps, time_step)
        self.forecast_hist.append(deepcopy(scenarios))

        num_checks = 7
        average_over = 7

        needed_time_steps = (num_checks + average_over) * 7 * 24
        hours_back = 24 + 24 * num_checks
        if len(prev_steps[f"non_shiftable_load_0"]) < hours_back:
            self.save_scenarios.append(deepcopy(scenarios))
            self.populate_naive_forec(prev_steps, average_over)
            return scenarios

        naive_forec_save = {
            key: None for key in list(self.forec_func.keys()) + ["forec"]
        }

        for i in range(self.num_buildings):
            load_pow = prev_steps[f"non_shiftable_load_{i}"]
            pv_pow = prev_steps[f"solar_generation_{i}"]
            load_pv_power = [load - pv_pow[j] for j, load in enumerate(load_pow)]

            forecast = [power for power in load_pv_power]
            for j in range(num_checks):
                for k in range(24):
                    forec_ind = -(j + 1) * 24 + k
                    forecast[forec_ind] = self.forecast_hist[-(j + 1) * 24 - 1][0][i][k]
            naive_forec_save["forec"] = forecast

            for forec_method, _ in self.forec_func.items():
                naive_forec_save[forec_method] = self.save_naive_forec[forec_method][i]

            checks = {
                "day": [[] for _ in range(24)],
                "week": [[] for _ in range(24)],
                "forec": [[] for _ in range(24)],
                "prev": [[]],
                "grad": [[]],
            }
            for j in range(num_checks):
                for k in range(24):
                    for forec_method in checks.keys():
                        if k != 0 and forec_method in ["grad", "prev"]:
                            continue
                        ind_err = -24 * (j + 1) + k
                        naive_forec = naive_forec_save[forec_method]
                        error = abs(naive_forec[ind_err] - load_pv_power[ind_err])
                        checks[forec_method][k].append(error)

            forec_errors = checks.pop("forec")
            for j in range(self.steps_ahead):
                min_error = 10000000
                best_method = "forec"
                for forec_method, errors in checks.items():
                    if j != 0 and forec_method in ["grad", "prev"]:
                        continue
                    all_better = True
                    for k, forec_err in enumerate(forec_errors[j]):
                        naive_better = forec_err > errors[j][k]
                        all_better = all_better and naive_better

                    mean_error = sum(errors[j]) / len(errors[j])
                    if mean_error < min_error and all_better:
                        best_method = forec_method
                        min_error = mean_error

                if best_method != "forec":
                    # print(f"house {i}")
                    # print(f"step {j}")
                    # print(best_method)
                    best_forec = self.forec_func[best_method](
                        load_pv_power, average_over, j
                    )
                    scenarios[0][i][j] = best_forec

        self.save_scenarios.append(deepcopy(scenarios))
        self.populate_naive_forec(prev_steps, average_over)

        return scenarios

    def populate_naive_forec(self, prev_steps, average_over):
        debug = False
        for i in range(self.num_buildings):
            load_pow = prev_steps[f"non_shiftable_load_{i}"]
            pv_pow = prev_steps[f"solar_generation_{i}"]
            load_pv_power = [load - pv_pow[j] for j, load in enumerate(load_pow)]

            for forec_method, forec_fun in self.forec_func.items():
                self.save_naive_forec[forec_method][i].append(
                    forec_fun(load_pv_power, average_over, 0)
                )

                if len(load_pv_power) % 500 == 0:
                    plt.plot(
                        [0] + self.save_naive_forec[forec_method][i],
                        label=forec_method,
                        linestyle="--",
                    )
            if len(load_pv_power) % 500 == 0 and debug:
                forecast = [power for power in load_pv_power]
                for j, forec in enumerate(self.forecast_hist):
                    forecast[j] = forec[0][i][0]
                """
                plt.figure()
                plt.plot([0] + forecast, label="forecast")
                plt.plot(load_pv_power, label="actual")
                plt.title(f"House {i}")
                plt.legend()
                """
                plt.figure()
                scenatrios = [power for power in load_pv_power]
                for j, forec in enumerate(self.save_scenarios):
                    scenatrios[j] = forec[0][i][0]

                plt.plot([0] + scenatrios, label="scenarios")
                plt.plot([0] + forecast, label="forecast")
                plt.plot(load_pv_power, label="actual")
                plt.title(f"House {i}")
                plt.legend()
        if len(load_pv_power) % 500 == 0 and debug:
            plt.show()


def single_week_forecasting(load_pv_power, average_over, step):
    return single_day_forecasting(load_pv_power, average_over, step, 7)


def single_day_forecasting(load_pv_power, average_over, step, num_days=1):
    default_val = 0.366777057557699
    to_average = 0
    num_averaged = 0
    for j in range(average_over):
        index_naive = step - (j + 1) * 24 * num_days

        if len(load_pv_power) >= -index_naive:
            to_average += load_pv_power[index_naive]
            num_averaged += 1
    if num_averaged == 0:
        forecast = default_val
    else:
        forecast = to_average / num_averaged
    return forecast


def single_prev_hour_forecasting(load_pv_power, average_over, step):
    return load_pv_power[len(load_pv_power) + step - 1]


def single_gradient_forecasting(load_pv_power, average_over, step):
    default_val = 0.366777057557699
    x = np.array([[1, 0], [1, 1], [1, 2]])
    first_elem = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x))

    forecast = default_val
    if len(load_pv_power) >= 3:
        to_ind = len(load_pv_power) + step
        y = np.array(load_pv_power[to_ind - 3 : to_ind])
        # slope, intercept, _, _, _ = stats.linregress(x, y)
        beta = np.dot(first_elem, y)
        slope = beta[1]
        intercept = beta[0]
        forec_pow = slope * 3 + intercept
        forecast = forec_pow

    return forecast


def naive_week_forecasting(load_pv_power, average_over):
    return naive_day_forecasting(load_pv_power, average_over, 7)


def naive_day_forecasting(load_pv_power, average_over, num_days=1):
    default_val = 0.366777057557699
    num_defaults = num_days * 24
    default_start = [default_val for _ in range(num_defaults)]
    forecasted = default_start

    for j in range(average_over):
        if len(forecasted) > len(load_pv_power):
            break

        array_to_average = list()

        for k in range(j + 1):
            from_ind = k * 24 * num_days
            if j != average_over - 1:
                to_ind = from_ind + 24 * num_days
            else:
                remaining_days = len(load_pv_power) - len(forecasted)
                to_ind = from_ind + remaining_days

            array_to_average.append(load_pv_power[from_ind:to_ind])
        np_to_avg = np.array(array_to_average)
        naive_average = np.average(np_to_avg, axis=0)
        forecasted += list(naive_average)

    return forecasted[: len(load_pv_power)]
    for i, _ in enumerate(load_pv_power):
        to_average = 0
        num_averaged = 0
        for j in range(average_over):
            index_naive = i - (j + 1) * 24 * num_days

            if index_naive >= 0:
                to_average += load_pv_power[index_naive]
                num_averaged += 1

        if num_averaged != 0:
            forecasted[i] = to_average / num_averaged

    return forecasted


def prev_hour_forecasting(load_pv_power, average_over):
    default_val = 0.366777057557699
    forecasted = [default_val] + load_pv_power[:-1]
    return forecasted


def gradient_forecasting(load_pv_power, average_over):
    default_val = 0.366777057557699
    x = np.array([[1, 0], [1, 1], [1, 2]])
    first_elem = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x))
    forecasted = list()
    forecasted = [default_val for _ in load_pv_power]
    for i, _ in enumerate(load_pv_power):
        if i >= 3:
            y = np.array(load_pv_power[i - 3 : i])
            # slope, intercept, _, _, _ = stats.linregress(x, y)
            beta = np.dot(first_elem, y)
            slope = beta[1]
            intercept = beta[0]
            forec_pow = slope * 3 + intercept
            forecasted[i] = forec_pow

    return forecasted


if __name__ == "__main__":
    file_name = "debug_logs/scenarios_recurrent_gaussian_qts_9000_1.csv"
    a = ScenarioFile(file_name)
    for i in range(10):
        print(a.generate_scenarios({}, 0)[0])
