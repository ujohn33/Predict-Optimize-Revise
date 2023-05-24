import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_run(filename, name="", file_perfect = "data/citylearn_challenge_2022_phase_1/milp_phase_1_log.csv",only_common=False):
    ax_list = []
    df_run = pd.read_csv(filename)
    num_buildings = sum([1 for i in df_run.columns if i.startswith("baseload_")])
    df_perfect_run = pd.read_csv(file_perfect)

    date_range = pd.date_range(start='2021-07-01',end="2022-07-01",freq='H')

    date_range = date_range[:len(df_run["baseload_0"])]
    df_run["date_range"] = date_range

    for i in range(num_buildings):
        df_run[f"perfect_load_{i}"]=df_perfect_run[f"final_load_{i}"]

    df_run["prices"] = df_run["prices"] * 10
    df_run["carbon_intensity"] = df_run["carbon_intensity"] * 30
    if not only_common:
        for i in range(num_buildings):
            ax = df_run.plot(x="date_range",drawstyle="steps",
                y=[
                    #"prices",
                    "carbon_intensity",
                    f"baseload_{i}",
                    f"final_load_{i}",
                    f"perfect_load_{i}",
                ],
                title=f"Battery_{i} {name}",
            )
            ax.axhline(0, linestyle="--")
            ax_list.append(ax)
            # ax.set_ylim(-4, 8)
            # ax.set_xlim(0, 140)
    
    baseload_cols = [f"baseload_{i}" for i in range(num_buildings)]
    final_load_cols = [f"final_load_{i}" for i in range(num_buildings)]
    charge_cols = [f"charge_power_{i}" for i in range(num_buildings)]
    perfect_cols = [f"perfect_load_{i}" for i in range(num_buildings)]


    df_run["tot_baseload"] = df_run[baseload_cols].sum(axis=1)
    df_run["tot_final_load"] = df_run[final_load_cols].sum(axis=1)
    df_run["charge_power_total"] = df_run[charge_cols].sum(axis=1)
    df_run["tot_perfect_load"] = df_run[perfect_cols].sum(axis=1)

    ax = df_run.plot(x="date_range",drawstyle="steps",
        y=[
            "prices",
            #"carbon_intensity",
            "tot_baseload",
            "tot_final_load",
            "tot_perfect_load",
        ],
        title=f"Total loads {name}",
    )
    ax.axhline(0, linestyle="--")
    ax.set_ylim(-15, 25)
    # ax.set_xlim(0, 8760)

    df_run["obj_func"] = df_run["price_eval"].cumsum() / (
        2 * df_run["price_eval_no_batt"].cumsum()
    ) + df_run["co2_eval"].cumsum() / (
        2 * df_run["co2_eval_no_batt"].cumsum()
    )
    ax_list.append(ax)

    return df_run, ax_list

def plot_mpc_decision(mpc_log_file, time_step, file_perfect = "data/citylearn_challenge_2022_phase_1/milp_phase_1_log.csv", only_common=True):
    real_power_file = f"debug_logs/real_power_mpc_{mpc_log_file}.csv"
    scenario_file = f"debug_logs/scen_mpc_{mpc_log_file}.csv"
    batt_pow_file = f"debug_logs/pow_mpc_{mpc_log_file}.csv"
    
    real_df = pd.read_csv(real_power_file)
    scen_df = pd.read_csv(scenario_file)
    batt_df = pd.read_csv(batt_pow_file)
    if file_perfect is not None:
        perf_df = pd.read_csv(file_perfect)
        perf_df.index = perf_df.index-1

    num_hours = int(scen_df.columns[-1][:-1])

    num_buildings = max(scen_df["building"])+1
    
    scenarios = [[] for _ in range(num_buildings)]
    batt_powers = list()
    perf_power = list()
    real_power = list()
    next_batt_power = list()
    for i in range(num_buildings):
        
        rows_scen = scen_df.loc[(scen_df["time_step"]==time_step) & (scen_df["building"]==i)]

        for j in list(rows_scen["scenario"]):
            scenario = [list(rows_scen[f"+{i}h"])[j] for i in range(num_hours)]
            scenarios[i].append(scenario)
        
        mean = np.array([np.mean(list(rows_scen[f"+{i}h"])) for i in range(num_hours)])
        std = np.array([np.std(list(rows_scen[f"+{i}h"])) for i in range(num_hours)])
        if file_perfect is not None:
            perf_batt_power = [perf_df[f"charge_power_{i}"][time_step+j] for j in range(num_hours)]
            perf_power.append(perf_batt_power)

        power_build = [real_df.loc[(real_df["time_step"]==time_step+j)][f"building_{i}"] for j in range(num_hours)]
        real_power.append(power_build)
        
        rows_batt = batt_df.loc[(batt_df["time_step"]==time_step) & (batt_df["building"]==i)]
        batt_pow = [float(rows_batt[f"+{j}h"]) for j in range(num_hours)]
        
        batt_powers.append(batt_pow)

        a = float(batt_df.loc[(batt_df["time_step"]==time_step+j) & (batt_df["building"]==i)]["+0h"])
        next_batt = [float(batt_df.loc[(batt_df["time_step"]==time_step+j) & (batt_df["building"]==i)]["+0h"]) for j in range(num_hours)]
        next_batt_power.append(next_batt)
        if not only_common:
            plt.figure()
            plt.title(f"Buidling {i}")
            plt.plot(mean,alpha=0.9, drawstyle='steps-post', c="tab:blue", linestyle='--', linewidth=0.8,label=f"{len(scenarios)} scenarios mean+std")
            plt.plot(mean+std,alpha=0.9, drawstyle='steps-post', linestyle='dotted', linewidth=0.8, c="tab:blue")
            plt.plot(mean-std,alpha=0.9, drawstyle='steps-post', linestyle='dotted', linewidth=0.8, c="tab:blue")
            plt.plot(power_build, drawstyle='steps-post', c="tab:red", linestyle='--',linewidth=0.8, label= "Real power")


            plt.plot(batt_pow, drawstyle='steps-post', c="tab:orange", label="MPC battery")
            if file_perfect is not None:
                plt.plot(perf_batt_power, drawstyle='steps-post', c="tab:green", label="Perfect battery")


            plt.legend()

    scen_comm = aggreg_scen(scenarios)
    batt_comm = sum_lists(batt_powers)
    if file_perfect is not None:
        perf_comm = sum_lists(perf_power)
    real_comm = sum_lists(real_power)
    real_comm = real_comm.reshape((real_comm.size))
    next_comm = sum_lists(next_batt_power)


    scen_mean = np.array([np.mean([scen_comm[j][i] for j in range(len(scen_comm))]) for i in range(num_hours)])
    scen_std = np.array([np.std([scen_comm[j][i] for j in range(len(scen_comm))]) for i in range(num_hours)])


    plt.figure()
    plt.title(f"All buildings")
    plt.plot(scen_mean,alpha=0.9, drawstyle='steps-post', c="tab:blue", linestyle='--', linewidth=0.8,label=f"{len(scenarios)} scenarios mean+std")
    plt.plot(scen_mean+scen_std,alpha=0.9, drawstyle='steps-post', linestyle='dotted', linewidth=0.8, c="tab:blue")
    plt.plot(scen_mean-scen_std,alpha=0.9, drawstyle='steps-post', linestyle='dotted', linewidth=0.8, c="tab:blue")
    plt.plot(real_comm, drawstyle='steps-post', c="tab:red", linestyle='--',linewidth=0.8, label= "Real power")


    plt.plot(batt_comm, drawstyle='steps-post', c="tab:orange", label="MPC battery")
    if file_perfect is not None:
        plt.plot(perf_comm, drawstyle='steps-post', c="tab:green", label="Perfect battery")


    plt.legend()

    plt.figure()
    plt.plot(real_comm+batt_comm, drawstyle='steps-post', label="MPC multi scenario")
    if file_perfect is not None:
        plt.plot(real_comm+perf_comm, drawstyle='steps-post', label="Perfect battery")
    plt.plot(real_comm+next_comm, drawstyle='steps-post', label="MPC next step")
    plt.legend()

def sum_lists(list_of_list):
    summed_list = [0 for _ in list_of_list[0]]
    for i, sub_list in enumerate(list_of_list):
        for j, val in enumerate(sub_list):
            summed_list[j] += val
    
    return np.array(summed_list)

def aggreg_scen(scenarios):
    summed_list = [[0 for _ in scenarios[0][0]] for _ in scenarios[0]]
    for i, building in enumerate(scenarios):
        for j, scenario in enumerate(building):
            for k, val in enumerate(scenario):
                summed_list[j][k] += val
    return summed_list


if __name__ == "__main__":
    #filename = "debug_logs/run_logs.csv"
    # filename = "data/citylearn_challenge_2022_phase_1/milp_phase_1_log.csv"
    #plot_run(filename, name="")
    mpc_log_file = "recurrent_quant_s10_p1_t200"
    file_perfect = "data/citylearn_challenge_2022_phase_1/milp_phase_1_log.csv"
    plot_mpc_decision(mpc_log_file, 155,file_perfect=file_perfect, only_common=True)

    plt.show()