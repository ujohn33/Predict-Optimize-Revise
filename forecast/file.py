import pandas as pd


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
            
            df_real = self.df_forec[f"pred_ds_{i+1}"][
                init_ind : init_ind + steps_ahead
            ]
            forec_real.append(list(df_real))

        return [forec_real]

class ScenarioFile():
    def __init__(self, scen_file, n_scenarios=10,steps_ahead=24):
        
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
                self.scen_dict[scen_num]=dict()
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


if __name__ == "__main__":
    file_name = "debug_logs/scenarios_recurrent_gaussian_qts_9000_1.csv"
    a = ScenarioFile(file_name)
    for i in range(10):
        print(a.generate_scenarios({}, 0)[0])

