import pandas as pd


class PerfectForecast:
    def __init__(self, steps_ahead):
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

        return [forec, forec]

class PerfectRealForecast:
    def __init__(self, steps_ahead):
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