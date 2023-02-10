import pandas as pd


class PerfectForecast:
    def __init__(self, steps_ahead):
        self.steps_ahead = steps_ahead
        consum_file = "data/citylearn_challenge_2022_phase_1/net_minmax_preds.csv"
        self.df_forec = pd.read_csv(consum_file).to_dict("list")

    def get_forecast(self, time_step, num_buildings):
        init_ind = time_step
        forec = list()
        for i in range(num_buildings):
            df_slice = self.df_forec[f"Net+1_{i+1}"][
                init_ind : init_ind + self.steps_ahead
            ]
            forec.append(list(df_slice))

        return [forec, forec]
