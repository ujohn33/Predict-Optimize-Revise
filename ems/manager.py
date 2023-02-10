import pandas as pd


class Manager:
    def __init__(self):
        carbon_file = "data/citylearn_challenge_2022_phase_1/carbon_intensity_full.csv"
        self.carb_df = pd.read_csv(carbon_file).to_dict("list")['kg_CO2/kWh']

        price_file = "data/citylearn_challenge_2022_phase_1/pricing.csv"
        self.price_df = pd.read_csv(price_file).to_dict("list")[
            "Electricity Pricing [$]"
        ]
