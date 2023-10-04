import pandas as pd
import numpy as np
import os
import warnings
import copy
from utils.util_functions import reduce_mem_usage

warnings.simplefilter('ignore')

def init_test_data(mode):
    print("Init test data")
    path_to_bu1 = "data/citylearn_challenge_2022_phase_1/Building_1.csv"
    path_to_bu = path_to_bu1
    path_to_weather = "data/citylearn_challenge_2022_phase_1/weather.csv"
    df_bu, df_we = pd.read_csv(path_to_bu), pd.read_csv(path_to_weather)
    df = pd.merge(df_bu, df_we, left_index=True, right_index=True)

    cols = ['Month', 'Hour', 'Day_Type', 'Daylight_Savings_Status', 'Indoor_Temperature',
            'Average_Unmet_Cooling_Setpoint_Difference',
            'Indoor_Relative_Humidity',
            'Equipment_Electric_Power',
            'DHW_Heating',
            'Cooling_Load',
            'Heating_Load',
            'Solar_Generation',
            'Outdoor_Drybulb_Temperature',
            'Relative_Humidity',
            'Diffuse_Solar_Radiation',
            'Direct_Solar_Radiation',
            '6h_Prediction_Outdoor_Drybulb_Temperature',
            '12h_Prediction_Outdoor_Drybulb_Temperature',
            '24h_Prediction_Outdoor_Drybulb_Temperature',
            '6h_Prediction_Relative_Humidity',
            '12h_Prediction_Relative_Humidity',
            '24h_Prediction_Relative_Humidity',
            '6h_Prediction_Diffuse_Solar_Radiation',
            '12h_Prediction_Diffuse_Solar_Radiation',
            '24h_Prediction_Diffuse_Solar_Radiation',
            '6h_Prediction_Direct_Solar_Radiation',
            '12h_Prediction_Direct_Solar_Radiation',
            '24h_Prediction_Direct_Solar_Radiation',
            ]
    df.columns = cols

    selected_cols = ['Month', 'Hour', 'Day_Type',
                     'Equipment_Electric_Power',
                     'Solar_Generation',
                     'Outdoor_Drybulb_Temperature',
                     'Relative_Humidity',
                     'Diffuse_Solar_Radiation',
                     'Direct_Solar_Radiation',
                     ]
    df = df[selected_cols]

    df['Hour'] = df.Hour % 24
    df["hour_x"] = np.cos(2*np.pi* df["Hour"] / 24)
    df["hour_y"] = np.sin(2*np.pi* df["Hour"] / 24)
    
    df["month_x"] = np.cos(2*np.pi* df["Month"] / (12))
    df["month_y"] = np.sin(2*np.pi*df["Month"] / (12))

    df["weekday_x"] = np.cos(2*np.pi* df["Day_Type"] / (7))
    df["weekday_y"] = np.sin(2*np.pi*df["Day_Type"] / (7))
    # drop columns Hour and Month
    df.drop(columns=['Hour', 'Month', 'Day_Type'], inplace=True)

    N = 24
    for i in range(N):
        df['Outdoor_Drybulb_Temperature_{}'.format(i)] = df['Outdoor_Drybulb_Temperature'].shift(-i - 1)
        df['Relative_Humidity_{}'.format(i)] = df['Relative_Humidity'].shift(-i - 1)
        df['Diffuse_Solar_Radiation_{}'.format(i)] = df['Diffuse_Solar_Radiation'].shift(-i - 1)
        df['Direct_Solar_Radiation_{}'.format(i)] = df['Direct_Solar_Radiation'].shift(-i - 1)
    for i in range(int(N * 1.25)):
        if mode == 'load':
            df['Load_Past_{}'.format(i)] = df['Equipment_Electric_Power'].shift(i+1)
        elif mode == 'solar':
            df['Solar_Past_{}'.format(i)] = df['Solar_Generation'].shift(i+1)

    df.drop(columns=['Equipment_Electric_Power', 'Solar_Generation'], inplace=True)
    print('init df shape:', df.shape)

    print('init df shape:', df.shape)
    df = reduce_mem_usage(df)
    # # drop rows with nan values
    # df_drop = df.dropna(inplace=False)
    targets = [item for item in df.columns if 'Net_Future_' in item]
    x_train = df.drop(targets, axis=1)
    return x_train
