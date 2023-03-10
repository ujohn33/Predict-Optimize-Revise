import gc
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
# import timedelta
from datetime import datetime, timedelta

TARGET = 'net_target+1'

#Function to define the forecast name from time
def get_time_label(time):
    if time.month<10: name_month= '0' + str(time.month)
    else: name_month = str(time.month)
    if time.day<10: name_day= '0' + str(time.day)
    else: name_day = str(time.day)
    if time.hour<10: name_hour= '0' + str(time.hour)
    else: name_hour = str(time.hour)
    if time.minute<10: name_minute= '0' + str(time.minute)
    else: name_minute = str(time.minute)
    time_name = '_'+str(time.year)+name_month+name_day+name_hour+name_minute
    return time_name

class expando:
    pass 

class modelEstimation:
    def __init__(self, data): 
        print('Splitting data for each lead time')
        self._split_times(data)
        print('Calculate correlation matrix')
        self._get_corr(data)

    def _split_times(self, data):
        self.uniform = expando()
        for hour in data.Hour.unique().astype(int):
            setattr(self.uniform, hour, expando())
            for leadT in range(1, 25):
                # get all the values in the data frame for the given lead time for the given hour
                reals = data.loc[data['Hour'] == hour, 'lead_time_'+str(leadT)]
                times = data.loc[data['Hour'] == hour, 'time_step']
                # reals = data.loc[data['Hour'] == leadT, TARGET]
                unif_aux = {}
                unif_aux['value'] = {}
                unif_aux['time'] = {}
                unif_aux['date'] = {}
                # create a dictionary with reals.index.values as keys
                unif_aux['t'] = dict(zip(range(len(reals)), times))
                for index in unif_aux['t'].keys():
                    unif_aux['value'][index] = reals.iloc[index]
                    unif_aux['time'][index] = unif_aux['t'][index] % 24
                    unif_aux['date'][index] = unif_aux['t'][index] // 24
                unif_aux = pd.DataFrame(unif_aux,columns=['t','value','time','date'])
                setattr(getattr(self.uniform, hour), str(leadT), unif_aux)
                del unif_aux, reals
                gc.collect()
        pass



    def _get_corr(self, data, horizon=24):
        #Next we estimate the correlation matrix for the uniform variables. To facilitate this the
        #uniform variables are put on an appropriate form for computing a correlation matrix. This
        #is done through using a pivot table      
        uniform_df = pd.DataFrame({'t': [], 'value': [], 'ltname': [],\
        'date': [], 'time': []})
        for leadT in data.Hour.unique().astype(int):
            for leadT in range(horizon):
                uniform_leadT = getattr(self.uniform, str(leadT))
                df_leadT_temp = pd.DataFrame({'t': uniform_leadT.t, \
                    'value': uniform_leadT.value, 'ltname': leadT, 'date': uniform_leadT.date, \
                    'time': uniform_leadT.time})
                uniform_df = pd.concat([uniform_df, df_leadT_temp], axis=0)
                del uniform_leadT, df_leadT_temp
        uniform_df['value']=uniform_df['value'].astype(float)
        uniform_pivot = uniform_df.pivot_table(index='date',columns=('ltname'),values='value')

        norm_df =  uniform_df
        norm_df['value'] = norm.ppf(uniform_df['value'])
        norm_pivot = norm_df.pivot_table(index='date',columns=('ltname'),values='value')

        #From the observations in the uniform domain we can now compute the correlation matrix. 
        #The correlation matrix specifies the Gaussian copula used for combining the different models. 
        #Where the computed correlation is NaN we set it to zero.
        correlation_matrix_na = norm_pivot.corr()
        where_are_NaNs = np.isnan(correlation_matrix_na)
        correlation_matrix = correlation_matrix_na
        correlation_matrix[where_are_NaNs] = 0.
        if not np.all(np.diag(correlation_matrix) == 1.):
            print('All diagonal values of correlation matrix are not 1!')
            np.fill_diagonal(correlation_matrix.values, 1.)
        print(correlation_matrix)
        self.corr = expando()
        self.corr.correlation_matrix = correlation_matrix
        self.corr.pivot_columns = uniform_pivot.columns
        pass

class Scenarios:
    def __init__(self, model, data, fore, nb_scenarios=10):
        self.nb_scenarios = nb_scenarios
        #self._get_covariance(model, data)
        self._get_scenarios(model, data, fore)

    def _get_covariance(self, model, data, horizon=24):
        self.correlation_matrix = pd.DataFrame(columns = model.corr.pivot_columns, 
                                               index = model.corr.pivot_columns)
        for leadT_ref in range(horizon):
            for leadT_loc in range(horizon):
                dleadT = abs(int(leadT_loc[6:]) - int(leadT_ref[6:]))
                self.correlation_matrix.loc[(leadT_ref),(leadT_loc)] = \
                model.corr.fit.combined.func(dleadT)
        self.correlation_matrix = self.correlation_matrix.astype(float)
        pass

    def _get_scenarios(self, model, data_test, fore, horizon=24):
        self.simulation = expando()      
        times_of_issue = pd.to_datetime(data_test.timestamp)
        for i_time_issue, time_issue in enumerate(times_of_issue):
            time_issue_name = get_time_label(time_issue)
            print('time_issue_name: ', time_issue_name)
            # get the data for the date of issue 
            setattr(self.simulation, time_issue_name, expando())
            t_actual = expando()
            for leadT in range(horizon):
                t_actual_temp = time_issue + timedelta(hours=int(leadT))
                setattr(t_actual, str(leadT), t_actual_temp)
            setattr(getattr(self.simulation, time_issue_name), 't_actual', t_actual)
            # put the forecast here instead of the means
            mean = fore[time_issue:(time_issue + timedelta(hours=24))][TARGET].values
            #First we simulate uniform variables with the appropriate interdependence structure. 
            #This is easily done by first simulating Gaussian varialbes with the same interdependence
            #sturcture and the transforming them to the uniform domain by their marginals.
            rv_mvnorm = multivariate_normal(mean, model.corr.correlation_matrix)
            simulation_mvnorm = rv_mvnorm.rvs(self.nb_scenarios)
            simulation_uniform = pd.DataFrame(data=norm.cdf(simulation_mvnorm), 
                                              columns = model.corr.pivot_columns.astype(str))
            #print(simulation_uniform)
            #Having obtained the simulated variables in the uniform domain, we need to get them into the transformed 
            #domain. This is done by using the inverse cummulative density function (inv_cdf) for each region and 
            #lead time. As the marginals depend on the predicted values, the predictions are required. 
            #Here the predictions that came with the data are used.
            #first we put the transformed predictions on the appropriate form. To do this we need a set of 
            #multi horizon point predictions spanning the locations considered and the prediction horizons.
            #Futher we need a starting time. In this implementation we simply choose a starting time from
            #the forecast data and choose the associated forecasts.
            scen_label = [None] * self.nb_scenarios
            for iscen in range(1, self.nb_scenarios+1):
                scen_label[iscen-1] = 'scan_' + str(iscen)
            scen_label.insert(0, 'forecasts') 
            #scen_label.insert(0, 'init_forecasts')
            self.scen_label = scen_label

            #We now create a dataframe with the transformed predictions
            simulation = pd.DataFrame(0, columns=data_test.Hour.unique(),
                                     index=scen_label)
            for leadT in data_test.Hour.unique().astype(int):
                for iscen in range(self.nb_scenarios):
                    #print(simulation_uniform.columns)quirt
                    simulation_transformed_temp = \
                    float((getattr(simulation_uniform, str(float(leadT)))[iscen]))
                    simulation.loc[scen_label[iscen+1], leadT] = simulation_transformed_temp
                # Save the modified input forecast
                setattr(getattr(self.simulation, date_issue_name), 'simulation', simulation)
            #print(simulation)

# if main
if __name__ == '__main__':
    # Input data directory
    data_train = pd.read_csv('./data/train_lead_times.csv', index_col=0, parse_dates=['timestamp'])
    data_test = pd.read_csv('./data/extra_test.csv', index_col=0, parse_dates=['timestamp'])
    # download the point forecast for the test data
    points = pd.read_csv('./data/point/test_fcst.csv', index_col=0, parse_dates=['timestamp'])
    # Calculate a covariance matrix on copula-transformed data
    model = modelEstimation(data_train)
    # Estimate the scenarios
    scs = Scenarios(model, data_test, points)
        # save the scenarios into a dataframe
    scs_df = pd.DataFrame()
    for i, timestamp in enumerate(data_test.timestamp):
        timestamp = pd.to_datetime(timestamp)
        time_name = get_time_label(timestamp)
        temp = getattr(getattr(scs.simulation, time_name), 'simulation').T.iloc[:, 1:]
        temp['real'] = data_test.loc[data_test.timestamp.dt.date == date.date(), :].net_target.reset_index(drop=True)
        temp['timestamp'] = data_test.loc[data_test.timestamp.dt.date == date.date(), :].timestamp.reset_index(drop=True)
        scs_df = pd.concat([scs_df, temp], axis=0)
    scs_df.to_csv('./data/scenarios_corr.csv')