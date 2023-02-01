import gc
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
# import timedelta
from datetime import datetime, timedelta
# import copula_model
from copula_model import modelEstimation

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

class Scenarios:
    def __init__(self, model, qs, data, nb_scenarios=10):
        self.nb_scenarios = nb_scenarios
        self._set_quantiles(qs)
        #self._get_covariance(model, data)
        self._get_scenarios(model, data, qs)

    def _set_quantiles(self, data): 
        # find quantiles for each hour of the day
        self.betas = expando()
        # slice the data 
        for leadT in data.hour.unique().astype(int):
            # slice the data
            quantiles = data.loc[data['hour'] == leadT, :].iloc[:, :-1].reset_index(drop=True)
            # set attr
            setattr(self.betas, str(leadT), quantiles)
            del quantiles
            gc.collect()

    def _get_covariance(self, model, data):
        self.correlation_matrix = pd.DataFrame(columns = model.corr.pivot_columns, 
                                               index = model.corr.pivot_columns)
        for leadT_ref in data.hour.unique():
            for leadT_loc in data.hour.unique():
                dleadT = abs(int(leadT_loc[6:]) - int(leadT_ref[6:]))
                self.correlation_matrix.loc[(leadT_ref),(leadT_loc)] = \
                model.corr.fit.combined.func(dleadT)
        self.correlation_matrix = self.correlation_matrix.astype(float)
        pass

    def _get_scenarios(self, model, data_test, fore):
        self.simulation = expando()      
        dates_of_issue = pd.to_datetime(data_test.timestamp.dt.date.unique())
        for i_date_issue, date_issue in enumerate(dates_of_issue):
            date_issue_name = get_time_label(date_issue)
            print('date_issue_name: ', date_issue_name)
            # get the data for the date of issue
            setattr(self.simulation, date_issue_name, expando())
            t_actual = expando()
            for ileadT, leadT in enumerate(data_test.Hour.unique()):
                t_actual_temp = date_issue + timedelta(hours=int(leadT))
                setattr(t_actual, str(leadT), t_actual_temp)
            setattr(getattr(self.simulation, date_issue_name), 't_actual', t_actual)
            mean = np.zeros(model.corr.correlation_matrix.shape[1])
            #First we simulate uniform variables with the appropriate interdependence structure. 
            #This is easily done by first simulating Gaussian varialbes with the same interdependence
            #sturcture and the transforming them to the uniform domain by their marginals.
            rv_mvnorm = multivariate_normal(mean, model.corr.correlation_matrix)
            simulation_mvnorm = rv_mvnorm.rvs(self.nb_scenarios)
            simulation_uniform = pd.DataFrame(data=norm.cdf(simulation_mvnorm), 
                                              columns = model.corr.pivot_columns.astype(str))
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
                predict_simulation = getattr(self.betas, str(leadT)).loc[i_date_issue]
                # make the transformation using the inverse cdf
                inv_cdf = getattr(model.inv_cdf, str(leadT))
                conditional_inv_cdf = inv_cdf(predict_simulation)
                for iscen in range(self.nb_scenarios):
                    #print(simulation_uniform.columns)quirt
                    simulation_transformed_temp = \
                    float(conditional_inv_cdf(getattr(simulation_uniform, str(float(leadT)))[iscen]))
                    simulation.loc[scen_label[iscen+1], leadT] = simulation_transformed_temp
                # Save the modified input forecast
                setattr(getattr(self.simulation, date_issue_name), 'simulation', simulation)
            print(simulation)

# if main
if __name__ == '__main__':
    input_dir = './data/quantile/'
    output = pd.read_csv(input_dir+'./year_qs.csv', index_col=0)
    # change all but last column names to float
    output.columns = [float(x) if x != 'hour' else x for x in output.columns]
    output_test = pd.read_csv(input_dir+'./year_qs_test.csv', index_col=0)
    # change all but last column names to float
    output_test.columns = [float(x) if x != 'hour' else x for x in output_test.columns]
    ID_test = 4644
    # Input data directory
    data_train = pd.read_csv('./data/extra_train.csv', index_col=0, parse_dates=['timestamp'])
    data_test = pd.read_csv('./data/extra_test.csv', index_col=0, parse_dates=['timestamp'])
    # Calculate a covariance matrix on copula-transformed data
    model = modelEstimation(output, data_train)
    # Estimate the scenarios
    scs = Scenarios(model, output_test, data_test)
    # save the scenarios into a dataframe
    scs_df = pd.DataFrame()
    for i, date in enumerate(data_test.timestamp.dt.date.unique()):
        date = pd.to_datetime(date)
        date_name = get_time_label(date)
        temp = getattr(getattr(scs.simulation, date_name), 'simulation').T.iloc[:, 1:]
        temp['real'] = data_test.loc[data_test.timestamp.dt.date == date.date(), :].net_target.reset_index(drop=True)
        temp['timestamp'] = data_test.loc[data_test.timestamp.dt.date == date.date(), :].timestamp.reset_index(drop=True)
        scs_df = pd.concat([scs_df, temp], axis=0)
    scs_df.to_csv('./data/scenarios.csv')


