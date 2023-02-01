import gc
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
# import timedelta
from datetime import datetime, timedelta


#cdf from conditional quantile regression
def cqr_cdf(row, cdf_keyword): 
    #prob = betas.loc[:,('probabilities')].values
    #print(row)
    prob = row.index.values
    quantiles = row.values
    #quantiles[quantiles < 0] = 0
    #quantiles[quantiles > 1] = 1
    quantiles_extended = np.concatenate([[0], sorted(quantiles), [1]])
    probabilities_extended = np.concatenate([[0],prob,[1]])
    if cdf_keyword == 'cdf':
        interpolation = interp1d(quantiles_extended, probabilities_extended, bounds_error=False, fill_value=(0, 1))
    elif cdf_keyword == 'inv_cdf':
        interpolation = interp1d(probabilities_extended, quantiles_extended, bounds_error=False, fill_value=(0, 1))
    return interpolation

class expando:
    pass 

class modelEstimation:
    def __init__(self, qs, data): 
        print('modelEstimation class initialized')
        print('Setting up quantiles')
        self._set_quantiles(qs)
        print('Cooking up the cdf distributions')
        self._set_cdf(qs)
        print('CDF transfrormation')
        self._apply_cdf(data)
        print('Calculate correlation matrix')
        self._get_corr(data)

    def _set_quantiles(self, data): 
        # find quantiles for each hour of the day
        self.betas = expando()
        # slice the data 
        for leadT in data.hour.unique():
            # slice the data
            quantiles = data.loc[data['hour'] == leadT, :].iloc[:, :-1].reset_index(drop=True)
            # set attr
            setattr(self.betas, str(leadT), quantiles)
            del quantiles
            gc.collect()

    def _set_cdf(self, data):
        self.cdf = expando()
        self.inv_cdf = expando()
        for leadT in data.hour.unique():
            cdf_loc_leadT = \
                lambda prediction, cdf_keyword='cdf': \
                cqr_cdf(prediction, cdf_keyword)
            setattr(self.cdf, str(leadT), cdf_loc_leadT)

            inv_cdf_loc_leadT = \
                lambda prediction, cdf_keyword='inv_cdf': \
                cqr_cdf(prediction, cdf_keyword)
            setattr(self.inv_cdf, str(leadT), inv_cdf_loc_leadT)
        pass


    def get_cdf(self, location, leadT):
        return getattr(getattr(self.cdf, location), leadT)

    def _apply_cdf(self, data):
        #Using the defined cummulative density function (cdf) we can now convert every observation 
        #into the uniform domain. This is done for every lead time.
        # it is assumed that the transformed data of the length of the forecasting horizon follows a gaussian distribution
        self.uniform = expando()
        for ileadT, leadT in enumerate(data.Hour.unique().astype(int)):
            cdf_loc_leadT = getattr(self.cdf, str(leadT))

            preds = getattr(self.betas, str(leadT))
            reals = data.loc[data['Hour'] == leadT, 'net_target']
            unif_aux = {}
            unif_aux['value'] = {}
            unif_aux['time'] = {}
            unif_aux['date'] = {}
            # create a dictionary with reals.index.values as keys
            unif_aux['t'] = dict(zip(range(len(reals)), reals.index))
            for index in unif_aux['t'].keys():
                conditional_cdf_loc_leadT = cdf_loc_leadT(preds.iloc[index])
                unif_aux['value'][index] = conditional_cdf_loc_leadT(reals.iloc[index])
                unif_aux['time'][index] = unif_aux['t'][index] % 24
                unif_aux['date'][index] = unif_aux['t'][index] // 24
                del conditional_cdf_loc_leadT
            unif_aux = pd.DataFrame(unif_aux,columns=['t','value','time','date'])
            setattr(self.uniform, str(leadT), unif_aux)
            del unif_aux, preds, reals
            gc.collect()
        pass

    def _get_corr(self, data):
        #Next we estimate the correlation matrix for the uniform variables. To facilitate this the
        #uniform variables are put on an appropriate form for computing a correlation matrix. This
        #is done through using a pivot table      
        uniform_df = pd.DataFrame({'t': [], 'value': [], 'ltname': [],\
        'date': [], 'time': []})
        for leadT in data.Hour.unique().astype(int):
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

