import pandas as pd
import numpy as np
from forecast.metrics import *
# plot CRPS scores for each horizon
import matplotlib.pyplot as plt
import itertools
import climpred
from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.metrics import Metric
import xarray as xr
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

dimType = Optional[Union[str, List[str]]]
metric_kwargsType = Any

def __compute_anomalies(a, b, weights, axis, skipna):
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    # Only do weighted sums if there are weights. Cannot have a
    # single generic function with weights of all ones, because
    # the denominator gets inflated when there are masked regions.
    if weights is not None:
        with suppress_warnings("invalid value encountered in true_divide"):
            ma = sumfunc(a * weights, axis=axis) / sumfunc(weights, axis=axis)
            mb = sumfunc(b * weights, axis=axis) / sumfunc(weights, axis=axis)
    else:
        with suppress_warnings("Mean of empty slice"):
            ma = meanfunc(a, axis=axis)
            mb = meanfunc(b, axis=axis)
    am, bm = a - ma, b - mb
    return am, bm

# smoothness
def calculate_cv_standard_variation(
    forecast: xr.Dataset,
    verif: xr.Dataset,
    dim: dimType = None,
    **metric_kwargs: metric_kwargsType,
) -> xr.Dataset:
    fm = __compute_anomalies(forecast, verif, weights=None, axis=dim, **metric_kwargs)
    diff = am[0:-1] - am[1::]
    smoothness = np.std(diff) - abs(np.mean(diff))
    return smoothness

# if main
if __name__ == "__main__":
    # open csv file for scenarios
    scens = pd.read_csv('debug_logs/scenarios_together+naive_9000_1.csv')
    scens = scens.set_index(['time_step', 'building'], drop=False)
    # remove rows with time_step 8759
    scens = scens[scens['time_step'] != 8759]
    # keep only indexes with time steps 0-8735
    scens = scens.loc[(scens['time_step'] >= 0) & (scens['time_step'] <= 8735)]
    # rename time_step to init
    scens = scens.rename(columns={'time_step': 'init', 'scenario': 'member'})

    reals = pd.read_csv('debug_logs/real_power_together+naive_9000_1.csv')
    # rename time_step to time
    reals = reals.rename(columns={'time_step': 'time'})
    reals = reals.iloc[1:, :]
    reals = reals.melt(id_vars=['time'], var_name='building', value_name='net')
    reals = reals.sort_values(by=['time', 'building'])
    reals['building'] = reals['building'].str[-1].astype(int)
    reals = reals[reals['time'] != -1]

    # take unique scenario numbers if scens and get a list of lists with unique combinations of 5 that exist
    scens_unique = scens['member'].unique()
    scens_unique = scens_unique.tolist()
    # get combinations of 5
    scens_combinations = list(itertools.combinations(scens_unique, 5))
    temp_scens = scens.loc[scens['member'].isin(scens_combinations[0])]
    temp_xrds = temp_scens.set_index(['init', 'member', 'building', 'lead']).to_xarray()
    temp_xrds['lead'].attrs['units'] = 'years'
    temp_xobs = reals.set_index(['time', 'building']).to_xarray()
    temp_ens = climpred.HindcastEnsemble(temp_xrds)
    temp_ens = temp_ens.add_observations(temp_xobs)

    xarray_dict = {}
    metric_dict = {}

    metrics = ['mae', 'rmse', 'acc', 'nrmse', 'nmae', 'smape', 'crps', 'crpss', 'crpss_es', 'threshold_brier_score', 'less']
    # start a csv with metrics as columns
    metric_file = open('metrics.csv', 'w+')
    # metrics as columns
    index = ['unique_id']
    line = metrics
    metric_file.write(','.join(index + line) + '\n')
    for combo in scens_combinations:
        # write the index of the scenario to the csv
        line_start = ''.join(str(e) for e in combo)
        metric_file.write(line_start + ',')
        for metric in metrics:
            xarray_dict[metric] = temp_ens.verify(metric=metric, comparison='m2o', dim=['member','init', 'building'], alignment='same_inits').assign_coords(metric=metric.upper())
            metric_dict[metric] = float(xarray_dict[metric]['net'].mean().values)
            metric_file.write(str(metric_dict[metric]) + ',')
        metric_file.write('\n')
    metric_file.close()