import numpy as np

def crps(y_true, y_pred):
    """
    Computes the Continuous Ranked Probability Score (CRPS).

    Args:
        y_true (array-like): True values of the target variable, of shape (n_samples,)
        y_pred (array-like): Predicted values of the target variable, of shape (n_samples, n_forecasts)

    Returns:
        float: CRPS score.
    """
    n_samples = y_true.shape[0]
    n_forecasts = y_pred.shape[1]

    # Compute the first term of the CRPS formula
    crps_term1 = 0
    for t in range(n_samples):
        for i in range(n_forecasts):
            crps_term1 += abs(y_pred[t, i] - y_true[t])

    crps_term1 /= n_samples * n_forecasts

    # Compute the second term of the CRPS formula
    crps_term2 = 0
    for t in range(n_samples):
        for i in range(n_forecasts):
            for j in range(n_forecasts):
                crps_term2 += abs(y_pred[t, i] - y_pred[t, j])

    crps_term2 /= 2 * n_samples * n_forecasts ** 2

    # Compute the CRPS score
    crps_score = crps_term1 - crps_term2

    return crps_term1, crps_term2, crps_score

def rank_bins(scens, reals):
    # for each lead time, find the bin that the real value is in
    bins = []
    # loop over steps ahead
    for step in scens['time_step'].unique():
        # loop over buildings
        for building in scens['building'].unique():
            temp = scens.loc[scens['time_step'] == step].loc[scens['building'] == building]
            for i in range(1, 25):
                build_col = 'building_{}'.format(building)
                real_value = reals.loc[reals['time_step'] == i+1, build_col].values[0]
                column_name = '+{}h'.format(i)
                bin_number = temp[column_name].sort_values().reset_index(drop=True).searchsorted(real_value)
                bins.append(bin_number)
    return bins


# Adapted to numpy from pyro.ops.stats.crps_empirical
# def crps(y_true, y_pred, sample_weight=None):
#     # transpose y_pred 
#     #y_pred = y_pred.T
#     num_samples = y_pred.shape[0]
#     absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

#     if num_samples == 1:
#         return np.average(absolute_error, weights=sample_weight)

#     y_pred = np.sort(y_pred, axis=0)
#     diff = y_pred[1:] - y_pred[:-1]
#     weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
#     weight = np.expand_dims(weight, -1)

#     per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
#     return np.average(per_obs_crps, weights=sample_weight), absolute_error, np.average(np.sum(diff * weight, axis=0) / num_samples**2)

def calculate_crps_scores(reals, scens):
    crps_scores = []
    abs_scores = []
    spread_scores = []
    for i in range(1, 25):
        crps_scores_B = []
        abs_scores_B = []
        spread_scores_B = []
        for build in range(5):
            obs_input = reals.loc[reals['building'] == build, ['net_target+' + str(i)]]
            scens_input = scens.loc[scens['building'] == build, ['scenario','+' + str(i) + 'h', 'time_step']]
            scens_input = scens_input.pivot(index='time_step',columns='scenario',  values='+' + str(i) + 'h')
            obs_input = obs_input.to_numpy()
            scens_input = scens_input.to_numpy()
            abs_diff, spread, score_B = crps(obs_input, scens_input)
            abs_scores_B.append(abs_diff)
            spread_scores_B.append(spread)
            crps_scores_B.append(score_B)
            print('CRPS score for building ' + str(build) + ' and horizon ' + str(i) + ' is ' + str(np.mean(crps_scores_B)))
        abs_scores.append(np.mean(abs_scores_B))
        spread_scores.append(np.mean(spread_scores_B))
        crps_scores.append(np.mean(crps_scores_B))
    return crps_scores, abs_scores, spread_scores


def rank_bins(scens, reals):
    # for each lead time, find the bin that the real value is in
    bins = []
    # loop over steps ahead
    for step in scens['time_step'].unique():
        # loop over buildings
        for building in scens['building'].unique():
            temp = scens.loc[scens['time_step'] == step].loc[scens['building'] == building]
            for i in range(1, 25):
                build_col = 'building_{}'.format(building)
                real_value = reals.loc[reals['time_step'] == i+1, build_col].values[0]
                column_name = '+{}h'.format(i)
                bin_number = temp[column_name].sort_values().reset_index(drop=True).searchsorted(real_value)
                bins.append(bin_number)
    return bins