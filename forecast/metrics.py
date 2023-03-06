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