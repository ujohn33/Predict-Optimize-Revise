import numpy as np
import matplotlib.pyplot as plt

def crps(y_true, y_pred):
    """
    Computes the Continuous Ranked Probability Score (CRPS).

    Args:
        y_true (array-like): True values of the target variable, of shape (n_samples,)
        y_pred (array-like): Predicted values of the target variable, of shape (n_samples, n_forecasts)

    Returns:
        float: CRPS score.
    """
    # Ensure y_pred is a numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute absolute differences between forecasts and true values
    abs_diff_obs = np.abs(y_pred - y_true[:, None])
    crps_term1 = np.mean(abs_diff_obs)

    # Compute absolute differences among all pairs of forecasts
    abs_diff_members = np.abs(y_pred[:, :, None] - y_pred[:, None, :])
    crps_term2 = np.mean(abs_diff_members) / 2

    # Compute the CRPS score
    crps_score = crps_term1 - crps_term2

    return crps_term1, crps_term2, crps_score


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
    num_buildings = reals['building'].nunique()
    for i in range(24):
        crps_scores_B = []
        abs_scores_B = []
        spread_scores_B = []
        for build in range(num_buildings):
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
            temp = scens.loc[(scens['time_step'] == step) & (scens['building'] == building)]
            for i in range(24):
                real_value = reals.loc[(reals['time_step'] == step) & (reals['building'] == building)]
                column_name = '+{}h'.format(i)
                net_target_col_name = 'net_target+{}'.format(i)
                bin_number = temp[column_name].sort_values().reset_index(drop=True).searchsorted(real_value[net_target_col_name].values[0])
                bins.append(bin_number)
    return bins

# def rank_bins(scens, reals):
#     reals_grouped = reals.reset_index(drop=True).groupby(['time_step', 'building'])
#     scens_grouped = scens.reset_index(drop=True).groupby(['time_step', 'building'])
#     # loop over steps ahead
#     bins = []
#     for i in range(24):
#         column_name = '+{}h'.format(i)
#         net_target_col_name = 'net_target+{}'.format(i)
#         # get the real values for this step ahead
#         real_values = reals_grouped[net_target_col_name].first().astype(float).values
#         # get the corresponding scenario values for this step ahead
#         scenario_values = scens_grouped[column_name].apply(np.sort).values
#         # find the bin number for each real value
#         bin_numbers = np.apply_along_axis(lambda x: np.digitize(real_values, x) - 1, axis=1, arr=np.vstack(scenario_values))
#         bins.extend(bin_numbers)
#     return bins

def rank_bins_metric(bins_list):
    # find the density of each bin
    # Calculate the histogram and bin edges
    hist, bin_edges = np.histogram(bins_list, bins='auto', density=True)
    # Calculate the bin widths
    bin_widths = np.diff(bin_edges)
    # Calculate the probability densities for each bin
    probability_densities = hist * bin_widths
    # calculate std deviation of probability densities
    std = np.std(probability_densities)
    return std


def plot_bins(bins_list):
    # number of unique bins
    n_bins = len(np.unique(bins_list))
    # plot bin frequencies in per unit for bins in probability per unit
    bin_counts, bin_edges = np.histogram(bins_list, bins='auto')
    bin_frequencies = bin_counts / len(bins_list)
    # calculate bin width
    bin_width = (max(bins_list) - min(bins_list)) / n_bins
    plt.bar(bin_edges[:-1], bin_frequencies, width=bin_width, color='grey', edgecolor='black')
    # make it pretty
    plt.title('Rank histogram: Calibration')
    plt.xlabel('Bin Number')
    plt.ylabel('Frequency [p.u.]')
    # set style to b&w
    plt.style.use('seaborn')
    # mark the probability of the real value falling in the bin
    plt.axhline(y=1/n_bins, color='black', linestyle='--', linewidth=2, label='Flat rank', zorder=3)
    plt.show()

def mean_ensemble_error(scens, reals):
    # calculate the mean distance from the mean of the ensemble members
    # make a dict of lists to store the mean error for each step ahead
    mean_error = {}
    # loop over steps ahead
    for i in range(24):
        column_name = '+{}h'.format(i)
        temp = scens.reset_index(drop=True).groupby(['time_step', 'building'], group_keys=True)[column_name].mean()
        temp_real = reals.reset_index(drop=True).groupby(['time_step', 'building'], group_keys=True)['net_target+{}'.format(i)].first()
        mean_error[i] = np.mean(temp - temp_real)
    return mean_error

# def mean_ensemble_error(scens):
#     # calculate the mean distance from the mean of the ensemble members
#     # make a dict of lists to store the mean error for each step ahead
#     mean_error = {}
#     # loop over steps ahead
#     for i in range(24):
#         column_name = '+{}h'.format(i)
#         mean_error[i] = []
#         for step in scens['time_step'].unique():
#             # loop over buildings
#             for building in scens['building'].unique():
#                 temp = scens.loc[(scens['time_step'] == step) & (scens['building'] == building)]
#                 mean_error[i].append(np.mean(temp[column_name] - temp[column_name].mean()))
#     return mean_error

# make a function that takes same inputs as mean_ensemble_error and returns the mean of its return value
def discrete_mean_ensemble_error(scens, reals):
    return np.mean(list(mean_ensemble_error(scens, reals).values()))

def discrete_ensemble_spread(scens):
    return np.mean(list(ensemble_std(scens).values()))

def ensemble_std(scens):
    # calculate the spread for each lead time
    spread = {} 
    for lead in range(24):
        spread[lead] = []
        column_name = '+{}h'.format(lead)
        # loop over buildings
        for scen_num in scens['scenario'].unique():
            for building in scens['building'].unique():
                temp = scens.loc[(scens['scenario'] == scen_num)&(scens['building'] == building)]
                spread[lead].append(temp[column_name].std())
        spread[lead] = np.mean(spread[lead])
    return spread

def equal_likelihood(scens, control):
    # a bar plot showing the number of cases when each scenario was the forecast closest to the control
    equal_likelihood_freq = {}
    for lead in range(24):
        equal_likelihood_freq[lead] = {}
        # loop over steps ahead
        for step in scens['time_step'].unique():
            # loop over buildings
            for building in scens['building'].unique():
                temp = scens.loc[(scens['time_step'] == step) & (scens['building'] == building)]
                temp = temp.set_index('scenario', drop=False)
                target_col = 'net_target+{}'.format(building)
                column_name = '+{}h'.format(lead)
                # find the scenario that is closest to the control
                closest = temp[column_name].sub(control.loc[(control['time_step'] == step) & (control['building'] == building), target_col].values[0]).abs().idxmin()
                equal_likelihood_freq[lead][closest] = equal_likelihood_freq[lead].get(closest, 0) + 1
        # find the frequency in percentage for each lead time
        # by dividing equal_likelihood_freq[lead][closest] by the total count over all scenarios
        total = sum(equal_likelihood_freq[lead].values())
        for key in equal_likelihood_freq[lead]:
            equal_likelihood_freq[lead][key] = equal_likelihood_freq[lead][key] / total
    return equal_likelihood_freq

def plot_equal_likelihood(equal_likelihood_freq):
    # plot the equal likelihood frequency
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for lead in equal_likelihood_freq.keys():
        ax.bar(lead, equal_likelihood_freq[lead][0], color='blue', alpha=0.5)
        ax.bar(lead, equal_likelihood_freq[lead][1], color='red', alpha=0.5)
        ax.bar(lead, equal_likelihood_freq[lead][2], color='green', alpha=0.5)
        ax.bar(lead, equal_likelihood_freq[lead][3], color='orange', alpha=0.5)
        ax.bar(lead, equal_likelihood_freq[lead][4], color='purple', alpha=0.5)
    ax.set_xlabel('Lead time (hours)')
    ax.set_ylabel('Frequency')
    ax.set_title('Equal likelihood frequency')
    ax.legend(['Scenario 0', 'Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'])
    plt.show()

def flatness_equal_likelihood(equal_likelihood_freq):
    # calculate the metric for the equal likelihood frequency as the standard deviation from equal likelihood for each scneario
    metric = {}
    for lead in equal_likelihood_freq.keys():
        for scenario in equal_likelihood_freq[lead].keys():
            metric[lead] = metric.get(lead, 0) + (equal_likelihood_freq[lead][scenario] - 0.2)**2
        # take an average over all scenarios
        metric[lead] = np.sqrt(metric[lead] / len(equal_likelihood_freq[lead].keys()))
    # take an average over all lead times
    metric = np.mean(list(metric.values()))
    return metric