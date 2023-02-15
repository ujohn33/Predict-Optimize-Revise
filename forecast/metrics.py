import numpy as np

def crps(forecasts, actuals):
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    n_timesteps, n_scenarios = forecasts.shape
    crps = 0.0
    for t in range(n_timesteps):
        forecast_cdf = np.sort(forecasts[t, :])
        actual = actuals[t]
        delta = forecast_cdf - actual
        pos = np.where(delta > 0, delta, 0)
        neg = np.where(delta < 0, delta, 0)
        crps += np.mean(0.5 * pos ** 2 - 0.5 * neg ** 2)
    return crps
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for a set of forecasted scenarios and actual values.

    Parameters
    ----------
    forecasts : list of lists
        A 2D list of forecasted scenarios, where each row represents a time step and each column represents a different scenario.
    actuals : list
        A 1D list of actual values corresponding to each time step in the forecasts.

    Returns
    -------
    float
        The CRPS score, which measures the difference between the predicted cumulative distribution function (CDF) and the actual CDF of the target variable.

    Examples
    --------
    >>> forecasts = [[0.1, 0.3, 0.2], [0.4, 0.6, 0.5]]
    >>> actuals = [0.15, 0.55]
    >>> crps(forecasts, actuals)
    0.135
    """
