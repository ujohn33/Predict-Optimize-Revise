import numpy as np

def CRPS_ensemble(obs,fc,fair=True,axis=0):
    """
    @author: Ole Wulff
    @date: 2020-07-08
    implementation of fair (adjusted) CRPS based on equation (6) from Leutbecher (2018, QJRMS, https://doi.org/10.1002/qj.3387)
    version with fair=False tested against properscoring implementation crps_ensemble (see https://pypi.org/project/properscoring/)
    INPUT:
        obs: observations as n-dimensional array
        fc: forecast ensemble as (n+1)-dimensional where the extra dimension (axis) carries the ensemble members
        fair: if True returns the fair version of the CRPS accounting for the limited ensemble size (see Leutbecher, 2018)
              if False returns the normal CRPS
        axis: axis of fc array that contains the ensemble members, defaults to 0
    OUTPUT:
        CRPS: n-dimensional array
    TODO:
        implement weights for ensemble member weighting
    """
    odims = obs.shape
    M = fc.shape[axis]
    if axis != 0:
        fc = np.swapaxes(fc,axis,0)

    # flatten all dimensions except for the ensemble member dimension:
    fc_flat = fc.reshape([M,-1])
    obs_flat = obs.reshape([-1])

    dsum = np.array([abs(fc_flat[jj] - fc_flat[kk]) for kk in range(M) for jj in range(M)]).sum(axis=axis)
    if fair:
        CRPS = 1/M * (abs(fc_flat - obs_flat)).sum(axis=axis) - 1/(2*M*(M-1)) * dsum
    else:
        CRPS = 1/M * (abs(fc_flat - obs_flat)).sum(axis=axis) - 1/(2*M**2) * dsum

    # is this necessary or even a good idea at all?
#     del dsum, fc_flat, obs_flat

    return CRPS.reshape([*odims])

def crps_ensemble(obs,fc,fair=True):
    """
    A xarray wrapper for CRPS_ensemble()
    """

    # obs = obs.broadcast_like(fc.mean('member'))
    obs,fc = xr.align(obs,fc,join='outer')
    print(obs)

    return xr.apply_ufunc(
            CRPS_ensemble, obs, fc,
            input_core_dims  = [['time'],['member','time']],
            output_core_dims = [['time']],
            vectorize=True
        )