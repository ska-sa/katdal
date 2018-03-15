# flake8: noqa

################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Function to apply TelescopeState calibration to visibilities.

Hold, interpolate and apply cal solutions from a katdal object.
Code largely pilfered from the katdal_loader in katsdpimager, but using a katdal object
rather than a filename, so that things are a bit more portable. """

import numpy as np
import scipy.interpolate

try:
    import cPickle as pickle
except ImportError:
    import pickle

from .lazy_indexer import LazyTransform


class CalibrationReadError(RuntimeError):
    """An error occurred in loading calibration values from file"""
    pass


class ComplexInterpolate1D(object):
    """Interpolator that separates magnitude and phase of complex values.

    The phase interpolation is done by first linearly interpolating the
    complex values, then normalising. This is not perfect because the angular
    velocity changes (slower at the ends and faster in the middle), but it
    avoids the loss of amplitude that occurs without normalisation.

    The parameters are the same as for :func:`scipy.interpolate.interp1d`,
    except that fill values other than nan and "extrapolate" should not be
    used.
    """
    def __init__(self, x, y, *args, **kwargs):
        mag = np.abs(y)
        phase = y / mag
        self._mag = scipy.interpolate.interp1d(x, mag, *args, **kwargs)
        self._phase = scipy.interpolate.interp1d(x, phase, *args, **kwargs)

    def __call__(self, x):
        mag = self._mag(x)
        phase = self._phase(x)
        return phase / np.abs(phase) * mag


def interpolate_nans_1d(y, *args, **kwargs):
    if np.isnan(y).all():
        return y  # nothing to do , all nan values
    if np.isfinite(y).all():
        return y  # nothing to do, all values valid

    # interpolate across nans (but you will loose the first and last values)
    nan_locs = np.isnan(y)
    nan_perc = float(nan_locs.sum()) / float(y.size)
    # if nan_perc > 0.1: # conservative values
    # if nan_perc > 1.:  # original values
    if nan_perc > 0.49:  # original values
        y[:] = np.nan
    else:
        X = np.nonzero(~nan_locs)[0]
        Y = y[X]
        f = ComplexInterpolate1D(X, Y, *args, **kwargs)
        y = f(range(len(y)))
    return y


def _get_cal_attr(key, katdal_obj, sensor=True):
    """Load a fixed attribute from file.
    If the attribute is presented as a sensor, it is checked to ensure that
    all the values are the same.
    Raises
    ------
    CalibrationReadError
        if there was a problem reading the value from file (sensor does not exist,
        does not unpickle correctly, inconsistent values etc)
    """
    try:
        value = katdal_obj.file['TelescopeState/{}'.format(key)]['value']
        if len(value) == 0:
            raise ValueError('empty sensor')
        value = [pickle.loads(x) for x in value]
    except (NameError, SyntaxError):
        raise
    except Exception as e:
        raise CalibrationReadError('Could not read {}: {}'.format(key, e))

    if not sensor:
        timestamps = katdal_obj.file['TelescopeState/{}'.format(key)]['timestamp']
        return timestamps, value

    if not all(np.array_equal(value[0], x) for x in value):
        raise CalibrationReadError('Could not read {}: inconsistent values'.format(key))
    return value[0]


def _get_cal_antlist(katdal_obj):
    """Load antenna list used for calibration.
    If the value does not match the antenna list in the katdal dataset,
    a :exc:`CalibrationReadError` is raised. Eventually this could be
    extended to allow for an antenna list that doesn't match by permuting
    the calibration solutions.
    """
    cal_antlist = _get_cal_attr('cal_antlist', katdal_obj)
    if cal_antlist != [ant.name for ant in katdal_obj.ants]:
        raise CalibrationReadError('cal_antlist does not match katdal antenna list')
    return cal_antlist


def _get_cal_pol_ordering(katdal_obj):
    """Load polarization ordering used by calibration solutions.

    Returns
    -------
    dict
        Keys are 'h' and 'v' and values are 0 and 1, in some order
    """
    cal_pol_ordering = _get_cal_attr('cal_pol_ordering', katdal_obj)
    try:
        cal_pol_ordering = np.array(cal_pol_ordering)
    except (NameError, SyntaxError):
        raise
    except Exception as e:
        raise CalibrationReadError(str(e))
    if cal_pol_ordering.shape != (4, 2):
        raise CalibrationReadError('cal_pol_ordering does not have expected shape')
    if cal_pol_ordering[0, 0] != cal_pol_ordering[0, 1]:
        raise CalibrationReadError('cal_pol_ordering[0] is not consistent')
    if cal_pol_ordering[1, 0] != cal_pol_ordering[1, 1]:
        raise CalibrationReadError('cal_pol_ordering[1] is not consistent')
    order = [cal_pol_ordering[0, 0], cal_pol_ordering[1, 0]]
    if set(order) != set('vh'):
        raise CalibrationReadError('cal_pol_ordering does not contain h and v')
    return {order[0]: 0, order[1]: 1}


def _get_cal_product(key, katdal_obj, **kwargs):
    """Loads calibration solutions from a katdal file.

    If an error occurs while loading the data, a warning is printed and the
    return value is ``None``. Any keyword args are passed to
    :func:`scipy.interpolate.interp1d` or `ComplexInterpolate1D`.

    Solutions that contain non-finite values are discarded.

    Parameters
    ----------
    key : str
        Name of the telescope state sensor

    Returns
    -------
    interp : callable
        Interpolation function which accepts timestamps and returns
        interpolated data with shape (time, channel, pol, antenna). If the
        solution is channel-independent, that axis will be present with
        size 1.
    """
    timestamps, values = _get_cal_attr(key, katdal_obj, sensor=False)
    values = np.asarray(values)

    if (timestamps[-1] < katdal_obj._timestamps[0]):
        print('All %s calibration solution ahead of observation, no overlap' % key)
    elif (timestamps[0] > katdal_obj._timestamps[-1]):
        print('All %s calibration solution after observation, no overlap' % key)

    if values.ndim == 3:
        # Insert a channel axis
        values = values[:, np.newaxis, ...]

    if values.ndim != 4:
        raise ValueError('Calibration solutions has wrong number of dimensions')

    # only use solutions with valid values, assuming matrix dimensions
    # (ts, chan, pol, ant)
    # - all values per antenna must be valid
    # - all values per polarisation must be valid
    # - some channels must be valid (you will interpolate over them later)
    ts_mask = np.isfinite(values).all(axis=-1).all(axis=-1).any(axis=-1)
    if (ts_mask.sum() / float(len(timestamps))) < 0.7:
        raise ValueError('no finite solutions')
    values = values[ts_mask, ...]
    timestamps = timestamps[ts_mask]

    if values.shape[1] > 1:
        # Only use channels selected in h5 file
        values = values[:, katdal_obj.channels, ...]
        for ts_idx in range(values.shape[0]):
            # Interpolate across nans in channel axis.
            for pol_idx in range(values.shape[2]):
                for ant_idx in range(values.shape[3]):
                    values[ts_idx, :, pol_idx, ant_idx] = interpolate_nans_1d(
                                                                              values[ts_idx, :, pol_idx, ant_idx],
                                                                              kind='linear',
                                                                              fill_value='extrapolate',
                                                                              assume_sorted=True,
                                                                             )

    # to interpolate you need >1 timestamps, else return only the single value
    # if values.shape[0] > 1:
    if timestamps.size > 1:
        kind = kwargs.get('kind', 'linear')  # default if none given
        if np.iscomplexobj(values) and kind not in ['zero', 'nearest']:
            interp = ComplexInterpolate1D
        else:
            interp = scipy.interpolate.interp1d
        return interp(
                      timestamps,
                      values,
                      axis=0,
                      fill_value='extrapolate',
                      assume_sorted=True,
                      **kwargs)
    return values

def _cal_setup(katdal_obj):
    katdal_obj._cal_pol_ordering = _get_cal_pol_ordering(katdal_obj)
    katdal_obj._cal_ant_ordering = _get_cal_antlist(katdal_obj)
    katdal_obj._data_channel_freqs = katdal_obj.channel_freqs
    katdal_obj._delay_to_phase = (-2j * np.pi * katdal_obj._data_channel_freqs)[np.newaxis, :, np.newaxis, np.newaxis]
    # baselines from the data files are given as <ant><pol>
    # for indexing this has to be swapped to <pol><ant>
    katdal_obj._cp_lookup = [[(katdal_obj._cal_pol_ordering[prod[0][-1]], katdal_obj._cal_ant_ordering.index(prod[0][:-1]),),
                              (katdal_obj._cal_pol_ordering[prod[1][-1]], katdal_obj._cal_ant_ordering.index(prod[1][:-1]),)]
                             for prod in katdal_obj.corr_products]
    # katdal_obj._cp_lookup = [[(katdal_obj._cal_ant_ordering.index(prod[0][:-1]), katdal_obj._cal_pol_ordering[prod[0][-1]],),
    #                           (katdal_obj._cal_ant_ordering.index(prod[1][:-1]), katdal_obj._cal_pol_ordering[prod[1][-1]],)]
    #                          for prod in katdal_obj.corr_products]
    katdal_obj._cp_lookup = np.asarray(katdal_obj._cp_lookup, dtype=int)
    # katdal_obj._cp_lookup[:, :,[0, 1]] = katdal_obj._cp_lookup[:, :,[1, 0]]

    for key in katdal_obj._cal_solns.keys():
        try:
            katdal_obj._cal_solns[key]['interp'] = _get_cal_product(
                                                                    'cal_product_' + key,
                                                                    katdal_obj,
                                                                    kind=katdal_obj._cal_solns[key]['kind'],
                                                                   )
        except CalibrationReadError:
            raise
        except:  # no delay cal solution from telstate
            # should raise a warning
            raise
    return katdal_obj


def applycal(katdal_obj):
    """
    Apply the K, B, G solutions to visibilities at provided timestamps.
    Optionally recompute the weights as well.

    Returns
    =======
    katdal_obj: containing vis and weights (optional) with cal solns applied.
    """
    initcal = lambda kind: {'interp': None, 'solns': None, 'kind': kind}
    katdal_obj._cal_solns = {
                             'K': initcal('linear'),
                             'B': initcal('zero'),
                             'G': initcal('linear'),
                            }
    katdal_obj._cal_coeffs = None

    def _cal_interp(timestamps):
        # Interpolate the calibration solutions for the selected range
        for key in katdal_obj._cal_solns.keys():
            if katdal_obj._cal_solns[key]['interp'] is not None:
                # interpolate over values, or repeat single value
                if hasattr(katdal_obj._cal_solns[key]['interp'], '__call__'):
                    solns = katdal_obj._cal_solns[key]['interp'](timestamps)
                else:
                    solns = np.repeat(katdal_obj._cal_solns[key]['interp'],
                                      timestamps.size,
                                      axis=0)
                if key == 'K':
                    solns = np.exp(solns * katdal_obj._delay_to_phase)
                # optimise shape for calibration calculation
                katdal_obj._cal_solns[key]['solns'] = np.rollaxis(np.rollaxis(solns, 3, 0), 3, 0)
                # katdal_obj._cal_solns[key]['solns'] = solns

    def _cal_calc1(katdal_obj):
        import time

        _cal_setup(katdal_obj)
        _cal_interp(katdal_obj.timestamps)

        _bls_len = katdal_obj._corrprod_keep.sum()
        _chan_len = katdal_obj._freq_keep.sum()
        _time_len = katdal_obj._time_keep.sum()

        if _bls_len != len(katdal_obj._cp_lookup):
            raise ValueError('Shape mismatch between correlation products.')
        if _chan_len != len(katdal_obj._data_channel_freqs):
            raise ValueError('Shape mismatch in frequency axis.')
        if _time_len != len(katdal_obj.timestamps):
            raise ValueError('Shape mismatch in timestamps.')

        # calibrate visibilities
        etime = time.time()
        katdal_obj._cal_coeffs = np.ones((_time_len, _chan_len, _bls_len), dtype=complex)
        K = katdal_obj._cal_solns['K']['solns']
        B = katdal_obj._cal_solns['B']['solns']
        G = katdal_obj._cal_solns['G']['solns']
        _cal_shape = [_time_len, _chan_len]
        # ---
        stime = time.time()
        for idx, cp in enumerate(katdal_obj._cp_lookup):
            # ((X*K)/B)/G
            # K
            scale = K[cp[0][0], cp[0][1], :, :] * K[cp[1][0], cp[1][1], :, :].conj()
            katdal_obj._cal_coeffs[:, :, idx] *= scale
            # B
            scale = B[cp[0][0], cp[0][1], :, :] * B[cp[1][0], cp[1][1], :, :].conj()
            katdal_obj._cal_coeffs[:, :, idx] /= scale
            # G
            scale = G[cp[0][0], cp[0][1], :, :] * G[cp[1][0], cp[1][1], :, :].conj()
            katdal_obj._cal_coeffs[:, :, idx]  *= np.reciprocal(scale)
        print 'total time', time.time()-stime
        print 'cal_coeffs', katdal_obj._cal_coeffs.shape

        return katdal_obj


    def _cal_calc2(katdal_obj):
        import time

        _cal_setup(katdal_obj)
        _cal_interp(katdal_obj.timestamps)

        _bls_len = katdal_obj._corrprod_keep.sum()
        _chan_len = katdal_obj._freq_keep.sum()
        _time_len = katdal_obj._time_keep.sum()

        if _bls_len != len(katdal_obj._cp_lookup):
            raise ValueError('Shape mismatch between correlation products.')
        if _chan_len != len(katdal_obj._data_channel_freqs):
            raise ValueError('Shape mismatch in frequency axis.')
        if _time_len != len(katdal_obj.timestamps):
            raise ValueError('Shape mismatch in timestamps.')

        # calibrate visibilities
        katdal_obj._cal_coeffs = np.zeros((_time_len, _chan_len, _bls_len), dtype=complex)
        _cal_shape = [_time_len, _chan_len]
        dummy = np.zeros(_cal_shape, dtype=complex)
        default = np.ones(_cal_shape, dtype=complex)
        # ---

        stime = time.time()
        for idx, cp in enumerate(katdal_obj._cp_lookup):
            caldata = np.empty(len(katdal_obj._cal_solns.keys()), dtype=object)
            # apply cal solutions in sequence: K, B, G
            for seq, caltype in enumerate(['K', 'B', 'G']):
                X = katdal_obj._cal_solns[caltype]['solns']
                if X is None:
                    print ('TelescopeState does not have %s calibration solutions' % seq)
                    scale = default
                else:
                    scale = dummy + X[cp[0][0], cp[0][1], :, :] * X[cp[1][0], cp[1][1], :, :].conj()
                caldata[seq] = scale
            caldata = np.array([(x,) for x in caldata]).squeeze()
            # ((X*K)/B)/G
            katdal_obj._cal_coeffs[:, :, idx] = (caldata[0, ...] / caldata[1, ...]) * np.reciprocal(caldata[2, ...])
        print 'total time', time.time()-stime
        print 'cal_coeffs', katdal_obj._cal_coeffs.shape
        return katdal_obj


    def _cal_vis(vis, keep):
        # if no calibration coefficient matrix exist, calculate one
        if katdal_obj._cal_coeffs is None:
            print 'Adding cal'
            _cal_calc1(katdal_obj)
            # _cal_calc2(katdal_obj)

        # after a select the visibilities will change, recalculate
        _bls_len = katdal_obj._corrprod_keep.sum()
        _chan_len = katdal_obj._freq_keep.sum()
        _time_len = katdal_obj._time_keep.sum()
        vis_size = _time_len*_chan_len*_bls_len
        if vis_size != katdal_obj._cal_coeffs.size:
            print 'Updating cal', vis_size, katdal_obj._cal_coeffs.size
            _cal_calc(katdal_obj)

        print 'here', np.shape(katdal_obj._cal_coeffs[keep])
        vis *= katdal_obj._cal_coeffs[keep]
        # return vis*katdal_obj._cal_coeffs[keep]
        return vis
    katdal_obj.cal_vis = LazyTransform('cal_vis', _cal_vis)


    def _cal_weights(weights, keep):
        _cal_interp(katdal_obj.timestamps)
        for idx, cp in enumerate(applycal._cp_lookup):
            # scale weights when appling cal solutions in sequence: B, G
            for seq, caltype in enumerate(['B', 'G']):
                X = applycal._cal_solns[caltype]['solns']
                if X is not None:
                    scale = X[:, :, cp[0][1], cp[0][0]] * X[:, :, cp[1][1], cp[1][0]].conj()
                    weights[:, :, idx] *= scale.real**2 + scale.imag**2
        return weights
    katdal_obj.cal_weights = LazyTransform('cal_weights', _cal_weights)

    return katdal_obj


# -fin-
