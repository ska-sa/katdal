################################################################################
# Copyright (c) 2017-2018, National Research Foundation (Square Kilometre Array)
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

import dask.array as da
import numpy as np

from .lazy_indexer import LazyTransform
from .sensordata import _safe_linear_interp


class CalibrationReadError(RuntimeError):
    """An error occurred in loading calibration values from file"""
    pass


class SimpleInterpolate1D(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __call__(self, x):
        y_re = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]))
        y_im = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]))
        # Interpolate across nans in time axis.
        for ch_idx in range(self._y.shape[1]):
            for pol_idx in range(self._y.shape[2]):
                for ant_idx in range(self._y.shape[3]):
                    y_re[:, ch_idx, pol_idx, ant_idx] = _safe_linear_interp(self._x,
                                                                            self._y[:, ch_idx, pol_idx, ant_idx].real,
                                                                            x)
                    y_im[:, ch_idx, pol_idx, ant_idx] = _safe_linear_interp(self._x,
                                                                            self._y[:, ch_idx, pol_idx, ant_idx].imag,
                                                                            x)
        y = y_re + y_im
        return y


class ComplexInterpolate1D(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._mag = np.abs(y)
        self._phase = y / self._mag

    def __call__(self, x):
        mag = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]))
        phase_re = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]))
        phase_im = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]))
        # Interpolate across nans in time axis.
        for ch_idx in range(self._y.shape[1]):
            for pol_idx in range(self._y.shape[2]):
                for ant_idx in range(self._y.shape[3]):
                    mag[:, ch_idx, pol_idx, ant_idx] = _safe_linear_interp(self._x,
                                                                           self._mag[:, ch_idx, pol_idx, ant_idx],
                                                                           x)
                    phase_re[:, ch_idx, pol_idx, ant_idx] = _safe_linear_interp(self._x,
                                                                                self._phase[:, ch_idx, pol_idx, ant_idx].real,
                                                                                x)
                    phase_im[:, ch_idx, pol_idx, ant_idx] = _safe_linear_interp(self._x,
                                                                                self._phase[:, ch_idx, pol_idx, ant_idx].imag,
                                                                                x)
        phase = phase_re + phase_im
        return phase / np.abs(phase) * mag


def interpolate_nans_1d(y):
    """Interpolate over nans in a timeseries.

    When interpolating the per-channel timeseries, the channels with more
    nan values will deviate from the neighboring channels appearing as spikes
    when the mean over time is plotted.
    """
    if np.isnan(y).all():
        return y  # nothing to do, all nan values
    if np.isfinite(y).all():
        return y  # nothing to do, all values valid

    # interpolate across nans
    nan_locs = np.isnan(y)
    xi = np.nonzero(~nan_locs)[0]
    yi = y[xi]
    x = np.arange(len(y))
    y = _safe_linear_interp(xi, yi, x)
    return y


def _get_cal_sensor(key, katdal_obj):
    """Load a calibrator solution from sensors.
    """
    values = []
    timestamps = []
    for sensor in katdal_obj.file['TelescopeState'].keys():
        if key in sensor:
            value = katdal_obj.sensor['TelescopeState/{}'.format(sensor)]
            timestamp = katdal_obj.sensor.timestamps
            try:
                values = np.vstack((values, value))
                timestamps = np.vstack((timestamps, timestamp))
            except ValueError:
                values = np.asarray(value)
                timestamps = np.asarray(timestamp)
    timestamps = np.reshape(timestamps, timestamps.size)
    if timestamps.size > 1:
        timestamps = timestamps.squeeze()
    return timestamps, values


def _get_cal_antlist(katdal_obj):
    """Load antenna list used for calibration.
    If the value does not match the antenna list in the katdal dataset,
    a :exc:`CalibrationReadError` is raised. Eventually this could be
    extended to allow for an antenna list that doesn't match by permuting
    the calibration solutions.
    """
    cal_antlist = katdal_obj._get_telstate_attr('cal_antlist')
    if isinstance(cal_antlist, np.ndarray):
        cal_antlist = cal_antlist.tolist()
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
    cal_pol_ordering = katdal_obj._get_telstate_attr('cal_pol_ordering')
    try:
        cal_pol_ordering = np.array(cal_pol_ordering)
    except (NameError, SyntaxError):
        raise
    except Exception as e:
        raise CalibrationReadError(str(e))

    if cal_pol_ordering.shape != (4, 2):
        # assume newer telstate format
        pol_dict = {}
        for idx, pol in enumerate(cal_pol_ordering):
            pol_dict[pol] = idx
        return pol_dict

    # older format needs more work
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
    :func:`SimpleInterplate1D` or `ComplexInterpolate1D`.

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

    timestamps, values = _get_cal_sensor(key, katdal_obj)

    # avoid extrapolation
    if (timestamps[-1] < katdal_obj._timestamps[0]):
        # All calibration solutions ahead of observation, use last one
        values = values[[-1], ...]
        timestamps = timestamps[[-1], ...]
    elif (timestamps[0] > katdal_obj._timestamps[-1]):
        # All calibration solutions after observation, use first one
        values = values[[0], ...]
        timestamps = timestamps[[0], ...]

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
                    values[ts_idx, :, pol_idx, ant_idx] = interpolate_nans_1d(values[ts_idx, :, pol_idx, ant_idx])

    # to interpolate you need >1 timestamps, else return only the single value
    if timestamps.size > 1:
        kind = kwargs.get('kind', 'linear')  # default if none given
        if np.iscomplexobj(values) and kind not in ['zero', 'nearest']:
            interp = ComplexInterpolate1D
        else:
            interp = SimpleInterpolate1D
        return interp(timestamps, values)
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
    katdal_obj._cp_lookup = np.asarray(katdal_obj._cp_lookup, dtype=int)
    # interpolation function over available solutions
    for key in katdal_obj._cal_solns.keys():
        try:
            katdal_obj._cal_solns[key]['interp'] = _get_cal_product('cal_product_' + key,
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
    katdal_obj._cal_solns = {'K': initcal('linear'),
                             'B': initcal('zero'),
                             'G': initcal('linear'),
                             }

    katdal_obj._cal_coeffs = []
    katdal_obj._wght_coeffs = []

    ts_chunk_size = 128
    ch_chunk_size = -1

    def _cal_interp(timestamps):
        """Interpolate values between calculated timestamps"""
        # Interpolate the calibration solutions for the selected range
        for key in katdal_obj._cal_solns.keys():
            if katdal_obj._cal_solns[key]['interp'] is not None:
                # interpolate over values, or repeat single value
                # expected size <ts><ch><pol><ant>
                if hasattr(katdal_obj._cal_solns[key]['interp'], '__call__'):
                    solns = katdal_obj._cal_solns[key]['interp'](timestamps)
                else:
                    solns = np.repeat(katdal_obj._cal_solns[key]['interp'],
                                      timestamps.size,
                                      axis=0)
                if key == 'K':
                    solns = np.exp(solns * katdal_obj._delay_to_phase)
                katdal_obj._cal_solns[key]['solns'] = da.from_array(solns,
                                                                    chunks=(ts_chunk_size, ch_chunk_size, 1, 1),
                                                                    )

    def _cal_calc(katdal_obj):
        """Calculate calibration and weight coefficients"""

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
        K = katdal_obj._cal_solns['K']['solns']
        B = katdal_obj._cal_solns['B']['solns']
        G = katdal_obj._cal_solns['G']['solns']

        # ((X*K)/B)/G
        for idx, cp in enumerate(katdal_obj._cp_lookup):
            scale_coeff = None
            scale_wght = None
            # K
            scale_coeff = K[:, :, cp[0][0], cp[0][1]] * K[:, :, cp[1][0], cp[1][1]].conj()
            # B
            scale = B[:, :, cp[0][0], cp[0][1]] * B[:, :, cp[1][0], cp[1][1]].conj()
            scale_coeff /= scale
            scale_wght = (scale.real**2 + scale.imag**2)
            # G
            scale = G[:, :, cp[0][0], cp[0][1]] * G[:, :, cp[1][0], cp[1][1]].conj()
            scale_coeff *= np.reciprocal(scale)
            scale_wght *= (scale.real**2 + scale.imag**2)
            katdal_obj._cal_coeffs.append(scale_coeff)
            katdal_obj._wght_coeffs.append(scale_wght)

        # coefficient matrices <ts><ch><bl>
        katdal_obj._cal_coeffs = da.stack(katdal_obj._cal_coeffs, axis=2)
        katdal_obj._wght_coeffs = da.stack(katdal_obj._wght_coeffs, axis=2)
        return katdal_obj
    _cal_calc(katdal_obj)

    def _cal_vis(vis, keep):
        vis_coeffs = None
        # only extract what you need from the full matrix
        bls = np.nonzero(katdal_obj._corrprod_keep)[0]
        chan = np.nonzero(katdal_obj._freq_keep)[0]
        time = np.nonzero(katdal_obj._time_keep)[0]
        vis_coeffs = katdal_obj._cal_coeffs.vindex[:, :, bls].vindex[:, :, chan].vindex[:, :, time].compute()
        # visibilities <ts><ch><bl>
        vis *= vis_coeffs
        return vis
    katdal_obj.cal_vis = LazyTransform('cal_vis', _cal_vis)

    def _cal_weights(weights, keep):
        wght_coeffs = None
        # only extract what you need from the full matrix
        bls = np.nonzero(katdal_obj._corrprod_keep)[0]
        chan = np.nonzero(katdal_obj._freq_keep)[0]
        time = np.nonzero(katdal_obj._time_keep)[0]
        wght_coeffs = katdal_obj._wght_coeffs.vindex[:, :, bls].vindex[:, :, chan].vindex[:, :, time].compute()
        # weights <ts><ch><bl>
        weights *= wght_coeffs
        return weights
    katdal_obj.cal_weights = LazyTransform('cal_weights', _cal_weights)

    return katdal_obj


# -fin-
