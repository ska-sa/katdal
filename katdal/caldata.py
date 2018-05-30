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
Code largely pilfered from the katdal_loader in katsdpimager,
but using a katdal object rather than a filename,
so that things are a bit more portable. """

import dask.array as da
import numpy as np

# from .lazy_indexer import LazyTransform


class CalibrationReadError(RuntimeError):
    """An error occurred in loading calibration values from file"""
    pass


class CalibrationValueError(ValueError):
    """An error occurred in the calculation of calibration coefficients"""
    pass


class SimpleInterpolate1D(object):
    def __init__(self, x, y):
        sort_idx = np.argsort(x)
        self._x = x[sort_idx]
        self._y = y[sort_idx]

    def __call__(self, x):
        y = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]), dtype=complex)
        # Interpolate across nans in time axis.
        for ch_idx in range(self._y.shape[1]):
            for pol_idx in range(self._y.shape[2]):
                for ant_idx in range(self._y.shape[3]):
                    y[:, ch_idx, pol_idx, ant_idx] = np.interp(x,
                                                               self._x,
                                                               self._y[:, ch_idx, pol_idx, ant_idx],
                                                               )
        return y


class ComplexInterpolate1D(object):
    def __init__(self, x, y):
        sort_idx = np.argsort(x)
        self._x = x[sort_idx]
        self._y = y[sort_idx]
        self._mag = np.abs(y)
        self._phase = y / self._mag

    def __call__(self, x):
        mag = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]), dtype=complex)
        phase = np.empty((x.shape[0], self._y.shape[1], self._y.shape[2], self._y.shape[3]), dtype=complex)
        # Interpolate across nans in time axis.
        for ch_idx in range(self._y.shape[1]):
            for pol_idx in range(self._y.shape[2]):
                for ant_idx in range(self._y.shape[3]):
                    mag[:, ch_idx, pol_idx, ant_idx] = np.interp(x,
                                                                 self._x,
                                                                 self._mag[:, ch_idx, pol_idx, ant_idx],
                                                                 )
                    phase[:, ch_idx, pol_idx, ant_idx] = np.interp(x,
                                                                   self._x,
                                                                   self._phase[:, ch_idx, pol_idx, ant_idx],
                                                                   )
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
    return np.interp(x, xi, yi)


def _get_cal_sensor(key, katdal_obj):
    """Load a calibrator solution from sensors.
    """
    values = np.array([])
    timestamps = katdal_obj.sensor.timestamps
    try:
        values = katdal_obj.sensor[key]
    except KeyError:
        pass  # cal solutions not available
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
    cal_antlist = katdal_obj.source.metadata.attrs['cal_antlist']
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
    cal_pol_ordering = katdal_obj.source.metadata.attrs['cal_pol_ordering']
    try:
        cal_pol_ordering = np.array(cal_pol_ordering)
    except (NameError, SyntaxError):
        raise
    except Exception as e:
        raise CalibrationReadError(str(e))

    pol_dict = {}
    for idx, pol in enumerate(cal_pol_ordering):
        pol_dict[pol] = idx
    return pol_dict


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

    if values.size < 1:
        return None

    if values.ndim == 3:
        # Insert a channel axis
        values = values[:, np.newaxis, ...]

    if values.ndim != 4:
        raise CalibrationReadError('Calibration solutions has wrong number of dimensions')

    # avoid extrapolation
    if (timestamps[-1] < katdal_obj.source.timestamps[0]):
        # All calibration solutions ahead of observation, use last one
        values = values[[-1], ...]
        timestamps = timestamps[[-1], ...]
    elif (timestamps[0] > katdal_obj.source.timestamps[-1]):
        # All calibration solutions after observation, use first one
        values = values[[0], ...]
        timestamps = timestamps[[0], ...]

    if values.shape[1] > 1:
        # Only use selected channels
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


class _cal_setup():
    def __init__(self, katdal_obj):
        def initcal(kind):
            return {'interp': None, 'solns': None, 'kind': kind}
        self._cal_available = ['K', 'B0', 'G', 'KCROSS_DIODE']
        _cal_interp = ['linear', 'zero', 'linear', 'linear']
        self._cal_solns = {}
        for idx, cal in enumerate(self._cal_available):
            self._cal_solns[cal] = initcal(_cal_interp[idx])

        self._cal_pol_ordering = _get_cal_pol_ordering(katdal_obj)
        self._cal_ant_ordering = _get_cal_antlist(katdal_obj)
        self._data_channel_freqs = katdal_obj.channel_freqs
        self._delay_to_phase = (-2j * np.pi * self._data_channel_freqs)[np.newaxis, :, np.newaxis, np.newaxis]
        # baselines from the data files are given as <ant><pol>
        # for indexing this has to be swapped to <pol><ant>
        self._cp_lookup = [[(self._cal_pol_ordering[prod[0][-1]], self._cal_ant_ordering.index(prod[0][:-1]),),
                            (self._cal_pol_ordering[prod[1][-1]], self._cal_ant_ordering.index(prod[1][:-1]),)]
                           for prod in katdal_obj.corr_products]
        self._cp_lookup = np.asarray(self._cp_lookup, dtype=int)
        # interpolation function over available solutions
        for key in self._cal_solns.keys():
            self._cal_solns[key]['interp'] = _get_cal_product('cal_product_' + key,
                                                              katdal_obj,
                                                              kind=self._cal_solns[key]['kind'],
                                                              )
        self._cal_interp(katdal_obj.timestamps)

    def _cal_interp(self, timestamps):
        """Interpolate values between calculated timestamps"""
        # Interpolate the calibration solutions for the selected range
        for key in self._cal_solns.keys():
            if self._cal_solns[key]['interp'] is not None:
                # interpolate over values, or repeat single value
                # expected size <ts><ch><pol><ant>
                if hasattr(self._cal_solns[key]['interp'], '__call__'):
                    solns = self._cal_solns[key]['interp'](timestamps)
                else:
                    solns = np.broadcast_to(self._cal_solns[key]['interp'], timestamps.shape)
                if key == 'K':
                    solns = np.exp(solns * self._delay_to_phase)
                self._cal_solns[key]['solns'] = da.from_array(solns,
                                                              chunks=solns.shape,
                                                              )


def applycal(katdal_obj):
    """
    Apply the K, B, G solutions to visibilities at provided timestamps.
    Optionally recompute the weights as well.

    Returns
    -------
    katdal_obj: containing vis and weights (optional) with cal solns applied.
    """

    katdal_obj.cal_coeffs = []
    katdal_obj.weight_coeffs = []

    def _cal_calc(katdal_obj):
        # """Calculate calibration and weight coefficients"""
        """Setup calibration members for use"""

        cal_obj = _cal_setup(katdal_obj)
        _bls_len = katdal_obj._corrprod_keep.sum()
        _chan_len = katdal_obj._freq_keep.sum()
        _time_len = katdal_obj._time_keep.sum()

        if _bls_len != len(cal_obj._cp_lookup):
            raise CalibrationValueError('Shape mismatch between correlation products.')
        if _chan_len != len(cal_obj._data_channel_freqs):
            raise CalibrationValueError('Shape mismatch in frequency axis.')
        if _time_len != len(katdal_obj.timestamps):
            raise CalibrationValueError('Shape mismatch in timestamps.')

        # calibration matrix
        _cal_shape = [_time_len, _chan_len]
        dummy = np.zeros(_cal_shape, dtype=float)
        default = np.ones(_cal_shape, dtype=float)
        # apply calibrations in order given by measurement set (((X*K)/B)/G)/D
        for idx, cp in enumerate(cal_obj._cp_lookup):
            scale_coeff = None
            scale_wght = None
            for seq, caltype in enumerate(cal_obj._cal_available):
                X = cal_obj._cal_solns[caltype]['solns']
                if X is None:
                    scale = default
                else:
                    scale = dummy + X[:, :, cp[0][0], cp[0][1]] * X[:, :, cp[1][0], cp[1][1]].conj()
                if seq == 0:
                    scale_coeff = scale
                    scale_wght = da.from_array(default, chunks=default.shape)
                else:
                    scale_coeff *= np.reciprocal(scale)
                    scale_wght *= scale.real**2 + scale.imag**2

            katdal_obj.cal_coeffs.append(scale_coeff)
            katdal_obj.weight_coeffs.append(scale_wght)

        # coefficient matrices <ts><ch><bl>
        katdal_obj.cal_coeffs = da.stack(katdal_obj.cal_coeffs, axis=2)
        katdal_obj.weight_coeffs = da.stack(katdal_obj.weight_coeffs, axis=2)
        return katdal_obj
    _cal_calc(katdal_obj)

    def _cal_vis(vis):
        # only extract what you need from the full matrix
        time = np.nonzero(katdal_obj._time_keep)[0]
        chan = np.nonzero(katdal_obj._freq_keep)[0]
        bls = np.nonzero(katdal_obj._corrprod_keep)[0].tolist()
        # vis <ts><ch><bl>
        vis *= katdal_obj.cal_coeffs[time, :, :][:, chan, :][:, :, bls]
        return vis
    katdal_obj.cal_vis = _cal_vis

    def _cal_weights(weights):
        # only extract what you need from the full matrix
        time = np.nonzero(katdal_obj._time_keep)[0]
        chan = np.nonzero(katdal_obj._freq_keep)[0]
        bls = np.nonzero(katdal_obj._corrprod_keep)[0].tolist()
        # weights <ts><ch><bl>
        weights *= katdal_obj.weight_coeffs[time, :, :][:, chan, :][:, :, bls]
        return weights
    katdal_obj.cal_weights = _cal_weights

    return katdal_obj


# -fin-
