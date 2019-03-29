###############################################################################
# Copyright (c) 2018-2019, National Research Foundation (Square Kilometre Array)
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
###############################################################################

"""Utilities for applying calibration solutions to visibilities and weights."""
from __future__ import print_function, division, absolute_import
from builtins import range, zip

import logging
import itertools
import operator

import numpy as np
import dask.array as da
import dask.base
import dask.utils
import toolz
import numba

from .categorical import CategoricalData, ComparableArrayWrapper
from .spectral_window import SpectralWindow
from .flags import POSTPROC


# A constant indicating invalid / absent gain (typically due to flagged data)
INVALID_GAIN = np.complex64(complex(np.nan, np.nan))
# All the calibration products katdal knows about
CAL_PRODUCTS = ('K', 'B', 'G')

logger = logging.getLogger(__name__)


def _call_from_block_function(func, shape, num_chunks, chunk_location, array_location, func_kwargs):
    block_info = {
        'shape': shape,
        'num-chunks': num_chunks,
        'chunk-location': chunk_location,
        'array-location': list(array_location)
    }
    return func(block_info, **func_kwargs)


# This has been submitted to dask as https://github.com/dask/dask/pull/4476.
# If it gets merged it can be used rather than copied here. There are also
# unit tests there.
def from_block_function(func, shape, chunks='auto', dtype=None, name=None, **kwargs):
    """
    Create an array from a function that builds individual blocks.

    For each block, the function is passed a dictionary with information about
    the block to construct, and should return a numpy array.

    >>> block_info      # doctest: +SKIP
    {'shape': (12, 20),
     'num-chunks': (3, 4),
     'chunk-location': (2, 1),
     'array-location': [(8, 12), (5, 10)]
    }

    The values in the dictionary are respectively the shape of the full
    array, the number of chunks in the full array in each dimension, the
    position of this block in chunks, and the position in the array
    (for example, the slice corresponding to ``8:12, 5:10``).

    Parameters
    ----------
    func : callable
        Function to produce every block in the array
    shape : Tuple[int]
        Shape of the resulting array.
    chunks : tuple, optional
        Chunk shape of resulting blocks. If not provided, a chunking scheme
        is chosen automatically.
    dtype : np.dtype, optional
        The ``dtype`` of the output array. It is recommended to provide this.
        If not provided, will be inferred by applying the function to a small
        set of fake data.
    name : str, optional
        The key name to use for the output array. If not provided,
        will be determined from `func`.
    **kwargs :
        Other keyword arguments to pass to function. Values must be constants
        (not dask.arrays)

    Examples
    --------
    This is a simplified version of :func:`eye` which only handles square
    arrays with the ones on the main diagonal.

    >>> def eye_chunk(block_info):
    ...     location = block_info['array-location']
    ...     r0, r1 = location[0]
    ...     c0, c1 = location[1]
    ...     if r0 == c0:
    ...         return np.eye(r1 - r0, c1 - c0)
    ...     else:
    ...         return np.zeros((r1 - r0, c1 - c0))
    >>> from_block_function(eye_chunk, (4, 4), chunks=2, dtype=float)
    dask.array<eye_chunk, shape=(4, 4), dtype=float64, chunksize=(2, 2)>
    >>> _.compute()
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """

    name = '%s-%s' % (name or dask.utils.funcname(func),
                      dask.base.tokenize(func, shape, dtype, chunks))

    if dtype is None:
        dummy_block_info = {
            'shape': shape,
            'num-chunks': shape,
            'chunk-location': (0,) * len(shape),
            'array-location': [(0, 1)] * len(shape)
        }
        dtype = da.apply_infer_dtype(func, [dummy_block_info], kwargs, 'from_block_function')

    chunks = da.core.normalize_chunks(chunks, shape, dtype=dtype)
    # Allow for shape=None when chunks are already in normalized form
    shape = tuple(sum(bd) for bd in chunks)

    keys = list(itertools.product([name], *[range(len(bd)) for bd in chunks]))
    aggdims = [list(toolz.accumulate(operator.add, (0,) + bd)) for bd in chunks]
    locdims = [list(zip(a[:-1], a[1:])) for a in aggdims]
    locations = list(itertools.product(*locdims))
    num_chunks = tuple(len(bd) for bd in chunks)
    dsk = {key: (_call_from_block_function, func, shape, num_chunks, key[1:], location, kwargs)
           for key, location in zip(keys, locations)}
    return da.Array(dsk, name, chunks, dtype=dtype)


def complex_interp(x, xi, yi, left=None, right=None):
    """Piecewise linear interpolation of magnitude and phase of complex values.

    Given discrete data points (`xi`, `yi`), this returns a 1-D piecewise
    linear interpolation `y` evaluated at the `x` coordinates, similar to
    `numpy.interp(x, xi, yi)`. While :func:`numpy.interp` interpolates the real
    and imaginary parts of `yi` separately, this function interpolates
    magnitude and (unwrapped) phase separately instead. This is useful when the
    phase of `yi` changes more rapidly than its magnitude, as in electronic
    gains.

    Parameters
    ----------
    x : 1-D sequence of float, length *M*
        The x-coordinates at which to evaluate the interpolated values
    xi : 1-D sequence of float, length *N*
        The x-coordinates of the data points, must be sorted in ascending order
    yi : 1-D sequence of complex, length *N*
        The y-coordinates of the data points, same length as `xi`
    left : complex, optional
        Value to return for `x < xi[0]`, default is `yi[0]`
    right : complex, optional
        Value to return for `x > xi[-1]`, default is `yi[-1]`

    Returns
    -------
    y : array of complex, length *M*
        The evaluated y-coordinates, same length as `x` and same dtype as `yi`
    """
    # Extract magnitude and unwrapped phase
    mag_i = np.abs(yi)
    phase_i = np.unwrap(np.angle(yi))
    # Prepare left and right interpolation extensions
    mag_left = phase_left = mag_right = phase_right = None
    if left is not None:
        mag_left = np.abs(left)
        phase_left = np.unwrap([phase_i[0], np.angle(left)])[1]
    if right is not None:
        mag_right = np.abs(right)
        phase_right = np.unwrap([phase_i[-1], np.angle(right)])[1]
    # Interpolate magnitude and phase separately, and reassemble
    mag = np.interp(x, xi, mag_i, left=mag_left, right=mag_right)
    phase = np.interp(x, xi, phase_i, left=phase_left, right=phase_right)
    y = np.empty_like(phase, dtype=np.complex128)
    np.cos(phase, out=y.real)
    np.sin(phase, out=y.imag)
    y *= mag
    return y.astype(yi.dtype)


def has_cal_product(cache, attrs, product):
    """Check if calibration solution `product` is available in `cache`."""
    key = 'cal_product_' + product
    try:
        parts = int(attrs[key + '_parts'])
    except KeyError:
        return key in cache
    else:
        return any(key + str(part) in cache for part in range(parts))


def get_cal_product(cache, attrs, product):
    """Extract calibration solution `product` from `cache` as a sensor.

    This takes care of stitching together multiple parts of the product
    if this is indicated in the `attrs` dict.
    """
    key = 'cal_product_' + product
    try:
        n_parts = int(attrs[key + '_parts'])
    except KeyError:
        return cache.get(key)
    # Handle multi-part cal product (as produced by "split cal")
    # First collect all the parts as sensors (and mark missing ones as None)
    parts = []
    valid_part = None
    for n in range(n_parts):
        try:
            valid_part = cache.get(key + str(n))
        except KeyError:
            parts.append(None)
        else:
            parts.append(valid_part)
    if valid_part is None:
        raise KeyError("No cal product '{}' parts found (expected {})"
                       .format(product, n_parts))
    # Convert each part to its sensor values (filling missing ones with NaNs)
    events = valid_part.events
    invalid_part = None
    for n in range(n_parts):
        if parts[n] is None:
            if invalid_part is None:
                # This assumes that each part has the same array shape
                invalid_part = [np.full_like(value, INVALID_GAIN)
                                for segment, value in valid_part.segments()]
            parts[n] = invalid_part
        else:
            if not np.array_equal(parts[n].events, events):
                raise ValueError("Cal product '{}' part {} does not align in "
                                 "time with the rest".format(product, n))
            parts[n] = [value for segment, value in parts[n].segments()]
    # Stitch all the value arrays together and form a new combined sensor
    values = np.concatenate(parts, axis=1)
    values = [ComparableArrayWrapper(v) for v in values]
    return CategoricalData(values, events)


def calc_delay_correction(sensor, index, data_freqs):
    """Calculate correction sensor from delay calibration solution sensor.

    Given the delay calibration solution `sensor`, this extracts the delay time
    series of the input specified by `index` (in the form (pol, ant)) and
    builds a categorical sensor for the corresponding complex correction terms
    (channelised by `data_freqs`).

    Invalid delays (NaNs) are replaced by zeros, since bandpass calibration
    still has a shot at fixing any residual delay.
    """
    delays = [np.nan_to_num(value[index]) for segm, value in sensor.segments()]
    # Delays produced by cal pipeline are raw phase slopes, i.e. exp(2 pi j d f)
    corrections = [np.exp(-2j * np.pi * d * data_freqs).astype('complex64')
                   for d in delays]
    corrections = [ComparableArrayWrapper(c) for c in corrections]
    return CategoricalData(corrections, sensor.events)


def calc_bandpass_correction(sensor, index, data_freqs, cal_freqs):
    """Calculate correction sensor from bandpass calibration solution sensor.

    Given the bandpass calibration solution `sensor`, this extracts the time
    series of bandpasses (channelised by `cal_freqs`) for the input specified
    by `index` (in the form (pol, ant)) and builds a categorical sensor for
    the corresponding complex correction terms (channelised by `data_freqs`).

    Invalid solutions (NaNs) are replaced by linear interpolations over
    frequency (separately for magnitude and phase), as long as some channels
    have valid solutions.
    """
    corrections = []
    for segment, value in sensor.segments():
        bp = value[(slice(None),) + index]
        valid = np.isfinite(bp)
        if valid.any():
            # Don't extrapolate to edges of band where gain typically drops off
            bp = complex_interp(data_freqs, cal_freqs[valid], bp[valid],
                                left=INVALID_GAIN, right=INVALID_GAIN)
        else:
            bp = np.full(len(data_freqs), INVALID_GAIN)
        corrections.append(ComparableArrayWrapper(np.reciprocal(bp)))
    return CategoricalData(corrections, sensor.events)


def calc_gain_correction(sensor, index):
    """Calculate correction sensor from gain calibration solution sensor.

    Given the gain calibration solution `sensor`, this extracts the time
    series of gains for the input specified by `index` (in the form (pol, ant))
    and interpolates them over time to get the corresponding complex correction
    terms.

    Invalid solutions (NaNs) are replaced by linear interpolations over time
    (separately for magnitude and phase), as long as some dumps have valid
    solutions.
    """
    dumps = np.arange(sensor.events[-1])
    events = sensor.events[:-1]
    gains = np.array([value[index] for segment, value in sensor.segments()])
    valid = np.isfinite(gains)
    if not valid.any():
        return CategoricalData([INVALID_GAIN], [0, len(dumps)])
    smooth_gains = complex_interp(dumps, events[valid], gains[valid])
    return np.reciprocal(smooth_gains)


def add_applycal_sensors(cache, attrs, data_freqs):
    """Add virtual sensors that store calibration corrections, to sensor cache.

    This maps receptor inputs to the relevant indices in each calibration
    product based on the ants and pols found in `attrs`. It then registers
    a virtual sensor per input and per cal product in the SensorCache `cache`,
    with template 'Calibration/{inp}_correction_{product}'. The virtual sensor
    function picks the appropriate correction calculator based on the cal
    product name, which also uses auxiliary info like the channel frequencies,
    `data_freqs`.
    """
    cal_ants = attrs.get('cal_antlist', [])
    cal_pols = attrs.get('cal_pol_ordering', [])
    cal_input_map = {ant + pol: (pol_idx, ant_idx)
                     for (pol_idx, pol) in enumerate(cal_pols)
                     for (ant_idx, ant) in enumerate(cal_ants)}
    if not cal_input_map:
        return
    try:
        cal_spw = SpectralWindow(attrs['cal_center_freq'], None,
                                 attrs['cal_n_chans'], sideband=1,
                                 bandwidth=attrs['cal_bandwidth'])
        cal_freqs = cal_spw.channel_freqs
    except KeyError:
        logger.warning('Missing cal spectral attributes, disabling applycal')
        return

    def calc_correction_per_input(cache, name, inp, product):
        """Calculate correction sensor for input `inp` from cal solutions."""
        product_sensor = get_cal_product(cache, attrs, product)
        try:
            index = cal_input_map[inp]
        except KeyError:
            raise KeyError("No calibration solutions available for input "
                           "'{}' - available ones are {}"
                           .format(inp, sorted(cal_input_map.keys())))
        if product == 'K':
            correction_sensor = calc_delay_correction(product_sensor, index,
                                                      data_freqs)
        elif product == 'B':
            correction_sensor = calc_bandpass_correction(product_sensor, index,
                                                         data_freqs, cal_freqs)
        elif product == 'G':
            correction_sensor = calc_gain_correction(product_sensor, index)
        else:
            raise KeyError("Unknown calibration product '{}'".format(product))
        cache[name] = correction_sensor
        return correction_sensor

    correction_sensor_template = 'Calibration/{inp}_correction_{product}'
    cache.virtual[correction_sensor_template] = calc_correction_per_input


@numba.jit(nopython=True, nogil=True)
def _correction_inputs_to_corrprods(g_per_cp, g_per_input, input1_index, input2_index):
    """Convert gains per input to gains per correlation product."""
    for i in range(g_per_cp.shape[0]):
        for j in range(g_per_cp.shape[1]):
            g_per_cp[i, j] = (g_per_input[i, input1_index[j]]
                              * np.conj(g_per_input[i, input2_index[j]]))


class CorrectionParams(object):
    """Data needed to compute corrections in :func:`calc_correction_per_corrprod`.

    Once constructed, the data in this class must not be modified, as it will
    be baked into dask graphs.

    Parameters
    ----------
    products : dict
        A dictionary (indexed by cal product name) of lists (indexed
        by input) of sequences (indexed by dump) of numpy arrays, with
        corrections to apply.
    inputs : list of str
        Names of inputs, in the same order as the input axis of products
    input1_index, input2_index : ndarray
        Indices into `inputs` of first and second items of correlation product
    """
    def __init__(self, inputs, input1_index, input2_index, products):
        self.inputs = inputs
        self.input1_index = input1_index
        self.input2_index = input2_index
        self.products = products


def calc_correction_per_corrprod(dump, channels, params):
    """Gain correction per channel per correlation product for a given dump.

    This calculates an array of complex gain correction terms of shape
    (n_chans, n_corrprods) that can be directly applied to visibility data.
    This incorporates all requested calibration products at the specified
    dump and channels.

    Parameters
    ----------
    dump : int
        Dump index (applicable to full data set, i.e. absolute)
    channels : slice
        Channel indices (applicable to full data set, i.e. absolute)
    params : :class:`CorrectionParams`
        Data for obtaining corrections to merge

    Returns
    -------
    gains : array of complex64, shape (n_chans, n_corrprods)
        Gain corrections per channel per correlation product

    Raises
    ------
    KeyError
        If input and/or cal product has no associated correction
    """
    n_channels = channels.stop - channels.start
    g_per_input = np.ones((len(params.inputs), n_channels), dtype='complex64')
    for product in params.products.values():
        for n in range(len(params.inputs)):
            sensor = product[n]
            g_product = sensor[dump]
            if np.shape(g_product) != ():
                g_product = g_product[channels]
            g_per_input[n] *= g_product
    # Transpose to (channel, input) order, and ensure C ordering
    g_per_input = np.ascontiguousarray(g_per_input.T)
    g_per_cp = np.empty((n_channels, len(params.input1_index)), dtype='complex64')
    _correction_inputs_to_corrprods(g_per_cp, g_per_input,
                                    params.input1_index, params.input2_index)
    return g_per_cp


def _correction_block(block_info, params):
    """Calculate applycal correction for a single time-freq-baseline chunk."""
    slices = tuple(slice(*l) for l in block_info['array-location'])
    block_shape = tuple(s.stop - s.start for s in slices)
    correction = np.empty(block_shape, np.complex64)
    # TODO: make calc_correction_per_corrprod multi-dump aware
    for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        correction[n] = calc_correction_per_corrprod(dump, slices[1], params)
    return correction


def calc_correction(chunks, cache, corrprods, cal_products):
    """Create a dask array containing applycal corrections.

    Parameters
    ----------
    chunks : tuple of tuple of int
        Chunking scheme of the resulting array, in normalized form (see
        :func:`dask.array.core.normalize_chunks`).
    cache : :class:`katdal.sensordata.SensorCache` object
        Sensor cache, used to look up individual correction sensors
    corrprods : sequence of (string, string)
        Selected correlation products as pairs of correlator input labels
    cal_products : sequence of string
        Calibration products that will contribute to corrections
    """
    shape = tuple(sum(bd) for bd in chunks)
    if len(chunks[2]) > 1:
        logger.warning('ignoring chunking on baseline axis')
        chunks = (chunks[0], chunks[1], (shape[2],))
    inputs = sorted(set(np.ravel(corrprods)))
    input1_index = np.array([inputs.index(cp[0]) for cp in corrprods])
    input2_index = np.array([inputs.index(cp[1]) for cp in corrprods])
    products = {}
    for product in cal_products:
        products[product] = []
        for i, inp in enumerate(inputs):
            sensor_name = 'Calibration/{}_correction_{}'.format(inp, product)
            sensor = cache.get(sensor_name)
            # Indexing CategoricalData by dump is relatively slow (tens of
            # microseconds), so expand it into a plain-old Python list.
            if isinstance(sensor, CategoricalData):
                data = [None] * sensor.events[-1]
                for s, v in sensor.segments():
                    for j in range(s.start, s.stop):
                        data[j] = v
            else:
                data = sensor
            products[product].append(data)
    params = CorrectionParams(inputs, input1_index, input2_index, products)
    name = 'corrections[{}]'.format(','.join(cal_products))
    return from_block_function(
        _correction_block, shape=shape, chunks=chunks, dtype=np.complex64,
        name=name, params=params)


@numba.jit(nopython=True, nogil=True)
def apply_vis_correction(data, correction):
    """Clean up and apply `correction` to visibility data in `data`."""
    out = np.empty_like(data)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                c = correction[i, j, k]
                if not np.isnan(c):
                    out[i, j, k] = data[i, j, k] * c
                else:
                    out[i, j, k] = data[i, j, k]
    return out


@numba.jit(nopython=True, nogil=True)
def apply_weights_correction(data, correction):
    """Clean up and apply `correction` to weight data in `data`."""
    out = np.empty_like(data)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                cc = correction[i, j, k]
                c = cc.real**2 + cc.imag**2
                if c > 0:   # Will be false if c is NaN
                    out[i, j, k] = data[i, j, k] / c
                else:
                    out[i, j, k] = 0
    return out


@numba.jit(nopython=True, nogil=True)
def apply_flags_correction(data, correction):
    """Set POSTPROC flag wherever `correction` is invalid."""
    out = np.copy(data)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                if np.isnan(correction[i, j, k]):
                    out[i, j, k] |= POSTPROC
    return out
