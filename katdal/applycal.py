###############################################################################
# Copyright (c) 2018-2021, National Research Foundation (SARAO)
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

import logging

import dask.array as da
import numba
import numpy as np

from .categorical import CategoricalData, ComparableArrayWrapper
from .flags import POSTPROC
from .sensordata import SensorGetter, SimpleSensorGetter
from .spectral_window import SpectralWindow

# A constant indicating invalid / absent gain (typically due to flagged data)
INVALID_GAIN = np.complex64(complex(np.nan, np.nan))
# All the calibration product types katdal knows about
CAL_PRODUCT_TYPES = ('K', 'B', 'G', 'GPHASE', 'GAMP_PHASE')

logger = logging.getLogger(__name__)


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
        with np.errstate(invalid='ignore'):
            phase_left = np.unwrap([phase_i[0], np.angle(left)])[1]
    if right is not None:
        mag_right = np.abs(right)
        with np.errstate(invalid='ignore'):
            phase_right = np.unwrap([phase_i[-1], np.angle(right)])[1]
    # Interpolate magnitude and phase separately, and reassemble
    mag = np.interp(x, xi, mag_i, left=mag_left, right=mag_right)
    phase = np.interp(x, xi, phase_i, left=phase_left, right=phase_right)
    y = np.empty_like(phase, dtype=np.complex128)
    np.cos(phase, out=y.real)
    np.sin(phase, out=y.imag)
    y *= mag
    return y.astype(yi.dtype)


def _parse_cal_product(cal_product):
    """Split `cal_product` into `cal_stream` and `product_type` parts."""
    fields = cal_product.rsplit('.', 1)
    if len(fields) != 2:
        raise ValueError(f'Calibration product {cal_product} is not in the format '
                         '<cal_stream>.<product_type>')
    return fields[0], fields[1]


def get_cal_product(cache, cal_stream, product_type):
    """Extract calibration solution from cache as a sensor.

    Parameters
    ----------
    cache : :class:`~katdal.sensordata.SensorCache` object
        Sensor cache serving cal product sensors
    cal_stream : string
        Name of calibration stream (e.g. "l1")
    product_type : string
        Calibration product type (e.g. "G")
    """
    sensor_name = f'Calibration/Products/{cal_stream}/{product_type}'
    return cache.get(sensor_name)


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


def calc_gain_correction(sensor, index, targets=None):
    """Calculate correction sensor from gain calibration solution sensor.

    Given the gain calibration solution `sensor`, this extracts the time
    series of gains for the input specified by `index` (in the form (pol, ant))
    and interpolates them over time to get the corresponding complex correction
    terms. The optional `targets` parameter is a :class:`CategoricalData` i.e.
    a sensor indicating the target associated with each dump. The targets can
    be actual :class:`katpoint.Target` objects or indices, as long as they
    uniquely identify the target. If provided, interpolate solutions derived
    from one target only at dumps associated with that target, which is what
    you want for self-calibration solutions (but not for standard calibration
    based on gain calibrator sources).

    Invalid solutions (NaNs) are replaced by linear interpolations over time
    (separately for magnitude and phase), as long as some dumps have valid
    solutions on the appropriate target.
    """
    dumps = np.arange(sensor.events[-1])
    events = []
    gains = []
    for segment, value in sensor.segments():
        # Discard "invalid gain" placeholder (typically the initial value)
        if value is INVALID_GAIN:
            continue
        events.append(segment.start)
        gains.append(value[(Ellipsis,) + index])
    if not events:
        return np.full((len(dumps), 1), INVALID_GAIN)
    events = np.array(events)
    # Let the gains be shaped either (cal_n_chans, n_events) or (1, n_events)
    gains = np.atleast_2d(np.array(gains).T)
    # Assume all dumps have the same target by default, i.e. interpolate freely
    if targets is None:
        targets = CategoricalData([0], [0, len(dumps)])
    smooth_gains = np.full((len(dumps), gains.shape[0]), INVALID_GAIN)
    # Iterate over number of channels / "IFs" / subbands in gain product
    for target in targets.unique_values:
        on_target = (targets == target)
        for chan, gains_per_chan in enumerate(gains):
            valid = np.isfinite(gains_per_chan) & on_target[events]
            if valid.any():
                smooth_gains[on_target, chan] = complex_interp(
                    dumps[on_target], events[valid], gains_per_chan[valid])
    return np.reciprocal(smooth_gains)


def calibrate_flux(sensor, targets, gaincal_flux):
    """Apply flux scale to calibrator gains (aka flux calibration).

    Given the gain calibration solution `sensor`, this identifies the target
    associated with each set of solutions by looking up the gain events in the
    `targets` sensor, and then scales the gains by the inverse square root of
    the relevant flux if a valid match is found in the `gaincal_flux` dict. This
    is equivalent to the final step of the AIPS GETJY and CASA fluxscale tasks.
    """
    # If no calibration info is available, do nothing
    if not gaincal_flux:
        return sensor
    calibrated_gains = []
    for segment, gains in sensor.segments():
        # Ignore "invalid gain" placeholder (typically the initial value)
        if gains is INVALID_GAIN:
            calibrated_gains.append(ComparableArrayWrapper(gains))
            continue
        # Find the target at the time of the gain solution (i.e. gain calibrator)
        target = targets[segment.start]
        for name in [target.name] + target.aliases:
            flux = gaincal_flux.get(name, np.nan)
            # Scale the gains if a valid flux density was found for this target
            if flux > 0.0:
                calibrated_gains.append(ComparableArrayWrapper(gains / np.sqrt(flux)))
                break
        else:
            calibrated_gains.append(ComparableArrayWrapper(gains))
    return CategoricalData(calibrated_gains, sensor.events)


def add_applycal_sensors(cache, attrs, data_freqs, cal_stream, cal_substreams=None,
                         gaincal_flux={}):
    """Register virtual sensors for one calibration stream.

    This operates on a single calibration stream called `cal_stream` (possibly
    an alias), which derives from one or more underlying cal streams listed in
    `cal_substreams` and has stream attributes in `attrs`.

    The first set of virtual sensors maps all cal products into a unified
    namespace (template 'Calibration/Products/`cal_stream`/{product_type}').
    Map receptor inputs to the relevant indices in each calibration product
    based on the ants and pols found in `attrs`. Then register a virtual sensor
    per product type and per input in the SensorCache `cache`, with template
    'Calibration/Corrections/`cal_stream`/{product_type}/{inp}'. The virtual
    sensor function picks the appropriate correction calculator based on the
    cal product type, which also uses auxiliary info like the channel
    frequencies, `data_freqs`.

    Parameters
    ----------
    cache : :class:`~katdal.sensordata.SensorCache` object
        Sensor cache serving cal product sensors and receiving correction sensors
    attrs : dict-like
        Calibration stream attributes (e.g. a "cal" telstate view)
    data_freqs : array of float, shape (*F*,)
        Centre frequency of each frequency channel of visibilities, in Hz
    cal_stream : string
        Name of (possibly virtual) calibration stream (e.g. "l1")
    cal_substreams : sequence of string, optional
        Names of actual underlying calibration streams (e.g. ["cal"]),
        defaults to [`cal_stream`] itself
    gaincal_flux : dict mapping string to float, optional
        Flux density (in Jy) per gaincal target name, used to flux calibrate
        the "G" product, overriding the measured flux stored in `attrs`
        (if available). A value of None disables flux calibration.

    Returns
    -------
    cal_freqs : 1D array of float, or None
        Centre frequency of each frequency channel of calibration stream, in Hz
        (or None if no sensors were registered)
    """
    if cal_substreams is None:
        cal_substreams = [cal_stream]
    cal_ants = attrs.get('antlist', [])
    cal_pols = attrs.get('pol_ordering', [])
    cal_input_map = {ant + pol: (pol_idx, ant_idx)
                     for (pol_idx, pol) in enumerate(cal_pols)
                     for (ant_idx, ant) in enumerate(cal_ants)}
    if not cal_input_map:
        return
    try:
        cal_spw = SpectralWindow(attrs['center_freq'], None,
                                 attrs['n_chans'], sideband=1,
                                 bandwidth=attrs['bandwidth'])
        cal_freqs = cal_spw.channel_freqs
    except KeyError:
        logger.warning("Disabling cal stream '%s' due to missing "
                       "spectral attributes", cal_stream)
        return
    targets = cache.get('Observation/target')
    # Override pipeline fluxes (or disable flux calibration)
    if gaincal_flux is None:
        gaincal_flux = {}
    else:
        measured_flux = attrs.get('measured_flux', {}).copy()
        measured_flux.update(gaincal_flux)
        gaincal_flux = measured_flux

    def indirect_cal_product_name(name, product_type):
        # XXX The first underscore below is actually a telstate separator...
        return name.split('/')[-2] + '_product_' + product_type

    def indirect_cal_product_raw(cache, name, product_type):
        # XXX The first underscore below is actually a telstate separator...
        product_str = '_product_' + product_type
        raw_products = []
        for stream in cal_substreams:
            sensor_name = stream + product_str
            raw_product = cache.get(sensor_name, extract=False)
            assert isinstance(raw_product, SensorGetter), \
                sensor_name + ' is already extracted'
            raw_products.append(raw_product)
        if len(raw_products) == 1:
            return raw_products[0]
        else:
            raw_products = [raw.get() for raw in raw_products]
            timestamps = np.concatenate([raw_product.timestamp for raw_product in raw_products])
            values = np.concatenate([raw_product.value for raw_product in raw_products])
            ordered = timestamps.argsort()
            timestamps = timestamps[ordered]
            values = values[ordered]
            return SimpleSensorGetter(indirect_cal_product_name(name, product_type),
                                      timestamps, values)

    def indirect_cal_product(cache, name, product_type):
        try:
            n_parts = int(attrs[f'product_{product_type}_parts'])
        except KeyError:
            return indirect_cal_product_raw(cache, name, product_type)
        # Handle multi-part cal product (as produced by "split cal")
        # First collect all the parts as sensors (and mark missing ones as None)
        parts = []
        for n in range(n_parts):
            try:
                part = indirect_cal_product_raw(cache, name + str(n), product_type + str(n))
            except KeyError:
                part = SimpleSensorGetter(name + str(n), np.array([]), np.array([]))
            parts.append(part)

        # Stitch together values with the same timestamp
        parts = [part.get() for part in parts]
        timestamps = []
        values = []
        part_indices = [0] * n_parts
        part_timestamps = [
            part.timestamp[0] if len(part.timestamp) else np.inf
            for part in parts
        ]
        while True:
            next_timestamp = min(part_timestamps)
            if next_timestamp == np.inf:
                break
            pieces = []
            for ts, ind, part in zip(part_timestamps, part_indices, parts):
                if ts == next_timestamp:
                    piece = ComparableArrayWrapper.unwrap(part.value[ind])
                    pieces.append(piece)
                else:
                    pieces.append(None)
            if any(piece is None for piece in pieces):
                invalid = np.full_like(piece, INVALID_GAIN)
                pieces = [piece if piece is not None else invalid for piece in pieces]
            timestamps.append(next_timestamp)
            value = np.concatenate(pieces, axis=0)
            values.append(ComparableArrayWrapper(value))
            for i, part in enumerate(parts):
                if part_timestamps[i] == next_timestamp:
                    ts = part.timestamp
                    part_indices[i] += 1
                    part_timestamps[i] = ts[part_indices[i]] if part_indices[i] < len(ts) else np.inf
        if not timestamps:
            raise KeyError(f"No cal product '{name}' parts found (expected {n_parts})")
        return SimpleSensorGetter(indirect_cal_product_name(name, product_type),
                                  np.array(timestamps), np.array(values))

    def calc_correction_per_input(cache, name, inp, product_type):
        """Calculate correction sensor for input `inp` from cal solutions."""
        product_sensor = get_cal_product(cache, cal_stream, product_type)
        try:
            index = cal_input_map[inp]
        except KeyError:
            raise KeyError(f"No calibration solutions available for input '{inp}' - "
                           f'available ones are {sorted(cal_input_map.keys())}')
        if product_type == 'K':
            correction_sensor = calc_delay_correction(product_sensor, index,
                                                      data_freqs)
        elif product_type == 'B':
            correction_sensor = calc_bandpass_correction(product_sensor, index,
                                                         data_freqs, cal_freqs)
        elif product_type == 'G':
            product_sensor = calibrate_flux(product_sensor, targets, gaincal_flux)
            correction_sensor = calc_gain_correction(product_sensor, index)
        elif product_type in ('GPHASE', 'GAMP_PHASE'):
            correction_sensor = calc_gain_correction(product_sensor, index, targets)
        else:
            raise KeyError(f"Unknown calibration product type '{product_type}' - "
                           f'available ones are {CAL_PRODUCT_TYPES}')
        cache[name] = correction_sensor
        return correction_sensor

    template = f'Calibration/Products/{cal_stream}/{{product_type}}'
    cache.virtual[template] = indirect_cal_product
    template = f'Calibration/Corrections/{cal_stream}/{{product_type}}/{{inp}}'
    cache.virtual[template] = calc_correction_per_input
    return cal_freqs


@numba.jit(nopython=True, nogil=True)
def _correction_inputs_to_corrprods(g_per_cp, g_per_input, input1_index, input2_index):
    """Convert gains per input to gains per correlation product."""
    for i in range(g_per_cp.shape[0]):
        for j in range(g_per_cp.shape[1]):
            g_per_cp[i, j] = (g_per_input[i, input1_index[j]]
                              * np.conj(g_per_input[i, input2_index[j]]))


class CorrectionParams:
    """Data needed to compute corrections in :func:`calc_correction_per_corrprod`.

    Once constructed, the data in this class must not be modified, as it will
    be baked into dask graphs.

    Parameters
    ----------
    inputs : list of str
        Names of inputs, in the same order as the input axis of products
    input1_index, input2_index : array of int
        Indices into `inputs` of first and second items of correlation product
    corrections : dict
        A dictionary (indexed by cal product name) of lists (indexed
        by input) of sequences (indexed by dump) of numpy arrays, with
        corrections to apply.
    channel_maps : dict
        A dictionary (indexed by cal product name) of functions (signature
        `g = channel_map(g, channels)`) that map the frequency axis of the
        cal product `g` onto the frequency axis of the visibility data, where
        the vis frequency axis will be indexed by the slice `channels`.
    """
    def __init__(self, inputs, input1_index, input2_index, corrections, channel_maps):
        self.inputs = inputs
        self.input1_index = input1_index
        self.input2_index = input2_index
        self.corrections = corrections
        self.channel_maps = channel_maps


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
        Corrections per input, together with correlation product indices

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
    for cal_product, product_corrections in params.corrections.items():
        channel_map = params.channel_maps[cal_product]
        for i in range(len(params.inputs)):
            sensor = product_corrections[i]
            g_per_channel = sensor[dump]
            g_per_input[i] *= channel_map(g_per_channel, channels)
    # Transpose to (channel, input) order, and ensure C ordering
    g_per_input = np.ascontiguousarray(g_per_input.T)
    g_per_cp = np.empty((n_channels, len(params.input1_index)), dtype='complex64')
    _correction_inputs_to_corrprods(g_per_cp, g_per_input,
                                    params.input1_index, params.input2_index)
    return g_per_cp


def _correction_block(block_info, params):
    """Calculate applycal correction for a single time-freq-baseline chunk."""
    slices = tuple(slice(*loc) for loc in block_info[None]['array-location'])
    block_shape = block_info[None]['chunk-shape']
    correction = np.empty(block_shape, np.complex64)
    # TODO: make calc_correction_per_corrprod multi-dump aware
    for n, dump in enumerate(range(slices[0].start, slices[0].stop)):
        correction[n] = calc_correction_per_corrprod(dump, slices[1], params)
    return correction


def calc_correction(chunks, cache, corrprods, cal_products, data_freqs,
                    all_cal_freqs, skip_missing_products=False):
    """Create a dask array containing applycal corrections.

    Parameters
    ----------
    chunks : tuple of tuple of int
        Chunking scheme of the resulting array, in normalized form (see
        :func:`dask.array.core.normalize_chunks`).
    cache : :class:`~katdal.sensordata.SensorCache` object
        Sensor cache, used to look up individual correction sensors
    corrprods : sequence of (string, string)
        Selected correlation products as pairs of correlator input labels
    cal_products : sequence of string
        Calibration products that will contribute to corrections (e.g. ["l1.G"])
    data_freqs : array of float, shape (*F*,)
        Centre frequency of each frequency channel of visibilities, in Hz
    all_cal_freqs : dict
        Dictionary mapping cal stream name (e.g. "l1") to array of associated
        frequencies
    skip_missing_products : bool
        If True, skip products with missing sensors instead of raising KeyError

    Returns
    -------
    final_cal_products : list of string
        List of calibration products in the order that they will be applied
        (potentially a subset of `cal_products` if skipping missing products)
    corrections : :class:`dask.array.Array` object, or None
        Dask array that produces corrections for entire vis array, or `None` if
        no calibration products were found (either `cal_products` is empty or all
        products had some missing sensors and `skip_missing_products` is True)

    Raises
    ------
    KeyError
        If a correction sensor for a given input and cal product is not found
        (and `skip_missing_products` is False)
    """
    shape = tuple(sum(bd) for bd in chunks)
    if len(chunks[2]) > 1:
        logger.warning('ignoring chunking on baseline axis')
        chunks = (chunks[0], chunks[1], (shape[2],))
    inputs = sorted(set(np.ravel(corrprods)))
    input1_index = np.array([inputs.index(cp[0]) for cp in corrprods])
    input2_index = np.array([inputs.index(cp[1]) for cp in corrprods])
    corrections = {}
    channel_maps = {}
    for cal_product in cal_products:
        cal_stream, product_type = _parse_cal_product(cal_product)
        sensor_prefix = f'Calibration/Corrections/{cal_stream}/{product_type}/'
        corrections_per_product = []
        for i, inp in enumerate(inputs):
            try:
                sensor = cache.get(sensor_prefix + inp)
            except KeyError:
                if skip_missing_products:
                    break
                else:
                    raise
            # Indexing CategoricalData by dump is relatively slow (tens of
            # microseconds), so expand it into a plain-old Python list.
            if isinstance(sensor, CategoricalData):
                data = [None] * sensor.events[-1]
                for s, v in sensor.segments():
                    for j in range(s.start, s.stop):
                        data[j] = v
            else:
                data = sensor
            corrections_per_product.append(data)
        else:
            corrections[cal_product] = corrections_per_product
            # Frequency configuration for *stream* (not necessarily for product)
            cal_stream_freqs = all_cal_freqs[cal_stream]
            # Get number of frequency channels of *corrections* by inspecting it
            # at first dump for each input and picking max to reject bad inputs.
            # Expected to be either 1, len(cal_stream_freqs) or len(data_freqs).
            correction_n_chans = max([len(np.atleast_1d(corr_per_input[0]))
                                      for corr_per_input in corrections_per_product])
            if correction_n_chans == 1:
                # Scalar values will be broadcast by NumPy - no slicing required
                channel_maps[cal_product] = lambda g, channels: g
            elif correction_n_chans == len(data_freqs) and (
                    # This test indicates that correction frequencies either differ
                    # from those of cal stream (i.e. already interpolated), or the
                    # cal stream matches the data freqs to within 1 mHz anyway.
                    len(cal_stream_freqs) != len(data_freqs)
                    or np.allclose(cal_stream_freqs, data_freqs, rtol=0, atol=1e-3)):
                # Corrections are already lined up with data - slice directly
                channel_maps[cal_product] = lambda g, channels: g[channels]
            else:
                # Pick closest cal channel for each data channel
                expand = np.abs(data_freqs[:, np.newaxis]
                                - cal_stream_freqs[np.newaxis, :]).argmin(axis=-1)
                channel_maps[cal_product] = lambda g, channels: g[expand[channels]]
    final_cal_products = list(corrections.keys())
    if not final_cal_products:
        return final_cal_products, None
    params = CorrectionParams(inputs, input1_index, input2_index,
                              corrections, channel_maps)
    name = 'corrections[{}]'.format(','.join(sorted(final_cal_products)))
    return (final_cal_products,
            da.map_blocks(_correction_block, dtype=np.complex64, chunks=chunks,
                          name=name, params=params))


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
                c = cc.real * cc.real + cc.imag * cc.imag
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
