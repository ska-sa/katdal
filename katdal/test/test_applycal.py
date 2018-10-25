###############################################################################
# Copyright (c) 2018, National Research Foundation (Square Kilometre Array)
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

"""Tests for :py:mod:`katdal.applycal`."""
from __future__ import print_function, division, absolute_import
from builtins import object, range

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_raises, assert_equal
import dask.array as da

from katdal.spectral_window import SpectralWindow
from katdal.sensordata import SensorCache
from katdal.categorical import ComparableArrayWrapper, CategoricalData
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.applycal import (complex_interp, calc_correction_per_corrprod,
                             get_cal_product, calc_delay_correction,
                             calc_bandpass_correction, calc_gain_correction,
                             add_applycal_sensors, add_applycal_transform,
                             apply_vis_correction)


POLS = ['v', 'h']
ANTS = ['m000', 'm001', 'm002', 'm003', 'm004']
N_DUMPS = 100
SAMPLE_RATE = 1712.0

CENTRE_FREQ = 1284.0
BANDWIDTH = 856.0
N_CHANS = 128
FREQS = SpectralWindow(CENTRE_FREQ, None, N_CHANS,
                       sideband=1, bandwidth=BANDWIDTH).channel_freqs
CAL_CENTRE_FREQ = 1200.0
CAL_BANDWIDTH = 800.0
CAL_N_CHANS = 64
CAL_FREQS = SpectralWindow(CAL_CENTRE_FREQ, None, CAL_N_CHANS,
                           sideband=1, bandwidth=CAL_BANDWIDTH).channel_freqs

INPUTS = [ant + pol for ant in ANTS for pol in POLS]
INDEX1, INDEX2 = np.triu_indices(len(INPUTS))
CORRPRODS = [(INPUTS[i1], INPUTS[i2]) for i1, i2 in zip(INDEX1, INDEX2)]
N_CORRPRODS = len(CORRPRODS)

SKIP_ANT = 0
BAD_DELAY_ANT = 1
BAD_BANDPASS_ANT = 2
BAD_CHANNELS = np.full(CAL_N_CHANS, False)
BAD_CHANNELS[30:40] = True
BAD_CHANNELS[50] = True
BANDPASS_PARTS = 4
GAIN_EVENTS = list(range(0, N_DUMPS, 10))
BAD_GAIN_ANT = 3
BAD_DUMPS = [20, 40]
CAL_PRODUCTS = ['K', 'B', 'G']

ATTRS = {'cal_antlist': ANTS, 'cal_pol_ordering': POLS,
         'cal_center_freq': CAL_CENTRE_FREQ, 'cal_bandwidth': CAL_BANDWIDTH,
         'cal_n_chans': CAL_N_CHANS, 'cal_product_B_parts': BANDPASS_PARTS}


def create_delay(pol, ant):
    """Synthesise a delay in seconds from `pol` and `ant` indices."""
    return (100 ** pol * ant / SAMPLE_RATE) if ant != BAD_DELAY_ANT else np.nan


def create_bandpass(pol, ant):
    """Synthesise a bandpass response from `pol` and `ant` indices."""
    bp = (CAL_N_CHANS * ant + np.arange(CAL_N_CHANS) +
          1j * (1 + pol) * np.ones(CAL_N_CHANS))
    if ant == BAD_BANDPASS_ANT:
        bp[:] = np.nan
    else:
        bp[BAD_CHANNELS] = np.nan
    return bp.astype(np.complex64)


def create_gain(pol, ant):
    """Synthesise a gain time series from `pol`, `ant` indices and events."""
    events = np.array(GAIN_EVENTS)
    gains = np.ones_like(events, dtype=np.complex64)
    gains *= (-1) ** pol * (1 + ant) * np.exp(2j * np.pi * events / 100.)
    if ant == BAD_GAIN_ANT:
        gains[:] = np.nan
    bad_events = [GAIN_EVENTS.index(dump) for dump in BAD_DUMPS]
    gains[bad_events] = np.nan
    return gains


def create_product(func):
    """Form calibration product by evaluating `func` for all pols and ants."""
    pols = range(len(POLS))
    ants = range(len(ANTS))
    values = [func(pol, ant) for pol in pols for ant in ants]
    values = np.array(values)
    pol_ant = (len(pols), len(ants))
    # Pols, ants are final 2 dims; values shape (8, 128) becomes (128, 2, 4)
    rest = values.shape[1:]
    return np.moveaxis(values, 0, -1).reshape(rest + pol_ant)


def create_sensor_cache():
    """Create a SensorCache for testing applycal sensors."""
    cache = {}
    # Add delay product
    delays = create_product(create_delay)
    cache['cal_product_K'] = CategoricalData([np.zeros_like(delays), delays],
                                             events=[0, 10, N_DUMPS])
    # Add bandpass product (multi-part)
    bandpasses = create_product(create_bandpass)
    for part, bp in enumerate(np.split(bandpasses, BANDPASS_PARTS)):
        cache['cal_product_B' + str(part)] = CategoricalData(
            [np.ones_like(bp), bp], events=[0, 12, N_DUMPS])
    # Add gain product
    gains = create_product(create_gain)
    gains = [ComparableArrayWrapper(g) for g in gains]
    cache['cal_product_G'] = CategoricalData(gains, GAIN_EVENTS + [N_DUMPS])
    # Construct sensor cache
    return SensorCache(cache, timestamps=np.arange(N_DUMPS, dtype=float),
                       dump_period=1.)


def delay_corrections(pol, ant):
    """Figure out N_CHANS delay corrections given `pol` and `ant` indices."""
    # Zero out missing delays (indicated by NaN)
    delay = np.nan_to_num(create_delay(pol, ant))
    return np.exp(2j * np.pi * delay * FREQS).astype('complex64')


def bandpass_corrections(pol, ant):
    """Figure out N_CHANS bandpass corrections given `pol`, `ant` indices."""
    bp = create_bandpass(pol, ant)
    valid = np.isfinite(bp)
    if valid.any():
        bp = complex_interp(FREQS, CAL_FREQS[valid], bp[valid],
                            left=np.nan, right=np.nan)
    else:
        bp = np.full(N_CHANS, np.nan + 1j * np.nan, dtype=bp.dtype)
    return np.reciprocal(bp)


def gain_corrections(pol, ant):
    """Figure out N_DUMPS gain corrections given `pol` and `ant` indices."""
    dumps = np.arange(N_DUMPS)
    events = np.array(GAIN_EVENTS)
    gains = create_gain(pol, ant)
    valid = np.isfinite(gains)
    if valid.any():
        gains = complex_interp(dumps, events[valid], gains[valid])
    else:
        gains = np.full(N_DUMPS, np.nan + 1j * np.nan, dtype=gains.dtype)
    return np.reciprocal(gains)


def gains_per_corrprod(dumps, channels, corrprods=()):
    """Predict corrprod correction for a time-frequency-baseline selection."""
    input_map = {ant + pol: (pol_idx, ant_idx)
                 for (pol_idx, pol) in enumerate(POLS)
                 for (ant_idx, ant) in enumerate(ANTS)}
    gains_per_input = np.ones((len(dumps), N_CHANS, len(INPUTS)),
                              dtype='complex64')
    if 'K' in CAL_PRODUCTS:
        gains_per_input *= np.array([delay_corrections(*input_map[inp])
                                     for inp in INPUTS]).T
    if 'B' in CAL_PRODUCTS:
        gains_per_input *= np.array([bandpass_corrections(*input_map[inp])
                                     for inp in INPUTS]).T
    if 'G' in CAL_PRODUCTS:
        gains_per_input *= np.array([gain_corrections(*input_map[inp])[dumps]
                                     for inp in INPUTS]).T[:, np.newaxis]
    gains_per_input = gains_per_input[:, channels, :]
    gain1 = gains_per_input[:, :, INDEX1[corrprods]]
    gain2 = gains_per_input[:, :, INDEX2[corrprods]]
    return gain1 * gain2.conj()


class TestComplexInterp(object):
    """Test the :func:`~katdal.applycal.complex_interp` function."""
    def setup(self):
        self.xi = np.arange(1., 10.)
        self.yi_unit = np.exp(2j * np.pi * self.xi / 10.)
        rs = np.random.RandomState(1234)
        self.x = 10 * rs.rand(100)
        self.yi = 10 * rs.rand(len(self.yi_unit)) * self.yi_unit

    def test_basic(self):
        y = complex_interp(self.x, self.xi, self.yi)
        mag_y = np.interp(self.x, self.xi, np.abs(self.yi))
        assert_allclose(np.abs(y), mag_y, rtol=1e-14)
        phase_y = np.interp(self.x, self.xi, np.unwrap(np.angle(self.yi)))
        assert_allclose(np.mod(np.angle(y), 2 * np.pi),
                        np.mod(phase_y, 2 * np.pi), rtol=1e-14)

    def test_exact_values(self):
        y = complex_interp(self.xi, self.xi, self.yi)
        assert_allclose(y, self.yi, rtol=1e-14)

    def test_correct_wrap(self):
        xi = np.arange(2)
        yi = np.array([-1+1j, -2-2j])
        x = 0.5
        y = -0.5 * (np.sqrt(2) + np.sqrt(8))   # np.interp has y = -1.5-0.5j
        assert_allclose(complex_interp(x, xi, yi), y, rtol=1e-14)

    def test_phase_only_interpolation(self):
        y = complex_interp(self.x, self.xi, self.yi_unit)
        assert_allclose(np.abs(y), 1.0, rtol=1e-14)

    def test_complex64_interpolation(self):
        yi = self.yi_unit.astype(np.complex64)
        y = complex_interp(self.x, self.xi, yi)
        assert_equal(y.dtype, yi.dtype)
        assert_allclose(np.abs(y), 1.0, rtol=1e-7)

    def test_left_right(self):
        # Extend yi[0] and yi[-1] at edges
        y = complex_interp(sorted(self.x), self.xi, self.yi)
        assert_allclose(y[0], self.yi[0], rtol=1e-14)
        assert_allclose(y[-1], self.yi[-1], rtol=1e-14)
        # Explicit edge values
        y = complex_interp(sorted(self.x), self.xi, self.yi, left=0, right=1j)
        assert_allclose(y[0], 0, rtol=1e-14)
        assert_allclose(y[-1], 1j, rtol=1e-14)


class TestCorrectionPerInput(object):
    """Test the :func:`~katdal.applycal.calc_*_correction` functions."""
    def setup(self):
        self.cache = create_sensor_cache()

    def test_get_cal_product_basic(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'K')
        product = create_product(create_delay)
        assert_array_equal(product_sensor[0], np.zeros_like(product))
        assert_array_equal(product_sensor[10], product)

    def test_get_cal_product_multipart(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'B')
        product = create_product(create_bandpass)
        assert_array_equal(product_sensor[0], np.ones_like(product))
        assert_array_equal(product_sensor[12], product)

    def test_calc_delay_correction(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'K')
        constant_bandpass = np.ones(N_CHANS, dtype='complex64')
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_delay_correction(product_sensor, (m, n), FREQS)
                assert_array_equal(sensor[n], constant_bandpass)
                assert_array_equal(sensor[10 + n], delay_corrections(m, n))

    def test_calc_bandpass_correction(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'B')
        constant_bandpass = np.ones(N_CHANS, dtype='complex64')
        constant_bandpass[FREQS < CAL_FREQS[0]] = np.nan
        constant_bandpass[FREQS > CAL_FREQS[-1]] = np.nan
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_bandpass_correction(product_sensor, (m, n),
                                                  FREQS, CAL_FREQS)
                assert_array_equal(sensor[n], constant_bandpass)
                assert_array_equal(sensor[12 + n], bandpass_corrections(m, n))

    def test_calc_gain_correction(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'G')
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_gain_correction(product_sensor, (m, n))
                assert_array_equal(sensor[:], gain_corrections(m, n))


class TestVirtualCorrectionSensors(object):
    """Test :func:`~katdal.applycal.add_applycal_sensors` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS)

    def test_add_sensors_does_nothing_if_no_ants_or_pols(self):
        cache = create_sensor_cache()
        n_virtuals_before = len(cache.virtual)
        add_applycal_sensors(cache, {}, [])
        n_virtuals_after = len(cache.virtual)
        assert_equal(n_virtuals_after, n_virtuals_before)

    def test_delay_sensors(self):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = 'Calibration/{}{}_correction_K'.format(ant, pol)
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[10 + n], delay_corrections(m, n))

    def test_bandpass_sensors(self):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = 'Calibration/{}{}_correction_B'.format(ant, pol)
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[12 + n], bandpass_corrections(m, n))

    def test_gain_sensors(self):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = 'Calibration/{}{}_correction_G'.format(ant, pol)
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[:], gain_corrections(m, n))

    def test_unknown_inputs_and_products(self):
        known_input = 'Calibration/{}{}'.format(ANTS[0], POLS[0])
        with assert_raises(KeyError):
            self.cache.get('Calibration/unknown_correction_K')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_unknown')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_K_unknown')


class TestCorrectionPerCorrprod(object):
    """Test :func:`~katdal.applycal.calc_correction_per_corrprod` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS)

    def test_correction_per_corrprod(self):
        dump = 15
        channels = list(range(22, 38))
        gains = calc_correction_per_corrprod(dump, channels, self.cache,
                                             INPUTS, INDEX1, INDEX2,
                                             CAL_PRODUCTS)[np.newaxis]
        expected_gains = gains_per_corrprod([dump], channels)
        assert_array_equal(gains, expected_gains)


class TestApplyCal(object):
    """Test :func:`~katdal.applycal.add_applycal_transform` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS)
        time_keep = np.full(N_DUMPS, False, dtype=np.bool_)
        time_keep[10:20] = True
        freq_keep = np.full(N_CHANS, False, dtype=np.bool_)
        freq_keep[22:38] = True
        corrprod_keep = np.full(N_CORRPRODS, True, dtype=np.bool_)
        # Throw out one antenna
        for n, inp in enumerate(INPUTS):
            if inp.startswith(ANTS[SKIP_ANT]):
                corrprod_keep[INDEX1 == n] = False
                corrprod_keep[INDEX2 == n] = False
        self.stage1 = (time_keep, freq_keep, corrprod_keep)
        # List of selected correlation products
        self.corrprods = [cp for n, cp in enumerate(CORRPRODS)
                          if corrprod_keep[n]]

    def test_applycal_vis(self):
        vis_real = np.random.randn(N_DUMPS, N_CHANS, N_CORRPRODS)
        vis_imag = np.random.randn(N_DUMPS, N_CHANS, N_CORRPRODS)
        vis = np.asarray(vis_real + 1j * vis_imag, dtype='complex64')
        vis_dask = da.from_array(vis, chunks=(10, 4, 6))
        indexer = DaskLazyIndexer(vis_dask, self.stage1)
        add_applycal_transform(indexer, self.cache, self.corrprods,
                               CAL_PRODUCTS, apply_vis_correction)
        # Apply stage 2 selection on top of stage 1
        stage2 = np.s_[5:7, 2:5, :]
        stage1_indices = tuple(k.nonzero()[0] for k in self.stage1)
        final_indices = tuple(i[s] for s, i in zip(stage2, stage1_indices))
        gains = gains_per_corrprod(*final_indices)
        # Leave visibilities alone where gains are NaN
        gains[np.isnan(gains)] = 1.0
        # Quick and dirty oindex of vis (yet another way doing axes in reverse)
        selected_vis = vis
        dims = reversed(range(vis.ndim))
        for dim, indices in zip(dims, reversed(final_indices)):
            selected_vis = np.take(selected_vis, indices, axis=dim)
        expected_vis = selected_vis * gains
        calibrated_vis = indexer[stage2]
        assert_array_equal(calibrated_vis, expected_vis)
