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
from katdal.applycal import (complex_interp,
                             has_cal_product, get_cal_product, INVALID_GAIN,
                             calc_delay_correction, calc_bandpass_correction,
                             calc_gain_correction, apply_vis_correction,
                             apply_weights_correction, apply_flags_correction,
                             add_applycal_sensors, calc_correction)
from katdal.flags import POSTPROC


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
BAD_GAIN_DUMPS = [20, 40]
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
        bp[:] = INVALID_GAIN
    else:
        bp[BAD_CHANNELS] = INVALID_GAIN
    return bp.astype(np.complex64)


def create_gain(pol, ant):
    """Synthesise a gain time series from `pol`, `ant` indices and events."""
    events = np.array(GAIN_EVENTS)
    gains = np.ones_like(events, dtype=np.complex64)
    gains *= (-1) ** pol * (1 + ant) * np.exp(2j * np.pi * events / 100.)
    if ant == BAD_GAIN_ANT:
        gains[:] = INVALID_GAIN
    bad_events = [GAIN_EVENTS.index(dump) for dump in BAD_GAIN_DUMPS]
    gains[bad_events] = INVALID_GAIN
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


def create_sensor_cache(bandpass_parts=BANDPASS_PARTS):
    """Create a SensorCache for testing applycal sensors."""
    cache = {}
    # Add delay product (single)
    delays = create_product(create_delay)
    cache['cal_product_K'] = CategoricalData([np.zeros_like(delays), delays],
                                             events=[0, 10, N_DUMPS])
    # Add bandpass product (multi-part)
    bandpasses = create_product(create_bandpass)
    for part, bp in enumerate(np.split(bandpasses, bandpass_parts)):
        cache['cal_product_B' + str(part)] = CategoricalData(
            [np.ones_like(bp), bp], events=[0, 12, N_DUMPS])
    # Add gain product (single multi-part as a corner case)
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
    return np.exp(-2j * np.pi * delay * FREQS).astype('complex64')


def bandpass_corrections(pol, ant):
    """Figure out N_CHANS bandpass corrections given `pol`, `ant` indices."""
    bp = create_bandpass(pol, ant)
    valid = np.isfinite(bp)
    if valid.any():
        bp = complex_interp(FREQS, CAL_FREQS[valid], bp[valid],
                            left=INVALID_GAIN, right=INVALID_GAIN)
    else:
        bp = np.full(N_CHANS, INVALID_GAIN)
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
        gains = np.full(N_DUMPS, INVALID_GAIN)
    return np.reciprocal(gains)


def corrections_per_corrprod(dumps, channels, corrprods=()):
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


class TestCalProductAccess(object):
    """Test the :func:`~katdal.applycal.*_cal_product` functions."""
    def setup(self):
        self.cache = create_sensor_cache()

    def test_has_cal_product(self):
        assert_equal(has_cal_product(self.cache, ATTRS, 'K'), True)
        assert_equal(has_cal_product(self.cache, ATTRS, 'B'), True)
        assert_equal(has_cal_product(self.cache, ATTRS, 'G'), True)
        assert_equal(has_cal_product(self.cache, ATTRS, 'haha'), False)
        # Remove parts of multi-part cal product one by one
        cache = create_sensor_cache()
        for n in range(BANDPASS_PARTS):
            assert_equal(has_cal_product(cache, ATTRS, 'B'), True)
            del cache['cal_product_B' + str(n)]
        # All parts of multi-part cal product gone
        assert_equal(has_cal_product(cache, ATTRS, 'B'), False)

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

    def test_get_cal_product_single_multipart(self):
        cache = create_sensor_cache(bandpass_parts=1)
        attrs = ATTRS.copy()
        attrs['cal_product_B_parts'] = 1
        product_sensor = get_cal_product(cache, attrs, 'B')
        product = create_product(create_bandpass)
        assert_array_equal(product_sensor[0], np.ones_like(product))
        assert_array_equal(product_sensor[12], product)

    def test_get_cal_product_missing_parts(self):
        cache = create_sensor_cache()
        product = create_product(create_bandpass)
        n_chans_per_part = CAL_N_CHANS // BANDPASS_PARTS
        # Remove parts of multi-part cal product one by one
        for n in range(BANDPASS_PARTS - 1):
            del cache['cal_product_B' + str(n)]
            product_sensor = get_cal_product(cache, ATTRS, 'B')
            part = slice(n * n_chans_per_part, (n + 1) * n_chans_per_part)
            product[part] = INVALID_GAIN
            assert_array_equal(product_sensor[12], product)
        # All parts gone triggers a KeyError
        del cache['cal_product_B' + str(BANDPASS_PARTS - 1)]
        with assert_raises(KeyError):
            get_cal_product(cache, ATTRS, 'B')

    def test_get_cal_product_gain(self):
        product_sensor = get_cal_product(self.cache, ATTRS, 'G')
        product = create_product(create_gain)
        assert_array_equal(product_sensor[GAIN_EVENTS], product)


class TestCorrectionPerInput(object):
    """Test the :func:`~katdal.applycal.calc_*_correction` functions."""
    def setup(self):
        self.cache = create_sensor_cache()

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
        constant_bandpass[FREQS < CAL_FREQS[0]] = INVALID_GAIN
        constant_bandpass[FREQS > CAL_FREQS[-1]] = INVALID_GAIN
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

    def test_add_sensors_does_nothing_if_no_ants_pols_or_spw(self):
        cache = create_sensor_cache()
        n_virtuals_before = len(cache.virtual)
        add_applycal_sensors(cache, {}, [])
        n_virtuals_after = len(cache.virtual)
        assert_equal(n_virtuals_after, n_virtuals_before)
        attrs = ATTRS.copy()
        del attrs['cal_center_freq']
        add_applycal_sensors(self.cache, attrs, FREQS)
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


class TestCalcCorrection(object):
    """Test :func:`~katdal.applycal.calc_correction` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS)

    def test_calc_correction(self):
        dump = 15
        channels = np.s_[22:38]
        shape = (N_DUMPS, N_CHANS, N_CORRPRODS)
        chunks = da.core.normalize_chunks((10, 5, -1), shape)
        corrections = calc_correction(chunks, self.cache, CORRPRODS, CAL_PRODUCTS)
        corrections = corrections[dump:dump+1, channels].compute()
        expected_corrections = corrections_per_corrprod([dump], channels)
        assert_array_equal(corrections, expected_corrections)


class TestApplyCal(object):
    """Test :func:`~katdal.applycal.apply_vis_correction` and friends"""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS)

    def _applycal(self, array, apply_correction):
        """Calibrate `array` with `apply_correction` and return all factors."""
        array_dask = da.from_array(array, chunks=(10, 4, 6))
        correction = calc_correction(array_dask.chunks, self.cache, CORRPRODS, CAL_PRODUCTS)
        corrected = da.core.elemwise(apply_correction, array_dask, correction, dtype=array_dask.dtype)
        return corrected.compute(), correction.compute()

    def test_applycal_vis(self):
        vis_real = np.random.randn(N_DUMPS, N_CHANS, N_CORRPRODS)
        vis_imag = np.random.randn(N_DUMPS, N_CHANS, N_CORRPRODS)
        vis = np.asarray(vis_real + 1j * vis_imag, dtype='complex64')
        calibrated_vis, corrections = self._applycal(vis, apply_vis_correction)
        # Leave visibilities alone where gains are NaN
        corrections[np.isnan(corrections)] = 1.0
        vis *= corrections
        assert_array_equal(calibrated_vis, vis)

    def test_applycal_weights(self):
        weights = np.random.rand(N_DUMPS, N_CHANS,
                                 N_CORRPRODS).astype('float32')
        calibrated_weights, corrections = self._applycal(weights, apply_weights_correction)
        # Zero the weights where the gains are NaN or zero
        corrections2 = corrections.real ** 2 + corrections.imag ** 2
        corrections2[np.isnan(corrections2)] = np.inf
        corrections2[corrections2 == 0] = np.inf
        weights /= corrections2
        assert_array_equal(calibrated_weights, weights)

    def test_applycal_flags(self):
        flags = np.random.randint(0, 128, (N_DUMPS, N_CHANS, N_CORRPRODS), np.uint8)
        calibrated_flags, corrections = self._applycal(flags, apply_flags_correction)
        flags |= np.where(np.isnan(corrections), np.uint8(POSTPROC), np.uint8(0))
        assert_array_equal(calibrated_flags, flags)
