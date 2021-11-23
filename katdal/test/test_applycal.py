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

"""Tests for :py:mod:`katdal.applycal`."""

from functools import partial

import dask.array as da
import katpoint
import numpy as np
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_allclose, assert_array_equal

from katdal.applycal import (INVALID_GAIN, add_applycal_sensors,
                             apply_flags_correction, apply_vis_correction,
                             apply_weights_correction,
                             calc_bandpass_correction, calc_correction,
                             calc_delay_correction, calc_gain_correction,
                             calibrate_flux, complex_interp, get_cal_product)
from katdal.categorical import (CategoricalData, ComparableArrayWrapper,
                                sensor_to_categorical)
from katdal.flags import POSTPROC
from katdal.sensordata import SensorCache, SimpleSensorGetter
from katdal.spectral_window import SpectralWindow
from katdal.visdatav4 import SENSOR_PROPS

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
DATA_TO_CAL_CHANNEL = np.abs(FREQS[:, np.newaxis]
                             - CAL_FREQS[np.newaxis, :]).argmin(axis=-1)

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
GAIN_EVENTS = list(range(10, N_DUMPS, 10))
BAD_GAIN_ANT = 3
BAD_GAIN_DUMPS = [20, 40]

TARGETS = np.array([katpoint.Target('gaincal1, radec, 0, -90'),
                    katpoint.Target('other | gaincal2, radec, 0, -80')])
TARGET_INDICES = np.arange(len(GAIN_EVENTS)) % 2
FLUX_VALUES = np.array([16.0, 4.0])
FLUX_SCALE_FACTORS = 1.0 / np.sqrt(FLUX_VALUES[TARGET_INDICES])
FLUXES = {'gaincal1': FLUX_VALUES[0], 'gaincal2': FLUX_VALUES[1]}
# The measured flux for gaincal1 is wrong on purpose so that we have to
# override it. There is also an extra unknown gain calibrator in the mix.
PIPELINE_FLUXES = {'gaincal1': FLUX_VALUES[0] / 2, 'gaincal2': FLUX_VALUES[1],
                   'unknown_gaincal': 10.0}
FLUX_OVERRIDES = {'gaincal1': FLUX_VALUES[0]}

CAL_STREAM = 'cal'
ATTRS = {'antlist': ANTS, 'pol_ordering': POLS,
         'center_freq': CAL_CENTRE_FREQ, 'bandwidth': CAL_BANDWIDTH,
         'n_chans': CAL_N_CHANS, 'product_B_parts': BANDPASS_PARTS,
         'measured_flux': PIPELINE_FLUXES}
CAL_PRODUCT_TYPES = ('K', 'B', 'G', 'GPHASE')
CAL_PRODUCTS = [f'{CAL_STREAM}.{prod}' for prod in CAL_PRODUCT_TYPES]


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


def create_gain(pol, ant, multi_channel=False, targets=False, fluxes=False):
    """Synthesise a gain time series from `pol`, `ant` indices and events.

    The gain can also vary with frequency a la GPHASE if `multi_channel` is
    True, in which case all valid gains are also normalised to magnitude 1 to
    resemble GPHASE. Optionally tweak the gains to reflect the effect of flux
    calibration and target-dependent gains.
    """
    events = np.array(GAIN_EVENTS)
    gains = np.ones_like(events, dtype=np.complex64)
    # The gain magnitude reflects the input or (ant, pol) index
    factor = len(POLS) * ant + pol + 1
    # The gain phase drifts as a function of time but over a limited range
    # so that the target index can be reflected in the sign of the gain
    gains *= factor * np.exp(2j * np.pi * events / N_DUMPS / 12)
    if fluxes:
        gains *= FLUX_SCALE_FACTORS
    if targets:
        gains *= (-1) ** np.arange(len(GAIN_EVENTS))
    if multi_channel:
        gains = np.outer(gains, create_bandpass(pol, ant))
        gains /= np.abs(gains)
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


def create_raw_sensor(timestamps, values):
    """Create a :class:`SimpleSensorGetter` from raw sensor data."""
    wrapped_values = [ComparableArrayWrapper(value) for value in values]
    return SimpleSensorGetter(None, np.array(timestamps), np.array(wrapped_values))


def create_categorical_sensor(timestamps, values, initial_value=None):
    """Create a :class:`CategoricalData` from raw sensor data."""
    wrapped_values = [ComparableArrayWrapper(value) for value in values]
    dump_midtimes = np.arange(N_DUMPS, dtype=float)
    return sensor_to_categorical(timestamps, wrapped_values, dump_midtimes,
                                 1.0, initial_value=initial_value)


TARGET_SENSOR = create_categorical_sensor(GAIN_EVENTS, TARGETS[TARGET_INDICES])


def create_sensor_cache(bandpass_parts=BANDPASS_PARTS):
    """Create a SensorCache for testing applycal sensors."""
    cache = {}
    cache['Observation/target'] = TARGET_SENSOR
    # Add delay product
    delays = create_product(create_delay)
    sensor = create_raw_sensor([3., 10.], [np.zeros_like(delays), delays])
    cache[CAL_STREAM + '_product_K'] = sensor
    # Add bandpass product (multi-part)
    bandpasses = create_product(create_bandpass)
    for part, bp in enumerate(np.split(bandpasses, bandpass_parts)):
        sensor = create_raw_sensor([2., 12.], [np.ones_like(bp), bp])
        cache[CAL_STREAM + '_product_B' + str(part)] = sensor
    # Add gain product (one value for entire band)
    gains = create_product(create_gain)
    sensor = create_raw_sensor(GAIN_EVENTS, gains)
    cache[CAL_STREAM + '_product_G'] = sensor
    # Add gain product (varying across frequency and time)
    gains = create_product(partial(create_gain, multi_channel=True, targets=True))
    sensor = create_raw_sensor(GAIN_EVENTS, gains)
    cache[CAL_STREAM + '_product_GPHASE'] = sensor
    # Construct sensor cache
    return SensorCache(cache, timestamps=np.arange(N_DUMPS, dtype=float),
                       dump_period=1., props=SENSOR_PROPS)


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


def gain_corrections(pol, ant, multi_channel=False, targets=False, fluxes=False):
    """Figure out N_DUMPS gain corrections given `pol` and `ant` indices."""
    dumps = np.arange(N_DUMPS)
    events = np.array(GAIN_EVENTS)
    gains = create_gain(pol, ant, multi_channel, targets, fluxes)
    gains = np.atleast_2d(gains.T)
    targets = TARGET_SENSOR if targets else CategoricalData([0], [0, len(dumps)])
    smooth_gains = np.full((N_DUMPS, gains.shape[0]), INVALID_GAIN, dtype=gains.dtype)
    for chan, gains_per_chan in enumerate(gains):
        for target in set(targets):
            on_target = (targets == target)
            valid = np.isfinite(gains_per_chan) & on_target[events]
            smooth_gains[on_target, chan] = INVALID_GAIN if not valid.any() else \
                complex_interp(dumps[on_target], events[valid], gains_per_chan[valid])
    return np.reciprocal(smooth_gains)


def corrections_per_corrprod(dumps, channels, cal_products):
    """Predict corrprod correction for a time-frequency-baseline selection."""
    input_map = {ant + pol: (pol_idx, ant_idx)
                 for (pol_idx, pol) in enumerate(POLS)
                 for (ant_idx, ant) in enumerate(ANTS)}
    gains_per_input = np.ones((len(dumps), N_CHANS, len(INPUTS)),
                              dtype='complex64')
    corrections = {
        CAL_STREAM + '.K': np.array([delay_corrections(*input_map[inp])
                                     for inp in INPUTS]).T,
        CAL_STREAM + '.B': np.array([bandpass_corrections(*input_map[inp])
                                     for inp in INPUTS]).T,
        CAL_STREAM + '.G': np.array([gain_corrections(*input_map[inp],
                                                      fluxes=True)[dumps]
                                     for inp in INPUTS]).T,
        CAL_STREAM + '.GPHASE': np.array([gain_corrections(*input_map[inp],
                                                           multi_channel=True,
                                                           targets=True)[dumps]
                                          for inp in INPUTS]
                                         ).transpose(1, 2, 0)[:, DATA_TO_CAL_CHANNEL]}
    # Apply (K, B, G, GPHASE) corrections in the same order
    # used by the system under test to get bit-exact results
    for cal_product in cal_products:
        gains_per_input *= corrections[cal_product]
    gains_per_input = gains_per_input[:, channels, :]
    gain1 = gains_per_input[:, :, INDEX1]
    gain2 = gains_per_input[:, :, INDEX2]
    return gain1 * gain2.conj()


def assert_categorical_data_equal(actual, desired):
    """Assert that two :class:`CategoricalData` objects are equal."""
    assert_array_equal(actual.events, desired.events)
    assert_equal(len(actual.unique_values), len(desired.unique_values))
    for a, d in zip(actual.unique_values, desired.unique_values):
        assert_array_equal(a, d)


class TestComplexInterp:
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


class TestCalProductAccess:
    """Test the :func:`~katdal.applycal.*_cal_product` functions."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS, CAL_STREAM, gaincal_flux=None)

    def test_get_cal_product_basic(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'K')
        product = create_product(create_delay)
        assert_array_equal(product_sensor[0], np.zeros_like(product))
        assert_array_equal(product_sensor[10], product)

    def test_get_cal_product_multipart(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'B')
        product = create_product(create_bandpass)
        assert_array_equal(product_sensor[0], np.ones_like(product))
        assert_array_equal(product_sensor[12], product)

    def test_get_cal_product_single_multipart(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'B')
        product = create_product(create_bandpass)
        assert_array_equal(product_sensor[0], np.ones_like(product))
        assert_array_equal(product_sensor[12], product)

    def test_get_cal_product_missing_parts(self):
        cache = create_sensor_cache()
        product = create_product(create_bandpass)
        n_chans_per_part = CAL_N_CHANS // BANDPASS_PARTS
        # Remove parts of multi-part cal product one by one
        for n in range(BANDPASS_PARTS - 1):
            del cache[CAL_STREAM + '_product_B' + str(n)]
            product_sensor = get_cal_product(cache, CAL_STREAM, 'B')
            part = slice(n * n_chans_per_part, (n + 1) * n_chans_per_part)
            product[part] = INVALID_GAIN
            assert_array_equal(product_sensor[12], product)
            # Recalculate on the next pass
            del cache[f'Calibration/Products/{CAL_STREAM}/B']
        # All parts gone triggers a KeyError
        del cache[CAL_STREAM + '_product_B' + str(BANDPASS_PARTS - 1)]
        with assert_raises(KeyError):
            get_cal_product(cache, CAL_STREAM, 'B')

    def test_get_cal_product_gain(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'G')
        product = create_product(create_gain)
        assert_array_equal(product_sensor[GAIN_EVENTS], product)

    def test_get_cal_product_selfcal_gain(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'GPHASE')
        product = create_product(partial(create_gain, multi_channel=True, targets=True))
        assert_array_equal(product_sensor[GAIN_EVENTS], product)


class TestCorrectionPerInput:
    """Test the :func:`~katdal.applycal.calc_*_correction` functions."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS, CAL_STREAM, gaincal_flux=None)

    def test_calc_delay_correction(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'K')
        constant_bandpass = np.ones(N_CHANS, dtype='complex64')
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_delay_correction(product_sensor, (m, n), FREQS)
                assert_array_equal(sensor[n], constant_bandpass)
                assert_array_equal(sensor[10 + n], delay_corrections(m, n))

    def test_calc_bandpass_correction(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'B')
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
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'G')
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_gain_correction(product_sensor, (m, n))
                assert_array_equal(sensor[:], gain_corrections(m, n))

    def test_calc_selfcal_gain_correction(self):
        product_sensor = get_cal_product(self.cache, CAL_STREAM, 'GPHASE')
        target_sensor = self.cache.get('Observation/target')
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_gain_correction(product_sensor, (m, n), target_sensor)
                assert_array_equal(sensor[:], gain_corrections(
                    m, n, multi_channel=True, targets=True))


class TestVirtualCorrectionSensors:
    """Test :func:`~katdal.applycal.add_applycal_sensors` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS, CAL_STREAM, gaincal_flux=None)

    def test_add_sensors_does_nothing_if_no_ants_pols_or_spw(self):
        cache = create_sensor_cache()
        n_virtuals_before = len(cache.virtual)
        add_applycal_sensors(cache, {}, [], CAL_STREAM, gaincal_flux=None)
        n_virtuals_after = len(cache.virtual)
        assert_equal(n_virtuals_after, n_virtuals_before)
        attrs = ATTRS.copy()
        del attrs['center_freq']
        add_applycal_sensors(self.cache, attrs, FREQS, CAL_STREAM, gaincal_flux=None)
        n_virtuals_after = len(cache.virtual)
        assert_equal(n_virtuals_after, n_virtuals_before)

    def test_delay_sensors(self, stream=CAL_STREAM):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = f'Calibration/Corrections/{stream}/K/{ant}{pol}'
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[10 + n], delay_corrections(m, n))

    def test_bandpass_sensors(self, stream=CAL_STREAM):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = f'Calibration/Corrections/{stream}/B/{ant}{pol}'
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[12 + n], bandpass_corrections(m, n))

    def test_gain_sensors(self, stream=CAL_STREAM):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = f'Calibration/Corrections/{stream}/G/{ant}{pol}'
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[:], gain_corrections(m, n))

    def test_selfcal_gain_sensors(self, stream=CAL_STREAM):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = f'Calibration/Corrections/{stream}/GPHASE/{ant}{pol}'
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[:], gain_corrections(
                    m, n, multi_channel=True, targets=True))

    def test_unknown_inputs_and_products(self):
        known_input = ANTS[0] + POLS[0]
        with assert_raises(KeyError):
            self.cache.get(f'Calibration/Corrections/{CAL_STREAM}/K/unknown')
        with assert_raises(KeyError):
            self.cache.get(f'Calibration/Corrections/{CAL_STREAM}/unknown/{known_input}')
        with assert_raises(KeyError):
            self.cache.get(f'Calibration/Corrections/{CAL_STREAM}/K_unknown/{known_input}')
        with assert_raises(KeyError):
            self.cache.get('Calibration/Corrections/unknown/K/' + known_input)
        with assert_raises(KeyError):
            self.cache.get(f'Calibration/Products/{CAL_STREAM}/K_unknown')
        with assert_raises(KeyError):
            self.cache.get('Calibration/Products/unknown/K')

    def test_indirect_cal_product(self):
        add_applycal_sensors(self.cache, ATTRS, FREQS, 'my_cal', [CAL_STREAM],
                             gaincal_flux=None)
        self.test_delay_sensors('my_cal')
        self.test_bandpass_sensors('my_cal')
        self.test_gain_sensors('my_cal')


class TestCalibrateFlux:
    """Test :func:`~katdal.applycal.calibrate_flux` function."""

    def setup(self):
        gains = create_product(create_gain)
        self.sensor = create_categorical_sensor(GAIN_EVENTS, gains, INVALID_GAIN)
        calibrated_gains = create_product(partial(create_gain, fluxes=True))
        self.calibrated_sensor = create_categorical_sensor(
            GAIN_EVENTS, calibrated_gains, INVALID_GAIN)

    def test_basic(self):
        fluxes = FLUXES.copy()
        calibrated_sensor = calibrate_flux(self.sensor, TARGET_SENSOR, fluxes)
        assert_categorical_data_equal(calibrated_sensor, self.calibrated_sensor)
        fluxes.update(gaincal3=10.0)
        calibrated_sensor = calibrate_flux(self.sensor, TARGET_SENSOR, fluxes)
        assert_categorical_data_equal(calibrated_sensor, self.calibrated_sensor)

    def test_missing_fluxes(self):
        calibrated_sensor = calibrate_flux(self.sensor, TARGET_SENSOR, {})
        assert_categorical_data_equal(calibrated_sensor, self.sensor)
        calibrated_sensor = calibrate_flux(self.sensor, TARGET_SENSOR,
                                           {'gaincal1': np.nan})
        assert_categorical_data_equal(calibrated_sensor, self.sensor)
        calibrated_sensor = calibrate_flux(self.sensor, TARGET_SENSOR,
                                           {'gaincal1': 1.0})
        assert_categorical_data_equal(calibrated_sensor, self.sensor)


class TestCalcCorrection:
    """Test :func:`~katdal.applycal.calc_correction` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        # Include fluxcal, which is also done in corrections_per_corrprod
        add_applycal_sensors(self.cache, ATTRS, FREQS, CAL_STREAM,
                             gaincal_flux=FLUX_OVERRIDES)

    def test_calc_correction(self):
        dump = 15
        channels = np.s_[22:38]
        shape = (N_DUMPS, N_CHANS, N_CORRPRODS)
        chunks = da.core.normalize_chunks((10, 5, -1), shape)
        final_cal_products, corrections = calc_correction(
            chunks, self.cache, CORRPRODS, CAL_PRODUCTS, FREQS, {'cal': CAL_FREQS})
        assert_equal(set(final_cal_products), set(CAL_PRODUCTS))
        corrections = corrections[dump:dump+1, channels].compute()
        expected_corrections = corrections_per_corrprod([dump], channels,
                                                        final_cal_products)
        assert_array_equal(corrections, expected_corrections)

    def test_skip_missing_products(self):
        dump = 15
        channels = np.s_[22:38]
        shape = (N_DUMPS, N_CHANS, N_CORRPRODS)
        chunks = da.core.normalize_chunks((10, 5, -1), shape)
        final_cal_products, corrections = calc_correction(
            chunks, self.cache, CORRPRODS, [], FREQS, {'cal': CAL_FREQS})
        assert_equal(final_cal_products, [])
        assert_equal(corrections, None)
        with assert_raises(ValueError):
            calc_correction(chunks, self.cache, CORRPRODS, ['INVALID'], FREQS,
                            {'cal': CAL_FREQS})
        unknown = CAL_STREAM + '.UNKNOWN'
        final_cal_products, corrections = calc_correction(
            chunks, self.cache, CORRPRODS, [unknown], FREQS, {'cal': CAL_FREQS},
            skip_missing_products=True)
        assert_equal(final_cal_products, [])
        assert_equal(corrections, None)
        cal_products = CAL_PRODUCTS + [unknown]
        with assert_raises(KeyError):
            calc_correction(chunks, self.cache, CORRPRODS, cal_products, FREQS,
                            {'cal': CAL_FREQS}, skip_missing_products=False)
        final_cal_products, corrections = calc_correction(
            chunks, self.cache, CORRPRODS, cal_products, FREQS, {'cal': CAL_FREQS},
            skip_missing_products=True)
        assert_equal(set(final_cal_products), set(CAL_PRODUCTS))
        corrections = corrections[dump:dump+1, channels].compute()
        expected_corrections = corrections_per_corrprod([dump], channels,
                                                        final_cal_products)
        assert_array_equal(corrections, expected_corrections)


class TestApplyCal:
    """Test :func:`~katdal.applycal.apply_vis_correction` and friends"""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, ATTRS, FREQS, CAL_STREAM, gaincal_flux=None)

    def _applycal(self, array, apply_correction):
        """Calibrate `array` with `apply_correction` and return all factors."""
        array_dask = da.from_array(array, chunks=(10, 4, 6))
        final_cal_products, correction = calc_correction(
            array_dask.chunks, self.cache, CORRPRODS, CAL_PRODUCTS, FREQS, {'cal': CAL_FREQS})
        corrected = da.core.elemwise(apply_correction, array_dask, correction,
                                     dtype=array_dask.dtype)
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
