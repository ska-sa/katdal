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
from numpy.testing import assert_array_equal
from nose.tools import assert_raises

from katdal.sensordata import SensorCache
from katdal.categorical import CategoricalData
from katdal.applycal import calc_delay_correction, add_applycal_sensors


CAL_POLS = ['v', 'h']
CAL_ANTS = ['m000', 'm001', 'm002', 'm003']
CENTRE_FREQ = 1284.0
BANDWIDTH = 856.0
N_CHANS = 128
N_DUMPS = 100
SAMPLE_RATE = 1712.
FREQS = CENTRE_FREQ + BANDWIDTH / N_CHANS * (np.arange(N_CHANS) - N_CHANS // 2)


def delay_gains(delay):
    return np.exp(2j * np.pi * delay / SAMPLE_RATE * FREQS).astype('complex64')


def create_sensor_cache():
    """Create a SensorCache for testing applycal sensors."""
    cache = {}
    delays = np.arange(len(CAL_ANTS))
    delays = np.array([delays, -delays]) / SAMPLE_RATE
    cache['cal_product_K'] = CategoricalData([np.zeros_like(delays), delays],
                                             events=[0, 10, N_DUMPS])
    return SensorCache(cache, timestamps=np.arange(N_DUMPS, dtype=float),
                       dump_period=1.)


class TestCalcCorrection(object):
    """Test the :func:`~katdal.applycal.calc_*_correction` functions."""
    def setup(self):
        self.cache = create_sensor_cache()

    def test_calc_delay_correction(self):
        for n in range(len(CAL_ANTS)):
            for m in range(len(CAL_POLS)):
                sensor = calc_delay_correction(self.cache, 'K', (m, n), FREQS)
                assert_array_equal(sensor[n],
                                   np.ones(N_CHANS, dtype='complex64'))
                expected_delay = (-1) ** m * n
                assert_array_equal(sensor[10 + n], delay_gains(expected_delay))


class TestVirtualCorrectionSensors(object):
    """Test :func:`~katdal.applycal.add_applycal_sensors` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, [], [], [])   # this does nothing
        add_applycal_sensors(self.cache, CAL_ANTS, CAL_POLS, FREQS)

    def test_delay_sensors(self):
        for n, ant in enumerate(CAL_ANTS):
            for m, pol in enumerate(CAL_POLS):
                sensor_name = 'Calibration/{}{}_correction_K'.format(ant, pol)
                sensor = self.cache.get(sensor_name)
                expected_delay = (-1) ** m * n
                assert_array_equal(sensor[10 + n], delay_gains(expected_delay))

    def test_unknown_inputs_and_products(self):
        known_input = 'Calibration/{}{}'.format(CAL_ANTS[0], CAL_POLS[0])
        with assert_raises(KeyError):
            self.cache.get('Calibration/unknown_correction_K')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_unknown')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_K_unknown')
