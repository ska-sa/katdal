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


POLS = ['v', 'h']
ANTS = ['m000', 'm001', 'm002', 'm003']
CENTRE_FREQ = 1284.0
BANDWIDTH = 856.0
N_CHANS = 128
N_DUMPS = 100
SAMPLE_RATE = 1712.

INPUTS = [ant + pol for ant in ANTS for pol in POLS]
FREQS = CENTRE_FREQ + BANDWIDTH / N_CHANS * (np.arange(N_CHANS) - N_CHANS // 2)
INDEX1, INDEX2 = np.triu_indices(len(INPUTS))
CORRPRODS = [(INPUTS[i1], INPUTS[i2]) for i1, i2 in zip(INDEX1, INDEX2)]
N_CORRPRODS = len(CORRPRODS)


def create_sensor_cache():
    """Create a SensorCache for testing applycal sensors."""
    cache = {}
    delays = np.arange(len(ANTS))
    delays = np.array([delays, -delays]) / SAMPLE_RATE
    cache['cal_product_K'] = CategoricalData([np.zeros_like(delays), delays],
                                             events=[0, 10, N_DUMPS])
    return SensorCache(cache, timestamps=np.arange(N_DUMPS, dtype=float),
                       dump_period=1.)


def delay_gains(pol, ant):
    delay = (-1) ** pol * ant
    return np.exp(2j * np.pi * delay / SAMPLE_RATE * FREQS).astype('complex64')


class TestCorrectionPerInput(object):
    """Test the :func:`~katdal.applycal.calc_*_correction` functions."""
    def setup(self):
        self.cache = create_sensor_cache()

    def test_calc_delay_correction(self):
        for n in range(len(ANTS)):
            for m in range(len(POLS)):
                sensor = calc_delay_correction(self.cache, 'K', (m, n), FREQS)
                assert_array_equal(sensor[n],
                                   np.ones(N_CHANS, dtype='complex64'))
                assert_array_equal(sensor[10 + n], delay_gains(m, n))


class TestVirtualCorrectionSensors(object):
    """Test :func:`~katdal.applycal.add_applycal_sensors` function."""
    def setup(self):
        self.cache = create_sensor_cache()
        add_applycal_sensors(self.cache, [], [], [])   # this does nothing
        add_applycal_sensors(self.cache, ANTS, POLS, FREQS)

    def test_delay_sensors(self):
        for n, ant in enumerate(ANTS):
            for m, pol in enumerate(POLS):
                sensor_name = 'Calibration/{}{}_correction_K'.format(ant, pol)
                sensor = self.cache.get(sensor_name)
                assert_array_equal(sensor[10 + n], delay_gains(m, n))

    def test_unknown_inputs_and_products(self):
        known_input = 'Calibration/{}{}'.format(ANTS[0], POLS[0])
        with assert_raises(KeyError):
            self.cache.get('Calibration/unknown_correction_K')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_unknown')
        with assert_raises(KeyError):
            self.cache.get(known_input + '_correction_K_unknown')
