################################################################################
# Copyright (c) 2011-2021, National Research Foundation (SARAO)
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

"""Tests for :py:mod:`katdal.categorical`."""

import numpy as np
from numpy.testing import assert_array_equal

from katdal.categorical import _single_event_per_dump, sensor_to_categorical


def test_dump_to_event_parsing():
    values = np.array(list('ABCDEFGH'))
    events = np.array([0, 0, 1, 3, 3, 4, 4, 6, 8])
    greedy = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    cleaned = list(_single_event_per_dump(events, greedy))
    new_values = values[cleaned]
    new_events = events[cleaned]
    assert_array_equal(cleaned, [0, 2, 4, 6, 7], 'Dump->event parser failed')
    assert_array_equal(new_values, list('ACEGH'), 'Dump->event parser failed')
    assert_array_equal(new_events, [0, 1, 3, 5, 6], 'Dump->event parser failed')


def test_categorical_sensor_creation():
    timestamps = [-363.784, 2.467, 8.839, 8.867, 15.924, 48.925, 54.897, 88.982]
    values = ['stop', 'slew', 'track', 'slew', 'track', 'slew', 'track', 'slew']
    dump_period = 8.
    dump_times = np.arange(4., 100., dump_period)
    categ = sensor_to_categorical(timestamps, values, dump_times, dump_period,
                                  greedy_values=('slew', 'stop'),
                                  initial_value='slew')
    assert_array_equal(categ.unique_values, ['slew', 'track'],
                       'Sensor->categorical failed')
    assert_array_equal(categ.events, [0, 2, 6, 7, 11, 12],
                       'Sensor->categorical failed')
    assert_array_equal(categ.indices, [0, 1, 0, 1, 0],
                       'Sensor->categorical failed')
