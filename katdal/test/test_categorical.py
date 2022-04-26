################################################################################
# Copyright (c) 2011-2022, National Research Foundation (SARAO)
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
from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from katdal.categorical import (
    CategoricalData, concatenate_categorical, _single_event_per_dump,
    sensor_to_categorical, tabulate_categorical, parse_categorical_table,
)


def categorical_from_string(s):
    """Turn sequence of characters `s` into a categorical sensor.

    Each sensor value is a single non-blank character, and its event dump
    corresponds to its position in the input string. Spaces / blanks are
    therefore ignored (except to determine event locations).
    """
    non_blanks = [(n, char) for n, char in enumerate(list(s)) if char != " "]
    events, values = zip(*non_blanks)
    return CategoricalData(values, events + (len(s),))


def categorical_to_string(sensor):
    """Turn character-based categorical `sensor` into a string.

    The sensor value strings are padded by the appropriate number of spaces
    so that the values are located at their corresponding event dumps in
    the string, e.g. '  A   B   C' for events at [2, 6, 10].
    """
    s = sensor.events[-1] * [' ']
    for segment, char in sensor.segments():
        s[segment.start] = char
    return ''.join(s)


def test_quick_and_dirty_categorical_from_string():
    s = 'a  b'
    sensor = categorical_from_string(s)
    assert_array_equal(sensor.unique_values, ['a', 'b'])
    assert_array_equal(sensor.events, [0, 3, 4])
    assert_array_equal(sensor.indices, [0, 1])
    assert_equal(categorical_to_string(sensor), s)
    s = '  0    1    0    1  '
    sensor = categorical_from_string(s)
    assert_array_equal(sensor.unique_values, ['0', '1'])
    assert_array_equal(sensor.events, [2, 7, 12, 17, 20])
    assert_array_equal(sensor.indices, [0, 1, 0, 1])
    assert_equal(categorical_to_string(sensor), s)


def test_categorical_data_add_remove():
    x = 'A    B     C    A      B    C     '
    sensor = categorical_from_string(x)
    sensor.add(2)
    y = 'A A  B     C    A      B    C     '
    assert_equal(categorical_to_string(sensor), y)
    sensor.add(7, 'D')
    y = 'A A  B D   C    A      B    C     '
    assert_equal(categorical_to_string(sensor), y)
    sensor.add(11, 'E')
    y = 'A A  B D   E    A      B    C     '
    assert_equal(categorical_to_string(sensor), y)
    sensor.remove('B')
    y = 'A A    D   E    A           C     '
    assert_equal(categorical_to_string(sensor), y)
    sensor.remove_repeats()
    y = 'A      D   E    A           C     '
    assert_equal(categorical_to_string(sensor), y)


def test_categorical_data_add_unmatched():
    x = 'A    B     C    A      B    C     '
    sensor = categorical_from_string(x)
    # All alignment events are within `match_dist` of existing events: no change
    s = '^     ^   ^     ^     ^      ^    '
    align_sensor = categorical_from_string(s)
    sensor.add_unmatched(align_sensor.events, match_dist=1)
    assert_equal(categorical_to_string(sensor), x)
    sensor.add_unmatched(align_sensor.events, match_dist=0)
    # Now some alignment events are unmatched: add extra events
    x = 'A    B     C    A      B    C     '
    y = 'A    BB   BC    A     AB    CC    '
    s = '^     ^   ^     ^     ^      ^    '
    assert_equal(categorical_to_string(sensor), y)


def test_categorical_data_align():
    x = 'A    B     C    A      B    C     '
    sensor = categorical_from_string(x)
    s = '^         ^     ^     ^      ^    '
    align_sensor = categorical_from_string(s)
    sensor.align(align_sensor.events)
    # Move events to nearest alignment event / segment start,
    # keeping the last one if they land on top of each other
    x = 'A    B     C    A      B    C     '
    y = 'B         C     A     B      C    '
    s = '^         ^     ^     ^      ^    '
    assert_equal(categorical_to_string(sensor), y)
    s = '^         ^      ^            ^   '
    align_sensor = categorical_from_string(s)
    sensor.align(align_sensor.events)
    # In this case all A's are overridden by B's after alignment
    x = 'A    B     C    A      B    C     '
    y = 'B         C      B            C   '
    s = '^         ^      ^            ^   '
    assert_equal(categorical_to_string(sensor), y)


def test_categorical_data_partition():
    x = 'A    B     C    A      B    C     '
    sensor = categorical_from_string(x)
    s = '^        ^    ^                   '
    align_sensor = categorical_from_string(s)
    # Cut up sensor at alignment events / segment starts
    sensors = sensor.partition(align_sensor.events)
    parts = ['A    B   ', 'B C  ', 'C A      B    C     ']
    for part, part_sensor in zip(parts, sensors):
        assert_equal(categorical_to_string(part_sensor), part)
    # Put parts back together again to obtain original sensor
    concat_sensor = concatenate_categorical(sensors)
    assert_equal(categorical_to_string(concat_sensor), x)


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


TABLE = """
Dumps  Label  Target  Scan
-----------------------------
0      "      A       track
5      track          track
306    track  B       track
308                   slew
353                   track
652    track  C       track
654                   slew
666
"""


def test_tabulate_categorical():
    sensors = parse_categorical_table(TABLE)
    derived_table = tabulate_categorical(sensors)
    assert_equal(derived_table, TABLE)
    assert_equal(sensors['Label'][0], '')
    assert_array_equal(sensors['Label'].events, [0, 5, 306, 652, 666])
    assert_array_equal(sensors['Target'].events, [0, 306, 652, 666])
    assert_array_equal(sensors['Scan'].events,
                       [0, 5, 306, 308, 353, 652, 654, 666])
