################################################################################
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
################################################################################

"""Tests for :py:mod:`katdal.sensordata`."""

from collections import OrderedDict
from unittest import mock

import numpy as np
from nose.tools import (assert_equal, assert_in, assert_is_instance,
                        assert_not_in, assert_raises)

from katdal.sensordata import (SensorCache, SensorData, SimpleSensorGetter,
                               remove_duplicates_and_invalid_values,
                               telstate_decode, to_str)


def assert_equal_typed(a, b):
    assert_equal(a, b)
    assert_equal(type(a), type(b))


class TestToStr:
    def test_non_str(self):
        assert_equal_typed(to_str(3), 3)
        assert_equal_typed(to_str(None), None)

    def test_simple_str(self):
        assert_equal_typed(to_str(b'hello'), 'hello')
        assert_equal_typed(to_str('hello'), 'hello')

    def test_non_ascii(self):
        assert_equal_typed(to_str(b'caf\xc3\xa9'), 'café')
        assert_equal_typed(to_str('café'), 'café')

    def test_list(self):
        assert_equal_typed(to_str([b'hello', 'world']), ['hello', 'world'])

    def test_tuple(self):
        assert_equal_typed(to_str((b'hello', 'world')), ('hello', 'world'))

    def test_dict(self):
        assert_equal_typed(to_str({b'hello': b'world', 'abc': 'xyz'}),
                           {'hello': 'world', 'abc': 'xyz'})

    def test_custom_dict(self):
        assert_equal_typed(to_str(OrderedDict([(b'hello', b'world'), ('abc', 'xyz')])),
                           OrderedDict([('hello', 'world'), ('abc', 'xyz')]))

    def test_numpy_str(self):
        a = np.array([[b'abc', b'def'], [b'ghi', b'jk']])
        b = np.array([['abc', 'def'], ['ghi', 'jk']])
        c = np.array([['abc', 'def'], ['ghi', 'jk']])
        np.testing.assert_array_equal(to_str(a), c)
        np.testing.assert_array_equal(to_str(b), c)

    def test_numpy_object(self):
        a = np.array([b'abc', 'def', (b'xyz', 'uvw')], dtype='O')
        b = np.array(['abc', 'def', ('xyz', 'uvw')], dtype='O')
        np.testing.assert_array_equal(to_str(a), b)


@mock.patch('katsdptelstate.encoding._allow_pickle', True)
@mock.patch('katsdptelstate.encoding._warn_on_pickle', False)
def test_telstate_decode():
    raw = "S'1'\n."
    assert telstate_decode(raw) == '1'
    assert telstate_decode(raw.encode()) == '1'
    assert telstate_decode(np.void(raw.encode())) == '1'
    assert telstate_decode('l', no_decode=('l', 's', 'u', 'x')) == 'l'
    raw_np = ("cnumpy.core.multiarray\nscalar\np1\n(cnumpy\ndtype\np2\n(S'f8'\nI0\nI1\ntRp3\n"
              "(I3\nS'<'\nNNNI-1\nI-1\nI0\ntbS'8\\xdf\\xd4(\\x89\\xfc\\xef?'\ntRp4\n.")
    value_np = telstate_decode(raw_np)
    assert value_np == 0.9995771214953271
    assert isinstance(value_np, np.float64)


class TestSensorCache:
    def _cache_data(self):
        sensors = [
            ('foo', [4.0, 7.0], [3.0, 6.0]),
            ('cat', [2.0, 6.0], ['hello', 'world'])
        ]
        cache_data = {}
        for name, ts, values in sensors:
            sd = SimpleSensorGetter(name, np.asarray(ts), np.asarray(values))
            cache_data[name] = sd
        return cache_data

    def setup(self):
        self.cache = SensorCache(self._cache_data(), timestamps=np.arange(10.), dump_period=1.0)

    def test_extract_float(self):
        data = self.cache.get('foo', extract=True)
        np.testing.assert_array_equal(data, [3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0])

    def test_extract_categorical(self):
        data = self.cache.get('cat', extract=True)
        H = 'hello'
        W = 'world'
        np.testing.assert_array_equal(data[:], [H, H, H, H, H, H, W, W, W, W])

    def test_alias(self):
        self.cache = SensorCache(
            self._cache_data(), timestamps=np.arange(10.), dump_period=1.0,
            aliases={'zz': 'at'})
        # Check that adding the alias didn't lead to extraction
        assert_is_instance(self.cache.get('czz', extract=False), SimpleSensorGetter)
        np.testing.assert_array_equal(self.cache['czz'], self.cache['cat'])

    def test_len(self):
        assert_equal(len(self.cache), 2)

    def test_keys(self):
        assert_equal(sorted(self.cache.keys()), ['cat', 'foo'])

    def test_contains(self):
        assert_in('cat', self.cache)
        assert_in('foo', self.cache)
        assert_not_in('dog', self.cache)
        template = 'Antennas/{ant}/{param1}_{param2}'
        self.cache.virtual[template] = lambda x: None
        assert_not_in(template, self.cache)

    def test_setitem_delitem(self):
        self.cache['bar'] = SimpleSensorGetter('bar', np.array([1.0]), np.array([0.0]))
        np.testing.assert_array_equal(self.cache['bar'], np.zeros(10))
        del self.cache['bar']
        assert_not_in('bar', self.cache)

    def test_sensor_time_offset(self):
        data = self.cache.get('foo', extract=True, time_offset=-1.0)
        np.testing.assert_array_equal(data, [3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0])

    def test_virtual_sensors(self):
        calculate_value = mock.Mock()

        def _check_sensor(cache, name, **kwargs):
            """Check that virtual sensor function gets the expected parameters."""
            assert_equal(kwargs, params)
            calculate_value()
            value = kwargs['param2']
            cache[name] = value
            return value

        # Set up a virtual sensor and trigger it to get a value
        params = {'ant': 'm000', 'param1': 'one', 'param2': 'two'}
        template = 'Antennas/{ant}/{param1}_{param2}'
        self.cache.virtual[template] = _check_sensor
        value = self.cache.get(template.format(**params))
        assert_equal(value, params['param2'])
        assert_equal(calculate_value.call_count, 1)
        # Check that the value was taken from the cache the second time around
        value = self.cache.get(template.format(**params))
        assert_equal(value, params['param2'])
        assert_equal(calculate_value.call_count, 1)
        # If your parameter values contain underscores, don't use it as delimiter
        params = {'ant': 'm000', 'param1': 'one', 'param2': 'two_three'}
        with assert_raises(AssertionError):
            self.cache.get(template.format(**params))
        template = 'Antennas/{ant}/{param1}/{param2}'
        # The updated template has not yet been added to the cache
        with assert_raises(KeyError):
            self.cache.get(template.format(**params))
        self.cache.virtual[template] = _check_sensor
        value = self.cache.get(template.format(**params))
        assert_equal(value, params['param2'])
        assert_equal(calculate_value.call_count, 2)

    # TODO: more tests required:
    # - extract=False
    # - selection


def test_sensor_cleanup():
    # The first sensor event has a status of "unknown" and is therefore invalid. It happened
    # after the second (valid) event, though, and snuck through due to a bug (now fixed).
    # This mirrors the behaviour of the cbf_1_wide_input_labelling sensor in CBID 1588667937.
    timestamp = np.array([1.0, 0.0, 3.0, 3.0, 3.0, 3.0, 2.0])
    value = np.array(['broke', 'a', 'c', 'c', 'c', 'd', 'b'])
    status = np.array(['unknown', 'nominal', 'nominal', 'nominal', 'warn', 'error', 'nominal'])
    dirty = SensorData('test', timestamp, value, status)
    clean = remove_duplicates_and_invalid_values(dirty)
    assert_equal(clean.status, None)
    np.testing.assert_array_equal(clean.value, np.array(['a', 'b', 'd']))
    np.testing.assert_array_equal(clean.timestamp, np.array([0.0, 2.0, 3.0]))
