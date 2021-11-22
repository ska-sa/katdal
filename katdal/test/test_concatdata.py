################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""Tests for :py:mod:`katdal.concatdata`."""

import numpy as np
from nose.tools import assert_equal, assert_in, assert_not_in, assert_raises

from katdal.categorical import CategoricalData
from katdal.concatdata import ConcatenatedSensorCache
from katdal.sensordata import SensorCache, SimpleSensorGetter


class TestConcatenatedSensorCache:
    @staticmethod
    def _make_cache(timestamps, sensors):
        cache_data = {}
        for name, ts, values in sensors:
            sd = SimpleSensorGetter(name, np.asarray(ts), np.asarray(values))
            cache_data[name] = sd
        return SensorCache(cache_data, timestamps, 2.0)

    def setup(self):
        self.timestamps1 = np.arange(100.0, 110.0, 2.0)
        self.timestamps2 = np.arange(1000.0, 1006.0, 2.0)
        sensors1 = [
            ('foo', [104.0, 107.0], [3.0, 6.0]),
            ('cat', [102.0, 110.0], ['hello', 'world']),
            ('int_missing', [105.0], [42])
        ]
        sensors2 = [
            ('foo', [995.0, 1010.0], [10.0, 25.0]),
            ('cat', [1000.0, 1002.0, 1004.0, 1006.0], ['world', 'hello', 'again', 'hello']),
            ('float_missing', [995.0], [3.0])
        ]
        self.cache1 = self._make_cache(self.timestamps1, sensors1)
        self.cache2 = self._make_cache(self.timestamps2, sensors2)
        self.keep = np.array([True, False, True, False, False, True, False, True])
        self.cache = ConcatenatedSensorCache([self.cache1, self.cache2], keep=self.keep)

    def test_timestamps(self):
        np.testing.assert_array_equal(
            self.cache.timestamps,
            np.concatenate([self.timestamps1, self.timestamps2])
        )

    def test_float(self):
        data = self.cache.get('foo')
        np.testing.assert_allclose(data, [3.0, 3, 3, 5, 6, 15, 17, 19])

    def test_categorical(self):
        data = self.cache.get('cat')
        assert_equal(data.unique_values, ['hello', 'world', 'again'])
        H = 'hello'
        W = 'world'
        A = 'again'
        np.testing.assert_array_equal(data[:], [H, H, H, H, H, W, H, A])

    def test_float_missing(self):
        data = self.cache.get('float_missing')
        np.testing.assert_array_equal(data, [np.nan] * 5 + [3.0] * 3)

    def test_int_missing(self):
        data = self.cache.get('int_missing')
        np.testing.assert_array_equal(data[:], [42] * 5 + [-1] * 3)

    def test_missing_select(self):
        data = self.cache['int_missing']
        np.testing.assert_array_equal(data[:], [42, 42, -1, -1])

    def test_float_select(self):
        data = self.cache['foo']
        np.testing.assert_allclose(data, [3.0, 3, 15, 19])

    def test_categorical_select(self):
        data = self.cache['cat']
        np.testing.assert_array_equal(data, ['hello', 'hello', 'world', 'again'])

    def test_no_extract(self):
        data = self.cache.get('foo', extract=False)
        values = data.get()
        np.testing.assert_array_equal(values.timestamp, [104.0, 107.0, 995.0, 1010.0])
        np.testing.assert_array_equal(values.value, [3.0, 6.0, 10.0, 25.0])

    def test_no_extract_missing(self):
        data = self.cache.get('float_missing', extract=False)
        values = data.get()
        np.testing.assert_array_equal(values.timestamp, [995.0])
        np.testing.assert_array_equal(values.value, [3.0])

    def test_missing_sensor(self):
        with assert_raises(KeyError):
            self.cache['sir_not_appearing_in_this_cache']

    def test_partially_extract(self):
        self.cache1['foo']
        data = self.cache.get('foo', extract=False)
        np.testing.assert_array_equal(data, self.cache.get('foo', extract=True))

    def test_setitem_categorical(self):
        data = CategoricalData(['x', 'y', 'x'], [0, 2, 4, 8])
        self.cache['dog'] = data
        ans = self.cache.get('dog')
        assert_equal(data.unique_values, ans.unique_values)
        np.testing.assert_array_equal(data.events, ans.events)
        np.testing.assert_array_equal(data.indices, ans.indices)

    def test_setitem_array(self):
        data = np.array([1.0, 2, 3, 5, 8, 13, 21, 34])
        self.cache['fib'] = data
        ans = self.cache.get('fib')
        np.testing.assert_array_equal(data, ans)

    def test_len(self):
        assert_equal(len(self.cache), 4)

    def test_keys(self):
        assert_equal(sorted(self.cache.keys()), ['cat', 'float_missing', 'foo', 'int_missing'])

    def test_contains(self):
        assert_in('cat', self.cache)
        assert_in('float_missing', self.cache)
        assert_in('int_missing', self.cache)
        assert_not_in('dog', self.cache)
        assert_not_in('', self.cache)
