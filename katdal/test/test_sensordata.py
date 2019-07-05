# -*- coding: utf-8

################################################################################
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
################################################################################

"""Tests for :py:mod:`katdal.sensordata`."""
from __future__ import print_function, division, absolute_import
from builtins import object

from collections import OrderedDict

import numpy as np
from nose.tools import assert_equal, assert_raises
import mock

from katdal.sensordata import SensorCache, to_str


def assert_equal_typed(a, b):
    assert_equal(a, b)
    assert_equal(type(a), type(b))


class TestToStr(object):
    def test_non_str(self):
        assert_equal_typed(to_str(3), 3)
        assert_equal_typed(to_str(None), None)

    def test_simple_str(self):
        assert_equal_typed(to_str(b'hello'), 'hello')
        assert_equal_typed(to_str(u'hello'), 'hello')

    def test_non_ascii(self):
        assert_equal_typed(to_str(b'caf\xc3\xa9'), 'café')
        assert_equal_typed(to_str(u'café'), 'café')

    def test_list(self):
        assert_equal_typed(to_str([b'hello', u'world']), ['hello', 'world'])

    def test_tuple(self):
        assert_equal_typed(to_str((b'hello', u'world')), ('hello', 'world'))

    def test_dict(self):
        assert_equal_typed(to_str({b'hello': b'world', u'abc': u'xyz'}),
                           {'hello': 'world', 'abc': 'xyz'})

    def test_custom_dict(self):
        assert_equal_typed(to_str(OrderedDict([(b'hello', b'world'), (u'abc', u'xyz')])),
                           OrderedDict([('hello', 'world'), ('abc', 'xyz')]))

    def test_numpy_str(self):
        a = np.array([[b'abc', b'def'], [b'ghi', b'jk']])
        b = np.array([[u'abc', u'def'], [u'ghi', u'jk']])
        c = np.array([['abc', 'def'], ['ghi', 'jk']])
        np.testing.assert_array_equal(to_str(a), c)
        np.testing.assert_array_equal(to_str(b), c)

    def test_numpy_object(self):
        a = np.array([b'abc', u'def', (b'xyz', u'uvw')], dtype='O')
        b = np.array(['abc', 'def', ('xyz', 'uvw')], dtype='O')
        np.testing.assert_array_equal(to_str(a), b)


class TestSensorCache(object):
    def setup(self):
        self.cache = SensorCache({}, timestamps=np.arange(10.), dump_period=1.0)

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
