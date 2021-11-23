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

"""Tests for :py:mod:`katdal.dataset`."""

import numpy as np
from katpoint import Antenna, Target, Timestamp, rad2deg
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

from katdal.categorical import CategoricalData
from katdal.dataset import (DEFAULT_VIRTUAL_SENSORS, DataSet, Subarray,
                            _selection_to_list, parse_url_or_path)
from katdal.sensordata import SensorCache
from katdal.spectral_window import SpectralWindow


class MinimalDataSet(DataSet):
    """Minimal data set containing a single target track."""
    def __init__(self, target, subarray, spectral_window, timestamps):
        super().__init__(name='test', ref_ant='array')
        num_dumps = len(timestamps)
        num_chans = spectral_window.num_chans
        num_corrprods = len(subarray.corr_products)
        dump_period = timestamps[1] - timestamps[0]

        def constant_sensor(value):
            return CategoricalData([value], [0, num_dumps])

        self.subarrays = [subarray]
        self.spectral_windows = [spectral_window]
        sensors = {}
        for ant in subarray.ants:
            sensors[f'Antennas/{ant.name}/antenna'] = constant_sensor(ant)
            az, el = target.azel(timestamps, ant)
            sensors[f'Antennas/{ant.name}/az'] = az
            sensors[f'Antennas/{ant.name}/el'] = el
        # Extract array reference position as 'array_ant' from last antenna
        array_ant_fields = ['array'] + ant.description.split(',')[1:5]
        array_ant = Antenna(','.join(array_ant_fields))
        sensors['Antennas/array/antenna'] = constant_sensor(array_ant)

        sensors['Observation/target'] = constant_sensor(target)
        for name in ('spw', 'subarray', 'scan', 'compscan', 'target'):
            sensors[f'Observation/{name}_index'] = constant_sensor(0)
        sensors['Observation/scan_state'] = constant_sensor('track')
        sensors['Observation/label'] = constant_sensor('track')

        self._timestamps = timestamps
        self._time_keep = np.full(num_dumps, True, dtype=np.bool_)
        self._freq_keep = np.full(num_chans, True, dtype=np.bool_)
        self._corrprod_keep = np.full(num_corrprods, True, dtype=np.bool_)
        self.dump_period = dump_period
        self.start_time = Timestamp(timestamps[0] - 0.5 * dump_period)
        self.end_time = Timestamp(timestamps[-1] + 0.5 * dump_period)
        self.sensor = SensorCache(sensors, timestamps, dump_period,
                                  keep=self._time_keep,
                                  virtual=DEFAULT_VIRTUAL_SENSORS)
        self.catalogue.add(target)
        self.catalogue.antenna = array_ant
        self.select(spw=0, subarray=0)

    @property
    def timestamps(self):
        return self._timestamps[self._time_keep]


def test_parse_url_or_path():
    # Normal URLs and empty strings pass right through
    assert_equal(parse_url_or_path('https://archive/file').geturl(), 'https://archive/file')
    assert_equal(parse_url_or_path('').geturl(), '')
    # Relative paths are turned into absolute paths and gain a 'file' scheme
    relative_file_url = parse_url_or_path('dir/filename.rdb')
    assert_equal(relative_file_url.scheme, 'file')
    parts = relative_file_url.path.rpartition('dir/filename.rdb')
    assert len(parts[0]) > 0
    assert_equal(parts[1], 'dir/filename.rdb')
    assert len(parts[2]) == 0
    # Absolute paths remain the same (just gaining a 'file' scheme)
    absolute_file_url = parse_url_or_path('/dir/filename.rdb')
    assert_equal(absolute_file_url.scheme, 'file')
    assert_equal(absolute_file_url.path, '/dir/filename.rdb')


def test_selection_to_list():
    # Empty
    assert_equal(_selection_to_list(''), [])
    assert_equal(_selection_to_list([]), [])
    # Names
    assert_equal(_selection_to_list('a,b,c'), ['a', 'b', 'c'])
    assert_equal(_selection_to_list('a, b,c'), ['a', 'b', 'c'])
    assert_equal(_selection_to_list(['a', 'b', 'c']), ['a', 'b', 'c'])
    assert_equal(_selection_to_list(('a', 'b', 'c')), ['a', 'b', 'c'])
    assert_equal(_selection_to_list('a'), ['a'])
    # Objects
    assert_equal(_selection_to_list([1, 2, 3]), [1, 2, 3])
    assert_equal(_selection_to_list(1), [1])
    # Groups
    assert_equal(_selection_to_list('all', all=['a', 'b']), ['a', 'b'])


class TestVirtualSensors:
    def setup(self):
        self.target = Target('PKS1934-638, radec, 19:39, -63:42')
        self.antennas = [Antenna('m000, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, '
                                 '-8.264 -207.29 8.5965'),
                         Antenna('m063, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, '
                                 '-3419.5845 -1840.48 16.3825')]
        corrprods = [('m000h', 'm000h'), ('m000v', 'm000v'),
                     ('m063h', 'm063h'), ('m063v', 'm063v'),
                     ('m000h', 'm063h'), ('m000v', 'm063v')]
        subarray = Subarray(self.antennas, corrprods)
        spw = SpectralWindow(centre_freq=1284e6, channel_width=0, num_chans=16,
                             sideband=1, bandwidth=856e6)
        # Pick a time when the source is up as that seems more realistic
        self.timestamps = 1234667890.0 + 1.0 * np.arange(10)
        self.dataset = MinimalDataSet(self.target, subarray, spw, self.timestamps)
        self.array_ant = self.dataset.sensor.get('Antennas/array/antenna')[0]

    def test_timestamps(self):
        mjd = Timestamp(self.timestamps[0]).to_mjd()
        assert_equal(self.dataset.mjd[0], mjd)
        lst = self.array_ant.local_sidereal_time(self.timestamps)
        # Convert LST from radians (katpoint) to hours (katdal)
        assert_array_equal(self.dataset.lst, lst * (12 / np.pi))

    def test_pointing(self):
        az, el = self.target.azel(self.timestamps, self.antennas[1])
        assert_array_equal(self.dataset.az[:, 1], rad2deg(az))
        assert_array_equal(self.dataset.el[:, 1], rad2deg(el))
        ra, dec = self.target.radec(self.timestamps, self.antennas[0])
        assert_array_almost_equal(self.dataset.ra[:, 0], rad2deg(ra), decimal=5)
        assert_array_almost_equal(self.dataset.dec[:, 0], rad2deg(dec), decimal=5)
        angle = self.target.parallactic_angle(self.timestamps, self.antennas[0])
        # TODO: Check why this is so poor... see SR-1882 for progress on this
        assert_array_almost_equal(self.dataset.parangle[:, 0], rad2deg(angle), decimal=0)
        x, y = self.target.sphere_to_plane(az, el, self.timestamps, self.antennas[1])
        assert_array_equal(self.dataset.target_x[:, 1], rad2deg(x))
        assert_array_equal(self.dataset.target_y[:, 1], rad2deg(y))

    def test_uvw(self):
        u0, v0, w0 = self.target.uvw(self.antennas[0], self.timestamps, self.array_ant)
        u1, v1, w1 = self.target.uvw(self.antennas[1], self.timestamps, self.array_ant)
        u = u0 - u1
        v = v0 - v1
        w = w0 - w1
        assert_array_equal(self.dataset.u[:, 4], u)
        assert_array_equal(self.dataset.v[:, 4], v)
        assert_array_equal(self.dataset.w[:, 4], w)
        # Check that both H and V polarisations have the same (u, v, w)
        assert_array_equal(self.dataset.u[:, 5], u)
        assert_array_equal(self.dataset.v[:, 5], v)
        assert_array_equal(self.dataset.w[:, 5], w)

    def test_selecting_antenna(self):
        self.dataset.select(ants='~m000')
        assert_array_equal(
            self.dataset.corr_products,
            [('m063h', 'm063h'), ('m063v', 'm063v')])
