###############################################################################
# Copyright (c) 2018-2019,2021-2024, National Research Foundation (SARAO)
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

import logging

import numpy as np
import pytest
from katpoint import Antenna, Target, Timestamp, rad2deg
from numpy.testing import assert_array_almost_equal, assert_array_equal

from katdal.categorical import CategoricalData
from katdal.dataset import (DEFAULT_VIRTUAL_SENSORS, DataSet, Subarray,
                            _selection_to_list, parse_url_or_path)
from katdal.sensordata import SensorCache
from katdal.spectral_window import SpectralWindow


ANTENNAS = [
    Antenna('m000, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -8.264 -207.29 8.5965'),
    Antenna('m063, -30:42:39.8, 21:26:38.0, 1086.6, 13.5, -3419.5845 -1840.48 16.3825')
]
CORRPRODS = [
    ('m000h', 'm000h'), ('m000v', 'm000v'),
    ('m063h', 'm063h'), ('m063v', 'm063v'),
    ('m000h', 'm063h'), ('m000v', 'm063v')
]
SUBARRAY = Subarray(ANTENNAS, CORRPRODS)
SPW = SpectralWindow(
    centre_freq=1284e6, channel_width=0, num_chans=16, sideband=1, bandwidth=856e6
)


class MinimalDataSet(DataSet):
    """Minimal data set containing a series of slews and tracks.

    The timestamps are divided evenly into compound scans (one per target).
    Each compound scan consists of a 1-dump slew followed by a track.

    This has to be a derived class instead of a factory function or fixture
    because :class:`DataSet` is abstract. (XXX Actually it only needs
    timestamps to be implemented for these tests to work, so it is nearly
    there.)

    Parameters
    ----------
    targets : list of :class:`katpoint.Target`
    timestamps : array of float
    subarray : :class:`katdal.dataset.Subarray`
    spectral_window : :class:`katdal.spectral_window.SpectralWindow`
    """
    def __init__(self, targets, timestamps, subarray=SUBARRAY, spectral_window=SPW):
        super().__init__(name='test', ref_ant='array')
        num_dumps = len(timestamps)
        num_chans = spectral_window.num_chans
        num_corrprods = len(subarray.corr_products)
        dump_period = timestamps[1] - timestamps[0]

        num_compscans = len(targets)
        num_dumps_per_compscan = num_dumps // num_compscans
        assert num_dumps_per_compscan * num_compscans == num_dumps, \
            "len(timestamps) should be an integer multiple of len(targets)"
        compscan_starts = np.arange(0, num_dumps, num_dumps_per_compscan)
        compscan_events = np.r_[compscan_starts, num_dumps]
        # Each slew contains just 1 dump to make things simple
        scan_events = sorted(np.r_[compscan_starts, compscan_starts + 1, num_dumps])
        target_sensor = CategoricalData(targets, compscan_events)

        def constant_sensor(value):
            return CategoricalData([value], [0, num_dumps])

        self.subarrays = [subarray]
        self.spectral_windows = [spectral_window]
        sensors = {}
        sensors['Observation/spw_index'] = constant_sensor(0)
        sensors['Observation/subarray_index'] = constant_sensor(0)
        for ant in subarray.ants:
            sensors[f'Antennas/{ant.name}/antenna'] = constant_sensor(ant)
            ant_az = []
            ant_el = []
            for segment, target in target_sensor.segments():
                az, el = target.azel(timestamps[segment], ant)
                ant_az.append(az)
                ant_el.append(el)
            sensors[f'Antennas/{ant.name}/az'] = np.concatenate(ant_az)
            sensors[f'Antennas/{ant.name}/el'] = np.concatenate(ant_el)
        array_ant = subarray.ants[0].array_reference_antenna()
        sensors['Antennas/array/antenna'] = constant_sensor(array_ant)

        compscan_sensor = CategoricalData(range(num_compscans), compscan_events)
        label_sensor = CategoricalData(['track'] * num_compscans, compscan_events)
        sensors['Observation/target'] = target_sensor
        sensors['Observation/compscan_index'] = compscan_sensor
        sensors['Observation/target_index'] = compscan_sensor
        sensors['Observation/label'] = label_sensor
        scan_sensor = CategoricalData(range(2 * num_compscans), scan_events)
        state_sensor = CategoricalData(['slew', 'track'] * num_compscans, scan_events)
        sensors['Observation/scan_index'] = scan_sensor
        sensors['Observation/scan_state'] = state_sensor

        self._timestamps = timestamps
        self._time_keep = np.full(num_dumps, True, dtype=bool)
        self._freq_keep = np.full(num_chans, True, dtype=bool)
        self._corrprod_keep = np.full(num_corrprods, True, dtype=bool)
        self.dump_period = dump_period
        self.start_time = Timestamp(timestamps[0] - 0.5 * dump_period)
        self.end_time = Timestamp(timestamps[-1] + 0.5 * dump_period)
        self.sensor = SensorCache(sensors, timestamps, dump_period,
                                  keep=self._time_keep,
                                  virtual=DEFAULT_VIRTUAL_SENSORS)
        self.catalogue.add(targets)
        self.catalogue.antenna = array_ant
        self.select(spw=0, subarray=0)

    @property
    def timestamps(self):
        return self._timestamps[self._time_keep]


def test_parse_url_or_path():
    # Normal URLs and empty strings pass right through
    assert parse_url_or_path('https://archive/file').geturl() == 'https://archive/file'
    assert parse_url_or_path('').geturl() == ''
    # Relative paths are turned into absolute paths and gain a 'file' scheme
    relative_file_url = parse_url_or_path('dir/filename.rdb')
    assert relative_file_url.scheme == 'file'
    parts = relative_file_url.path.rpartition('dir/filename.rdb')
    assert len(parts[0]) > 0
    assert parts[1] == 'dir/filename.rdb'
    assert len(parts[2]) == 0
    # Absolute paths remain the same (just gaining a 'file' scheme)
    absolute_file_url = parse_url_or_path('/dir/filename.rdb')
    assert absolute_file_url.scheme == 'file'
    assert absolute_file_url.path == '/dir/filename.rdb'


def test_selection_to_list():
    # Empty
    assert _selection_to_list('') == []
    assert _selection_to_list([]) == []
    # Names
    assert _selection_to_list('a,b,c') == ['a', 'b', 'c']
    assert _selection_to_list('a, b,c') == ['a', 'b', 'c']
    assert _selection_to_list(['a', 'b', 'c']) == ['a', 'b', 'c']
    assert _selection_to_list(('a', 'b', 'c')) == ['a', 'b', 'c']
    assert _selection_to_list('a') == ['a']
    # Objects
    assert _selection_to_list([1, 2, 3]) == [1, 2, 3]
    assert _selection_to_list(1) == [1]
    # Groups
    assert _selection_to_list('all', all=['a', 'b']) == ['a', 'b']


class TestVirtualSensors:
    def setup_method(self):
        self.target = Target('PKS1934-638, radec, 19:39, -63:42')
        # Pick a time when the source is up as that seems more realistic
        self.timestamps = 1234667890.0 + 1.0 * np.arange(10)
        self.dataset = MinimalDataSet([self.target], self.timestamps)
        self.antennas = self.dataset.subarrays[0].ants
        self.array_ant = self.dataset.sensor.get('Antennas/array/antenna')[0]

    def test_timestamps(self):
        mjd = Timestamp(self.timestamps[0]).to_mjd()
        assert self.dataset.mjd[0] == mjd
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


@pytest.fixture
def dataset():
    """A basic dataset used to test the selection mechanism."""
    targets = [
        # It would have been nice to have radec = 19:39, -63:42 but then
        # selection by description string does not work because the catalogue's
        # description string pads it out to radec = 19:39:00.00, -63:42:00.0.
        # (XXX Maybe fix Target comparison in katpoint to support this?)
        Target('J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6'),
        Target('J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6'),
        Target('J0408-6545 | PKS 0408-65, radec bpcal, 4:08:20.38, -65:45:09.1'),
        Target('J1346-6024 | Cen B, radec, 13:46:49.04, -60:24:29.4'),
    ]
    # Ensure that len(timestamps) is an integer multiple of len(targets)
    timestamps = 1234667890.0 + 1.0 * np.arange(12)
    return MinimalDataSet(targets, timestamps)


def test_selecting_antenna(dataset):
    dataset.select(ants='~m000')
    assert_array_equal(
        dataset.corr_products,
        [('m063h', 'm063h'), ('m063v', 'm063v')])


@pytest.mark.parametrize(
    'scans,expected_dumps',
    [
            ('track', [1, 2, 4, 5, 7, 8, 10, 11]),
            ('slew', [0, 3, 6, 9]),
            ('~slew', [1, 2, 4, 5, 7, 8, 10, 11]),
            (('track', 'slew'), np.arange(12)),
            (('track', '~slew'), [1, 2, 4, 5, 7, 8, 10, 11]),
            (1, [1, 2]),
            ([1, 3, 4], [1, 2, 4, 5, 6]),
            # Empty selections
            ('scan', []),
            (-1, []),
            (100, []),
    ]
)
def test_select_scans(dataset, scans, expected_dumps):
    dataset.select(scans=scans)
    assert_array_equal(dataset.dumps, expected_dumps)


@pytest.mark.parametrize(
    'compscans,expected_dumps',
    [
            ('track', np.arange(12)),
            (1, [3, 4, 5]),
            ([1, 3], [3, 4, 5, 9, 10, 11]),
            # Empty selections
            ('~track', []),
            (-1, []),
            (100, []),
    ]
)
def test_select_compscans(dataset, compscans, expected_dumps):
    dataset.select(compscans=compscans)
    assert_array_equal(dataset.dumps, expected_dumps)


@pytest.mark.parametrize(
    'targets,expected_dumps',
    [
        ('PKS1934-638', [0, 1, 2]),
        ('J1939-6342', [0, 1, 2, 3, 4, 5]),
        (('J0408-6545', 'Cen B'), [6, 7, 8, 9, 10, 11]),
        ('J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6', [3, 4, 5]),
        (Target('J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6'), [3, 4, 5]),
        (
            [
                Target('J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6'),
                'J1346-6024 | Cen B, radec, 13:46:49.04, -60:24:29.4',
            ],
            [3, 4, 5, 9, 10, 11]
        ),
        (1, [3, 4, 5]),
        ([1, 3], [3, 4, 5, 9, 10, 11]),
        # Empty selections
        ('Moon', []),
        ('J1939-6342, radec gaincal, 19:39, -63:42', []),
        (Target('Sun, special'), []),
    ]
)
def test_select_targets(caplog, dataset, targets, expected_dumps):
    with caplog.at_level(logging.WARNING, logger='katdal.dataset'):
        dataset.select(targets=targets)
    # If the target is not found, check that a warning is logged
    if len(expected_dumps) == 0:
        assert 'Skipping unknown selected target' in caplog.text
    assert_array_equal(dataset.dumps, expected_dumps)


@pytest.mark.parametrize(
    'target_tags,expected_dumps',
    [
        ('radec', np.arange(12)),
        ('bpcal', [0, 1, 2, 6, 7, 8]),
        ('gaincal', [3, 4, 5]),
        ('gaincal,incorrect_tag', [3, 4, 5]),
        (['gaincal'], [3, 4, 5]),
        ('bpcal,gaincal', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (('bpcal', 'gaincal'), [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ('incorrect_tag', []),
    ]
)
def test_select_target_tags(caplog, dataset, target_tags, expected_dumps):
    with caplog.at_level(logging.WARNING, logger='katdal.dataset'):
        dataset.select(target_tags=target_tags)
    if 'incorrect_tag' in target_tags:
        assert 'Skipping unknown selected target tag' in caplog.text
    assert_array_equal(dataset.dumps, expected_dumps)
