###############################################################################
# Copyright (c) 2019,2021-2023, National Research Foundation (SARAO)
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

"""A minimal DataSet that is useful for unit tests."""

import numpy as np
from katpoint import Antenna, Timestamp

from katdal.categorical import CategoricalData
from katdal.dataset import DEFAULT_VIRTUAL_SENSORS, DataSet, Subarray
from katdal.lazy_indexer import LazyIndexer
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
        self.version = '4.0'
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
        self._vis = np.zeros(self.shape, dtype=np.complex64)
        self._flags = np.zeros(self.shape, dtype=bool)
        self._weights = np.zeros(self.shape, dtype=np.float32)

    @property
    def _stage1(self):
        return (self._time_keep, self._freq_keep, self._corrprod_keep)

    @property
    def timestamps(self):
        return self._timestamps[self._time_keep]

    @property
    def vis(self):
        return LazyIndexer(self._vis, self._stage1)

    @property
    def flags(self):
        return LazyIndexer(self._flags, self._stage1)

    @property
    def weights(self):
        return LazyIndexer(self._weights, self._stage1)
