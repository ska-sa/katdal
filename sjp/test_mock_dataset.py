from datetime import datetime, timedelta
import itertools

import numpy as np
from six.moves import range
import ephem

from katdal.dataset import (DataSet, Subarray,
                        SpectralWindow)
from katdal.h5datav3 import VIRTUAL_SENSORS, SENSOR_PROPS

from katdal.sensordata import SensorCache
from katdal.categorical import CategoricalData

import katpoint

ANTENNA_DESCRIPTIONS = [
    'm000, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -8.2435 -207.289 1.18 5872.443 5873.171, 0:01:23.5 0 -0:02:25.8 -0:03:01.9 0:00:42.6 -0:00:10.3 -0:11:37.7 0:03:03.5, 1.22',
    'm002, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -32.098 -224.248 1.202 5870.487 5871.204, 0:28:42.0 0 0:01:38.9 0:01:50.4 0:00:21.6 0:00:03.2 0:02:57.6, 1.22',
    'm005, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -102.08 -283.1155 1.454 5876.408 5876.854, 0:01:11.1 0 -0:00:50.9 -0:06:06.7 0:00:21.4 -0:00:10.6 -0:00:29.3, 1.22',
    'm017, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 199.641 -112.277 1.499 5868.708 5869.414, 0:47:17.0 0 -0:03:13.0 -0:06:45.8 0:00:42.8 0:00:05.4 0:06:26.5 0:03:35.6, 1.22',
    'm020, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 97.0245 -299.636 2.452 5852.26 5833.148, -1:37:44.7 0 0:04:29.3 0:04:55.8 0:00:24.9 0:00:04.9 0:03:37.0 0:02:17.1, 1.22',
    'm021, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -295.9485 -327.2465 0.6725 5885.674 5866.658, -0:22:35.3 0 0:02:41.4 -0:03:24.5 0:00:33.3 0:00:26.3 0:02:10.4 -0:04:09.4, 1.22',
    'm034, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 357.8245 -28.3215 1.515 5861.427 5861.286, 0:11:27.2 0 0:12:28.4 0:08:44.2 -0:00:34.1 0:01:29.7 -0:24:03.2 0:02:38.4, 1.22',
    'm041, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -287.531 -661.6855 2.5065 5870.621 5870.585, 0:06:55.8 0 -0:03:18.5 0:00:28.2 0:00:23.8 0:00:28.8 0:04:47.4 0:02:08.3, 1.22',
    'm042, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -361.6965 -460.3195 1.059 5870.792 5870.692, -0:07:55.0 0 0:00:53.5 -0:00:10.7 -0:00:16.1 0:00:08.4 0:02:22.5 0:01:37.9, 1.22',
    'm043, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -629.8345 -128.3255 -2.1695 5860.531 5861.26, 0:21:49.1 0 -0:00:00.8 0:00:47.2 0:00:03.0 0:00:34.9 -0:09:30.3 0:01:51.5, 1.22',
    'm048, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -2805.594 2686.8905 -17.153 5869.018 5850.242, 1:21:25.0 0 0:05:18.8 0:11:29.4 0:00:17.0 0:00:05.6 0:04:21.3 -0:02:36.1, 1.22',
    'm049, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -3605.9285 436.507 -4.6515 5881.679 5882.156, 0:22:43.8 0 0:12:37.6 0:25:26.6 0:00:37.1 0:00:33.8 -0:08:53.6 -0:01:09.3, 1.22',
    'm050, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -2052.322 -843.685 -2.0395 5882.69 5882.889, 0:26:30.1 0 -0:01:18.0 -0:05:14.5 -0:00:01.8 -0:00:13.0 -0:12:05.9 0:02:41.4, 1.22',
    'm054, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 871.9915 -499.8355 5.8945 5878.312 5878.298, 0:01:58.0 0 -0:06:01.7 -0:03:59.5 0:00:35.1 -0:00:45.8 0:06:50.6 -0:00:58.1, 1.22',
    'm055, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1201.798 96.4835 2.577 5849.908 5850.499, -0:00:02.5 0 0:02:37.7 -0:05:55.1 -0:00:01.2 0:01:30.5 0:06:12.9 0:03:27.3, 1.22',
    'm056, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 1598.4195 466.6315 -0.502 5873.236 5854.456, -0:25:01.4 0 0:02:18.3 0:07:57.5 -0:00:19.6 0:00:57.1 0:12:21.6 0:01:28.5, 1.22']

MEERKAT_LOW_FREQ = .856e9
MEERKAT_BANDWIDTH = .856e9
MEERKAT_CENTRE_FREQ = MEERKAT_LOW_FREQ + MEERKAT_BANDWIDTH / 2.
MEERKAT_CHANNELS = 32768
MEERKAT_CHANNEL_WIDTH = MEERKAT_BANDWIDTH / MEERKAT_CHANNELS

EPOCH = datetime.utcfromtimestamp(0)
OBSERVATION_START = datetime.now()
OBSERVATION_LENGTH = timedelta(hours=8)
OBSERVATION_END = OBSERVATION_START + OBSERVATION_LENGTH
OBSERVATION_START = (OBSERVATION_START - EPOCH).total_seconds()
OBSERVATION_END = (OBSERVATION_END - EPOCH).total_seconds()
OBSERVATION_LENGTH = OBSERVATION_LENGTH.total_seconds()

TARGETS = [katpoint.Target('%s, star' % t) for t in ('Achernar', 'Rigel', 'Sirius', 'Procyon')]

# Number of compound scans
NCOMPOUNDSCANS = 1000

def _create_refant_data(mock, ants, ndumps):
    """
    Setup reference antenna and related sensors
    """

    # Reference antenna and it's sensor data
    mock.ref_ant = ants[0].name

    def _generate_ref_ant_compound_scans(ndumps):
        """
        Divide dumps into periods of slewing and tracking at a target,
        yielding (dump_index, 'slew'/'track', target).
        """

        target_cycler = itertools.cycle(TARGETS)
        dumps_per_compscan = ndumps / float(NCOMPOUNDSCANS+1)
        slew_start = 0
        track_start = int(0.15*dumps_per_compscan)

        dump_range = range(0, ndumps, int(dumps_per_compscan))

        for i, target in zip(dump_range, target_cycler):
            yield (i + slew_start, 'slew', target)
            yield (i + track_start, 'track', target)

    # Generate compound scans for the reference antenna
    ref_ant_compound_scans = list(_generate_ref_ant_compound_scans(ndumps))

    # Labels seem to only involve tracks, separate them out
    label_scans = [tup for tup in ref_ant_compound_scans if tup[1] == 'track']
    events, values, _ = zip(*label_scans)
    label = CategoricalData(values, events + (ndumps,))

    # Generate dump indexes (events) and 'slew'/'track' (values)
    # and targets for the reference antenna
    events, values, targets = zip(*(_generate_ref_ant_compound_scans(ndumps)))
    refant = CategoricalData(values, events + (ndumps,))
    # DO THIS BCOS h5datav3.py does it
    refant.add_unmatched(label.events)
    #mock.sensor['Antennas/%s/activity' % mock.ref_ant] = refant

    # Derive scan states and indices from reference antenna data
    scan_index = CategoricalData(range(len(refant)), refant.events)

    mock.sensor['Observation/scan_state'] = refant
    mock.sensor['Observation/scan_index'] = scan_index

    # DO THIS BCOS h5datav3.py does it
    label.align(refant.events)

    # First track event unlikely to happen at dump 0
    if label.events[0] > 0:
        label.add(0, '')

    # Derive compound scan index from the label
    compscan_index = CategoricalData(range(len(label)), label.events)
    mock.sensor['Observation/label'] = label
    mock.sensor['Observation/compscan_index'] = compscan_index

    # Create categorical data for our targets
    targets = CategoricalData(targets, (events + (ndumps,)))
    # DO THIS BCOS h5datav3.py does it
    targets.align(refant.events)

    target_index = CategoricalData(targets.indices, targets.events)

    mock.sensor['Observation/target'] = targets
    mock.sensor['Observation/target_index'] = target_index

class MockDataSet(DataSet):
    def __init__(self):
        super(MockDataSet, self).__init__(name='mock')
        self.observer = 'mock'
        self.description = "Mock observation"

        # Observation times
        self.start_time = katpoint.Timestamp(OBSERVATION_START)
        self.end_time = katpoint.Timestamp(OBSERVATION_END)
        self.dump_period = 4.0
        self.dumps = np.arange(OBSERVATION_LENGTH / self.dump_period)
        ndumps = self.dumps.shape[0]
        self._timestamps = np.linspace(OBSERVATION_START, OBSERVATION_END,
                                            num=ndumps, endpoint=True)


        # Now create the sensors cache now that
        # we have dump period and timestamps
        cache = {}
        self.sensor = SensorCache(cache, self._timestamps, self.dump_period,
                                    keep=self._time_keep, props=SENSOR_PROPS,
                                    virtual=VIRTUAL_SENSORS)

        # Antenna and their sensor data
        ants = [katpoint.Antenna(d) for d in ANTENNA_DESCRIPTIONS]

        for ant in ants:
            ant_catdata = CategoricalData([ant], [0, ndumps])
            self.sensor['Antennas/%s/antenna' % (ant.name,)] = ant_catdata

        _create_refant_data(self, ants, ndumps)

        # Generate correlation products for all antenna pairs
        # auto-correlations included
        corr_products = np.array([
            (a1.name + c1, a2.name + c2)
                for i1, a1 in enumerate(ants)
                for i2, a2 in enumerate(ants[i1:])
                for c1 in ('h', 'v')
                for c2 in ('h', 'v')],
            dtype='|S5')

        # Single Subarray formed from all correlation products
        subarray = Subarray(ants, corr_products)
        self.subarrays = [subarray]
        self.subarray = 0
        self.inputs = subarray.inputs
        self.ants = subarray.ants
        self.corr_products = subarray.corr_products

        subarray_catdata = CategoricalData(self.subarrays, [0, ndumps])
        subarray_index_catdata = CategoricalData([self.subarray], [0, ndumps])
        self.sensor['Observation/subarray'] = subarray_catdata
        self.sensor['Observation/subarray_index'] = subarray_index_catdata

        # Single MEERKAT SPW and associated sensors
        spw = SpectralWindow(MEERKAT_CENTRE_FREQ, MEERKAT_CHANNEL_WIDTH,
                                                        MEERKAT_CHANNELS)

        self.spectral_windows = [spw]
        self.spw = 0
        self.channel_width = spw.channel_width
        self.freqs = self.channel_freqs = spw.channel_freqs
        self.channels = np.arange(MEERKAT_CHANNELS)

        spw_catdata = CategoricalData(self.spectral_windows, [0, ndumps])
        spw_index_catdata = CategoricalData([self.spw], [0, ndumps])
        self.sensor['Observation/spw'] = spw_catdata
        self.sensor['Observation/spw_index'] = spw_index_catdata

        # Add targets to catalogue
        for t in TARGETS:
            self.catalogue.add(t)

        ncorrproducts = self.corr_products.shape[0]
        nchan = self.channels.shape[0]

        # Select everything upfront (weights + flags already set to all in superclass)
        self._time_keep = np.ones(ndumps, dtype=np.bool)
        self._corrprod_keep = np.ones(ncorrproducts, dtype=np.bool)
        self._freq_keep = np.ones(nchan, dtype=np.bool)
        self.select(spw=0, subarray=0)

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def vis(self):
        vis = np.empty(self.shape, dtype=np.complex64)
        vis[:,:,:].real = np.full(self.shape, self.scan_indices[0])
        vis[:,:,:].imag = self.dumps[:,None,None]
        return vis

mock = MockDataSet()

for si, state, target in mock.scans():
    print mock._selection
    print si, state, target, mock.shape
    vis = mock.vis
    print vis[0,0,0], vis[-1,0,0]
