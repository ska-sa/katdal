"""Data accessor class for HDF5 files produced by KAT-7 correlator."""

import logging

import numpy as np
import h5py
import katpoint

from .dataset import DataSet, WrongVersion, BrokenFile, Subarray, \
                     SpectralWindow, DEFAULT_VIRTUAL_SENSORS, _robust_target
from .sensordata import SensorData, SensorCache
from .categorical import CategoricalData, sensor_to_categorical
from .lazy_indexer import LazyIndexer

logger = logging.getLogger('katfile.h5datav2')

# Simplify the scan activities to derive the basic state of the antenna (slewing, scanning, tracking, stopped)
SIMPLIFY_STATE = {'scan_ready': 'slew', 'scan': 'scan', 'scan_complete': 'scan', 'track': 'track', 'slew': 'slew'}

SENSOR_PROPS = {
    '*activity' : {'greedy_values' : ('slew', 'stop'), 'initial_value' : 'slew',
                   'transform' : lambda act: SIMPLIFY_STATE.get(act, 'stop')},
    '*target' : {'initial_value' : '', 'transform' : _robust_target},
    'RFE/center-frequency-hz' : {'categorical' : True},
    'RFE/rfe7.lo1.frequency' : {'categorical' : True},
    'Observation/label' : {'initial_value' : '', 'transform' : str, 'allow_repeats' : True}
}

def _calc_azel(cache, name, ant):
    """Calculate virtual (az, el) sensors from actual ones in sensor cache."""
    real_sensor = 'Antennas/%s/%s' % (ant, 'pos.actual-scan-azim' if name.endswith('az') else 'pos.actual-scan-elev')
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data

VIRTUAL_SENSORS = dict(DEFAULT_VIRTUAL_SENSORS)
VIRTUAL_SENSORS.update({'Antennas/{ant}/az' : _calc_azel, 'Antennas/{ant}/el' : _calc_azel})

#--------------------------------------------------------------------------------------------------
#--- Utility functions
#--------------------------------------------------------------------------------------------------

def get_single_value(group, name):
    """Return single value from attribute or dataset with given name in group.

    If `name` is an attribute of the HDF5 group `group`, it is returned,
    otherwise it is interpreted as an HDF5 dataset of `group` and the last value
    of `name` is returned. This is meant to retrieve static configuration values
    that potentially get set more than once during capture initialisation, but
    then does not change during actual capturing.

    Parameters
    ----------
    group : :class:`h5py.Group` object
        HDF5 group to query
    name : string
        Name of HDF5 attribute or dataset to query

    Returns
    -------
    value : object
        Attribute or last value of dataset

    """
    return group.attrs[name] if name in group.attrs else group[name].value[-1]

#--------------------------------------------------------------------------------------------------
#--- CLASS :  H5DataV2
#--------------------------------------------------------------------------------------------------

class H5DataV2(DataSet):
    """Load HDF5 format version 2 file produced by KAT-7 correlator.

    For more information on attributes, see the :class:`DataSet` docstring.

    Parameters
    ----------
    filename : string
        Name of HDF5 file
    ref_ant : string, optional
        Name of reference antenna, used to partition data set into scans
        (default is first antenna in use)
    time_offset : float, optional
        Offset to add to all correlator timestamps, in seconds

    Attributes
    ----------
    file : :class:`h5py.File` object
        Underlying HDF5 file, exposed via :mod:`h5py` interface

    """
    def __init__(self, filename, ref_ant='', time_offset=0.0):
        DataSet.__init__(self, filename, ref_ant, time_offset)

        # Load file
        self.file = f = h5py.File(filename, 'r')

        # Only continue if file is correct version and has been properly augmented
        self.version = f.attrs.get('version', '1.x')
        if not self.version.startswith('2.'):
            raise WrongVersion("Attempting to load version '%s' file with version 2 loader" % (self.version,))
        if not 'augment_ts' in f.attrs:
            raise BrokenFile('HDF5 file not augmented - please run k7_augment.py (provided by katcapture package)')

        # Load main HDF5 groups
        data_group, sensors_group, config_group = f['Data'], f['MetaData/Sensors'], f['MetaData/Configuration']
        markup_group = f['Markup']
        # Get observation script attributes, with defaults
        script_attrs = config_group['Observation'].attrs
        self.observer = script_attrs.get('script_observer', '')
        self.description = script_attrs.get('script_description', '')
        self.experiment_id = script_attrs.get('script_experiment_id', '')

        # ------ Extract timestamps ------

        self.dump_period = get_single_value(config_group['Correlator'], 'int_time')
        # Obtain visibility data and timestamps
        self._vis = data_group['correlator_data']
        self._timestamps = data_group['timestamps']
        num_dumps = len(self._timestamps)
        # Discard the last sample if the timestamp is a duplicate (caused by stop packet in k7_capture)
        num_dumps = (num_dumps - 1) if num_dumps > 1 and (self._timestamps[-1] == self._timestamps[-2]) else num_dumps
        # Estimate timestamps by assuming they are uniformly spaced (much quicker than loading them from file).
        # This is useful for the purpose of segmenting data set, where accurate timestamps are not that crucial.
        # The real timestamps are still loaded when the user explicitly asks for them.
        # Do quick test for uniform spacing of timestamps (necessary but not sufficient).
        if abs((self._timestamps[num_dumps - 1] - self._timestamps[0]) / self.dump_period + 1 - num_dumps) < 0.01:
            # Estimate the timestamps as being uniformly spaced
            data_timestamps = self._timestamps[0] + self.dump_period * np.arange(num_dumps)
        else:
            # Load the real timestamps instead and warn the user, as this is anomalous
            data_timestamps = self._timestamps[:num_dumps]
            expected_dumps = (self._timestamps[num_dumps - 1] - self._timestamps[0]) / self.dump_period + 1
            logger.warning(("Irregular timestamps detected in file '%s':"
                           "expected %.3f dumps based on dump period and start/end times, got %d instead") %
                           (filename, expected_dumps, num_dumps))
        # Move timestamps from start of each dump to the middle of the dump
        data_timestamps += 0.5 * self.dump_period + self.time_offset
        if data_timestamps[0] < 1e9:
            logger.warning("File '%s' has invalid first correlator timestamp (%f)" % (filename, data_timestamps[0],))
        self._time_keep = np.ones(num_dumps, dtype=np.bool)
        self.start_time = katpoint.Timestamp(data_timestamps[0] - 0.5 * self.dump_period)
        self.end_time = katpoint.Timestamp(data_timestamps[-1] + 0.5 * self.dump_period)

        # ------ Extract sensors ------

        # Populate sensor cache with all HDF5 datasets below sensor group that fit the description of a sensor
        cache = {}
        def register_sensor(name, obj):
            """A sensor is defined as a non-empty dataset with expected dtype."""
            if isinstance(obj, h5py.Dataset) and obj.shape != () and obj.dtype.names == ('timestamp','value','status'):
                cache[name] = SensorData(obj, name)
        sensors_group.visititems(register_sensor)
        # Use estimated data timestamps for now, to speed up data segmentation
        self.sensor = SensorCache(cache, data_timestamps, self.dump_period, keep=self._time_keep,
                                  props=SENSOR_PROPS, virtual=VIRTUAL_SENSORS)

        # ------ Extract subarrays ------

        # Original list of correlation products as pairs of input labels
        corrprods = get_single_value(config_group['Correlator'], 'bls_ordering')
        if len(corrprods) != self._vis.shape[2]:
            raise BrokenFile('Number of baseline labels received from correlator '
                           '(%d) differs from number of baselines in data (%d)' % (len(corrprods), self._vis.shape[2]))
        # All antennas in configuration as katpoint Antenna objects
        ants = [katpoint.Antenna(config_group['Antennas'][name].attrs['description'])
                for name in config_group['Antennas']]
        self.subarrays = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(self.subarrays, [0, len(data_timestamps)])
        self.sensor['Observation/subarray_index'] = CategoricalData([0], [0, len(data_timestamps)])
        # Store antenna objects in sensor cache too, for use in virtual sensor calculations
        for ant in ants:
            self.sensor['Antennas/%s/antenna' % (ant.name,)] = CategoricalData([ant], [0, len(data_timestamps)])
        # By default, only pick antennas that were in use by the script
        script_ants = config_group['Observation'].attrs['script_ants'].split(',')
        self.ref_ant = script_ants[0] if not ref_ant else ref_ant

        # ------ Extract spectral windows / frequencies ------

        centre_freq = self.sensor.get('RFE/center-frequency-hz')
        num_chans = get_single_value(config_group['Correlator'], 'n_chans')
        if num_chans != self._vis.shape[1]:
            raise BrokenFile('Number of channels received from correlator '
                             '(%d) differs from number of channels in data (%d)' % (num_chans, self._vis.shape[1]))
        channel_width = get_single_value(config_group['Correlator'], 'bandwidth') / num_chans
        self.spectral_windows = [SpectralWindow(spw_centre, channel_width, num_chans)
                                 for spw_centre in centre_freq.unique_values]
        self.sensor['Observation/spw'] = CategoricalData(self.spectral_windows, centre_freq.events)
        self.sensor['Observation/spw_index'] = CategoricalData(centre_freq.indices, centre_freq.events)

        # ------ Extract scans / compound scans / targets ------

        # Use the activity sensor of reference antenna to partition the data set into scans (and to set their states)
        scan = self.sensor.get('Antennas/%s/activity' % (self.ref_ant,))
        self.sensor['Observation/scan_state'] = scan
        self.sensor['Observation/scan_index'] = CategoricalData(range(len(scan)), scan.events)
        # Use labels to partition the data set into compound scans
        label = sensor_to_categorical(markup_group['labels']['timestamp'], markup_group['labels']['label'],
                                      data_timestamps, self.dump_period, **SENSOR_PROPS['Observation/label'])
        # Discard empty labels (typically found in raster scans, where first scan has proper label and rest are empty)
        label.remove('')
        # Move proper label events onto the nearest scan start
        # ASSUMPTION: Number of labels <= number of scans (i.e. only a single label allowed per scan)
        label.align(scan.events)
        # If one or more scans at start of data set have no corresponding label, add a default label for them
        if label.events[0] > 0:
            label.add(0, '')
        self.sensor['Observation/label'] = label
        self.sensor['Observation/compscan_index'] = CategoricalData(range(len(label)), label.events)
        # Use the target sensor of reference antenna to set the target for each scan
        target = self.sensor.get('Antennas/%s/target' % (self.ref_ant,))
        # Move target events onto the nearest scan start
        # ASSUMPTION: Number of targets <= number of scans (i.e. only a single target allowed per scan)
        target.align(scan.events)
        self.sensor['Observation/target'] = target
        self.sensor['Observation/target_index'] = CategoricalData(target.indices, target.events)
        # Set up catalogue containing all targets in file, with reference antenna as default antenna
        self.catalogue.add(target.unique_values)
        self.catalogue.antenna = self.sensor['Antennas/%s/antenna' % (self.ref_ant,)][0]

        # Avoid storing reference to self in transform closure below, as this hinders garbage collection
        dump_period, time_offset = self.dump_period, self.time_offset
        # Restore original (slow) timestamps so that subsequent sensors (e.g. pointing) will have accurate values
        self.sensor.timestamps = LazyIndexer(self._timestamps, keep=slice(num_dumps),
                                             transform=lambda t, keep: t + 0.5 * dump_period + time_offset)
        # Apply default selection and initialise all members that depend on selection in the process
        self.select(spw=0, subarray=0, ants=script_ants)

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since epoch.

        The timestamps are returned as an array indexer of float64, shape (*T*,),
        with one timestamp per integration aligned with the integration
        *midpoint*. To get the data array itself from the indexer `x`, do `x[:]`
        or perform any other form of indexing on it.

        """
        # Avoid storing reference to self in transform closure below, as this hinders garbage collection
        dump_period, time_offset = self.dump_period, self.time_offset
        return LazyIndexer(self._timestamps, keep=self._time_keep,
                           transform=lambda t, keep: t + 0.5 * dump_period + time_offset)

    @property
    def vis(self):
        """Complex visibility data as a function of time, frequency and baseline.

        The visibility data is returned as an array indexer of complex64, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The returned array always has all three dimensions,
        even for scalar (single) values. The number of integrations *T* matches
        the length of :meth:`timestamps`, the number of frequency channels *F*
        matches the length of :meth:`freqs` and the number of correlation
        products *B* matches the length of :meth:`corr_products`. To get the
        data array itself from the indexer `x`, do `x[:]` or perform any other
        form of indexing on it. Only then will data be loaded into memory.

        """
        def extract_vis(vis, keep):
            same_ndim = tuple([(np.newaxis if np.isscalar(dim_keep) else slice(None)) for dim_keep in keep[:3]] + [0])
            return vis.view(np.complex64)[same_ndim]
        return LazyIndexer(self._vis, (self._time_keep, self._freq_keep, self._corrprod_keep),
                           transform=extract_vis, shape_transform=lambda shape:shape[:-1], dtype=np.complex64)
