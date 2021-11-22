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

"""Data accessor class for HDF5 files produced by KAT-7 correlator."""

import logging
import pathlib
import secrets

import h5py
import katpoint
import numpy as np

from .categorical import CategoricalData, sensor_to_categorical
from .dataset import (DEFAULT_SENSOR_PROPS, DEFAULT_VIRTUAL_SENSORS,
                      BrokenFile, DataSet, Subarray, WrongVersion,
                      _robust_target, _selection_to_list)
from .flags import DESCRIPTIONS as FLAG_DESCRIPTIONS
from .flags import NAMES as FLAG_NAMES
from .lazy_indexer import LazyIndexer, LazyTransform
from .sensordata import RecordSensorGetter, SensorCache, to_str
from .spectral_window import SpectralWindow

logger = logging.getLogger(__name__)

# Simplify the scan activities to derive the basic state of the antenna (slewing, scanning, tracking, stopped)
SIMPLIFY_STATE = {'scan_ready': 'slew', 'scan': 'scan', 'scan_complete': 'scan', 'track': 'track', 'slew': 'slew'}

SENSOR_PROPS = dict(DEFAULT_SENSOR_PROPS)
SENSOR_PROPS.update({
    '*activity': {'greedy_values': ('slew', 'stop'), 'initial_value': 'slew',
                  'transform': lambda act: SIMPLIFY_STATE.get(act, 'stop')},
    '*target': {'initial_value': '', 'transform': _robust_target},
    # These float sensors are actually categorical by nature as they represent user settings
    'RFE/center-frequency-hz': {'categorical': True},
    'RFE/rfe7.lo1.frequency': {'categorical': True},
    '*attenuation': {'categorical': True},
    '*attenuator.horizontal': {'categorical': True},
    '*attenuator.vertical': {'categorical': True},
})

SENSOR_ALIASES = {
    'nd_coupler': 'rfe3.rfe15.noise.coupler.on',
    'nd_pin': 'rfe3.rfe15.noise.pin.on',
}


def _calc_azel(cache, name, ant):
    """Calculate virtual (az, el) sensors from actual ones in sensor cache."""
    base_name = 'pos.actual-scan-azim' if name.endswith('az') else 'pos.actual-scan-elev'
    real_sensor = f'Antennas/{ant}/{base_name}'
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data


VIRTUAL_SENSORS = dict(DEFAULT_VIRTUAL_SENSORS)
VIRTUAL_SENSORS.update({'Antennas/{ant}/az': _calc_azel, 'Antennas/{ant}/el': _calc_azel})

WEIGHT_NAMES = ('precision',)
WEIGHT_DESCRIPTIONS = ('visibility precision (inverse variance, i.e. 1 / sigma^2)',)

# -------------------------------------------------------------------------------------------------
# -- Utility functions
# -------------------------------------------------------------------------------------------------


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
    attrs = group.attrs
    value = attrs[name] if name in attrs else group[name][-1]
    return to_str(value)


def dummy_dataset(name, shape, dtype, value):
    """Dummy HDF5 dataset containing a single value.

    This creates a dummy HDF5 dataset in memory containing a single value. It
    can have virtually unlimited size as the dataset is highly compressed.

    Parameters
    ----------
    name : string
        Name of dataset
    shape : sequence of int
        Shape of dataset
    dtype : :class:`numpy.dtype` object or equivalent
        Type of data stored in dataset
    value : object
        All elements in the dataset will equal this value

    Returns
    -------
    dataset : :class:`h5py.Dataset` object
        Dummy HDF5 dataset

    """
    # It is important to randomise the filename as h5py does not allow two writable file objects with the same name
    # Without this randomness katdal can only open one file requiring a dummy dataset
    dummy_file = h5py.File(f'{name}_{secrets.token_hex(8)}.h5', 'x', driver='core', backing_store=False)
    return dummy_file.create_dataset(name, shape=shape, maxshape=shape,
                                     dtype=dtype, fillvalue=value, compression='gzip')

# -------------------------------------------------------------------------------------------------
# -- CLASS :  H5DataV2
# -------------------------------------------------------------------------------------------------


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
    mode : string, optional
        HDF5 file opening mode (e.g. 'r+' to open file in write mode)
    quicklook : {False, True}
        True if synthesised timestamps should be used to partition data set even
        if real timestamps are irregular, thereby avoiding the slow loading of
        real timestamps at the cost of slightly inaccurate label borders
    keepdims : {False, True}, optional
        Force vis / weights / flags to be 3-dimensional, regardless of selection
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    Attributes
    ----------
    file : :class:`h5py.File` object
        Underlying HDF5 file, exposed via :mod:`h5py` interface

    """

    def __init__(self, filename, ref_ant='', time_offset=0.0, mode='r',
                 quicklook=False, keepdims=False, **kwargs):
        # The closest thing to a capture block ID is the Unix timestamp in the original filename
        # There is only one (unnamed) output stream, so leave off the stream name
        cbid = pathlib.Path(filename).stem
        DataSet.__init__(self, cbid, ref_ant, time_offset, url=filename)

        # Load file
        self.file, self.version = H5DataV2._open(filename, mode)
        f = self.file

        # Load main HDF5 groups
        data_group, sensors_group, config_group = f['Data'], f['MetaData/Sensors'], f['MetaData/Configuration']
        markup_group = f['Markup']
        # Get observation script parameters, with defaults
        for k, v in config_group['Observation'].items():
            # For KAT-7 (v2.1) data, strip the 'script_' prefix from most parameters
            k = k if self.version > '2.1' or k in ('script_name', 'script_arguments') else k[7:]
            self.obs_params[str(k)] = to_str(v)
        self.observer = self.obs_params.get('observer', '')
        self.description = self.obs_params.get('description', '')
        self.experiment_id = self.obs_params.get('experiment_id', '')
        # Get script log from History group
        self.obs_script_log = f['History/script_log']['log'].tolist()

        # ------ Extract timestamps ------

        self.dump_period = get_single_value(config_group['Correlator'], 'int_time')
        # Obtain visibility data and timestamps
        self._vis = data_group['correlator_data']
        self._timestamps = data_group['timestamps']
        num_dumps = len(self._timestamps)
        if num_dumps != self._vis.shape[0]:
            raise BrokenFile(f'Number of timestamps received from k7_capture ({num_dumps}) '
                             f'differs from number of dumps in data ({self._vis.shape[0]})')
        # Discard the last sample if the timestamp is a duplicate (caused by stop packet in k7_capture)
        num_dumps = (num_dumps - 1) if num_dumps > 1 and (self._timestamps[-1] == self._timestamps[-2]) else num_dumps
        # Do quick test for uniform spacing of timestamps (necessary but not sufficient)
        expected_dumps = (self._timestamps[num_dumps - 1] - self._timestamps[0]) / self.dump_period + 1
        # The expected_dumps should always be an integer (like num_dumps), unless the timestamps and/or dump period
        # are messed up in the file, so the threshold of this test is a bit arbitrary (e.g. could use > 0.5)
        irregular = abs(expected_dumps - num_dumps) >= 0.01
        if irregular:
            # Warn the user, as this is anomalous
            logger.warning("Irregular timestamps detected in file '%s': expected %.3f dumps "
                           "based on dump period and start/end times, got %d instead",
                           filename, expected_dumps, num_dumps)
            if quicklook:
                logger.warning("Quicklook option selected - partitioning data based on synthesised timestamps instead")
        if not irregular or quicklook:
            # Estimate timestamps by assuming they are uniformly spaced (much quicker than loading them from file).
            # This is useful for the purpose of segmenting data set, where accurate timestamps are not that crucial.
            # The real timestamps are still loaded when the user explicitly asks for them.
            data_timestamps = self._timestamps[0] + self.dump_period * np.arange(num_dumps)
        else:
            # Load the real timestamps instead (could take several seconds on a large data set)
            data_timestamps = self._timestamps[:num_dumps]
        # Move timestamps from start of each dump to the middle of the dump
        data_timestamps += 0.5 * self.dump_period + self.time_offset
        if data_timestamps[0] < 1e9:
            logger.warning("File '%s' has invalid first correlator timestamp (%f)", filename, data_timestamps[0])
        self._time_keep = np.ones(num_dumps, dtype=np.bool)
        self.start_time = katpoint.Timestamp(data_timestamps[0] - 0.5 * self.dump_period)
        self.end_time = katpoint.Timestamp(data_timestamps[-1] + 0.5 * self.dump_period)
        self._keepdims = keepdims

        # ------ Extract flags ------

        # Check if flag group is present, else use dummy flag data
        self._flags = markup_group['flags'] if 'flags' in markup_group else \
            dummy_dataset('dummy_flags', shape=self._vis.shape[:-1], dtype=np.uint8, value=0)
        # Obtain flag descriptions from file or recreate default flag description table
        self._flags_description = to_str(markup_group['flags_description'][:]) \
            if 'flags_description' in markup_group else np.array(list(zip(FLAG_NAMES, FLAG_DESCRIPTIONS)))
        self._flags_select = np.array([0], dtype=np.uint8)
        self._flags_keep = 'all'

        # ------ Extract weights ------

        # Check if weight group present, else use dummy weight data
        self._weights = markup_group['weights'] if 'weights' in markup_group else \
            dummy_dataset('dummy_weights', shape=self._vis.shape[:-1], dtype=np.float32, value=1.0)
        # Obtain weight descriptions from file or recreate default weight description table
        self._weights_description = to_str(markup_group['weights_description'][:]) \
            if 'weights_description' in markup_group else np.array(list(zip(WEIGHT_NAMES, WEIGHT_DESCRIPTIONS)))
        self._weights_select = []
        self._weights_keep = 'all'

        # ------ Extract sensors ------

        # Populate sensor cache with all HDF5 datasets below sensor group that fit the description of a sensor
        cache = {}

        def register_sensor(name, obj):
            """A sensor is defined as a non-empty dataset with expected dtype."""
            if isinstance(obj, h5py.Dataset) and obj.shape != () and \
               obj.dtype.names == ('timestamp', 'value', 'status'):
                # Rename pedestal sensors from the old regime to become sensors of the corresponding antenna
                name = ('Antennas/ant' + name[13:]) if name.startswith('Pedestals/ped') else name
                cache[name] = RecordSensorGetter(obj, name)
        sensors_group.visititems(register_sensor)
        # Use estimated data timestamps for now, to speed up data segmentation
        self.sensor = SensorCache(cache, data_timestamps, self.dump_period, keep=self._time_keep,
                                  props=SENSOR_PROPS, virtual=VIRTUAL_SENSORS, aliases=SENSOR_ALIASES)

        # ------ Extract subarrays ------

        # By default, only pick antennas that were in use by the script
        script_ants = to_str(config_group['Observation'].attrs['script_ants']).split(',')
        self.ref_ant = script_ants[0] if not ref_ant else ref_ant
        # Original list of correlation products as pairs of input labels
        corrprods = get_single_value(config_group['Correlator'], 'bls_ordering')
        if len(corrprods) != self._vis.shape[2]:
            # Apply k7_capture baseline mask after the fact, in the hope that it fixes correlation product mislabelling
            corrprods = np.array([cp for cp in corrprods if cp[0][:-1] in script_ants and cp[1][:-1] in script_ants])
            # If there is still a mismatch between labels and data shape, file is considered broken (maybe bad labels?)
            if len(corrprods) != self._vis.shape[2]:
                raise BrokenFile('Number of baseline labels (containing expected antenna names) '
                                 'received from correlator (%d) differs from number of baselines in data (%d)' %
                                 (len(corrprods), self._vis.shape[2]))
            else:
                logger.warning('Reapplied k7_capture baseline mask to fix unexpected number of baseline labels')
        # All antennas in configuration as katpoint Antenna objects
        ants = [katpoint.Antenna(to_str(config_group['Antennas'][name].attrs['description']))
                for name in config_group['Antennas']]
        self.subarrays = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(self.subarrays, [0, len(data_timestamps)])
        self.sensor['Observation/subarray_index'] = CategoricalData([0], [0, len(data_timestamps)])
        # Store antenna objects in sensor cache too, for use in virtual sensor calculations
        for ant in ants:
            self.sensor[f'Antennas/{ant.name}/antenna'] = CategoricalData([ant], [0, len(data_timestamps)])
        # Extract array reference from first antenna (first 5 fields of description)
        array_ant_fields = ['array'] + ants[0].description.split(',')[1:5]
        array_ant = katpoint.Antenna(','.join(array_ant_fields))
        self.sensor['Antennas/array/antenna'] = CategoricalData([array_ant], [0, len(data_timestamps)])

        # ------ Extract spectral windows / frequencies ------

        # Ideally we would like to use calculated center-frequency-hz sensor produced by k7_capture (better for nband)
        if self.version >= '2.1':
            centre_freq = self.sensor.get('RFE/center-frequency-hz')
        else:
            # Fall back to basic RFE7 LO frequency, as this supported multiple spectral windows before k7_capture did
            # This assumes WBC mode, though (NBC modes only fully supported since HDF5 v2.1)
            centre_freq = self.sensor.get('RFE/rfe7.lo1.frequency')
            centre_freq.unique_values = [freq - 4200e6 for freq in centre_freq.unique_values]
        num_chans = get_single_value(config_group['Correlator'], 'n_chans')
        if num_chans != self._vis.shape[1]:
            raise BrokenFile(f'Number of channels received from correlator ({num_chans}) '
                             f'differs from number of channels in data ({self._vis.shape[1]})')
        bandwidth = get_single_value(config_group['Correlator'], 'bandwidth')
        channel_width = bandwidth / num_chans
        try:
            mode = self.sensor.get('DBE/dbe.mode').unique_values[0]
        except (KeyError, IndexError):
            # Guess the mode for version 2.0 files that haven't been re-augmented
            mode = 'wbc' if num_chans <= 1024 else 'wbc8k' if bandwidth > 200e6 else 'nbc'
        self.spectral_windows = [SpectralWindow(spw_centre, channel_width, num_chans, mode)
                                 for spw_centre in centre_freq.unique_values]
        self.sensor['Observation/spw'] = CategoricalData([self.spectral_windows[idx] for idx in centre_freq.indices],
                                                         centre_freq.events)
        self.sensor['Observation/spw_index'] = CategoricalData(centre_freq.indices, centre_freq.events)

        # ------ Extract scans / compound scans / targets ------

        # Use the activity sensor of reference antenna to partition the data set into scans (and to set their states)
        scan = self.sensor.get(f'Antennas/{self.ref_ant}/activity')
        # If the antenna starts slewing on the second dump, incorporate the first dump into the slew too.
        # This scenario typically occurs when the first target is only set after the first dump is received.
        # The workaround avoids putting the first dump in a scan by itself, typically with an irrelevant target.
        if len(scan) > 1 and scan.events[1] == 1 and scan[1] == 'slew':
            scan.events, scan.indices = scan.events[1:], scan.indices[1:]
            scan.events[0] = 0
        # Use labels to partition the data set into compound scans
        label = sensor_to_categorical(markup_group['labels']['timestamp'], to_str(markup_group['labels']['label'][:]),
                                      data_timestamps, self.dump_period, **SENSOR_PROPS['Observation/label'])
        # Discard empty labels (typically found in raster scans, where first scan has proper label and rest are empty)
        # However, if all labels are empty, keep them, otherwise whole data set will be one pathological compscan...
        if len(label.unique_values) > 1:
            label.remove('')
        # Create duplicate scan events where labels are set during a scan (i.e. not at start of scan)
        # ASSUMPTION: Number of scans >= number of labels (i.e. each label should introduce a new scan)
        scan.add_unmatched(label.events)
        self.sensor['Observation/scan_state'] = scan
        self.sensor['Observation/scan_index'] = CategoricalData(list(range(len(scan))), scan.events)
        # Move proper label events onto the nearest scan start
        # ASSUMPTION: Number of labels <= number of scans (i.e. only a single label allowed per scan)
        label.align(scan.events)
        # If one or more scans at start of data set have no corresponding label, add a default label for them
        if label.events[0] > 0:
            label.add(0, '')
        self.sensor['Observation/label'] = label
        self.sensor['Observation/compscan_index'] = CategoricalData(list(range(len(label))), label.events)
        # Use the target sensor of reference antenna to set the target for each scan
        target = self.sensor.get(f'Antennas/{self.ref_ant}/target')
        # Move target events onto the nearest scan start
        # ASSUMPTION: Number of targets <= number of scans (i.e. only a single target allowed per scan)
        target.align(scan.events)
        self.sensor['Observation/target'] = target
        self.sensor['Observation/target_index'] = CategoricalData(target.indices, target.events)
        # Set up catalogue containing all targets in file, with reference antenna as default antenna
        self.catalogue.add(target.unique_values)
        self.catalogue.antenna = self.sensor[f'Antennas/{self.ref_ant}/antenna'][0]
        # Ensure that each target flux model spans all frequencies in data set if possible
        self._fix_flux_freq_range()

        # Avoid storing reference to self in transform closure below, as this hinders garbage collection
        dump_period, time_offset = self.dump_period, self.time_offset
        # Restore original (slow) timestamps so that subsequent sensors (e.g. pointing) will have accurate values
        extract_time = LazyTransform('extract_time', lambda t, keep: t + 0.5 * dump_period + time_offset)
        self.sensor.timestamps = LazyIndexer(self._timestamps, keep=slice(num_dumps), transforms=[extract_time])
        # Apply default selection and initialise all members that depend on selection in the process
        self.select(spw=0, subarray=0, ants=script_ants)

    @staticmethod
    def _open(filename, mode='r'):
        """Open file and do basic version and augmentation sanity check."""
        f = h5py.File(filename, mode)
        version = to_str(f.attrs.get('version', '1.x'))
        if not version.startswith('2.'):
            raise WrongVersion(f"Attempting to load version '{version}' file with version 2 loader")
        if 'augment_ts' not in f.attrs:
            raise BrokenFile('HDF5 file not augmented - please run '
                             'k7_augment.py (provided by katcapture package)')
        return f, version

    @staticmethod
    def _get_ants(filename):
        """Quick look function to get the list of antennas in a data file.

        This is intended to be called without createing a full katdal object.

        Parameters
        ----------
        filename : string
            Data file name

        Returns
        -------
        antennas : list of :class:'katpoint.Antenna' objects

        """
        f, version = H5DataV2._open(filename)
        config_group = f['MetaData/Configuration']
        all_ants = [ant for ant in config_group['Antennas']]
        script_ants = to_str(config_group['Observation'].attrs.get('script_ants'))
        script_ants = script_ants.split(',') if script_ants else all_ants
        return [katpoint.Antenna(to_str(config_group['Antennas'][ant].attrs['description']))
                for ant in script_ants if ant in all_ants]

    @staticmethod
    def _get_targets(filename):
        """Quick look function to get the list of targets in a data file.

        This is intended to be called without createing a full katdal object.

        Parameters
        ----------
        filename : string
            Data file name

        Returns
        -------
        targets : :class:'katpoint.Catalogue' object
            All targets in file

        """
        f, version = H5DataV2._open(filename)
        # Use the delay-tracking centre as the one and only target
        # Try two different sensors for the DBE target
        try:
            target_list = f['MetaData/Sensors/DBE/target']
        except Exception:
            # Since h5py errors have varied over the years, we need Exception
            target_list = f['MetaData/Sensors/Beams/Beam0/target']
        all_target_strings = [to_str(target_data[1]) for target_data in target_list]
        return katpoint.Catalogue(np.unique(all_target_strings))

    def __str__(self):
        """Verbose human-friendly string representation of data set."""
        descr = [super().__str__()]
        # append the process_log, if it exists, for non-concatenated h5 files
        if 'process_log' in self.file['History']:
            descr.append('-------------------------------------------------------------------------------')
            descr.append('Process log:')
            for proc in self.file['History']['process_log']:
                # proc has a structured dtype and to_str doesn't work on it, so
                # we have to to_str each element.
                param_list = f'{to_str(proc[0]):>15}:'
                for param in to_str(proc[1]).split(','):
                    param_list += f'  {param}'
                descr.append(param_list)
        return '\n'.join(descr)

    @property
    def _weights_keep(self):
        known_weights = [row[0] for row in getattr(self, '_weights_description', [])]
        return [known_weights[ind] for ind in self._weights_select]

    @_weights_keep.setter
    def _weights_keep(self, names):
        known_weights = [row[0] for row in getattr(self, '_weights_description', [])]
        # Ensure a sequence of weight names
        names = _selection_to_list(names, all=known_weights)
        # Create index list for desired weights
        selection = []
        for name in names:
            try:
                selection.append(known_weights.index(name))
            except ValueError:
                logger.warning("%r is not a legitimate weight type for this file, "
                               "supported ones are %s", name, known_weights)
        if known_weights and not selection:
            logger.warning('No valid weights were selected - setting all weights to 1.0 by default')
        self._weights_select = selection

    @property
    def _flags_keep(self):
        if not hasattr(self, '_flags_description'):
            return []
        known_flags = [row[0] for row in self._flags_description]
        # The KAT-7 flagger uses the np.packbits convention (bit 0 = MSB) so don't flip
        selection = np.unpackbits(self._flags_select)
        assert len(known_flags) == len(selection), \
            f'Expected {len(selection)} flag types in file, got {self._flags_description}'
        return [name for name, bit in zip(known_flags, selection) if bit]

    @_flags_keep.setter
    def _flags_keep(self, names):
        if not hasattr(self, '_flags_description'):
            self._flags_select = np.array([0], dtype=np.uint8)
            return
        known_flags = [row[0] for row in self._flags_description]
        # Ensure `names` is a sequence of valid flag names (or an empty list)
        names = _selection_to_list(names, all=known_flags)
        # Create boolean list for desired flags
        selection = np.zeros(8, dtype=np.uint8)
        assert len(known_flags) == len(selection), \
            f'Expected {len(selection)} flag types in file, got {self._flags_description}'
        for name in names:
            try:
                selection[known_flags.index(name)] = 1
            except ValueError:
                logger.warning("%r is not a legitimate flag type for this file, "
                               "supported ones are %s", name, known_flags)
        # Pack index list into bit mask
        # The KAT-7 flagger uses the np.packbits convention (bit 0 = MSB) so don't flip
        flagmask = np.packbits(selection)
        if known_flags and not flagmask:
            logger.warning('No valid flags were selected - setting all flags to False by default')
        self._flags_select = flagmask

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array indexer of float64, shape (*T*,),
        with one timestamp per integration aligned with the integration
        *midpoint*. To get the data array itself from the indexer `x`, do `x[:]`
        or perform any other form of indexing on it.

        """
        # Avoid storing reference to self in transform closure below, as this hinders garbage collection
        dump_period, time_offset = self.dump_period, self.time_offset
        extract_time = LazyTransform('extract_time', lambda t, keep: t + 0.5 * dump_period + time_offset)
        return LazyIndexer(self._timestamps, keep=self._time_keep, transforms=[extract_time])

    def _vislike_indexer(self, dataset, extractor):
        """Lazy indexer for vis-like datasets (vis / weights / flags).

        This operates on datasets with shape (*T*, *F*, *B*) and potentially
        different dtypes. The data type conversions are all left to the provided
        extractor transform, while this method takes care of the common
        selection issues, such as preserving singleton dimensions and dealing
        with duplicate final dumps.

        Parameters
        ----------
        dataset : :class:`h5py.Dataset` object or equivalent
            Underlying vis-like dataset on which lazy indexing will be done
        extractor : function, signature ``data = f(data, keep)``
            Transform to apply to data (`keep` is user-provided 2nd-stage index)

        Returns
        -------
        indexer : :class:`LazyIndexer` object
            Lazy indexer with appropriate selectors and transforms included

        """
        # Create first-stage index from dataset selectors
        time_keep = self._time_keep
        # If there is a duplicate final dump, these lengths don't match -> ignore last dump in file
        if len(time_keep) == len(dataset) - 1:
            time_keep = np.zeros(len(dataset), dtype=np.bool)
            time_keep[:len(self._time_keep)] = self._time_keep
        stage1 = (time_keep, self._freq_keep, self._corrprod_keep)

        def _force_3dim(data, keep):
            """Keep singleton dimensions in stage 2 (i.e. final) indexing."""
            # Ensure that keep tuple has length of 3 (truncate or pad with blanket slices as necessary)
            keep = keep[:3] + (slice(None),) * (3 - len(keep))
            # Final indexing ensures that returned data are always 3-dimensional (i.e. keep singleton dimensions)
            keep_singles = [(np.newaxis if np.isscalar(dim_keep) else slice(None))
                            for dim_keep in keep]
            return data[tuple(keep_singles)]
        force_3dim = LazyTransform('force_3dim', _force_3dim)
        transforms = [extractor, force_3dim] if self._keepdims else [extractor]
        return LazyIndexer(dataset, stage1, transforms)

    @property
    def vis(self):
        r"""Complex visibility data as a function of time, frequency and baseline.

        The visibility data are returned as an array indexer of complex64, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The returned array always has all three dimensions,
        even for scalar (single) values. The number of integrations *T* matches
        the length of :meth:`timestamps`, the number of frequency channels *F*
        matches the length of :meth:`freqs` and the number of correlation
        products *B* matches the length of :meth:`corr_products`. To get the
        data array itself from the indexer `x`, do `x[:]` or perform any other
        form of indexing on it. Only then will data be loaded into memory.

        The sign convention of the imaginary part is consistent with an
        electric field of :math:`e^{i(\omega t - jz)}` i.e. phase that
        increases with time.
        """
        extract = LazyTransform('extract_vis',
                                # Discard the 4th / last dimension as this is subsumed in complex view
                                # The visibilities are conjugated due to using the lower sideband
                                lambda vis, keep: vis.view(np.complex64)[..., 0].conjugate(),
                                lambda shape: shape[:-1], np.complex64)
        return self._vislike_indexer(self._vis, extract)

    @property
    def weights(self):
        """Visibility weights as a function of time, frequency and baseline.

        The weights data are returned as an array indexer of float32, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`. To get the data array
        itself from the indexer `x`, do `x[:]` or perform any other form of
        indexing on it. Only then will data be loaded into memory.

        """
        # We currently only cater for a single weight type (i.e. either select it or fall back to 1.0)
        def transform(weights, keep):
            return weights.astype(np.float32) if self._weights_select else \
                np.ones_like(weights, dtype=np.float32)
        extract = LazyTransform('extract_weights', transform, dtype=np.float32)
        return self._vislike_indexer(self._weights, extract)

    @property
    def flags(self):
        """Flags as a function of time, frequency and baseline.

        The flags data are returned as an array indexer of bool, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`. To get the data array
        itself from the indexer `x`, do `x[:]` or perform any other form of
        indexing on it. Only then will data be loaded into memory.

        """
        def transform(flags, keep):
            """Use flagmask to blank out the flags we don't want."""
            # Then convert uint8 to bool -> if any flag bits set, flag is set
            return np.bool_(np.bitwise_and(self._flags_select, flags))
        extract = LazyTransform('extract_flags', transform, dtype=np.bool)
        return self._vislike_indexer(self._flags, extract)

    @property
    def temperature(self):
        """Air temperature in degrees Celsius."""
        return self.sensor['Enviro/asc.air.temperature']

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        return self.sensor['Enviro/asc.air.pressure']

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        return self.sensor['Enviro/asc.air.relative-humidity']

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        return self.sensor['Enviro/asc.wind.speed']

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        return self.sensor['Enviro/asc.wind.direction']
