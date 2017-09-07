################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
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

"""Data accessor class for data and metadata from various sources in v4 format."""

import logging

import numpy as np
import katpoint

from .dataset import (DataSet, BrokenFile, Subarray, SpectralWindow,
                      DEFAULT_SENSOR_PROPS, DEFAULT_VIRTUAL_SENSORS,
                      _robust_target)
from .sensordata import SensorCache
from .categorical import CategoricalData
from .lazy_indexer import LazyIndexer, LazyTransform


logger = logging.getLogger(__name__)

# Simplify the scan activities to derive the basic state of the antenna
# (slewing, scanning, tracking, stopped)
SIMPLIFY_STATE = {'scan_ready': 'slew', 'scan': 'scan', 'scan_complete': 'scan',
                  'load_scan': 'scan', 'track': 'track', 'slew': 'slew'}

SENSOR_PROPS = dict(DEFAULT_SENSOR_PROPS)
SENSOR_PROPS.update({
    '*activity': {'greedy_values': ('slew', 'stop'), 'initial_value': 'slew',
                  'transform': lambda act: SIMPLIFY_STATE.get(act, 'stop')},
    '*ap_indexer_position': {'initial_value': ''},
    '*ku_frequency': {'categorical': True},
    '*noise_diode': {'categorical': True, 'greedy_values': (True,),
                     'initial_value': 0.0, 'transform': lambda x: x > 0.0},
    '*serial_number': {'initial_value': 0},
    '*target': {'initial_value': '', 'transform': _robust_target},
})

SENSOR_ALIASES = {
    'nd_coupler': 'dig_noise_diode',
}


def _calc_azel(cache, name, ant):
    """Calculate virtual (az, el) sensors from actual ones in sensor cache."""
    real_sensor = 'Antennas/%s/pos_actual_scan_%s' % \
                  (ant, 'azim' if name.endswith('az') else 'elev')
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data


VIRTUAL_SENSORS = dict(DEFAULT_VIRTUAL_SENSORS)
VIRTUAL_SENSORS.update({'Antennas/{ant}/az': _calc_azel,
                        'Antennas/{ant}/el': _calc_azel})

FLAG_NAMES = ('reserved0', 'static', 'cam', 'data_lost',
              'ingest_rfi', 'predicted_rfi', 'cal_rfi', 'reserved7')
FLAG_DESCRIPTIONS = ('reserved - bit 0',
                     'predefined static flag list',
                     'flag based on live CAM information',
                     'no data was received',
                     'RFI detected in ingest',
                     'RFI predicted from space based pollutants',
                     'RFI detected in calibration',
                     'reserved - bit 7')
WEIGHT_NAMES = ('precision',)
WEIGHT_DESCRIPTIONS = ('visibility precision (inverse variance, i.e. 1 / sigma^2)',)

# -----------------------------------------------------------------------------
# --- Utility functions
# -----------------------------------------------------------------------------


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
    # # It is important to randomise the filename as h5py does not allow two writable file objects with the same name
    # # Without this randomness katdal can only open one file requiring a dummy dataset
    # random_string = ''.join(['%02x' % (x,) for x in np.random.randint(256, size=8)])
    # dummy_file = h5py.File('%s_%s.h5' % (name, random_string), driver='core', backing_store=False)
    # return dummy_file.create_dataset(name, shape=shape, maxshape=shape,
    #                                  dtype=dtype, fillvalue=value, compression='gzip')

# -----------------------------------------------------------------------------
# -- CLASS :  VisibilityDataV4
# -----------------------------------------------------------------------------


class VisibilityDataV4(DataSet):
    """Access format version 4 visibility data and metadata.

    For more information on attributes, see the :class:`DataSet` docstring.

    Parameters
    ----------
    metadata : :class:`AttrsSensors` object
        Metadata attributes and sensors
    timestamps : array-like of float, length *T*
        Timestamps at centroids of visibilities in UTC seconds since Unix epoch
    data : None or :class:`VisFlagsWeights` object, optional
        Correlator data (visibilities, flags and weights)
    ref_ant : string, optional
        Name of reference antenna, used to partition data set into scans
        (default is first antenna in use)
    time_offset : float, optional
        Offset to add to all correlator timestamps, in seconds
    centre_freq : float or None, optional
        Override centre frequency if provided, in Hz
    band : string or None, optional
        Override receiver band if provided (e.g. 'l') - used to find ND models
    keepdims : {False, True}, optional
        Force vis / weights / flags to be 3-dimensional, regardless of selection
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    """
    def __init__(self, metadata, timestamps, data=None, ref_ant='',
                 time_offset=0.0, centre_freq=None, band=None, keepdims=False,
                 **kwargs):
        name = metadata.name
        if data and data.name != name:
            name += ' | ' + data.name
        DataSet.__init__(self, name, ref_ant, time_offset)

        # ------ Extract vis and timestamps ------

        self.file = {}
        self.version = '4.0'
        self.dump_period = metadata.attrs['sdp_l0_int_time']
        self._timestamps = timestamps[:]
        self._vis = data.vis if data else None
        self._keepdims = keepdims

        # Check dimensions of timestamps vs those of visibility data
        num_dumps = len(timestamps)
        if data and (num_dumps != data.shape[0]):
            raise BrokenFile('Number of timestamps received from ingest '
                             '(%d) differs from number of dumps in data (%d)' %
                             (num_dumps, data.shape[0]))
        # The expected_dumps should always be an integer (like num_dumps),
        # unless the timestamps and/or dump period are messed up in the file,
        # so threshold of this test is a bit arbitrary (e.g. could use > 0.5)
        expected_dumps = (timestamps[-1] - timestamps[0]) / self.dump_period + 1
        if abs(expected_dumps - num_dumps) >= 0.01:
            # Warn the user, as this is anomalous
            logger.warning("Irregular timestamps detected: expected %.3f dumps "
                           "based on dump period and start/end times, "
                           "got %d instead", expected_dumps, num_dumps)
        self._timestamps += self.time_offset
        if self._timestamps[0] < 1e9:
            logger.warning("Data set has invalid first correlator timestamp "
                           "(%f)", self._timestamps[0])
        half_dump = 0.5 * self.dump_period
        self.start_time = katpoint.Timestamp(self._timestamps[0] - half_dump)
        self.end_time = katpoint.Timestamp(self._timestamps[-1] + half_dump)
        self._time_keep = np.ones(num_dumps, dtype=np.bool)

        # Assemble sensor cache
        self.sensor = SensorCache(metadata.sensors, self._timestamps,
                                  self.dump_period, self._time_keep,
                                  SENSOR_PROPS, VIRTUAL_SENSORS, SENSOR_ALIASES)

        # ------ Extract flags ------

        # TODO
        # # Check if flag group is present, else use dummy flag data
        # self._flags = data_group['flags'] if 'flags' in data_group else \
        #     dummy_dataset('dummy_flags', shape=self._vis.shape[:-1], dtype=np.uint8, value=0)
        # # Obtain flag descriptions from file or recreate default flag description table
        # self._flags_description = data_group['flags_description'] if 'flags_description' in data_group else \
        #     np.array(zip(FLAG_NAMES, FLAG_DESCRIPTIONS))
        # self._flags_select = np.array([0], dtype=np.uint8)
        # self._flags_keep = 'all'

        # ------ Extract weights ------

        # TODO
        # # Check if weights and weights_channel datasets are present, else use dummy weight data
        # self._weights = data_group['weights'] if 'weights' in data_group else \
        #     dummy_dataset('dummy_weights', shape=self._vis.shape[:-1], dtype=np.float32, value=1.0)
        # self._weights_channel = data_group['weights_channel'] if 'weights_channel' in data_group else \
        #     dummy_dataset('dummy_weights_channel', shape=self._vis.shape[:-2], dtype=np.float32, value=1.0)
        # # Obtain weight descriptions from file or recreate default weight description table
        # self._weights_description = data_group['weights_description'] if 'weights_description' in data_group else \
        #     np.array(zip(WEIGHT_NAMES, WEIGHT_DESCRIPTIONS))
        # self._weights_select = []
        # self._weights_keep = 'all'

        # ------ Extract observation parameters and script log ------

        self.obs_params = {}
        # Replay obs_params sensor if available and update obs_params dict accordingly
        try:
            obs_params = self.sensor.get('obs_params', extract=False)['value']
        except KeyError:
            obs_params = []
        for obs_param in obs_params:
            if obs_param:
                key, val = obs_param.split(' ', 1)
                self.obs_params[key] = np.lib.utils.safe_eval(val)
        # Get observation script parameters, with defaults
        self.observer = self.obs_params.get('observer', '')
        self.description = self.obs_params.get('description', '')
        self.experiment_id = self.obs_params.get('experiment_id', '')
        # Extract script log data verbatim (it is not a standard sensor anyway)
        try:
            self.obs_script_log = self.sensor.get('obs_script_log',
                                                  extract=False)['value'].tolist()
        except KeyError:
            self.obs_script_log = []

        # ------ Extract subarrays ------

        # List of correlation products as pairs of input labels
        corrprods = metadata.attrs['sdp_l0_bls_ordering']
        # Crash if there is mismatch between labels and data shape (bad labels?)
        if data and (len(corrprods) != data.shape[2]):
            raise BrokenFile('Number of baseline labels (containing expected '
                             'antenna names) received from correlator (%d) '
                             'differs from number of baselines in data (%d)' %
                             (len(corrprods), data.shape[2]))
        # Find all antennas in subarray with valid katpoint Antenna objects
        ants = []
        for resource in metadata.attrs['sub_pool_resources'].split(','):
            try:
                ant_description = self.sensor.get(resource + '_observer')[0]
                ants.append(katpoint.Antenna(ant_description))
            except (KeyError, ValueError):
                continue
        # Keep the basic list sorted as far as possible
        ants = sorted(ants)
        cam_ants = set(ant.name for ant in ants)
        # Find names of all antennas with associated correlator data
        cbf_ants = set([cp[0][:-1] for cp in corrprods] +
                       [cp[1][:-1] for cp in corrprods])
        # By default, only pick antennas that were in use by the script
        obs_ants = self.obs_params.get('ants')
        # Otherwise fall back to the list of antennas common to CAM and CBF
        obs_ants = obs_ants.split(',') if obs_ants else list(cam_ants & cbf_ants)
        self.ref_ant = obs_ants[0] if not ref_ant else ref_ant

        self.subarrays = subs = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(subs, [0, num_dumps])
        self.sensor['Observation/subarray_index'] = CategoricalData([0], [0, num_dumps])
        # Store antenna objects in sensor cache too, for use in virtual sensors
        for ant in ants:
            sensor_name = 'Antennas/%s/antenna' % (ant.name,)
            self.sensor[sensor_name] = CategoricalData([ant], [0, num_dumps])

        # ------ Extract spectral windows / frequencies ------

        # Get the receiver band identity ('l', 's', 'u', 'x') if not overridden
        band = metadata.attrs.get('sub_band', '') if not band else band
        if not band:
            logger.warning('Could not figure out receiver band - '
                           'please provide it via band parameter')
        # Populate antenna -> receiver mapping and figure out noise diode
        for ant in cam_ants:
            rx_sensor = 'Antennas/%s/rsc_rx%s_serial_number' % (ant, band)
            rx_serial = self.sensor[rx_sensor][0] if rx_sensor in self.sensor else 0
            if band:
                self.receivers[ant] = '%s.%d' % (band, rx_serial)
            nd_sensor = 'TelescopeState/%s_dig_%s_band_noise_diode' % (ant, band)
            if nd_sensor in self.sensor:
                # A sensor alias would be ideal for this but it only deals with suffixes ATM
                new_nd_sensor = 'Antennas/%s/nd_coupler' % (ant,)
                self.sensor[new_nd_sensor] = self.sensor.get(nd_sensor, extract=False)
        # Mapping describing current receiver information on MeerKAT
        # XXX Update this as new receivers come online
        rx_table = {
            # Standard L-band receiver
            'l': dict(band='L', centre_freq=1284e6, sideband=1),
            # Standard UHF receiver
            'u': dict(band='UHF', centre_freq=816e6, sideband=1),
            # Custom Ku receiver for RTS
            'x': dict(band='Ku', sideband=1),
        }
        spw_params = rx_table.get(band, dict(band='', sideband=1))
        # XXX Cater for future narrowband mode at some stage
        num_chans = metadata.attrs['cbf_n_chans']
        bandwidth = metadata.attrs['cbf_bandwidth']
        # Cater for non-standard receivers, starting with Ku-band
        if spw_params['band'] == 'Ku':
            if 'anc_siggen_ku_frequency' in self.sensor:
                siggen_freq = self.sensor.get('anc_siggen_ku_frequency')[0]
                spw_params['centre_freq'] = 100. * siggen_freq + 1284e6
        # "Fake UHF": a real receiver + L-band digitiser and flipped spectrum
        elif spw_params['band'] == 'UHF' and bandwidth == 856e6:
            spw_params['centre_freq'] = 428e6
            spw_params['sideband'] = -1
        # Get channel width from original CBF parameters
        spw_params['channel_width'] = bandwidth / num_chans
        # Continue with different channel count, but invalidate centre freq
        # (keep channel width though)
        if data and (num_chans != data.shape[1]):
            logger.warning('Number of channels received from correlator (%d) '
                           'differs from number of channels in data (%d) - '
                           'trusting the latter', num_chans, data.shape[1])
            num_chans = data.shape[1]
            spw_params.pop('centre_freq', None)
        # Override centre frequency if provided
        if centre_freq:
            spw_params['centre_freq'] = centre_freq
        if 'centre_freq' not in spw_params:
            # Choose something really obviously wrong but continue otherwise
            spw_params['centre_freq'] = 0.0
            logger.warning('Could not figure out centre frequency, setting it to '
                           '0 Hz - please provide it via centre_freq parameter')
        spw_params['num_chans'] = num_chans
        spw_params['product'] = metadata.attrs.get('sub_product', '')
        # We only expect a single spectral window within a single v4 data set
        self.spectral_windows = spws = [SpectralWindow(**spw_params)]
        self.sensor['Observation/spw'] = CategoricalData(spws, [0, num_dumps])
        self.sensor['Observation/spw_index'] = CategoricalData([0], [0, num_dumps])

        # ------ Extract scans / compound scans / targets ------

        # Use the activity sensor of reference antenna to partition the data
        # set into scans (and to set their states)
        scan = self.sensor.get(self.ref_ant + '_activity')
        # If the antenna starts slewing on the second dump, incorporate the
        # first dump into the slew too. This scenario typically occurs when the
        # first target is only set after the first dump is received.
        # The workaround avoids putting the first dump in a scan by itself,
        # typically with an irrelevant target.
        if len(scan) > 1 and scan.events[1] == 1 and scan[1] == 'slew':
            scan.events, scan.indices = scan.events[1:], scan.indices[1:]
            scan.events[0] = 0
        # Use labels to partition the data set into compound scans
        try:
            label = self.sensor.get('obs_label')
        except KeyError:
            label = CategoricalData([''], [0, num_dumps])
        # Discard empty labels (typically found in raster scans, where first
        # scan has proper label and rest are empty) However, if all labels are
        # empty, keep them, otherwise whole data set will be one pathological
        # compscan...
        if len(label.unique_values) > 1:
            label.remove('')
        # Create duplicate scan events where labels are set during a scan
        # (i.e. not at start of scan)
        # ASSUMPTION: Number of scans >= number of labels
        # (i.e. each label should introduce a new scan)
        scan.add_unmatched(label.events)
        self.sensor['Observation/scan_state'] = scan
        self.sensor['Observation/scan_index'] = CategoricalData(range(len(scan)),
                                                                scan.events)
        # Move proper label events onto the nearest scan start
        # ASSUMPTION: Number of labels <= number of scans
        # (i.e. only a single label allowed per scan)
        label.align(scan.events)
        # If one or more scans at start of data set have no corresponding label,
        # add a default label for them
        if label.events[0] > 0:
            label.add(0, '')
        self.sensor['Observation/label'] = label
        self.sensor['Observation/compscan_index'] = CategoricalData(range(len(label)),
                                                                    label.events)
        # Use the target sensor of reference antenna to set target for each scan
        target = self.sensor.get(self.ref_ant + '_target')
        # Remove initial blank target (typically because antenna starts stopped)
        if len(target) > 1 and target[0] == 'Nothing, special':
            target.events, target.indices = target.events[1:], target.indices[1:]
            target.events[0] = 0
        # Move target events onto the nearest scan start
        # ASSUMPTION: Number of targets <= number of scans
        # (i.e. only a single target allowed per scan)
        target.align(scan.events)
        self.sensor['Observation/target'] = target
        self.sensor['Observation/target_index'] = CategoricalData(target.indices,
                                                                  target.events)
        # Set up catalogue containing all targets in file, with reference
        # antenna as default antenna
        self.catalogue.add(target.unique_values)
        ref_sensor = 'Antennas/%s/antenna' % (self.ref_ant,)
        self.catalogue.antenna = self.sensor.get(ref_sensor)[0]
        # Ensure that each target flux model spans all frequencies
        # in data set if possible
        self._fix_flux_freq_range()

        # Apply default selection and initialise all members that depend
        # on selection in the process
        self.select(spw=0, subarray=0, ants=obs_ants)

    @staticmethod
    def _open(filename, mode='r'):
        """Open file and do basic version sanity check."""
        f = h5py.File(filename, mode)
        version = f.attrs.get('version', '1.x')
        if not version.startswith('3.'):
            raise WrongVersion("Attempting to load version '%s' file with version 3 loader" % (version,))
        return f, version

    @staticmethod
    def _get_ants(filename):
        """Quick look function to get the list of antennas in a data file.

        This is intended to be called without creating a complete katdal object.

        Parameters
        ----------
        filename : string
            Data file name

        Returns
        -------
        antennas : list of :class:'katpoint.Antenna' objects

        """
        f, version = VisibilityDataV4._open(filename)
        obs_params = {}
        tm_group = f['TelescopeModel']
        ants = []
        for name in tm_group:
            if tm_group[name].attrs.get('class') != 'AntennaPositioner':
                continue
            try:
                ant_description = tm_group[name]['observer']['value'][0]
            except KeyError:
                try:
                    ant_description = tm_group[name].attrs['observer']
                except KeyError:
                    ant_description = tm_group[name].attrs['description']
            ants.append(katpoint.Antenna(ant_description))
        cam_ants = set(ant.name for ant in ants)
        # Original list of correlation products as pairs of input labels
        corrprods = VisibilityDataV4._get_corrprods(f)
        # Find names of all antennas with associated correlator data
        cbf_ants = set([cp[0][:-1] for cp in corrprods] + [cp[1][:-1] for cp in corrprods])
        # By default, only pick antennas that were in use by the script
        tm_params = tm_group['obs/params']
        for obs_param in tm_params['value']:
            if obs_param:
                key, val = obs_param.split(' ', 1)
                obs_params[key] = np.lib.utils.safe_eval(val)
        obs_ants = obs_params.get('ants')
        # Otherwise fall back to the list of antennas common to CAM and CBF
        obs_ants = obs_ants.split(',') if obs_ants else list(cam_ants & cbf_ants)
        return [ant for ant in ants if ant.name in obs_ants]

    @staticmethod
    def _get_targets(filename):
        """Quick look function to get the list of targets in a data file.

        This is intended to be called without creating a complete katdal object.

        Parameters
        ----------
        filename : string
            Data file name

        Returns
        -------
        targets : :class:'katpoint.Catalogue' object
            All targets in file

        """
        f, version = VisibilityDataV4._open(filename)
        target_list = f['TelescopeModel/cbf/target']
        all_target_strings = [target_data[1] for target_data in target_list]
        return katpoint.Catalogue(np.unique(all_target_strings))

    @property
    def _weights_keep(self):
        known_weights = [row[0] for row in getattr(self, '_weights_description', [])]
        return [known_weights[ind] for ind in self._weights_select]

    @_weights_keep.setter
    def _weights_keep(self, names):
        known_weights = [row[0] for row in getattr(self, '_weights_description', [])]
        # Ensure a sequence of weight names
        names = known_weights if names == 'all' else \
            names.split(',') if isinstance(names, basestring) else names
        # Create index list for desired weights
        selection = []
        for name in names:
            try:
                selection.append(known_weights.index(name))
            except ValueError:
                logger.warning("%r is not a legitimate weight type for this file, "
                               "supported ones are %s" % (name, known_weights))
        if known_weights and not selection:
            logger.warning('No valid weights were selected - setting all weights to 1.0 by default')
        self._weights_select = selection

    @property
    def _flags_keep(self):
        if not hasattr(self, '_flags_description'):
            return []
        known_flags = [row[0] for row in self._flags_description]
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        selection = np.flipud(np.unpackbits(self._flags_select))
        assert len(known_flags) == len(selection), \
            'Expected %d flag types in file, got %s' % (len(selection), self._flags_description)
        return [name for name, bit in zip(known_flags, selection) if bit]

    @_flags_keep.setter
    def _flags_keep(self, names):
        if not hasattr(self, '_flags_description'):
            self._flags_select = np.array([0], dtype=np.uint8)
            return
        known_flags = [row[0] for row in self._flags_description]
        # Ensure a sequence of flag names
        names = known_flags if names == 'all' else \
            names.split(',') if isinstance(names, basestring) else names
        # Create boolean list for desired flags
        selection = np.zeros(8, dtype=np.uint8)
        assert len(known_flags) == len(selection), \
            'Expected %d flag types in file, got %d' % (len(selection), self._flags_description)
        for name in names:
            try:
                selection[known_flags.index(name)] = 1
            except ValueError:
                logger.warning("%r is not a legitimate flag type for this file, "
                               "supported ones are %s" % (name, known_flags))
        # Pack index list into bit mask
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        flagmask = np.packbits(np.flipud(selection))
        if known_flags and not flagmask:
            logger.warning('No valid flags were selected - setting all flags to False by default')
        self._flags_select = flagmask

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array of float64, shape (*T*,),
        with one timestamp per integration aligned with the integration
        *midpoint*.

        """
        return self._timestamps[self._time_keep]

    def _vislike_indexer(self, dataset, extractor=None, dims=3):
        """Lazy indexer for vis-like datasets (vis / weights / flags).

        This operates on datasets with shape (*T*, *F*, *B*) and potentially
        different dtypes. The data type conversions are all left to the provided
        optional extractor transform, while this method takes care of the common
        selection issues, such as preserving singleton dimensions and dealing
        with duplicate final dumps. By reducing the *dims* parameter, this
        method also works for datasets with shape (*T*, *F*) or even (*T*,).

        Parameters
        ----------
        dataset : :class:`h5py.Dataset` object or equivalent
            Underlying vis-like dataset on which lazy indexing will be done
        extractor : None or function, signature ``data = f(data, keep)``, optional
            Transform to apply to data (`keep` is user-provided 2nd-stage index)
            (None means no transform is applied)
        dims : integer, optional
            Number of dimensions in dataset (default has all three standard
            dimensions, while smaller values get rid of trailing dimensions)

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
        stage1 = (time_keep, self._freq_keep, self._corrprod_keep)[:dims]

        def _force_full_dim(data, keep):
            """Keep singleton dimensions in stage 2 (i.e. final) indexing."""
            # Ensure that keep tuple has length of dims (truncate or pad with blanket slices as necessary)
            keep = keep[:dims] + (slice(None),) * (dims - len(keep))
            # Final indexing ensures that returned data are always dims-dimensional (i.e. keep singleton dimensions)
            keep_singles = [(np.newaxis if np.isscalar(dim_keep) else slice(None))
                            for dim_keep in keep]
            return data[tuple(keep_singles)]
        force_full_dim = LazyTransform('force_full_dim', _force_full_dim)
        transforms = []
        if extractor:
            transforms.append(extractor)
        if self._keepdims:
            transforms.append(force_full_dim)
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
        if self.spectral_windows[self.spw].sideband == 1:
            # Discard the 4th / last dimension as this is subsumed in complex view
            convert = lambda vis, keep: vis.view(np.complex64)[..., 0]
        else:
            # Lower side-band has the conjugate visibilities, and this isn't
            # corrected in the correlator.
            convert = lambda vis, keep: vis.view(np.complex64)[..., 0].conjugate()
        extract = LazyTransform('extract_vis',
                                convert,
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
        # Build a lazy indexer for high-resolution weights (one per channel)
        weights_channel = self._vislike_indexer(self._weights_channel, dims=2)
        # Don't keep singleton dimensions in order to match lo-res weight shape
        # after second-stage indexing (but before any final keepdims transform)
        weights_channel.transforms = []

        # We currently only cater for a single weight type (i.e. either select it or fall back to 1.0)
        def transform(lo_res_weights, keep):
            hi_res_weights = weights_channel[keep]
            # Add corrprods dimension to hi-res weights to enable broadcasting
            if lo_res_weights.ndim > hi_res_weights.ndim:
                hi_res_weights = hi_res_weights[..., np.newaxis]
            return lo_res_weights * hi_res_weights if self._weights_select else \
                np.ones_like(lo_res_weights, dtype=np.float32)
        extract = LazyTransform('extract_weights', transform, dtype=np.float32)
        indexer = self._vislike_indexer(self._weights, extract)
        if weights_channel.name.find('dummy') < 0:
            indexer.name += ' * ' + weights_channel.name
        return indexer

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
        names = ['anc_weather_temperature']
        return self.sensor.get_with_fallback('temperature', names)

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        names = ['anc_weather_pressure']
        return self.sensor.get_with_fallback('pressure', names)

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        names = ['anc_weather_humidity']
        return self.sensor.get_with_fallback('humidity', names)

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        names = ['anc_weather_wind_speed']
        return self.sensor.get_with_fallback('wind_speed', names)

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        names = ['anc_weather_wind_direction']
        return self.sensor.get_with_fallback('wind_direction', names)
