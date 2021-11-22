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

"""Data accessor class for HDF5 files produced by RTS correlator."""

import logging
import pathlib
import secrets
from collections import Counter

import h5py
import katpoint
import katsdptelstate
import numpy as np

from .categorical import CategoricalData
from .dataset import (DEFAULT_SENSOR_PROPS, DEFAULT_VIRTUAL_SENSORS,
                      BrokenFile, DataSet, Subarray, WrongVersion,
                      _robust_target, _selection_to_list)
from .flags import DESCRIPTIONS as FLAG_DESCRIPTIONS
from .flags import NAMES as FLAG_NAMES
from .lazy_indexer import LazyIndexer, LazyTransform
from .sensordata import (H5TelstateSensorGetter, RecordSensorGetter,
                         SensorCache, telstate_decode, to_str)
from .spectral_window import SpectralWindow

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
    base_name = 'pos_actual_scan_azim' if name.endswith('az') else 'pos_actual_scan_elev'
    real_sensor = f'Antennas/{ant}/{base_name}'
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data


VIRTUAL_SENSORS = dict(DEFAULT_VIRTUAL_SENSORS)
VIRTUAL_SENSORS.update({'Antennas/{ant}/az': _calc_azel, 'Antennas/{ant}/el': _calc_azel})

WEIGHT_NAMES = ('precision',)
WEIGHT_DESCRIPTIONS = ('visibility precision (inverse variance, i.e. 1 / sigma^2)',)

# Number of bits in ADC sample counter, used to timestamp correlator data in original SPEAD stream
ADC_COUNTER_BITS = 48

# -------------------------------------------------------------------------------------------------
# --- Utility functions
# -------------------------------------------------------------------------------------------------


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


class _AttributeFound(Exception):
    """This indicates that an attribute has been found and contains its value."""

# -------------------------------------------------------------------------------------------------
# -- CLASS :  H5DataV3
# -------------------------------------------------------------------------------------------------


class H5DataV3(DataSet):
    """Load HDF5 format version 3 file produced by RTS correlator.

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
    time_scale : float or None, optional
        Resynthesise timestamps using this scale factor
    time_origin : float or None, optional
        Resynthesise timestamps using this sync time / epoch
    rotate_bls : {False, True}, optional
        Rotate baseline label list to work around early RTS correlator bug
    centre_freq : float or None, optional
        Override centre frequency if provided, in Hz
    band : string or None, optional
        Override receiver band if provided (e.g. 'l') - used to find ND models
    keepdims : {False, True}, optional
        Force vis / weights / flags to be 3-dimensional, regardless of selection
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    Attributes
    ----------
    file : :class:`h5py.File` object
        Underlying HDF5 file, exposed via :mod:`h5py` interface
    stream_name : string
        Name of L0 data stream, for finding corresponding telescope state keys

    Notes
    -----
    The timestamps can be resynchronised from the original sample counter
    values by specifying *time_scale* and/or *time_origin*. The basic formula
    is given by::

      timestamp = sample_counter / time_scale + time_origin

    """

    def __init__(self, filename, ref_ant='', time_offset=0.0, mode='r',
                 time_scale=None, time_origin=None, rotate_bls=False,
                 centre_freq=None, band=None, keepdims=False, **kwargs):
        # The closest thing to a capture block ID is the Unix timestamp in the original filename
        cbid = pathlib.Path(filename).stem
        DataSet.__init__(self, cbid, ref_ant, time_offset, url=filename)

        # Load file
        self.file, self.version = H5DataV3._open(filename, mode)
        f = self.file

        # Load main HDF5 groups
        data_group, tm_group = f['Data'], f['TelescopeModel']
        self.stream_name = to_str(data_group.attrs.get('stream_name', 'sdp_l0'))
        self.name = f'{cbid}_{self.stream_name}'
        # Pick first group with appropriate class as CBF
        cbfs = [comp for comp in tm_group
                if to_str(tm_group[comp].attrs.get('class')) == 'CorrelatorBeamformer']
        cbf_group = tm_group[cbfs[0]]
        # Get SDP group, if present
        sdp_group = tm_group.get('sdp')

        # ------ Extract sensors ------

        # Populate sensor cache with all HDF5 datasets below TelescopeModel group that fit the description of a sensor
        cache = {}

        def register_sensor(name, obj):
            """A sensor is defined as a non-empty dataset with expected dtype."""
            if isinstance(obj, h5py.Dataset) and obj.shape != () and \
               obj.dtype.names == ('timestamp', 'value', 'status'):
                comp_name, sensor_name = to_str(name).split('/', 1)
                comp_type = to_str(tm_group[comp_name].attrs.get('class'))
                # Mapping from specific components to generic sensor groups
                # Put antenna sensors in virtual Antenna group, the rest according to component type
                group_lookup = {'AntennaPositioner': 'Antennas/' + comp_name}
                group_name = group_lookup.get(comp_type, comp_type) if comp_type else comp_name
                name = '/'.join((group_name, sensor_name))
                cache[name] = RecordSensorGetter(obj, name)
        tm_group.visititems(register_sensor)
        # Also load sensors from TelescopeState for what it's worth
        if 'TelescopeState' in f.file:
            def register_telstate_sensor(name, obj):
                """A sensor is defined as a non-empty dataset with expected dtype."""
                # Before 2016-05-09 the dtype was ('value', 'timestamp')
                if isinstance(obj, h5py.Dataset) and obj.shape != () and \
                   set(obj.dtype.names) == {'timestamp', 'value'}:
                    name = 'TelescopeState/' + to_str(name)
                    cache[name] = H5TelstateSensorGetter(obj, name)
            f.file['TelescopeState'].visititems(register_telstate_sensor)

        # ------ Extract vis and timestamps ------

        # Get SDP L0 dump period if available, else fall back to CBF dump period
        cbf_dump_period = cbf_group.attrs.get('int_time')
        self.dump_period = self._get_l0_attr('int_time', cbf_group, sdp_group)
        # Determine if timestamps are already aligned with middle of dumps
        try:
            ts_ref = to_str(data_group['timestamps'].attrs['timestamp_reference'])
            assert ts_ref == 'centroid', f"Don't know timestamp reference {ts_ref!r}"
            offset_to_middle_of_dump = 0.0
        except KeyError:
            # Two possible cases:
            #   - RTS: SDP dump = CBF dump, timestamps at start of each dump
            #   - Early AR1 (before Oct 2015): SDP dump = mean of starts of CBF dumps
            # Fortunately, both result in the same offset of 1/2 a CBF dump
            if cbf_dump_period is None:
                raise BrokenFile('Timestamps are not centred and CBF dump period unknown')
            offset_to_middle_of_dump = 0.5 * cbf_dump_period
        # Obtain visibilities and timestamps (load the latter explicitly, but obviously not the former...)
        if 'correlator_data' in data_group:
            self._vis = data_group['correlator_data']
        else:
            raise BrokenFile('File contains no visibility data')
        self._timestamps = data_group['timestamps'][:]
        self._keepdims = keepdims

        # Resynthesise timestamps from sample counter based on a different
        # scale factor or origin. For this we need to get the CBF scale factor
        # and origin, ignoring any SDP parameters.
        old_scale = self._get_cbf_attr('scale_factor_timestamp', cbf_group)
        old_origin = self._get_cbf_attr('sync_time', cbf_group)
        # If no new scale factor or origin is given, just use old ones - timestamps should be identical
        time_scale = old_scale if time_scale is None else time_scale
        time_origin = old_origin if time_origin is None else time_origin
        # Work around wraps in ADC sample counter
        adc_wrap_period = 2 ** ADC_COUNTER_BITS / time_scale
        # Get second opinion of the observation start time from regular sensors
        regular_sensors = ('air_temperature', 'air_relative_humidity', 'air_pressure',
                           'pos_actual_scan_azim', 'pos_actual_scan_elev', 'script_log')
        data_duration = self._timestamps[-1] + self.dump_period - self._timestamps[0]
        sensor_start_time = 0.0
        # Pick first regular sensor with longer data record than data (hopefully straddling it)
        for sensor_name, sensor_data in cache.items():
            if sensor_name.endswith(regular_sensors) and sensor_data:
                sensor_times = sensor_data.get().timestamp
                proposed_sensor_start_time = sensor_times[0]
                sensor_duration = sensor_times[-1] - proposed_sensor_start_time
                if sensor_duration > data_duration:
                    sensor_start_time = proposed_sensor_start_time
                    break
        # If CBF sync time was too long ago, move it forward in steps of wrap period
        while sensor_start_time - time_origin > adc_wrap_period:
            time_origin += adc_wrap_period
        if time_origin != old_origin:
            logger.warning("CBF sync time overridden or moved forward to avoid sample counter wrapping")
            logger.warning("Sync time changed from %s to %s (UTC)" %
                           (katpoint.Timestamp(old_origin), katpoint.Timestamp(time_origin)))
            logger.warning("THE DATA MAY BE CORRUPTED with e.g. delay tracking errors - proceed at own risk!")
        # Resynthesise the timestamps using the final scale and origin
        samples = old_scale * (self._timestamps - old_origin)
        self._timestamps = samples / time_scale + time_origin
        # Now remove any time wraps within the observation
        time_deltas = np.diff(self._timestamps)
        # Assume that a large decrease in timestamp is due to wrapping of ADC sample counter
        time_wraps = np.nonzero(time_deltas < -adc_wrap_period / 2.)[0]
        if time_wraps:
            time_deltas[time_wraps] += adc_wrap_period
            self._timestamps = np.cumsum(np.r_[self._timestamps[0], time_deltas])
            for wrap in time_wraps:
                logger.warning('Time wrap found and corrected at: %s UTC' %
                               (katpoint.Timestamp(self._timestamps[wrap])))
            logger.warning("THE DATA MAY BE CORRUPTED with e.g. delay tracking errors - proceed at own risk!")
        # Warn if there are any remaining decreases in timestamps not associated with wraps
        backward_jumps = np.nonzero(time_deltas < 0.0)[0]
        for jump in backward_jumps:
            logger.warning('CBF timestamps going backward at: %s UTC (%g seconds)' %
                           (katpoint.Timestamp(self._timestamps[jump]), time_deltas[jump]))

        # Check dimensions of timestamps vs those of visibility data
        num_dumps = len(self._timestamps)
        if num_dumps != self._vis.shape[0]:
            raise BrokenFile(f'Number of timestamps received from ingest ({num_dumps}) '
                             f'differs from number of dumps in data ({self._vis.shape[0]})')
        # Discard the last sample if the timestamp is a duplicate (caused by stop packet in k7_capture)
        num_dumps = (num_dumps - 1) if num_dumps > 1 and (self._timestamps[-1] == self._timestamps[-2]) else num_dumps
        self._timestamps = self._timestamps[:num_dumps]
        # The expected_dumps should always be an integer (like num_dumps),
        # unless the timestamps and/or dump period are messed up in the file,
        # so the threshold of this test is a bit arbitrary (e.g. could use > 0.5).
        # The last dump might only be partially filled by ingest, so ignore it.
        if num_dumps > 1:
            expected_dumps = (self._timestamps[-2] - self._timestamps[0]) / self.dump_period + 2
            if abs(expected_dumps - num_dumps) >= 0.01:
                # Warn the user, as this is anomalous
                logger.warning("Irregular timestamps detected in file '%s': expected %.3f dumps "
                               "based on dump period and start/end times, got %d instead",
                               filename, expected_dumps, num_dumps)
        # Ensure timestamps are aligned with the middle of each dump
        self._timestamps += offset_to_middle_of_dump + self.time_offset
        if self._timestamps[0] < 1e9:
            logger.warning("File '%s' has invalid first correlator timestamp (%f)", filename, self._timestamps[0])
        self._time_keep = np.ones(num_dumps, dtype=np.bool)
        self.start_time = katpoint.Timestamp(self._timestamps[0] - 0.5 * self.dump_period)
        self.end_time = katpoint.Timestamp(self._timestamps[-1] + 0.5 * self.dump_period)
        # Populate sensor cache with all HDF5 datasets below TelescopeModel group that fit the description of a sensor
        self.sensor = SensorCache(cache, self._timestamps, self.dump_period, keep=self._time_keep,
                                  props=SENSOR_PROPS, virtual=VIRTUAL_SENSORS, aliases=SENSOR_ALIASES)

        # ------ Extract flags ------

        # Check if flag group is present, else use dummy flag data
        self._flags = data_group['flags'] if 'flags' in data_group else \
            dummy_dataset('dummy_flags', shape=self._vis.shape[:-1], dtype=np.uint8, value=0)
        # Obtain flag descriptions from file or recreate default flag description table
        self._flags_description = data_group['flags_description'] if 'flags_description' in data_group else \
            np.array(list(zip(FLAG_NAMES, FLAG_DESCRIPTIONS)))
        self._flags_select = np.array([0], dtype=np.uint8)
        self._flags_keep = 'all'

        # ------ Extract weights ------

        # Check if weights and weights_channel datasets are present, else use dummy weight data
        self._weights = data_group['weights'] if 'weights' in data_group else \
            dummy_dataset('dummy_weights', shape=self._vis.shape[:-1], dtype=np.float32, value=1.0)
        self._weights_channel = data_group['weights_channel'] if 'weights_channel' in data_group else \
            dummy_dataset('dummy_weights_channel', shape=self._vis.shape[:-2], dtype=np.float32, value=1.0)
        # Obtain weight descriptions from file or recreate default weight description table
        self._weights_description = data_group['weights_description'] if 'weights_description' in data_group else \
            np.array(list(zip(WEIGHT_NAMES, WEIGHT_DESCRIPTIONS)))
        self._weights_select = []
        self._weights_keep = 'all'

        # ------ Extract observation parameters and script log ------

        self.obs_params = {}
        # obs_params is a telstate attribute in v3.9 so try that first
        if 'capture_block_id' in f.attrs:
            attr_name = to_str(f.attrs['capture_block_id']) + '_obs_params'
            self.obs_params = self._get_telstate_attr(attr_name, {})
        else:
            try:
                # Replay obs_params sensor if available
                obs_params = self.sensor.get('Observation/params',
                                             extract=False).get().value
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
            self.obs_script_log = self.sensor.get('Observation/script_log', extract=False).get().value.tolist()
        except KeyError:
            self.obs_script_log = []

        # ------ Extract subarrays ------

        # All antennas in configuration as katpoint Antenna objects
        ants = []
        for name in tm_group:
            if to_str(tm_group[name].attrs.get('class')) != 'AntennaPositioner':
                continue
            try:
                ant_description = self.sensor[f'Antennas/{name}/observer'][0]
            except KeyError:
                try:
                    ant_description = to_str(tm_group[name].attrs['observer'])
                except KeyError:
                    ant_description = to_str(tm_group[name].attrs['description'])
            ants.append(katpoint.Antenna(ant_description))
        # Keep the basic list sorted as far as possible
        ants = sorted(ants)
        cam_ants = {ant.name for ant in ants}
        # Original list of correlation products as pairs of input labels
        corrprods = self._get_corrprods(f, self.stream_name)
        # Find names of all antennas with associated correlator data
        cbf_ants = set([cp[0][:-1] for cp in corrprods] + [cp[1][:-1] for cp in corrprods])
        # By default, only pick antennas that were in use by the script
        obs_ants = self.obs_params.get('ants')
        # Otherwise fall back to the list of antennas common to CAM and CBF
        obs_ants = obs_ants.split(',') if obs_ants else sorted(cam_ants & cbf_ants)
        self.ref_ant = obs_ants[0] if not ref_ant else ref_ant

        if len(corrprods) != self._vis.shape[2]:
            # Apply k7_capture baseline mask after the fact, in the hope that it fixes correlation product mislabelling
            corrprods = np.array([cp for cp in corrprods if cp[0][:-1] in obs_ants and cp[1][:-1] in obs_ants])
            # If there is still a mismatch between labels and data shape, file is considered broken (maybe bad labels?)
            if len(corrprods) != self._vis.shape[2]:
                raise BrokenFile('Number of baseline labels (containing expected antenna names) '
                                 'received from correlator (%d) differs from number of baselines in data (%d)' %
                                 (len(corrprods), self._vis.shape[2]))
            else:
                logger.warning('Reapplied k7_capture baseline mask to fix unexpected number of baseline labels')
        self.subarrays = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(self.subarrays, [0, num_dumps])
        self.sensor['Observation/subarray_index'] = CategoricalData([0], [0, num_dumps])
        # Store antenna objects in sensor cache too, for use in virtual sensor calculations
        for ant in ants:
            self.sensor[f'Antennas/{ant.name}/antenna'] = CategoricalData([ant], [0, num_dumps])
        # Extract array reference from first antenna (first 5 fields of description)
        array_ant_fields = ['array'] + ants[0].description.split(',')[1:5]
        array_ant = katpoint.Antenna(','.join(array_ant_fields))
        self.sensor['Antennas/array/antenna'] = CategoricalData([array_ant], [0, num_dumps])

        # ------ Extract spectral windows / frequencies ------

        # Get the receiver band identity ('l', 's', 'u', 'x') if not overridden
        if not band:
            if 'TelescopeState' in f.file:
                # Newer RTS, AR1 and beyond use the subarray band attribute / sensor
                band = self._get_telstate_attr('sub_band', default='',
                                               no_decode=('l', 's', 'u', 'x'))
            else:
                # Fallback for the original RTS before 2016-07-21 (not reliable)
                # Find the most common valid indexer position in the subarray
                positions = Counter()
                for ant in cam_ants:
                    pos_sensor = f'Antennas/{ant}/ap_indexer_position'
                    try:
                        pos = self.sensor[pos_sensor][-1]
                    except KeyError:
                        pos = ''
                    if pos in ('l', 's', 'u', 'x'):
                        positions[pos] += 1
                try:
                    band = positions.most_common(1)[0][0]
                except IndexError:
                    # An empty counter -> no valid positions were found
                    band = ''
                else:
                    logger.warning('Guessed receiver band from most common '
                                   'indexer position - this is not reliable!')
        if not band:
            logger.warning('Could not figure out receiver band - '
                           'please provide it via band parameter')
        # Populate antenna -> receiver mapping and figure out noise diode
        for ant in cam_ants:
            rx_sensor_options = (
                # Since 2018-01-16 MKAT / ARx only has this version
                f'TelescopeState/{ant}_rsc_rx{band}_serial_number',
                # RTS since 2017-11-15
                f'TelescopeState/{ant}_rx_serial_number',
                # Original TelescopeModel version
                f'Antennas/{ant}/rsc_rx{band}_serial_number')
            rx_serial = 0
            for rx_sensor in rx_sensor_options:
                if rx_sensor in self.sensor:
                    rx_serial = self.sensor[rx_sensor][0]
                    break
            if band:
                self.receivers[ant] = f'{band}.{rx_serial}'
            nd_sensor = f'TelescopeState/{ant}_dig_{band}_band_noise_diode'
            if nd_sensor in self.sensor:
                # A sensor alias would be ideal for this but it only deals with suffixes ATM
                new_nd_sensor = f'Antennas/{ant}/nd_coupler'
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
        num_chans = self._get_l0_attr('n_chans', cbf_group, sdp_group)
        bandwidth = self._get_l0_attr('bandwidth', cbf_group, sdp_group)
        # Work around a bc856M4k CBF bug active from 2016-04-28 to 2016-06-01 that got the bandwidth wrong
        if bandwidth == 857152196.0:
            logger.warning('Worked around CBF bandwidth bug (857.152 MHz -> 856.000 MHz)')
            bandwidth = 856000000.0
        # Cater for non-standard receivers
        if spw_params['band'] == 'Ku':
            siggen_freq = 0.0
            if 'Ancillary/siggen_ku_frequency' in self.sensor:
                siggen_freq = self.sensor['Ancillary/siggen_ku_frequency'][0]
            elif 'TelescopeState' in f.file:
                try:
                    siggen_freq = self.sensor['TelescopeState/anc_siggen_ku_frequency'][0]
                except KeyError:
                    pass
            if siggen_freq:
                spw_params['centre_freq'] = 100. * siggen_freq + 1284e6
        # "Fake UHF": a real receiver + L-band digitiser and flipped spectrum
        elif spw_params['band'] == 'UHF' and bandwidth == 856e6:
            spw_params['centre_freq'] = 428e6
            spw_params['sideband'] = -1
        l0_centre_freq = self._get_l0_attr('center_freq', sdp_group=sdp_group,
                                           required=False)
        if l0_centre_freq is not None:
            spw_params['centre_freq'] = l0_centre_freq
        # Get channel width from original CBF / SDP parameters
        spw_params['channel_width'] = bandwidth / num_chans
        # Continue with different channel count, but invalidate centre freq (keep channel width though)
        if num_chans != self._vis.shape[1]:
            logger.warning('Number of channels reported in metadata (%d) differs '
                           'from actual number of channels in data (%d) - trusting the latter',
                           num_chans, self._vis.shape[1])
            num_chans = self._vis.shape[1]
            spw_params.pop('centre_freq', None)
        # Override centre frequency if provided
        if centre_freq:
            spw_params['centre_freq'] = centre_freq
        if 'centre_freq' not in spw_params:
            # Choose something really obviously wrong but don't prevent opening the file
            spw_params['centre_freq'] = 0.0
            logger.warning('Could not figure out centre frequency, setting it to 0 Hz - '
                           'please provide it via centre_freq parameter')
        spw_params['num_chans'] = num_chans
        # The data product is set by the script or passed to it via schedule block
        spw_params['product'] = self.obs_params.get('product', '')
        # We only expect a single spectral window within a single v3 file,
        # as changing the centre freq is like changing the CBF mode
        self.spectral_windows = [SpectralWindow(**spw_params)]
        self.sensor['Observation/spw'] = CategoricalData(self.spectral_windows, [0, num_dumps])
        self.sensor['Observation/spw_index'] = CategoricalData([0], [0, num_dumps])

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
        try:
            label = self.sensor.get('Observation/label')
        except KeyError:
            label = CategoricalData([''], [0, num_dumps])
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
        # RTS workaround: Remove an initial blank target (typically because the antenna is stopped at the start)
        if len(target) > 1 and target[0] == 'Nothing, special':
            target.events, target.indices = target.events[1:], target.indices[1:]
            target.events[0] = 0
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

        # Apply default selection and initialise all members that depend on selection in the process
        self.select(spw=0, subarray=0, ants=obs_ants)

    def _get_telstate_attr(self, key, default=None, no_decode=()):
        """Retrieve an attribute from the TelescopeState.

        If there is no TelescopeState group, the key is missing, or it cannot
        be decoded, returns `default` instead.

        If the raw value is a member of `no_decode`, returns it directly
        rather than attempting to decode it. This is to support older files
        (created before 2016-11-30) in which the attributes were not encoded.
        """
        try:
            # Note: don't apply to_str to value: if it is a binary encoding it
            # needs to stay binary
            value = self.file['TelescopeState'].attrs[key]
            return telstate_decode(value, no_decode)
        except (KeyError, katsdptelstate.DecodeError):
            # In some cases the value is placed in a sensor instead. Return
            # the most recent value.
            try:
                return self.sensor['TelescopeState/' + key][-1]
            except (KeyError, IndexError):
                return default

    def _raise_if_attr_found(self, *attr_name_parts, **kwargs):
        """Look for attribute and if found, return value by raising exception.

        Parameters
        ----------
        attr_name_parts : sequence of strings
            Join these with '_' to form attribute name (last part is base name)
        h5_group : :class:`h5py.Group`, optional, keyword-only
            HDF5 group in TelescopeModel to use instead of TelescopeState

        Raises
        ------
        _AttributeFound(value)
            If attribute is found, return value via exception (else do nothing)
        """
        attr_name = '_'.join(attr_name_parts)
        key = attr_name_parts[-1]
        h5_group = kwargs.get('h5_group')
        if h5_group is not None:
            try:
                value = to_str(h5_group.attrs[attr_name])
            except KeyError:
                pass
            else:
                logger.debug("Found %s=%s in h5.file[%r].attrs[%r]",
                             key, value, h5_group.name, attr_name)
                raise _AttributeFound(value)
        else:
            NOTFOUND = object()     # Dummy to distinguish "not found" from None
            value = self._get_telstate_attr(attr_name, NOTFOUND)
            if value is not NOTFOUND:
                logger.debug('Found %s=%s in telstate[%r]', key, value, attr_name)
                raise _AttributeFound(value)

    def _get_cbf_attr(self, key, cbf_group=None, required=True, default=None):
        """Retrieve attribute associated with the CBF stream.

        It is searched for in several places, using the first match from the
        following:

        - :samp:`{stream_name}_{key}` in TelescopeState, for each stream
          upstream from the SDP output stream (considering the first in a list
          only each time).
        - :samp:`{instrument}_{key}` in TelescopeState, where `instrument`
          is the CBF instrument of the root stream.
        - :samp:`cbf_{key}` in TelescopeState, if `cbf_group` is given
        - :samp:`{key}` in `cbf_group` in TelescopeModel, if given
        - `default`, unless `required` is true

        Parameters
        ----------
        key : string
            Base name of the attribute
        cbf_group : :class:`h5py.Group`, optional
            HDF5 group for the CBF in TelescopeModel
        required : bool, optional
            If true (default), raise :exc:`BrokenFile` if the key is not found
        default : object, optional
            Value to return if the key is not found and `required` is false
        """
        srcs = self._get_telstate_attr(self.stream_name + '_src_streams')
        stream = None
        counter = 10   # Sanity check to catch loops in _src_streams
        try:
            while srcs:
                stream = srcs[0]
                self._raise_if_attr_found(stream, key)
                self._raise_if_attr_found('cbf', stream, key)
                srcs = self._get_telstate_attr(stream + '_src_streams')
                counter -= 1
                if counter == 0:
                    raise BrokenFile('Too many levels of streams in *_src_streams')
            if stream is not None:
                instrument = self._get_telstate_attr(stream + '_instrument_dev_name')
                if instrument:
                    self._raise_if_attr_found(instrument, key)
                    self._raise_if_attr_found('cbf', instrument, key)
            if cbf_group is not None:
                self._raise_if_attr_found('cbf', key)
                self._raise_if_attr_found(key, h5_group=cbf_group)
        except _AttributeFound as exc:
            return exc.args[0]
        if required:
            raise BrokenFile(f'File does not contain {key!r}')
        return default

    def _get_l0_attr(self, key, cbf_group=None, sdp_group=None, required=True,
                     default=None):
        """Retrieve attribute associated with the L0 data stream.

        It is searched for in several places, using the first match from the
        following:

        - :samp:`{stream_name}_{key}` in TelescopeState
        - :samp:`l0_{key}` in `sdp_group` in TelescopeModel, if given
        - Places searched by :meth:`_get_cbf_attr`

        Parameters
        ----------
        key : string
            Base name of the attribute
        cbf_group : :class:`h5py.Group`, optional
            HDF5 group for the CBF in TelescopeModel
        sdp_group : :class:`h5py.Group`, optional
            HDF5 group for the SDP in TelescopeModel
        required : bool, optional
            If true (default), raise :exc:`BrokenFile` if the key is not found
        default : object, optional
            Value to return if the key is not found and `required` is false
        """
        try:
            self._raise_if_attr_found(self.stream_name, key)
            if sdp_group is not None:
                self._raise_if_attr_found('l0', key, h5_group=sdp_group)
        except _AttributeFound as exc:
            return exc.args[0]
        return self._get_cbf_attr(key, cbf_group, required, default)

    @staticmethod
    def _get_corrprods(f, stream_name, rotate_bls=False):
        """Load the correlation products list from an open file.

        Parameters
        ----------
        f : :class:`h5py.File`
            Open HDF5 file
        stream_name : str
            Name of the L0 stream, for fetching corresponding TelescopeState entries
        rotate_bls : {False, True}, optional
            Rotate baseline label list to work around early RTS correlator bug

        Returns
        -------
        corrprods : list
            list of pairs of input labels
        """
        try:
            # If <stream_name>_bls_ordering is present, it should be used in preference
            # to cbf_bls_ordering.
            corrprods = telstate_decode(f['TelescopeState'].attrs[stream_name + '_bls_ordering'])
        except KeyError:
            # Prior to about Nov 2016, ingest would rewrite cbf_bls_ordering in
            # place.
            tm_group = f['TelescopeModel']
            # Pick first group with appropriate class as CBF
            cbfs = [comp for comp in tm_group
                    if to_str(tm_group[comp].attrs.get('class')) == 'CorrelatorBeamformer']
            cbf_group = tm_group[cbfs[0]]
            corrprods = to_str(cbf_group.attrs['bls_ordering'])
            # Work around early RTS correlator bug by re-ordering labels
            if rotate_bls:
                corrprods = corrprods[list(range(1, len(corrprods))) + [0]]
        return corrprods

    @staticmethod
    def _open(filename, mode='r'):
        """Open file and do basic version sanity check."""
        f = h5py.File(filename, mode)
        version = to_str(f.attrs.get('version', '1.x'))
        if not version.startswith('3.'):
            raise WrongVersion(f"Attempting to load version '{version}' file with version 3 loader")
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
        f, version = H5DataV3._open(filename)
        tm_group = f['TelescopeModel']
        stream_name = to_str(f['Data'].attrs.get('stream_name', 'sdp_l0'))
        ants = []
        for name in tm_group:
            if to_str(tm_group[name].attrs.get('class')) != 'AntennaPositioner':
                continue
            try:
                ant_description = tm_group[name]['observer']['value'][0]
            except KeyError:
                try:
                    ant_description = to_str(tm_group[name].attrs['observer'])
                except KeyError:
                    ant_description = to_str(tm_group[name].attrs['description'])
            ants.append(katpoint.Antenna(ant_description))
        cam_ants = {ant.name for ant in ants}
        # Original list of correlation products as pairs of input labels
        corrprods = H5DataV3._get_corrprods(f, stream_name)
        # Find names of all antennas with associated correlator data
        cbf_ants = set([cp[0][:-1] for cp in corrprods] + [cp[1][:-1] for cp in corrprods])
        # obs_params is a telstate attribute in v3.9 so try that first
        obs_params = {}
        if 'capture_block_id' in f.attrs:
            attr_name = to_str(f.attrs['capture_block_id']) + '_obs_params'
            value = f['TelescopeState'].attrs.get(attr_name)
            if value is not None:
                obs_params = telstate_decode(value)
        # Fall back to old obs_params location
        else:
            tm_params = tm_group['obs/params']
            for obs_param in tm_params['value']:
                if obs_param:
                    obs_param = to_str(obs_param)
                    key, val = obs_param.split(' ', 1)
                    obs_params[key] = np.lib.utils.safe_eval(val)
        # By default, only pick antennas that were in use by the script
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
        f, version = H5DataV3._open(filename)
        target_list = f['TelescopeModel/cbf/target']
        all_target_strings = [to_str(target_data[1]) for target_data in target_list]
        return katpoint.Catalogue(np.unique(all_target_strings))

    def __str__(self):
        """Verbose human-friendly string representation of data set."""
        descr = [super().__str__()]
        # append the process_log, if it exists, for non-concatenated h5 files
        if 'History' in self.file and 'process_log' in self.file['History']:
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
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        selection = np.flipud(np.unpackbits(self._flags_select))
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
            def convert(vis, keep):
                return vis.view(np.complex64)[..., 0]
        else:
            # Lower side-band has the conjugate visibilities, and this isn't
            # corrected in the correlator.
            def convert(vis, keep):
                return vis.view(np.complex64)[..., 0].conjugate()
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
        names = ['Enviro/air_temperature', 'TelescopeState/anc_weather_temperature']
        return self.sensor.get_with_fallback('temperature', names)

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        names = ['Enviro/air_pressure', 'TelescopeState/anc_weather_pressure']
        return self.sensor.get_with_fallback('pressure', names)

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        names = ['Enviro/air_relative_humidity', 'TelescopeState/anc_weather_humidity']
        return self.sensor.get_with_fallback('humidity', names)

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        names = ['Enviro/mean_wind_speed', 'Enviro/wind_speed', 'TelescopeState/anc_weather_wind_speed']
        return self.sensor.get_with_fallback('wind_speed', names)

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        names = ['Enviro/wind_direction', 'TelescopeState/anc_weather_wind_direction']
        return self.sensor.get_with_fallback('wind_direction', names)
