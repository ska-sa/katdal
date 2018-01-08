################################################################################
# Copyright (c) 2017-2018, National Research Foundation (Square Kilometre Array)
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

# -----------------------------------------------------------------------------
# -- CLASS :  VisibilityDataV4
# -----------------------------------------------------------------------------


class VisibilityDataV4(DataSet):
    """Access format version 4 visibility data and metadata.

    For more information on attributes, see the :class:`DataSet` docstring.

    Parameters
    ----------
    source : :class:`DataSource` object
        Correlator data (visibilities, flags and weights) and metadata
    ref_ant : string, optional
        Name of reference antenna, used to partition data set into scans
        (default is first antenna in use)
    time_offset : float, optional
        Offset to add to all correlator timestamps, in seconds
    centre_freq : float, optional
        Override centre frequency if provided, in Hz
    band : string, optional
        Override receiver band if provided (e.g. 'l') - used to find ND models
    keepdims : {False, True}, optional
        Force vis / weights / flags to be 3-dimensional, regardless of selection
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    """
    def __init__(self, source, ref_ant='', time_offset=0.0, centre_freq=None,
                 band=None, keepdims=False, **kwargs):
        DataSet.__init__(self, source.name, ref_ant, time_offset)
        attrs = source.metadata.attrs

        # ------ Extract timestamps ------

        self.source = source
        self.file = {}
        self.version = '4.0'
        stream_name = 'sdp_l0'
        self.dump_period = attrs[stream_name + '_int_time']
        self._timestamps = source.timestamps[:]
        self._keepdims = keepdims

        # Check dimensions of timestamps vs those of visibility data
        num_dumps = len(self._timestamps)
        if source.data and (num_dumps != source.data.shape[0]):
            raise BrokenFile('Number of timestamps received from ingest '
                             '(%d) differs from number of dumps in data (%d)' %
                             (num_dumps, source.data.shape[0]))
        # The expected_dumps should always be an integer (like num_dumps),
        # unless the timestamps and/or dump period are messed up in the file,
        # so threshold of this test is a bit arbitrary (e.g. could use > 0.5).
        # The last dump might only be partially filled by ingest, so ignore it.
        if num_dumps > 1:
            expected_dumps = 2 + (self._timestamps[-2] -
                                  self._timestamps[0]) / self.dump_period
            if abs(expected_dumps - num_dumps) >= 0.01:
                # Warn the user, as this is anomalous
                logger.warning("Irregular timestamps detected: expected %.3f "
                               "dumps based on dump period and start/end times, "
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
        self.sensor = SensorCache(source.metadata.sensors, self._timestamps,
                                  self.dump_period, self._time_keep,
                                  SENSOR_PROPS, VIRTUAL_SENSORS, SENSOR_ALIASES)

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
        corrprods = attrs[stream_name + '_bls_ordering']
        # Crash if there is mismatch between labels and data shape (bad labels?)
        if source.data and (len(corrprods) != source.data.shape[2]):
            raise BrokenFile('Number of baseline labels (containing expected '
                             'antenna names) received from correlator (%d) '
                             'differs from number of baselines in data (%d)' %
                             (len(corrprods), source.data.shape[2]))
        # Find all antennas in subarray with valid katpoint Antenna objects
        ants = []
        for resource in attrs['sub_pool_resources'].split(','):
            try:
                ant_description = self.sensor.get(resource + '_observer')[0]
                ants.append(katpoint.Antenna(ant_description))
            except (KeyError, ValueError):
                continue
        # Keep the basic list sorted as far as possible
        ants = sorted(ants)
        cam_ants = set(ant.name for ant in ants)
        # Find names of all antennas with associated correlator data
        sdp_ants = set([cp[0][:-1] for cp in corrprods] +
                       [cp[1][:-1] for cp in corrprods])
        # By default, only pick antennas that were in use by the script
        obs_ants = self.obs_params.get('ants')
        # Otherwise fall back to the list of antennas common to CAM and SDP / CBF
        obs_ants = obs_ants.split(',') if obs_ants else list(cam_ants & sdp_ants)
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
        band = attrs.get('sub_band', '') if not band else band
        if not band:
            logger.warning('Could not figure out receiver band - '
                           'please provide it via band parameter')
        # Populate antenna -> receiver mapping and figure out noise diode
        for ant in cam_ants:
            # Try sanitised version of RX serial number first
            rx_sensor = 'TelescopeState/%s_rx_serial_number' % (ant,)
            rx_serial = self.sensor[rx_sensor][0] if rx_sensor in self.sensor else 0
            if rx_serial == 0:
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
        # Use CBF spectral parameters by default
        num_chans = attrs['cbf_n_chans']
        bandwidth = attrs['cbf_bandwidth']
        # Cater for non-standard receivers, starting with Ku-band
        if spw_params['band'] == 'Ku':
            if 'anc_siggen_ku_frequency' in self.sensor:
                siggen_freq = self.sensor.get('anc_siggen_ku_frequency')[0]
                spw_params['centre_freq'] = 100. * siggen_freq + 1284e6
        # "Fake UHF": a real receiver + L-band digitiser and flipped spectrum
        elif spw_params['band'] == 'UHF' and bandwidth == 856e6:
            spw_params['centre_freq'] = 428e6
            spw_params['sideband'] = -1
        # If the file has SDP output stream parameters, use those instead
        num_chans = attrs.get(stream_name + '_n_chans', num_chans)
        bandwidth = attrs.get(stream_name + '_bandwidth', bandwidth)
        stream_centre_freq = attrs.get(stream_name + '_center_freq')
        if stream_centre_freq is not None:
            spw_params['centre_freq'] = stream_centre_freq
        # Get channel width from original CBF / SDP parameters
        spw_params['channel_width'] = bandwidth / num_chans
        # Continue with different channel count, but invalidate centre freq
        # (keep channel width though)
        if source.data and (num_chans != source.data.shape[1]):
            logger.warning('Number of channels reported in metadata (%d) differs'
                           ' from actual number of channels in data (%d) - '
                           'trusting the latter', num_chans, source.data.shape[1])
            num_chans = source.data.shape[1]
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
        spw_params['product'] = attrs.get('sub_product', '')
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

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array of float64, shape (*T*,),
        with one timestamp per integration aligned with the integration
        *midpoint*.

        """
        return self._timestamps[self._time_keep]

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
