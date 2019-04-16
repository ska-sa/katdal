################################################################################
# Copyright (c) 2017-2019, National Research Foundation (Square Kilometre Array)
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
from __future__ import print_function, division, absolute_import
from builtins import zip, range

import logging

import numpy as np
import katpoint
import dask.array as da

from .dataset import (DataSet, BrokenFile, Subarray, DEFAULT_SENSOR_PROPS,
                      DEFAULT_VIRTUAL_SENSORS, _robust_target,
                      _selection_to_list)
from .datasources import VisFlagsWeights
from .spectral_window import SpectralWindow
from .sensordata import SensorCache
from .categorical import CategoricalData
from .lazy_indexer import DaskLazyIndexer
from .applycal import (add_applycal_sensors, calc_correction,
                       apply_vis_correction, apply_weights_correction,
                       apply_flags_correction, has_cal_product, CAL_PRODUCTS)
from .flags import NAMES as FLAG_NAMES, DESCRIPTIONS as FLAG_DESCRIPTIONS


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
    'obs_label': {'initial_value': '', 'allow_repeats': True},
})

SENSOR_ALIASES = {
    'nd_coupler': 'dig_noise_diode',
}


def _calc_azel(cache, name, ant):
    """Calculate virtual (az, el) sensors from actual ones in sensor cache."""
    real_sensor = '%s_pos_actual_scan_%s' % \
                  (ant, 'azim' if name.endswith('az') else 'elev')
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data


def _add_sensor_alias(cache, new_name, old_name):
    """Add an optional alias for single sensor in sensor cache."""
    try:
        cache[new_name] = cache.get(old_name, extract=False)
    except KeyError:
        pass


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
        Name of reference antenna, used to partition data set into scans,
        to determine the targets and as antenna for the data set catalogue
        (no relation to the *calibration* reference antenna...). The default
        is to use the observation activity sensor for scan partitioning,
        the CBF target and the array reference position as catalogue antenna.
    time_offset : float, optional
        Offset to add to all correlator timestamps, in seconds
    applycal : string or sequence of strings, optional
        List of names of calibration products to apply to vis/weights/flags,
        as a sequence or string of comma-separated names. An empty string or
        sequence means no calibration will be applied (the default for now),
        while the keyword 'all' means all available products will be applied.
        *NB* In future the default will probably change to 'all'.
        *NB* This is still very much an experimental feature...
    sensor_store : string, optional
        Hostname / endpoint of katstore webserver to access additional sensors
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    """
    def __init__(self, source, ref_ant='', time_offset=0.0, applycal='',
                 sensor_store=None, **kwargs):
        DataSet.__init__(self, source.name, ref_ant, time_offset)
        attrs = source.metadata.attrs

        # ------ Extract timestamps ------

        def _before(date):
            return source.timestamps[0] < katpoint.Timestamp(date).secs

        self.source = source
        self.file = {}
        self.version = '4.0'
        self.dump_period = attrs['int_time']
        # The CBF dump period is not in the lite RDB version
        self.cbf_dump_period = attrs.get(
            'i0_baseline_correlation_products_int_time', None)
        num_dumps = len(source.timestamps)
        source.timestamps += self.time_offset
        if _before('2000-01-01'):
            logger.warning("Data set has invalid first correlator timestamp "
                           "(%f)", source.timestamps[0])
        # Workaround for one-CBF-dump offset (SR-1625), reflecting these updates:
        # - CMC2 aka cbf_dev_N 4k fixed since at least 2019-02-11
        # - CMC2 aka cbf_dev_N 1k fixed since at least 2019-03-03
        # - CMC1 aka cbf_N fixed since 2019-03-15
        cmc2 = 'cbf_dev' in attrs['sub_pool_resources']
        cbf4k = 'c856M4k' in attrs['sub_product']
        if (_before('2019-02-11')
            or _before('2019-03-03') and not (cmc2 and cbf4k)
                or _before('2019-03-15') and not cmc2):
            if self.cbf_dump_period is not None:
                source.timestamps -= self.cbf_dump_period
                # Record workaround in time_offset to make it easy to verify
                self.time_offset -= self.cbf_dump_period
                logger.info('Corrected data timestamps backwards by 1 CBF dump '
                            '(see JIRA ticket SR-1625 for more info)')
            else:
                logger.warning('Could not correct timestamps as CBF int time is unknown:'
                               ' consider using full RDB or explicit time_offset')
        half_dump = 0.5 * self.dump_period
        self.start_time = katpoint.Timestamp(source.timestamps[0] - half_dump)
        self.end_time = katpoint.Timestamp(source.timestamps[-1] + half_dump)
        self._time_keep = np.full(num_dumps, True, dtype=np.bool_)
        all_dumps = [0, num_dumps]

        # Assemble sensor cache
        self.sensor = SensorCache(source.metadata.sensors, source.timestamps,
                                  self.dump_period, self._time_keep,
                                  SENSOR_PROPS, VIRTUAL_SENSORS, SENSOR_ALIASES,
                                  sensor_store)

        # ------ Extract flags ------

        # Internal flag mask overridden whenever _flags_keep is set via select()
        self._flags_select = np.array([255], dtype=np.uint8)
        self._flags_keep = 'all'

        # ------ Extract observation parameters and script log ------

        self.obs_params = attrs['obs_params']
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
        corrprods = attrs['bls_ordering']
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
                ant_description = attrs[resource + '_observer']
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
        obs_ants = obs_ants.split(',') if obs_ants else sorted(cam_ants & sdp_ants)
        self.ref_ant = 'array' if not ref_ant else ref_ant
        valid_ref_ants = cam_ants | {'array'}
        if self.ref_ant not in valid_ref_ants:
            raise KeyError("Unknown ref_ant '%s', should be one of %s"
                           % (self.ref_ant, valid_ref_ants))

        self.subarrays = subs = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(subs, all_dumps)
        self.sensor['Observation/subarray_index'] = CategoricalData([0], all_dumps)
        # Store antenna objects in sensor cache too, for use in virtual
        # sensors, and make aliases for old-style target + activity sensors
        for ant in ants:
            prefix = 'Antennas/%s/' % (ant.name,)
            self.sensor[prefix + 'antenna'] = CategoricalData([ant], all_dumps)
            _add_sensor_alias(self.sensor, prefix + 'activity', ant.name + '_activity')
            _add_sensor_alias(self.sensor, prefix + 'target', ant.name + '_target')
        # Extract array reference from first antenna (first 5 fields of description)
        array_ant_fields = ['array'] + ants[0].description.split(',')[1:5]
        array_ant = katpoint.Antenna(','.join(array_ant_fields))
        # Cobble together "array" antenna sensors from various sources
        self.sensor['Antennas/array/antenna'] = CategoricalData(
            [array_ant], all_dumps)
        _add_sensor_alias(self.sensor, 'Antennas/array/activity', 'obs_activity')
        _add_sensor_alias(self.sensor, 'Antennas/array/target', 'cbf_target')

        # ------ Extract spectral windows / frequencies ------

        # Get the receiver band identity ('l', 's', 'u', 'x')
        band = attrs['sub_band']
        # Populate antenna -> receiver mapping and figure out noise diode
        for ant in cam_ants:
            # Try sanitised version of RX serial number first
            rx_serial = attrs.get('%s_rsc_rx%s_serial_number' % (ant, band), 0)
            self.receivers[ant] = '%s.%d' % (band, rx_serial)
            nd_sensor = '%s_dig_%s_band_noise_diode' % (ant, band)
            if nd_sensor in self.sensor:
                # A sensor alias would be ideal for this but it only deals with suffixes ATM
                new_nd_sensor = 'Antennas/%s/nd_coupler' % (ant,)
                self.sensor[new_nd_sensor] = self.sensor.get(nd_sensor, extract=False)
        num_chans = attrs['n_chans']
        bandwidth = attrs['bandwidth']
        centre_freq = attrs['center_freq']
        channel_width = bandwidth / num_chans
        # Continue with different channel count, but invalidate centre freq
        # (keep channel width though)
        if source.data and (num_chans != source.data.shape[1]):
            logger.warning('Number of channels reported in metadata (%d) differs'
                           ' from actual number of channels in data (%d) - '
                           'trusting the latter', num_chans, source.data.shape[1])
            num_chans = source.data.shape[1]
            centre_freq = 0.0
        product = attrs.get('sub_product', '')
        sideband = 1
        band_map = dict(l='L', s='S', u='UHF', x='X')   # noqa: E741
        spw_params = (centre_freq, channel_width, num_chans, product, sideband,
                      band_map[band])
        # We only expect a single spectral window within a single v4 data set
        self.spectral_windows = spws = [SpectralWindow(*spw_params)]
        self.sensor['Observation/spw'] = CategoricalData(spws, all_dumps)
        self.sensor['Observation/spw_index'] = CategoricalData([0], all_dumps)

        # ------ Extract scans / compound scans / targets ------

        # Use activity sensor of reference antenna to partition the data set into scans
        scan = self.sensor.get('Antennas/%s/activity' % (self.ref_ant,))
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
            label = CategoricalData([''], all_dumps)
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
        self.sensor['Observation/scan_index'] = CategoricalData(list(range(len(scan))),
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
        self.sensor['Observation/compscan_index'] = CategoricalData(list(range(len(label))),
                                                                    label.events)
        # Use target sensor of reference antenna to set the target for each scan
        target = self.sensor.get('Antennas/%s/target' % (self.ref_ant,))
        # Move target events onto the nearest scan start
        # ASSUMPTION: Number of targets <= number of scans
        # (i.e. only a single target allowed per scan)
        target.align(scan.events)
        # Remove repeats introduced by scan alignment (e.g. when sequence of
        # targets [A, B, A] becomes [A, A] if B and second A are in same scan)
        target.remove_repeats()
        # Remove initial target if antennas start in mode STOP
        # (typically left over from previous capture block)
        for segment, scan_state in scan.segments():
            # Keep going until first non-STOP scan or a new target is set
            if scan_state == 'stop' and target[segment.start] is target[0]:
                continue
            # Only remove initial target event if we move to a different target
            if target[segment.start] is not target[0]:
                # Only lose 1 event because target sensor doesn't allow repeats
                target.events = target.events[1:]
                target.indices = target.indices[1:]
                target.events[0] = 0
                # Remove initial target from target.unique_values if not used
                target.align(target.events)
            break
        self.sensor['Observation/target'] = target
        self.sensor['Observation/target_index'] = CategoricalData(target.indices,
                                                                  target.events)
        # Set up catalogue containing all targets in file, with reference antenna as default antenna
        self.catalogue.add(target.unique_values)
        self.catalogue.antenna = self.sensor['Antennas/%s/antenna' % (self.ref_ant,)][0]
        # Ensure that each target flux model spans all frequencies
        # in data set if possible
        self._fix_flux_freq_range()

        # ------ Register applycal virtual sensors and products ------

        freqs = self.spectral_windows[0].channel_freqs
        add_applycal_sensors(self.sensor, attrs, freqs)
        available_products = [product for product in CAL_PRODUCTS
                              if has_cal_product(self.sensor, attrs, product)]
        self._applycal = _selection_to_list(applycal, all=available_products)
        if not self.source.data or not self._applycal:
            self._corrections = None
            self._corrected = self.source.data
        else:
            self._corrections = calc_correction(self.source.data.vis.chunks, self.sensor,
                                                self.subarrays[self.subarray].corr_products,
                                                self._applycal)
            corrected_vis = self._make_corrected(apply_vis_correction, self.source.data.vis)
            corrected_flags = self._make_corrected(apply_flags_correction, self.source.data.flags)
            corrected_weights = self._make_corrected(apply_weights_correction, self.source.data.weights)
            name = self.source.data.name
            # Acknowledge that the applycal step is making the L1 product
            if 'sdp_l0' in name:
                name = name.replace('sdp_l0', 'sdp_l1')
            else:
                name = name + ' (corrected)'
            self._corrected = VisFlagsWeights(corrected_vis, corrected_flags, corrected_weights,
                                              name=name)

        # Apply default selection and initialise all members that depend
        # on selection in the process
        self.select(spw=0, subarray=0, ants=obs_ants)

    def _make_corrected(self, apply_correction, data):
        return da.core.elemwise(apply_correction, data, self._corrections, dtype=data.dtype)

    @property
    def _flags_keep(self):
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        selection = np.flipud(np.unpackbits(self._flags_select))
        assert len(FLAG_NAMES) == len(selection), \
            'Expected %d flag types, got %s' % (len(selection), FLAG_NAMES)
        return [name for name, bit in zip(FLAG_NAMES, selection) if bit]

    @_flags_keep.setter
    def _flags_keep(self, names):
        # Ensure `names` is a sequence of valid flag names (or an empty list)
        names = _selection_to_list(names, all=FLAG_NAMES)
        # Create boolean list for desired flags
        selection = np.zeros(8, dtype=np.uint8)
        assert len(FLAG_NAMES) == len(selection), \
            'Expected %d flag types, got %d' % (len(selection), FLAG_NAMES)
        for name in names:
            try:
                selection[FLAG_NAMES.index(name)] = 1
            except ValueError:
                logger.warning("%r is not a legitimate flag type, "
                               "supported ones are %s", name, FLAG_NAMES)
        # Pack index list into bit mask
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        flagmask = np.packbits(np.flipud(selection))
        if not flagmask:
            logger.warning('No valid flags were selected - setting all flags '
                           'to False by default')
        self._flags_select = flagmask

    def _set_keep(self, time_keep=None, freq_keep=None, corrprod_keep=None,
                  weights_keep=None, flags_keep=None):
        """Set time, frequency and/or correlation product selection masks.

        Set the selection masks for those parameters that are present. Also
        include weights and flags selections as options.

        Parameters
        ----------
        time_keep : array of bool, shape (*T*,), optional
            Boolean selection mask with one entry per timestamp
        freq_keep : array of bool, shape (*F*,), optional
            Boolean selection mask with one entry per frequency channel
        corrprod_keep : array of bool, shape (*B*,), optional
            Boolean selection mask with one entry per correlation product
        weights_keep : 'all' or string or sequence of strings, optional
            Names of selected weight types (or 'all' for the lot)
        flags_keep : 'all' or string or sequence of strings, optional
            Names of selected flag types (or 'all' for the lot)

        """
        DataSet._set_keep(self, time_keep, freq_keep, corrprod_keep, weights_keep, flags_keep)
        update_all = time_keep is not None or freq_keep is not None or corrprod_keep is not None
        update_flags = update_all or flags_keep is not None
        if not self.source.data:
            self._vis = self._weights = self._flags = None
        elif update_flags:
            # Create first-stage index from dataset selectors. Note: use
            # the member variables, not the parameters, because the parameters
            # can be None to indicate no change
            stage1 = (self._time_keep, self._freq_keep, self._corrprod_keep)
            if update_all:
                # Cache dask graphs for the data fields
                self._vis = DaskLazyIndexer(self._corrected.vis, stage1)
                self._weights = DaskLazyIndexer(self._corrected.weights, stage1)
            flag_transforms = []
            if ~self._flags_select != 0:
                # Copy so that the lambda isn't affected by future changes
                select = self._flags_select.copy()
                flag_transforms.append(lambda flags: da.bitwise_and(select, flags))
            flag_transforms.append(lambda flags: flags.view(np.bool_))
            self._flags = DaskLazyIndexer(self._corrected.flags, stage1, flag_transforms)

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array of float64, shape (*T*,),
        with one timestamp per integration aligned with the integration
        *midpoint*.

        """
        return self.source.timestamps[self._time_keep]

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
        if self._vis is None:
            raise ValueError('Visibilities are not available since dataset '
                             'was opened with metadata only')
        return self._vis

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
        if self._weights is None:
            raise ValueError('Weights are not available since dataset '
                             'was opened with metadata only')
        return self._weights

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
        if self._flags is None:
            raise ValueError('Flags are not available since dataset '
                             'was opened with metadata only')
        return self._flags

    @property
    def temperature(self):
        """Air temperature in degrees Celsius."""
        names = ['anc_air_temperature']
        return self.sensor.get_with_fallback('temperature', names)

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        names = ['anc_air_pressure']
        return self.sensor.get_with_fallback('pressure', names)

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        names = ['anc_air_relative_humidity']
        return self.sensor.get_with_fallback('humidity', names)

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        names = ['anc_mean_wind_speed']
        return self.sensor.get_with_fallback('wind_speed', names)

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        names = ['anc_wind_direction']
        return self.sensor.get_with_fallback('wind_direction', names)
