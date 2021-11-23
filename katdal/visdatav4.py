################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
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

import dask.array as da
import katpoint
import numpy as np

from .applycal import (CAL_PRODUCT_TYPES, INVALID_GAIN, add_applycal_sensors,
                       apply_flags_correction, apply_vis_correction,
                       apply_weights_correction, calc_correction)
from .categorical import CategoricalData, ComparableArrayWrapper
from .dataset import (DEFAULT_SENSOR_PROPS, DEFAULT_VIRTUAL_SENSORS,
                      BrokenFile, DataSet, Subarray, _robust_target,
                      _selection_to_list)
# FLAG_DESCRIPTIONS isn't used, but it's kept here for compatibility with
# external code that might get it from here
from .flags import DESCRIPTIONS as FLAG_DESCRIPTIONS  # noqa: F401
from .flags import NAMES as FLAG_NAMES  # noqa: F401
from .lazy_indexer import DaskLazyIndexer
from .sensordata import SensorCache, SimpleSensorGetter
from .spectral_window import SpectralWindow
from .vis_flags_weights import VisFlagsWeights

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
    '*_product_G': {'initial_value': INVALID_GAIN},
    '*_product_GPHASE': {'initial_value': INVALID_GAIN},
    '*_product_GAMP_PHASE': {'initial_value': INVALID_GAIN},
    'Calibration/Products/*/G': {'initial_value': INVALID_GAIN},
    'Calibration/Products/*/GPHASE': {'initial_value': INVALID_GAIN},
    'Calibration/Products/*/GAMP_PHASE': {'initial_value': INVALID_GAIN}
})

SENSOR_ALIASES = {
    'nd_coupler': 'dig_noise_diode',
}


def _calc_azel(cache, name, ant):
    """Calculate virtual (az, el) sensors from actual ones in sensor cache."""
    suffix = 'azim' if name.endswith('az') else 'elev'
    real_sensor = f'{ant}_pos_actual_scan_{suffix}'
    cache[name] = sensor_data = katpoint.deg2rad(cache.get(real_sensor))
    return sensor_data


def _calc_delay(cache, name, inp):
    """Extract virtual applied delay/phase sensors from raw CBF sensors."""
    # Obtain the relevant CBF attributes from the cache
    stream = cache.get('Correlator/antenna_channelised_voltage_stream')[0]
    sync_time = cache.get('Correlator/sync_time')[0]
    scale_factor_timestamp = cache.get('Correlator/scale_factor_timestamp')[0]
    # Don't extract the sensor since the real timestamps are hidden in the values
    getter = cache.get(f'{stream}_{inp}_delay', extract=False)
    sensor_data = getter.get()
    # Unwrap and unpack the five components of the CBF sensor
    values = [ComparableArrayWrapper.unwrap(v) for v in sensor_data.value]
    adc_sample_counts, delays, delay_rates, phases, phase_rates = zip(*values)
    # Convert the ADC sample count of each delay update to proper Unix timestamp
    times = sync_time + np.array(adc_sample_counts) / scale_factor_timestamp
    # Ensure that we can at least extrapolate to the final timestamp in the cache
    final_time = max(times[-1], cache.timestamps[-1]) + 1
    # Insert interpolation endpoint just before next update for strict monotonicity
    next_times = np.r_[times[1:] - 1e-6, final_time]
    # Use actual delay/phase rate used by F-engine to interpolate between updates
    next_delays = delays + delay_rates * (next_times - times)
    next_phases = phases + phase_rates * (next_times - times)
    times = np.c_[times, next_times].ravel()
    delays = np.c_[delays, next_delays].ravel()
    phases = np.c_[phases, next_phases].ravel()
    delay_data = SimpleSensorGetter(name, times, delays)
    phase_data = SimpleSensorGetter(name, times, phases)
    cache[name.replace('applied_phase', 'applied_delay')] = delay_data
    cache[name.replace('applied_delay', 'applied_phase')] = phase_data
    return delay_data if name.endswith('delay') else phase_data


def _calc_gain(cache, name, inp):
    """Extract virtual applied F-engine gain sensors from raw CBF sensors."""
    stream = cache.get('Correlator/antenna_channelised_voltage_stream')[0]
    # The real/imag parts are cast to int16 in the F-engine but the CBF sensor
    # seems to report back the CAM request, so round them here to be more accurate
    sensor_data = cache.get(f'{stream}_{inp}_eq',
                            transform=lambda g: np.array(g, dtype=np.complex64).round())
    cache[name] = sensor_data
    return sensor_data


VIRTUAL_SENSORS = dict(DEFAULT_VIRTUAL_SENSORS)
VIRTUAL_SENSORS.update({'Antennas/{ant}/az': _calc_azel,
                        'Antennas/{ant}/el': _calc_azel,
                        'Correlator/Inputs/{inp}/applied_delay': _calc_delay,
                        'Correlator/Inputs/{inp}/applied_phase': _calc_delay,
                        'Correlator/Inputs/{inp}/applied_gain': _calc_gain})

DEFAULT_CAL_PRODUCTS = ('l1.K', 'l1.B', 'l1.G', 'l2.GPHASE')


def _cbf_attrs(attrs):
    """Extract attributes from the various CBF streams and instruments."""
    correlator_stream = attrs['src_streams'][0]
    # XXX: should use telstate.join if attrs is a telstate
    int_time = attrs[correlator_stream + '_int_time']
    n_accs = attrs[correlator_stream + '_n_accs']
    f_engine_stream = attrs[correlator_stream + '_src_streams'][0]
    f_engine_instrument = attrs[f_engine_stream + '_instrument_dev_name']
    scale_factor_timestamp = attrs[f_engine_instrument + '_scale_factor_timestamp']
    return int_time, n_accs, f_engine_stream, scale_factor_timestamp


def _add_sensor_alias(cache, new_name, old_name):
    """Add an optional alias for single sensor in sensor cache."""
    try:
        cache[new_name] = cache.get(old_name, extract=False)
    except KeyError:
        pass


def _relative_view(telstate, name):
    """Create a telstate view by appending `name` to all existing namespaces."""
    prefix = telstate.prefixes[-1]
    view = telstate.view(prefix + name, exclusive=True)
    for prefix in reversed(telstate.prefixes[:-1]):
        view = view.view(prefix + name)
    return view


def _normalise_cal_products(products, cal_streams):
    """Expand user-supplied list of cal products into fully qualified versions."""
    requested_cal_products = _selection_to_list(products, all=cal_streams,
                                                default=DEFAULT_CAL_PRODUCTS)
    skip_missing_products = products in ('all', 'default') or any(
        '.' not in product for product in requested_cal_products)
    normalised_cal_products = []
    for product in requested_cal_products:
        if '.' in product:
            normalised_cal_products.append(product)
        elif product in cal_streams:
            normalised_cal_products.extend(['.'.join((product, product_type))
                                            for product_type in CAL_PRODUCT_TYPES])
        elif product in CAL_PRODUCT_TYPES:
            normalised_cal_products.extend(['.'.join((stream, product))
                                            for stream in cal_streams])
        else:
            streams = ','.join(cal_streams)
            streams = f' (one of {streams})' if streams else ' (none found)'
            product_types = ','.join(CAL_PRODUCT_TYPES)
            raise ValueError(f"Unknown calibration product '{product}': it should be a "
                             f'stream{streams}, product type (one of {product_types}) '
                             'or <stream>.<product_type>')
    return normalised_cal_products, skip_missing_products

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
    gaincal_flux : dict mapping string to float, optional
        Flux density (in Jy) per gaincal target name, used to flux calibrate
        the "G" product, overriding the measured flux produced by cal pipeline
        (if available). A value of None disables flux calibration.
    sensor_store : string, optional
        Hostname / endpoint of katstore webserver to access additional sensors
    preselect : dict, optional
        Subset of the data to select. See :class:`.TelstateDataSource` for
        details. This selection is permanent, and further selections made
        by :meth:`.DataSet.select` are relative to this subset.
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other formats and ignored

    """
    def __init__(self, source, ref_ant='', time_offset=0.0, applycal='',
                 gaincal_flux={}, sensor_store=None,
                 preselect=None, **kwargs):
        DataSet.__init__(self, source.name, ref_ant, time_offset, source.url)
        attrs = source.metadata.attrs

        # ------ Extract timestamps ------

        def _before(date):
            return source.timestamps[0] < katpoint.Timestamp(date).secs

        self.source = source
        self.file = {}
        self.version = '4.0'
        self.dump_period = attrs['int_time']
        # The CBF dump period is not in the lite RDB version
        try:
            (self.cbf_dump_period, cbf_n_accs,
             f_engine_stream, scale_factor_timestamp) = _cbf_attrs(attrs)
        except (KeyError, IndexError):
            self.cbf_dump_period = self.accumulations_per_dump = None
            f_engine_stream = scale_factor_timestamp = None
        else:
            cbf_dumps_per_sdp_dump = round(self.dump_period / self.cbf_dump_period)
            self.accumulations_per_dump = cbf_n_accs * cbf_dumps_per_sdp_dump
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
                                                  extract=False).get().value.tolist()
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
        cam_ants = {ant.name for ant in ants}
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
            raise KeyError(f"Unknown ref_ant '{self.ref_ant}', should be one of {valid_ref_ants}")

        self.subarrays = subs = [Subarray(ants, corrprods)]
        self.sensor['Observation/subarray'] = CategoricalData(subs, all_dumps)
        self.sensor['Observation/subarray_index'] = CategoricalData([0], all_dumps)
        # Store antenna objects in sensor cache too, for use in virtual
        # sensors, and make aliases for old-style target + activity sensors
        for ant in ants:
            prefix = f'Antennas/{ant.name}/'
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

        # ------ Extract correlator settings ------

        if f_engine_stream is not None:
            self.sensor['Correlator/antenna_channelised_voltage_stream'] = CategoricalData(
                [f_engine_stream], all_dumps)
        sync_time = attrs.get('sync_time')
        if sync_time is not None:
            self.sensor['Correlator/sync_time'] = CategoricalData(
                [sync_time], all_dumps)
        if scale_factor_timestamp is not None:
            self.sensor['Correlator/scale_factor_timestamp'] = CategoricalData(
               [scale_factor_timestamp], all_dumps)

        # ------ Extract spectral windows / frequencies ------

        # Get the receiver band identity ('l', 's', 'u', 'x')
        band = attrs['sub_band']
        # Populate antenna -> receiver mapping and figure out noise diode
        for ant in cam_ants:
            # Try sanitised version of RX serial number first
            rx_serial = attrs.get(f'{ant}_rsc_rx{band}_serial_number', 0)
            self.receivers[ant] = f'{band}.{rx_serial}'
            nd_sensor = f'{ant}_dig_{band}_band_noise_diode'
            if nd_sensor in self.sensor:
                # A sensor alias would be ideal for this but it only deals with suffixes ATM
                new_nd_sensor = f'Antennas/{ant}/nd_coupler'
                self.sensor[new_nd_sensor] = self.sensor.get(nd_sensor, extract=False)
        num_chans = attrs['n_chans']
        bandwidth = attrs['bandwidth']
        centre_freq = attrs['center_freq']
        channel_width = bandwidth / num_chans
        product = attrs.get('sub_product', '')
        sideband = 1
        band_map = dict(l='L', s='S', u='UHF', x='X')   # noqa: E741
        spw = SpectralWindow(centre_freq, channel_width, num_chans, product, sideband,
                             band_map[band])
        if preselect is None:
            preselect = {}
        if 'channels' in preselect:
            start, stop, stride = preselect['channels'].indices(num_chans)
            assert stride == 1    # Checked by TelstateDataSource
            spw = spw.subrange(start, stop)
        # Continue with different channel count, but invalidate centre freq
        # (keep channel width though)
        if source.data and (spw.num_chans != source.data.shape[1]):
            logger.warning('Number of channels reported in metadata (%d) differs'
                           ' from actual number of channels in data (%d) - '
                           'trusting the latter', spw.num_chans, source.data.shape[1])
            num_chans = source.data.shape[1]
            centre_freq = 0.0
            spw = SpectralWindow(centre_freq, channel_width, num_chans, product, sideband,
                                 band_map[band])
        # We only expect a single spectral window within a single v4 data set
        self.spectral_windows = spws = [spw]
        self.sensor['Observation/spw'] = CategoricalData(spws, all_dumps)
        self.sensor['Observation/spw_index'] = CategoricalData([0], all_dumps)

        # ------ Extract scans / compound scans / targets ------

        # Use activity sensor of reference antenna to partition the data set into scans
        scan = self.sensor.get(f'Antennas/{self.ref_ant}/activity')
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
        target = self.sensor.get(f'Antennas/{self.ref_ant}/target')
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
        self.catalogue.antenna = self.sensor[f'Antennas/{self.ref_ant}/antenna'][0]
        # Ensure that each target flux model spans all frequencies
        # in data set if possible
        self._fix_flux_freq_range()

        # ------ Register applycal virtual sensors and products ------

        cal_freqs = self._register_standard_cal_streams(gaincal_flux)
        normalised_cal_products, skip_missing_products = _normalise_cal_products(
            applycal, cal_freqs.keys())
        if not self.source.data or not normalised_cal_products:
            self._corrections = None
            self._corrected = self.source.data
        else:
            freqs = self.spectral_windows[0].channel_freqs
            corrprods = self.subarrays[self.subarray].corr_products
            self.applycal_products, self._corrections = calc_correction(
                self.source.data.vis.chunks, self.sensor, corrprods,
                normalised_cal_products, freqs, cal_freqs, skip_missing_products)
            if self._corrections is None:
                self._corrected = self.source.data
            else:
                corrected_vis = self._make_corrected(apply_vis_correction,
                                                     self.source.data.vis)
                corrected_flags = self._make_corrected(apply_flags_correction,
                                                       self.source.data.flags)
                corrected_weights = self._make_corrected(apply_weights_correction,
                                                         self.source.data.weights)
                unscaled_weights = self.source.data.unscaled_weights
                # Acknowledge that the applycal step is making the L1 product
                cal_streams = {cp.split('.')[0] for cp in self.applycal_products}
                if 'sdp_l0' in self.name and 'l1' in cal_streams:
                    self.name = self.name.replace('sdp_l0', 'sdp_l1')
                    if 'l2' in cal_streams:
                        self.name = self.name.replace('sdp_l1', 'sdp_l2')
                self._corrected = VisFlagsWeights(corrected_vis, corrected_flags,
                                                  corrected_weights, unscaled_weights)

        # Apply default selection and initialise all members that depend
        # on selection in the process
        self.select(spw=0, subarray=0, ants=obs_ants)

    def _register_standard_cal_streams(self, gaincal_flux):
        """Find L1 and L2 cal streams and register their virtual sensors."""
        # XXX This assumes that `attrs` is a telstate and not a dict-like
        attrs = self.source.metadata.attrs
        # Find first L1 and L2 underlying cal streams by trawling through archived streams
        l1_stream = ''
        l2_streams = []
        archived_streams = attrs.get('sdp_archived_streams', [])
        for stream in archived_streams:
            stream_attrs = _relative_view(attrs, stream)
            stream_type = stream_attrs.get('stream_type')
            if not l1_stream and stream_type == 'sdp.cal':
                l1_stream = stream
            elif not l2_streams and stream_type == 'sdp.continuum_image':
                targets = stream_attrs.get('targets', {})
                l2_streams = [attrs.join(stream, target + '_selfcal')
                              for target in targets.values()]
        # The default L1 cal stream, useful for older files
        if not l1_stream:
            l1_stream = 'cal'
        # Register virtual sensors for all streams, noting their channelisation
        freqs = self.spectral_windows[0].channel_freqs
        cal_freqs = {}
        l1_attrs = _relative_view(attrs, l1_stream)
        l1_freqs = add_applycal_sensors(self.sensor, l1_attrs, freqs, cal_stream='l1',
                                        cal_substreams=[l1_stream], gaincal_flux=gaincal_flux)
        if l1_freqs is not None:
            cal_freqs['l1'] = l1_freqs
        if l2_streams:
            l2_attrs = _relative_view(attrs, l2_streams[0])
            l2_freqs = add_applycal_sensors(self.sensor, l2_attrs, freqs, cal_stream='l2',
                                            cal_substreams=l2_streams, gaincal_flux=None)
            if l2_freqs is not None:
                cal_freqs['l2'] = l2_freqs
        return cal_freqs

    def _make_corrected(self, apply_correction, data):
        return da.core.elemwise(apply_correction, data, self._corrections, dtype=data.dtype)

    @property
    def _flags_keep(self):
        # Reverse flag indices as np.packbits has bit 0 as the MSB (we want LSB)
        selection = np.flipud(np.unpackbits(self._flags_select))
        assert len(FLAG_NAMES) == len(selection), \
            f'Expected {len(selection)} flag types, got {FLAG_NAMES}'
        return [name for name, bit in zip(FLAG_NAMES, selection) if bit]

    @_flags_keep.setter
    def _flags_keep(self, names):
        # Ensure `names` is a sequence of valid flag names (or an empty list)
        names = _selection_to_list(names, all=FLAG_NAMES)
        # Create boolean list for desired flags
        selection = np.zeros(8, dtype=np.uint8)
        assert len(FLAG_NAMES) == len(selection), \
            f'Expected {len(selection)} flag types, got {FLAG_NAMES}'
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
        super()._set_keep(time_keep, freq_keep, corrprod_keep, weights_keep, flags_keep)
        if not self.source.data:
            self._vis = self._weights = self._flags = self._raw_flags = self._excision = None
            return
        # Create first-stage index from dataset selectors. Note: use
        # the member variables, not the parameters, because the parameters
        # can be None to indicate no change
        stage1 = (self._time_keep, self._freq_keep, self._corrprod_keep)
        # Cache dask graphs for the data fields
        self._vis = DaskLazyIndexer(self._corrected.vis, stage1)
        self._weights = DaskLazyIndexer(self._corrected.weights, stage1)
        self._raw_flags = DaskLazyIndexer(self._corrected.flags, stage1)

        # Create flags indexer based on current flag selection
        flag_transforms = []
        if ~self._flags_select != 0:
            # Copy so that the closure isn't affected by future changes
            select = self._flags_select.copy()
            def bitwise_and(flags): return da.bitwise_and(select, flags)
            flag_transforms.append(bitwise_and)
        # View uint8 as bool (can still be undone by flags.view(np.uint8))
        def view_as_bool(flags): return flags.view(np.bool_)
        flag_transforms.append(view_as_bool)
        self._flags = DaskLazyIndexer(self._raw_flags, transforms=flag_transforms)

        # Create excision indexer based on unscaled weights
        unscaled_weights = self._corrected.unscaled_weights
        if unscaled_weights is None or self.accumulations_per_dump is None:
            self._excision = None
        else:
            # The maximum / expected number of CBF dumps per SDP dump
            cbf_dumps_per_sdp_dump = round(self.dump_period / self.cbf_dump_period)
            accs_per_sdp_dump = np.float32(self.accumulations_per_dump)
            accs_per_cbf_dump = accs_per_sdp_dump / np.float32(cbf_dumps_per_sdp_dump)
            # Each unscaled weight represents the actual number of accumulations per SDP dump.
            # Correct most of the weight compression artefacts by forcing each weight to be
            # an integer multiple of CBF n_accs, and then convert it to an excision fraction.
            def integer_cbf_dumps(w): return da.round(w / accs_per_cbf_dump) * accs_per_cbf_dump
            def excision_fraction(w): return (accs_per_sdp_dump - w) / accs_per_sdp_dump
            excision_transforms = [integer_cbf_dumps, excision_fraction]
            self._excision = DaskLazyIndexer(unscaled_weights, stage1, excision_transforms)

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
    def raw_flags(self):
        """Raw flags as a function of time, frequency and baseline.

        The flags data are returned as an array indexer of uint8, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`. To get the data array
        itself from the indexer `x`, do `x[:]` or perform any other form of
        indexing on it. Only then will data be loaded into memory.

        """
        if self._raw_flags is None:
            raise ValueError('Raw flags are not available since dataset '
                             'was opened with metadata only')
        return self._raw_flags

    @property
    def excision(self):
        """Excision as a function of time, frequency and baseline.

        The fraction of each visibility that has been excised in the SDP ingest
        pipeline is returned as an array indexer of bool, shape (*T*, *F*, *B*)
        with time along the first dimension, frequency along the second dimension
        and correlation product ("baseline") index along the third dimension.
        The number of integrations *T* matches the length of :meth:`timestamps`,
        the number of frequency channels *F* matches the length of :meth:`freqs`
        and the number of correlation products *B* matches the length of
        :meth:`corr_products`. To get the data array itself from the indexer `x`,
        do `x[:]` or perform any other form of indexing on it. Only then will
        data be loaded into memory.

        """
        if self._excision is None:
            raise ValueError('Excision is not available (maybe lite dataset or '
                             'dataset opened with metadata only?)')
        return self._excision

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
