"""Data accessor classes for HDF5 files produced by KAT correlators."""

import re

import numpy as np
import h5py
import katpoint

from .simplevisdata import SimpleVisData, WrongVersion, ScanIteratorStopped

#--------------------------------------------------------------------------------------------------
#--- Utility functions
#--------------------------------------------------------------------------------------------------

def has_data(group, dataset_name):
    """Check whether HDF5 group contains a non-empty dataset with given name.

    Parameters
    ----------
    group : :class:`h5py.Group` object
        HDF5 group to query
    dataset_name : string
        Name of HDF5 dataset to query

    Returns
    -------
    has_data : {True, False}
        True if dataset exists in group and is non-empty

    """
    return dataset_name in group and group[dataset_name].shape != ()

def remove_duplicates(sensor):
    """Remove duplicate timestamp values from sensor data.

    This sorts the 'timestamp' field of the sensor record array and removes any
    duplicate values, updating the corresponding 'value' and 'status' fields as
    well. If more than one timestamp have the same value, the value and status
    of the last of these timestamps are selected. If the values differ for the
    same timestamp, a warning is logged (and the last one is still picked).

    Parameters
    ----------
    sensor : :class:`h5py.Dataset` object, shape (N,)
        Sensor dataset, which acts like a record array with fields 'timestamp',
        'value' and 'status'

    Returns
    -------
    unique_sensor : record array, shape (M,)
        Sensor data with duplicate timestamps removed (M <= N)

    """
    x = np.atleast_1d(sensor['timestamp'])
    y = np.atleast_1d(sensor['value'])
    z = np.atleast_1d(sensor['status'])
    # Sort x via mergesort, as it is usually already sorted and stability is important
    sort_ind = np.argsort(x, kind='mergesort')
    x, y = x[sort_ind], y[sort_ind]
    # Array contains True where an x value is unique or the last of a run of identical x values
    last_of_run = np.asarray(list(np.diff(x) != 0) + [True])
    # Discard the False values, as they represent duplicates - simultaneously keep last of each run of duplicates
    unique_ind = last_of_run.nonzero()[0]
    # Determine the index of the x value chosen to represent each original x value (used to pick y values too)
    replacement = unique_ind[len(unique_ind) - np.cumsum(last_of_run[::-1])[::-1]]
    # All duplicates should have the same y and z values - complain otherwise, but continue
    if not np.all(y[replacement] == y) or not np.all(z[replacement] == z):
        print "WARNING: Sensor '%s' has duplicate timestamps with different values or statuses" % (sensor.name,)
        for ind in (y[replacement] != y).nonzero()[0]:
            print "DEBUG: At %s, sensor '%s' has values of %s and %s - keeping last one" % \
                  (katpoint.Timestamp(x[ind]).local(), sensor.name, y[ind], y[replacement][ind])
        for ind in (z[replacement] != z).nonzero()[0]:
            print "DEBUG: At %s, sensor '%s' has statuses of '%s' and '%s' - keeping last one" % \
                  (katpoint.Timestamp(x[ind]).local(), sensor.name, z[ind], z[replacement][ind])
    return np.rec.fromarrays([x[unique_ind], y[unique_ind], z[unique_ind]], dtype=sensor.dtype)

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
#--- CLASS :  H5DataV1
#--------------------------------------------------------------------------------------------------

class H5DataV1(SimpleVisData):
    """Load HDF5 format version 1 file produced by Fringe Finder correlator.

    Parameters
    ----------
    filename : string
        Name of HDF5 file
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    channel_range : sequence of 2 ints, optional
        Index of first and last frequency channel to load (defaults to all)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds

    """
    def __init__(self, filename, ref_ant='', channel_range=None, time_offset=0.0):
        SimpleVisData.__init__(self, filename, ref_ant, channel_range, time_offset)

        # Load file
        self.file = f = h5py.File(filename, 'r')

        # Only continue if file is correct version and has been properly augmented
        self.version = f.attrs.get('version', '1.x')
        if not self.version.startswith('1.'):
            raise WrongVersion("Attempting to load version '%s' file with version 1 loader" % (self.version,))
        if not 'augment' in f.attrs:
            raise ValueError('HDF5 file not augmented - please run k7_augment.py')

        # Get observation script attributes, with defaults
        self.observer = f.attrs.get('observer', '')
        self.description = f.attrs.get('description', '')
        self.experiment_id = f.attrs.get('experiment_id', '')

        # Find connected antennas and build Antenna objects for them
        ant_groups = f['Antennas'].keys()
        self.ants = [katpoint.Antenna(f['Antennas'][group].attrs['description']) for group in ant_groups]
        self.ref_ant = self.ants[0].name if not ref_ant else ref_ant

        # Map from (old-style) DBE input label (e.g. '0x') to the new antenna-based input label (e.g. 'ant1h')
        # Ensure all input labels are lower-case to avoid mixed-case issues
        input_label = dict([(f['Antennas'][group]['H'].attrs['dbe_input'], ant.name + 'h')
                            for ant, group in zip(self.ants, ant_groups) if 'H' in f['Antennas'][group]])
        input_label.update(dict([(f['Antennas'][group]['V'].attrs['dbe_input'], ant.name + 'v')
                                 for ant, group in zip(self.ants, ant_groups) if 'V' in f['Antennas'][group]]))
        # List input labels in order of system-wide DBE inputs
        self.inputs = [input_label[inp] for inp in sorted(input_label.keys())]
        # Split DBE input product string into its separate inputs
        split_product = re.compile(r'(\d+[xy])(\d+[xy])')
        self.corrprod_map = {}
        # Unpack map from DBE input product string to correlation product index so that it maps
        # pairs of input labels to the correlation product index instead
        for corrind, product in f['Correlator']['input_map']:
            match = split_product.match(product)
            if match is None:
                raise ValueError("Unknown DBE input product '%s' in input map (expected e.g. '0x1y')" % (product,))
            self.corrprod_map[tuple([input_label[inp] for inp in match.groups()])] = corrind

        # Extract frequency information
        band_center = f['Correlator'].attrs['center_frequency_hz']
        num_chans = f['Correlator'].attrs['num_freq_channels']
        self.channel_bw = f['Correlator'].attrs['channel_bandwidth_hz']
        # Assume that lower-sideband downconversion has been used, which flips frequency axis
        # Also subtract half a channel width to get frequencies at center of each channel
        self.channel_freqs = band_center - self.channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
        # Select subset of channels
        self._first_chan, self._last_chan = (channel_range[0], channel_range[1]) \
                                            if channel_range is not None else (0, num_chans - 1)
        self.channel_freqs = self.channel_freqs[self._first_chan:self._last_chan + 1]
        self.dump_rate = f['Correlator'].attrs['dump_rate_hz']
        self._time_offset = time_offset
        # Check whether there is any data at all (and get first timestamp)
        try:
            self._scan_group = f['Scans']['CompoundScan0']['Scan0']
        except (KeyError, h5py.H5Error):
            raise ValueError('HDF5 file contains no vis data (/Scans/CompoundScan0/Scan0 group absent)')
        self.start_time = self.timestamps()[0] - 0.5 / self.dump_rate
        # Find last scan and corresponding last timestamp
        last_compscan = 'CompoundScan' + str(len(f['Scans']) - 1)
        last_scan = 'Scan' + str(len(f['Scans'][last_compscan]) - 1)
        self._scan_group = f['Scans'][last_compscan][last_scan]
        self.end_time = self.timestamps()[-1] + 0.5 / self.dump_rate
        # Reset scan group - this file format only permits data access via scans() interface
        self._scan_group = None

    def scans(self):
        """Generator that iterates through scans in file.

        In addition to the variables yielded by the iterator, the :meth:`vis`
        and :meth:`timestamps` methods are adjusted to apply to the current scan.

        Yields
        ------
        scan_index : int
            Index of current scan (starts at 0)
        compscan_index : int
            Index of compound scan associated with this scan (starts at 0)
        state : {'stop', 'slew', 'scan', 'track'}
            State of reference antenna during scan
        target : :class:`katpoint.Target` object
            Target of observation during scan

        """
        scan_index = 0
        for compscan_index in range(len(self.file['Scans'])):
            compscan_group = self.file['Scans']['CompoundScan' + str(compscan_index)]
            target = katpoint.Target(compscan_group.attrs['target'])
            compscan_label = compscan_group.attrs['label']
            for scan in range(len(compscan_group)):
                self._scan_group = compscan_group['Scan' + str(scan)]
                state = self._scan_group.attrs['label']
                if state == 'scan' and compscan_label == 'track':
                    state = 'track'
                yield scan_index, compscan_index, state, target
                scan_index += 1
        self._scan_group = None

    def vis(self, corrprod, zero_missing_data=False):
        """Extract complex visibility data for current scan.

        Parameters
        ----------
        corrprod : (string, string) pair
            Correlation product to extract from visibility data, as a pair of
            correlator input labels, e.g. ('ant1h', 'ant2v')
        zero_missing_data : {False, True}
            True if an array of zeros of the appropriate shape should be returned
            when the requested correlation product could not be found (as opposed
            to raising an exception)

        Returns
        -------
        vis : array of complex64, shape (*T_k*, *F*)
            Visibility data as an array with time along the first dimension and
            frequency along the second dimension. The number of integrations for
            the current scan *T_k* matches the length of the output of
            :meth:`timestamps`, while the number of frequency channels *F*
            matches the size of `channel_freqs`.

        """
        if self._scan_group is None:
            raise ScanIteratorStopped('HDF5 v1 format only supports scan iterator interface - call scans() method first')
        try:
            corr_id, conj = self.corr_product(corrprod[0], corrprod[1])
            vis = self._scan_group['data'][str(corr_id)][:, self._first_chan:self._last_chan + 1]
            return vis.conj() if conj else vis
        except (KeyError, ValueError):
            if zero_missing_data:
                return np.zeros((len(self._scan_group['timestamps']), len(self.channel_freqs)), dtype=np.complex64)
            else:
                raise

    def timestamps(self):
        """Extract timestamps for current scan.

        Returns
        -------
        timestamps : array of float64, shape (*T_k*,)
            Sequence of timestamps, one per integration (in UTC seconds since
            epoch). These timestamps should be in *middle* of each integration.

        """
        if self._scan_group is None:
            raise ScanIteratorStopped('HDF5 v1 format only supports scan iterator interface - call scans() method first')
        return self._scan_group['timestamps'].value.astype(np.float64) / 1000. + 0.5 / self.dump_rate + self._time_offset

#--------------------------------------------------------------------------------------------------
#--- CLASS :  H5DataV2
#--------------------------------------------------------------------------------------------------

class H5DataV2(SimpleVisData):
    """Load HDF5 format version 2 file produced by KAT-7 correlator.

    Parameters
    ----------
    filename : string
        Name of HDF5 file
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    channel_range : sequence of 2 ints, optional
        Index of first and last frequency channel to load (defaults to all)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds

    """
    def __init__(self, filename, ref_ant='', channel_range=None, time_offset=0.0):
        SimpleVisData.__init__(self, filename, ref_ant, channel_range, time_offset)

        # Load file
        self.file = f = h5py.File(filename, 'r')

        # Only continue if file is correct version and has been properly augmented
        self.version = f.attrs.get('version', '1.x')
        if not self.version.startswith('2.'):
            raise WrongVersion("Attempting to load version '%s' file with version 2 loader" % (self.version,))
        if not 'augment_ts' in f.attrs:
            raise ValueError('HDF5 file not augmented - please run k7_augment.py')

        # Load main HDF5 groups
        data_group, sensors_group, config_group = f['Data'], f['MetaData/Sensors'], f['MetaData/Configuration']
        # Get observation script attributes, with defaults
        script_attrs = config_group['Observation'].attrs
        self.observer = script_attrs.get('script_observer', '')
        self.description = script_attrs.get('script_description', '')
        self.experiment_id = script_attrs.get('script_experiment_id', '')
        # Only pick antennas that were in use by the script
        ant_names = config_group['Observation'].attrs['script_ants'].split(',')
        self.ref_ant = ant_names[0] if not ref_ant else ref_ant
        # Build Antenna objects for them
        self.ants = [katpoint.Antenna(config_group['Antennas'][name].attrs['description']) for name in ant_names]

        # List of correlator input labels, in order of system-wide DBE inputs (assume they are sequential in array)
        # Ensure all input labels are lower-case to avoid mixed-case issues
        self.inputs = [labels[0].lower() for labels in get_single_value(config_group['Correlator'], 'input_labelling')]
        # Map from input label pair to correlation product index (which typically follows Miriad-style numbering)
        self.corrprod_map = dict([((input_pair[0].lower(), input_pair[1].lower()), corr_id) for corr_id, input_pair in
                                  enumerate(get_single_value(config_group['Correlator'], 'bls_ordering'))])

        # Extract frequency information
        band_center = sensors_group['RFE']['center-frequency-hz']['value'][0]
        num_chans = get_single_value(config_group['Correlator'], 'n_chans')
        self.channel_bw = get_single_value(config_group['Correlator'], 'bandwidth') / num_chans
        # Assume that lower-sideband downconversion has been used, which flips frequency axis
        # Also subtract half a channel width to get frequencies at center of each channel
        self.channel_freqs = band_center - self.channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
        # Select subset of channels
        self._first_chan, self._last_chan = (channel_range[0], channel_range[1]) \
                                            if channel_range is not None else (0, num_chans - 1)
        self.channel_freqs = self.channel_freqs[self._first_chan:self._last_chan + 1]
        sample_period = get_single_value(config_group['Correlator'], 'int_time')
        self.dump_rate = 1.0 / sample_period

        # Obtain visibility data and timestamps
        self._vis = data_group['correlator_data']
        # Load timestamps as UT seconds since Unix epoch, and move them from start of each sample to the middle
        self._data_timestamps = data_group['timestamps'].value + 0.5 * sample_period + time_offset
        dump_endtimes = self._data_timestamps + 0.5 * sample_period
        # Discard the last sample if the timestamp is a duplicate (caused by stop packet in k7_capture)
        if len(dump_endtimes) > 1 and (dump_endtimes[-1] == dump_endtimes[-2]):
            dump_endtimes = dump_endtimes[:-1]
        self.start_time = self._data_timestamps[0] - 0.5 * sample_period
        self.end_time = self._data_timestamps[-1] + 0.5 * sample_period

        # Use sensors of reference antenna to dissect data set
        ant_sensors = sensors_group['Antennas'][self.ref_ant]

        # Use the activity sensor of reference antenna to partition the data set into scans (and to label the scans)
        activity_sensor = remove_duplicates(ant_sensors['activity'])
        activity, activity_timestamps = activity_sensor['value'], activity_sensor['timestamp']
        # Simplify the activities to derive the basic state of the antenna (slewing, scanning, tracking, stopped)
        simplify = {'scan': 'scan', 'track': 'track', 'slew': 'slew', 'scan_ready': 'slew', 'scan_complete': 'slew'}
        state = np.array([simplify.get(act, 'stop') for act in activity])
        # Cull spurious short-lived activities (e.g. track immediately preceding scan_ready)
        state_durations = np.diff(np.r_[activity_timestamps, dump_endtimes[-1]])
        non_spurious = state_durations > 0.5 * sample_period
        state, activity_timestamps = state[non_spurious], activity_timestamps[non_spurious]
        # Identify times where the state changes - these become scan boundaries
        state_changes = [n for n in xrange(len(state)) if (n == 0) or (state[n] != state[n - 1])]
        self._scan_states, scan_timestamps = state[state_changes], activity_timestamps[state_changes]
        # Convert scan boundary times to sample indices
        self._scan_starts = dump_endtimes.searchsorted(scan_timestamps)
        self._scan_ends = np.r_[self._scan_starts[1:] - 1, len(dump_endtimes) - 1]

        # Use the target sensor of reference antenna to partition the data set into compound scans
        target_sensor = remove_duplicates(ant_sensors['target'])
        target, target_timestamps = target_sensor['value'], target_sensor['timestamp']
        # Ignore empty and repeating targets (but keep any target following an empty one, as well as first target)
        target_changes = [n for n in xrange(len(target)) if target[n] and ((n == 0) or (target[n] != target[n - 1]))]
        compscan_targets, target_timestamps = target[target_changes], target_timestamps[target_changes]
        compscan_starts = dump_endtimes.searchsorted(target_timestamps)
        # If target changes within a dump, move start of compound scan to the next dump (which fully contains target)
        dump_offset = dump_endtimes[compscan_starts] - target_timestamps
        compscan_starts[dump_offset < sample_period] += 1
        compscan_starts = np.clip(compscan_starts, 0, len(dump_endtimes) - 1)
        self._compscan_targets = [katpoint.Target(tgt) for tgt in compscan_targets]
        # TODO: Split scans at compscan boundaries
        self._scan_compscans = compscan_starts.searchsorted(self._scan_starts, side='right') - 1
        self._scan_compscans[self._scan_compscans < 0] = 0
         # lump first scan in with the first compound scan
        self._first_sample, self._last_sample = 0, len(self._data_timestamps) - 1

    def scans(self):
        """Generator that iterates through scans in file.

        In addition to the variables yielded by the iterator, the :meth:`vis`
        and :meth:`timestamps` methods are adjusted to apply to the current scan.

        Yields
        ------
        scan_index : int
            Index of current scan (starts at 0)
        compscan_index : int
            Index of compound scan associated with this scan (starts at 0)
        state : {'stop', 'slew', 'scan', 'track'}
            State of reference antenna during scan
        target : :class:`katpoint.Target` object
            Target of observation during scan

        """
        for scan_index in range(len(self._scan_states)):
            compscan_index = self._scan_compscans[scan_index]
            state = self._scan_states[scan_index]
            target = self._compscan_targets[compscan_index]
            self._first_sample = self._scan_starts[scan_index]
            self._last_sample = self._scan_ends[scan_index]
            yield scan_index, compscan_index, state, target
        self._first_sample, self._last_sample = 0, len(self._data_timestamps) - 1

    def vis(self, corrprod, zero_missing_data=False):
        """Extract complex visibility data for current scan.

        Parameters
        ----------
        corrprod : (string, string) pair
            Correlation product to extract from visibility data, as a pair of
            correlator input labels, e.g. ('ant1h', 'ant2v')
        zero_missing_data : {False, True}
            True if an array of zeros of the appropriate shape should be returned
            when the requested correlation product could not be found (as opposed
            to raising an exception)
 
        Returns
        -------
        vis : array of complex64, shape (*T_k*, *F*)
            Visibility data as an array with time along the first dimension and
            frequency along the second dimension. The number of integrations for
            the current scan *T_k* matches the length of the output of
            :meth:`timestamps`, while the number of frequency channels *F*
            matches the size of `channel_freqs`.

        """
        try:
            corr_id, conj = self.corr_product(corrprod[0], corrprod[1])
            vis = self._vis[self._first_sample:self._last_sample + 1,
                            self._first_chan:self._last_chan + 1, corr_id].view(np.complex64)[:, :, 0]
            return vis.conj() if conj else vis
        except (KeyError, ValueError):
            if zero_missing_data:
                return np.zeros((self._last_sample + 1 - self._first_sample,
                                 len(self.channel_freqs)), dtype=np.complex64)
            else:
                raise

    def timestamps(self):
        """Extract timestamps for current scan.

        Returns
        -------
        timestamps : array of float64, shape (*T_k*,)
            Sequence of timestamps, one per integration (in UTC seconds since
            epoch). These timestamps should be in *middle* of each integration.

        """
        return self._data_timestamps[self._first_sample:self._last_sample + 1]
