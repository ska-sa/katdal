"""Class for concatenating visibility data files."""

import katpoint

from .simplevisdata import SimpleVisData, ScanIteratorStopped

class CannotConcatenateFiles(Exception):
    """Visibility files cannot be concatenated due to incompatible settings."""
    pass

class ConcatVisData(SimpleVisData):
    """Class for concatenating visibility data files.

    This provides a :class:`SimpleVisData` interface to a list of visibility
    files with commensurate experimental settings and the same file format, in
    effect concatenating them. The files should use the same antennas / signal
    paths and have the same correlator and RF setup in order to be concatenated.

    Parameters
    ----------
    format : subclass of :class:`SimpleVisData`
        Appropriate file loader used to open each file
    filenames : sequence of strings
        List of filenames
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    channel_range : sequence of 2 ints, optional
        Index of first and last frequency channel to load (defaults to all)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds

    Attributes
    ----------
    filenames : list of string
        List of data filenames
    files : list of :class:`SimpleVisData` objects
        :class:`SimpleVisData` objects for individual data files
    version : string
        Format version string
    observer : string
        Name of person that recorded the data set (taken from first file)
    description : string
        Short description of the purpose of the data set (a concatenation of
        individual file descriptions)
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc. (taken from first file)
    ants : list of :class:`katpoint.Antenna` objects
        List of antennas present in file and used in experiment (i.e. subarray)
    ref_ant : string
        Name of reference antenna, used to partition data set into scans
    inputs : list of strings
        List of available correlator input labels ('ant1h'), in DBE input order
    corrprod_map : dict mapping tuple of 2 strings to object
        Map from a pair of correlator input labels, e.g. ('ant1h', 'ant2v'), to
        objects that index the visibility data array (typically an integer index)
    channel_bw : float
        Channel bandwidth, in Hz
    channel_freqs : array of float, shape (*F*,)
        Center frequency of each channel, in Hz
    dump_rate : float
        Dump rate, in Hz
    start_time : float
        Timestamp of start of earliest sample, in UT seconds since Unix epoch
    end_time : float
        Timestamp of end of latest sample, in UT seconds since Unix epoch

    """
    def __init__(self, format, filenames, ref_ant='', channel_range=None, time_offset=0.0):
        # Open all data sets
        self.filenames = filenames
        self.files = [format(filename, ref_ant, channel_range, time_offset) for filename in filenames]
        self.version = self.files[0].version
        self.observer = self.files[0].observer
        self.description = '\n'.join([("%s: '%s'" % (f.filename, f.description)) for f in self.files])
        self.experiment_id = self.files[0].experiment_id
        # Extract antenna / input / correlator setup and ensure it is compatible between data sets
        self.ants = self.files[0].ants
        if not all([(f.ants == self.ants) for f in self.files]):
            raise CannotConcatenateFiles('Different antennas in use:\n%s' %
                                         '\n'.join(['%s: %s' % (f.filename, ' '.join([ant.name for ant in f.ants]))
                                                    for f in self.files]))
        self.ref_ant = self.files[0].ref_ant
        if not all([(f.ref_ant == self.ref_ant) for f in self.files]):
            raise CannotConcatenateFiles('Different reference antennas in use:\n%s' %
                                         '\n'.join(['%s: %s' % (f.filename, f.ref_ant) for f in self.files]))
        self.inputs = self.files[0].inputs
        if not all([(f.inputs == self.inputs) for f in self.files]):
            raise CannotConcatenateFiles('Different inputs / signal paths in use:\n%s' %
                                         '\n'.join(['%s: %s' % (f.filename, f.inputs) for f in self.files]))
        self.corrprod_map = self.files[0].corrprod_map
        if not all([(f.corrprod_map == self.corrprod_map) for f in self.files]):
            raise CannotConcatenateFiles('Different correlator input map in use:\n%s' %
                                         '\n'.join(['%s: %s' % (f.filename, f.corrprod_map) for f in self.files]))
        self.channel_bw = self.files[0].channel_bw
        if not all([(f.channel_bw == self.channel_bw) for f in self.files]):
            raise CannotConcatenateFiles('Different channel bandwidth in use:\n%s' %
                                         '\n'.join(['%s: %s Hz' % (f.filename, f.channel_bw) for f in self.files]))
        self.channel_freqs = self.files[0].channel_freqs
        if not all([all(f.channel_freqs == self.channel_freqs) for f in self.files]):
            raise CannotConcatenateFiles('Different channel frequencies in use:\n%s' %
                                         '\n'.join(['%s: %s Hz' % (f.filename, f.channel_freqs) for f in self.files]))
        self.dump_rate = self.files[0].dump_rate
        if not all([(f.dump_rate == self.dump_rate) for f in self.files]):
            raise CannotConcatenateFiles('Different dump rate in use:\n%s' %
                                         '\n'.join(['%s: %s Hz' % (f.filename, f.dump_rate) for f in self.files]))
        self.start_time = min([f.start_time for f in self.files])
        self.end_time = max([f.end_time for f in self.files])
        self._current_file = None

    def __str__(self):
        """Verbose human-friendly string representation of data object."""
        inputs_used = [inp for inp in self.inputs if inp[:-1] in [ant.name for ant in self.ants]]
        descr = ['%s (version %s)' % (' '.join(self.filenames), self.version),
                 "description: %s | %s" % (self.experiment_id if self.experiment_id else 'No experiment ID',
                                           self.observer if self.observer else 'No observer'),
                 "%s" % (self.description if self.description else 'No description',),
                 'antennas: %s' % (' '.join([(ant.name + ' (*ref*)' if ant.name == self.ref_ant else ant.name)
                                             for ant in self.ants]),),
                 'inputs: %d, corrprods: %d' % (len(inputs_used), len(self.all_corr_products(inputs_used))),
                 'channels: %d (%.3f - %.3f MHz), %.3f MHz wide' % (len(self.channel_freqs), self.channel_freqs[0] / 1e6,
                                                                    self.channel_freqs[-1] / 1e6, self.channel_bw / 1e6),
                 'first sample starts at ' + katpoint.Timestamp(self.start_time).local()]
        num_samples, scan_ind, compscan_ind = 0, 0, 0
        for f in self.files:
            descr.append("%s starts at %s" % (f.filename, katpoint.Timestamp(f.start_time).local()))
            for s, cs, state, target in f.scans():
                ts = f.timestamps()
                descr.append("scan %2d (compscan %2d) %s '%s' for %d samples" %
                             (s + scan_ind, cs + compscan_ind, state, target.name, len(ts)))
                num_samples += len(ts)
            scan_ind += s + 1
            compscan_ind += cs + 1
            descr.append("%s ends at %s" % (f.filename, katpoint.Timestamp(f.end_time).local()))
        descr.append('last sample ends at ' + katpoint.Timestamp(self.end_time).local())
        descr.append('time samples: %d at %.3f Hz dump rate (%.1f min)' %
                     (num_samples, self.dump_rate, num_samples / self.dump_rate / 60.))
        return '\n'.join(descr)

    def scans(self):
        """Generator that iterates through scans in multiple files.

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
        scan_ind = compscan_ind = 0
        for f in self.files:
            self._current_file = f
            for s, cs, state, target in f.scans():
                yield scan_ind + s, compscan_ind + cs, state, target
            scan_ind += s + 1
            compscan_ind += cs + 1
        self._current_file = None

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
        if self._current_file is None:
            raise ScanIteratorStopped('Concatenated format only supports scan iterator interface - call scans() method first')
        return self._current_file.vis(corrprod, zero_missing_data)

    def timestamps(self):
        """Extract timestamps for current scan.

        Returns
        -------
        timestamps : array of float64, shape (*T_k*,)
            Sequence of timestamps, one per integration (in UTC seconds since
            epoch). These timestamps should be in *middle* of each integration.

        """
        if self._current_file is None:
            raise ScanIteratorStopped('Concatenated format only supports scan iterator interface - call scans() method first')
        return self._current_file.timestamps()
