"""Base class for loading a visibility data file."""

import katpoint

class SimpleVisData(object):
    """Base class for loading a visibility data file.

    This provides a very simple interface to a generic file containing visibility
    data (both single-dish and interferometer data supported). It is assumed that
    the data is partitioned into *scans* inside the file, and that the user
    typically want to access the data one scan at a time.

    Parameters
    ----------
    filename : string
        Name of file
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    channel_range : sequence of 2 ints, optional
        Index of first and last frequency channel to load (defaults to all)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds

    Attributes
    ----------
    filename : string
        Name of data file
    version : string
        Format version string
    observer : string
        Name of person that recorded the data set
    description : string
        Short description of the purpose of the data set
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    ants : list of :class:`katpoint.Antenna` objects
        List of antennas present in file and used in experiment (i.e. subarray)
    ref_ant : string
        Name of reference antenna, used to partition data set into scans
    input_map : dict mapping string to string
        Map from signal path labels ('ant1H') to correlator input labels ('0x')
    corrprod_map : dict mapping string to object
        Map from DBE correlation product strings ('0x1y') to objects that index
        the visibility data array (typically integer indices or pairs of indices)
    channel_bw : float
        Channel bandwidth, in Hz
    channel_freqs : array of float, shape (*F*,)
        Center frequency of each channel, in Hz
    dump_rate : float
        Dump rate, in Hz
    start_time : float
        Timestamp of the first sample in file, in UT seconds since Unix epoch

    """
    def __init__(self, filename, ref_ant='', channel_range=None, time_offset=0.0):
        self.filename = filename
        self.version = ''
        self.observer = ''
        self.description = ''
        self.experiment_id = ''
        self.ants = []
        self.ref_ant = ref_ant
        self.input_map = {}
        self.corrprod_map = {}
        self.channel_bw = 0.0
        self.channel_freqs = []
        self.dump_rate = 0.0
        self.start_time = 0.0

    def __str__(self):
        """Verbose human-friendly string representation of data object."""
        signals_used = [signal for signal in self.input_map if signal[:-1] in [ant.name for ant in self.ants]]
        descr = ['%s (version %s)' % (self.filename, self.version),
                 "description: %s | %s" % (self.experiment_id if self.experiment_id else 'No experiment ID',
                                           self.observer if self.observer else 'No observer'),
                 "'%s'" % (self.description if self.description else 'No description',),
                 'antennas: %s' % (' '.join([(ant.name + ' (*ref*)' if ant.name == self.ref_ant else ant.name)
                                             for ant in self.ants]),),
                 'inputs: %d, corrprods: %d' % (len(signals_used), len(self.corr_products(signals_used))),
                 'channels: %d (%.3f - %.3f MHz), %.3f MHz wide, %.3f Hz dump rate' %
                 (len(self.channel_freqs), self.channel_freqs[0] * 1e-6, self.channel_freqs[-1] * 1e-6,
                  self.channel_bw * 1e-6, self.dump_rate),
                 'first sample at %s' % (katpoint.Timestamp(self.start_time).local(),)]
        ts = []
        for s, cs, state, target in self.scans():
            ts = self.timestamps()
            descr.append("scan %2d (compscan %2d) %s '%s' for %d samples" % (s, cs, state, target.name, len(ts)))
        if len(ts) > 0:
            descr.append('last sample at %s' % (katpoint.Timestamp(ts[-1]).local(),))
        return '\n'.join(descr)

    def corr_input(self, signal):
        """Correlator input corresponding to signal path, with error reporting.

        Parameters
        ----------
        signal : string
            Label of signal path, as antenna name + pol name, e.g. 'ant1H'

        Returns
        -------
        dbe_input : string
            Label of corresponding correlator input in DBE format, e.g. '0x'

        Raises
        ------
        KeyError
            If requested signal path is not connected to correlator

        """
        try:
            return self.input_map[signal]
        except KeyError:
            raise KeyError("Signal path '%s' not connected to correlator (available signals are '%s')" %
                           (signal, "', '".join(self.input_map.keys())))

    def corr_products(self, signals):
        """Correlation products in data set involving the desired signals.

        This finds all the non-redundant correlation products in the file where
        both signal paths in the product are present in the provided list. It
        will return both autocorrelations and cross-correlations.

        Parameters
        ----------
        signals : sequence of strings
            List of signal path labels (e.g. ['ant1H', 'ant1V'])

        Returns
        -------
        corrprods : list of (int, int) pairs
            List of correlation products available in data file, as pairs of
            indices into the list of signal paths (e.g. [(0, 0), (0, 1), (1, 1)])

        """
        # DBE inputs that correspond to signal paths
        inputs = [self.corr_input(signal) for signal in signals]
        # Build all correlation products (and corresponding signal index pairs) from DBE input strings
        # For N antennas this results in N * N products
        dbestr_signalpairs = [(inputA + inputB, indexA, indexB) for indexA, inputA in enumerate(inputs)
                                                                for indexB, inputB in enumerate(inputs)]
        # Filter correlation products to keep the ones actually in the file (typically N * (N + 1) / 2 products)
        corrprods = [(indexA, indexB) for dbestr, indexA, indexB in dbestr_signalpairs if dbestr in self.corrprod_map]
        # If baseline A-B and its reverse B-A are both in the file, only keep the one where A < B
        for cp in corrprods:
            if (cp[1], cp[0]) in corrprods and cp[1] < cp[0]:
                corrprods.remove(cp)
        return corrprods

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
        raise NotImplementedError

    def vis(self, corrprod):
        """Extract complex visibility data for current scan.

        Parameters
        ----------
        corrprod : string or (string, string) pair
            Correlation product to extract from visibility data, either as
            string of concatenated correlator input labels (e.g. '0x1y') or a
            pair of signal path labels (e.g. ('ant1H', 'ant2V'))
 
        Returns
        -------
        vis : array of complex64, shape (*T_k*, *F*)
            Visibility data as an array with time along the first dimension and
            frequency along the second dimension. The number of integrations for
            the current scan *T_k* matches the length of the output of
            :meth:`timestamps`, while the number of frequency channels *F*
            matches the size of `channel_freqs`.

        """
        raise NotImplementedError

    def timestamps(self):
        """Extract timestamps for current scan.

        Returns
        -------
        timestamps : array of float64, shape (*T_k*,)
            Sequence of timestamps, one per integration (in UTC seconds since
            epoch). These timestamps should be in *middle* of each integration.

        """
        raise NotImplementedError
