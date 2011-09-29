"""Base class for loading a visibility data file."""

import katpoint

class WrongVersion(Exception):
    """Trying to access file using accessor class with the wrong version."""
    pass

class ScanIteratorStopped(Exception):
    """Scan iterator stopped (or not started yet) and file format does not allow direct data access."""
    pass

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
    file : object
        File object providing access to data file (e.g. h5py file handle)
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
        Timestamp of start of first sample in file, in UT seconds since Unix epoch
    end_time : float
        Timestamp of end of last sample in file, in UT seconds since Unix epoch

    """
    def __init__(self, filename, ref_ant='', channel_range=None, time_offset=0.0):
        self.filename = filename
        self.file = None
        self.version = ''
        self.observer = ''
        self.description = ''
        self.experiment_id = ''
        self.ants = []
        self.ref_ant = ref_ant
        self.inputs = set()
        self.corrprod_map = {}
        self.channel_bw = 0.0
        self.channel_freqs = []
        self.dump_rate = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def __str__(self):
        """Verbose human-friendly string representation of data object."""
        inputs_used = [inp for inp in self.inputs if inp[:-1] in [ant.name for ant in self.ants]]
        descr = ['%s (version %s)' % (self.filename, self.version),
                 "description: %s | %s" % (self.experiment_id if self.experiment_id else 'No experiment ID',
                                           self.observer if self.observer else 'No observer'),
                 "'%s'" % (self.description if self.description else 'No description',),
                 'antennas: %s' % (' '.join([(ant.name + ' (*ref*)' if ant.name == self.ref_ant else ant.name)
                                             for ant in self.ants]),),
                 'inputs: %d, corrprods: %d' % (len(inputs_used), len(self.all_corr_products(inputs_used))),
                 'channels: %d (%.3f - %.3f MHz), %.3f MHz wide' % (len(self.channel_freqs), self.channel_freqs[0] / 1e6,
                                                                    self.channel_freqs[-1] / 1e6, self.channel_bw / 1e6),
                 'first sample starts at ' + katpoint.Timestamp(self.start_time).local()]
        num_samples = 0
        for s, cs, state, target in self.scans():
            ts = self.timestamps()
            descr.append("scan %2d (compscan %2d) %s '%s' for %d samples" % (s, cs, state, target.name, len(ts)))
            num_samples += len(ts)
        descr.append('last sample ends at ' + katpoint.Timestamp(self.end_time).local())
        descr.append('time samples: %d at %.3f Hz dump rate (%.1f min)' %
                     (num_samples, self.dump_rate, num_samples / self.dump_rate / 60.))
        return '\n'.join(descr)

    def validate_inputs(self, inputs):
        """Validate a sequence of correlator input labels.

        This checks that all the provided input labels are present in the file,
        raising a :exc:`KeyError` exception if any are not.

        Parameters
        ----------
        inputs : sequence of strings
            List of correlator input labels (e.g. ['ant1h', 'ant1v'])

        Raises
        ------
        KeyError
            If any label was not found in file

        """
        for inp in inputs:
            if inp.lower() not in self.inputs:
                raise KeyError("Correlator does not have input labelled '%s' (available inputs are '%s')" %
                               (inp.lower(), "', '".join(self.inputs)))

    def corr_product(self, inputA, inputB):
        """Correlation product associated with input A x input B.

        This looks for the correlation product <A, B*> in the file, and returns
        the appropriate visibility index. If the direct product is not found,
        the reverse product <B, A*> is looked up instead, which will be the
        conjugate of the desired correlation product.

        Parameters
        ----------
        inputA, inputB : string
            Labels of correlator inputs to correlate (e.g. 'ant1h', 'ant2v')

        Returns
        -------
        corrprod_index : object
            Index into vis data array (typically integer index)
        conjugate : {False, True}
            True if visibility data should be conjugated

        Raises
        ------
        KeyError
            If requested input label is not available
        ValueError
            If requested correlation product is not available

        """
        # Normalise input labels to be lower-case
        inputA, inputB = inputA.lower(), inputB.lower()
        # Look for direct product (A x B) first
        corrprod = (inputA, inputB)
        self.validate_inputs(corrprod)
        if corrprod in self.corrprod_map:
            return self.corrprod_map[corrprod], False
        # Now look for reversed product (B x A), which will require conjugation of vis data
        corrprod = (inputB, inputA)
        if corrprod in self.corrprod_map:
            return self.corrprod_map[corrprod], True
        raise ValueError("Correlation product ('%s', '%s') or its reverse could not be found" % (inputA, inputB))

    def all_corr_products(self, inputs):
        """Correlation products in data set involving the desired inputs.

        This finds all the non-redundant correlation products in the file for
        which both inputs forming the product are present in the provided list.
        It will return both autocorrelations and cross-correlations.

        Parameters
        ----------
        inputs : sequence of strings
            List of correlator input labels (e.g. ['ant1h', 'ant1v'])

        Returns
        -------
        corrprods : list of (int, int) pairs
            List of correlation products available in data file, as pairs of
            indices into the list of input labels (e.g. [(0, 0), (0, 1), (1, 1)])

        """
        # Normalise input labels to be lower-case
        inputs = [inp.lower() for inp in inputs]
        self.validate_inputs(inputs)
        # Build all correlation products (and corresponding input index pairs) from DBE input strings
        # For N antennas this results in N * N products
        input_pairs = [((inputA, inputB), indexA, indexB) for indexA, inputA in enumerate(inputs)
                                                          for indexB, inputB in enumerate(inputs)]
        # Filter correlation products to keep the ones actually in the file (typically N * (N + 1) / 2 products)
        corrprods = [(indexA, indexB) for pair, indexA, indexB in input_pairs if pair in self.corrprod_map]
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
