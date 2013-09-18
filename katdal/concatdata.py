"""Class for concatenating visibility data sets."""

import os.path

import numpy as np

from .lazy_indexer import LazyIndexer
from .sensordata import SensorData, SensorCache, dummy_sensor_data
from .categorical import CategoricalData, unique_in_order, concatenate_categorical
from .dataset import DataSet


class ConcatenationError(Exception):
    """Sequence of objects could not be concatenated due to incompatibility."""
    pass

#--------------------------------------------------------------------------------------------------
#--- CLASS :  ConcatenatedLazyIndexer
#--------------------------------------------------------------------------------------------------

class ConcatenatedLazyIndexer(LazyIndexer):
    """Two-stage deferred indexer that concatenates multiple indexers.

    This indexer concatenates a sequence of indexers along the first (i.e. time)
    axis. The index specification is broken down into chunks along this axis,
    sent to the applicable underlying indexers and the returned data are
    concatenated again before returning it.

    Parameters
    ----------
    indexers : sequence of :class:`LazyIndexer` objects
        Sequence of indexers to be concatenated
    transforms : list of :class:`LazyTransform` objects or None, optional
        Extra chain of transforms to be applied to data after final indexing

    Attributes
    ----------
    name : string
        Name of first non-empty indexer (or empty string otherwise)

    Raises
    ------
    InvalidTransform
        If transform chain does not obey restrictions on changing the data shape

    """
    def __init__(self, indexers, transforms=None):
        # Only keep those indexers that have any data selected on first axis (unless nothing at all is selected)
        self.indexers = [indexer for indexer in indexers if indexer.shape[0]]
        if not self.indexers:
            self.indexers = indexers[:1]
        self.transforms = [] if transforms is None else transforms
        # Pick the first non-empty indexer name as overall name, or failing that, an empty string
        names = unique_in_order([indexer.name for indexer in self.indexers if indexer.name])
        self.name = (names[0] + ' etc.') if len(names) > 1 else names[0] if len(names) == 1 else ''
        # Test validity of shape and dtype
        self.shape, self.dtype

    def __str__(self):
        """Verbose human-friendly string representation of lazy indexer object."""
        shape, dtype = self._initial_shape, self._initial_dtype
        descr = [self._name_shape_dtype(self.name, shape, dtype)]
        for n, indexer in enumerate(self.indexers):
            indexer_descr = str(indexer).split('\n')
            descr += [('- Indexer %03d: ' % (n,)) + indexer_descr[0]]
            descr += ['               ' + indescr for indescr in indexer_descr[1:]]
        for transform in self.transforms:
            shape, dtype = transform.new_shape(shape), transform.dtype if transform.dtype is not None else dtype
            descr += ['-> ' + self._name_shape_dtype(transform.name, shape, dtype)]
        return '\n'.join(descr)

    def __getitem__(self, keep):
        """Extract a concatenated array from the underlying indexers.

        This applies the given second-stage (global) index on top of the existing
        first-stage (local) indices of each indexer, and concatenates the arrays
        extracted from each indexer along the first dimension.

        Parameters
        ----------
        keep : tuple of int or slice or sequence of int or sequence of bool
            Second-stage (global) index as a valid index or slice specification
            (supports arbitrary slicing or advanced indexing on any dimension)

        Returns
        -------
        data : array
            Concatenated output array

        """
        ndim = len(self._initial_shape)
        # Ensure that keep is a tuple (then turn it into a list to simplify further processing)
        keep = list(keep) if isinstance(keep, tuple) else [keep]
        # The original keep tuple will be passed to data transform chain
        original_keep = tuple(keep)
        # Ensure that keep is same length as first-stage data shape (truncate or pad with blanket slices as necessary)
        keep = keep[:ndim] + [slice(None)] * (ndim - len(keep))
        keep_head, keep_tail = keep[0], keep[1:]
        # Figure out the final shape on the fixed tail dimensions
        shape_tails = [len(np.atleast_1d(np.arange(dim_len)[dim_keep]))
                       for dim_keep, dim_len in zip(keep[1:], self._initial_shape[1:])]
        indexer_starts = np.cumsum([0] + [len(indexer) for indexer in self.indexers[:-1]])
        find_indexer = lambda index: indexer_starts.searchsorted(index, side='right') - 1
        # Interpret selection on first dimension, along which data will be concatenated
        if np.isscalar(keep_head):
            # If selection is a scalar, pass directly to appropriate indexer (after removing offset)
            keep_head = len(self) + keep_head if keep_head < 0 else keep_head
            ind = find_indexer(keep_head)
            out_data = self.indexers[ind][tuple([keep_head - indexer_starts[ind]] + keep_tail)]
        elif isinstance(keep_head, slice):
            # If selection is a slice, split it into smaller slices that span individual indexers
            # Start by normalising slice to full first-stage range
            start, stop, stride = keep_head.indices(len(self))
            chunks = []
            # Step through indexers that overlap with slice (it's guaranteed that some will overlap)
            for ind in range(find_indexer(start), find_indexer(stop) + 1):
                chunk_start = start - indexer_starts[ind] \
                              if start >= indexer_starts[ind] else ((start - indexer_starts[ind]) % stride)
                chunk_stop = stop - indexer_starts[ind]
                # The final .reshape is needed to upgrade any scalar or singleton chunks to full dimension
                chunks.append(self.indexers[ind][tuple([slice(chunk_start, chunk_stop, stride)] +
                                                       keep_tail)].reshape(tuple([-1] + shape_tails)))
            out_data = np.concatenate(chunks)
        else:
            # Anything else is advanced indexing via bool or integer sequences
            keep_head = np.atleast_1d(keep_head)
            # A boolean mask is simpler to handle (no repeated or out-of-order indexing) - partition mask over indexers
            if keep_head.dtype == np.bool and len(keep_head) == len(self):
                chunks = []
                for ind in range(len(self.indexers)):
                    chunk_start = indexer_starts[ind]
                    chunk_stop = indexer_starts[ind + 1] if ind < len(indexer_starts) - 1 else len(self)
                    chunks.append(self.indexers[ind][tuple([keep_head[chunk_start:chunk_stop]] +
                                                           keep_tail)].reshape(tuple([-1] + shape_tails)))
                out_data = np.concatenate(chunks)
            else:
                # Form sequence of relevant indexer indices and local data indices with indexer offsets removed
                indexers = find_indexer(keep_head)
                local_indices = keep_head - indexer_starts[indexers]
                # Determine output data shape after second-stage selection
                final_shape = [len(np.atleast_1d(np.arange(dim_len)[dim_keep]))
                               for dim_keep, dim_len in zip(keep, self._initial_shape)]
                out_data = np.empty(final_shape, dtype=self.dtype)
                for ind in range(len(self.indexers)):
                    chunk_mask = (indexers == ind)
                    # Insert all selected data originating from same indexer into final array
                    if chunk_mask.any():
                        out_data[chunk_mask] = self.indexers[ind][tuple([local_indices[chunk_mask]] + keep_tail)]
        # Apply transform chain to output data, if any
        return reduce(lambda data, transform: transform(data, original_keep), self.transforms, out_data)

    @property
    def _initial_shape(self):
        """Shape of data array after first-stage indexing and before transformation."""
        # Each component must have the same shape except for the first dimension (length)
        # The overall length will be the sum of component lengths
        shape_tails = set([indexer.shape[1:] for indexer in self.indexers])
        if len(shape_tails) != 1:
            raise ConcatenationError("Incompatible shapes among sub-indexers making up indexer '%s':\n%s" %
                                     (self.name, '\n'.join([repr(indexer) for indexer in self.indexers])))
        return tuple([np.sum([len(indexer) for indexer in self.indexers])] + list(shape_tails.pop()))

    @property
    def _initial_dtype(self):
        """Type of data array before transformation."""
        # Each component must have the same dtype, which becomes the overall dtype
        dtypes = set([indexer.dtype for indexer in self.indexers])
        if len(dtypes) == 1:
            return dtypes.pop()
        elif np.all([np.issubdtype(dtype, np.str) for dtype in dtypes]):
            # Strings of different lengths have different dtypes (e.g. '|S1' vs '|S10') but can be safely concatenated
            return np.dtype('|S%d' % (max([dt.itemsize for dt in dtypes]),))
        else:
            raise ConcatenationError("Incompatible dtypes among sub-indexers making up indexer '%s':\n%s" %
                                     (self.name, '\n'.join([repr(indexer) for indexer in self.indexers])))

#--------------------------------------------------------------------------------------------------
#--- CLASS :  ConcatenatedSensorData
#--------------------------------------------------------------------------------------------------

class ConcatenatedSensorData(SensorData):
    """The concatenation of multiple raw (uncached) sensor data sets.

    This is a convenient container for returning raw (uncached) sensor data sets
    from a :class:`ConcatenatedSensorCache` object. It only accesses the
    underlying data sets when explicitly asked to via the __getitem__ interface,
    but provides quick access to metadata such as sensor dtype, name and number
    of data points.

    Parameters
    ----------
    data : sequence of recarray-like with fields ('timestamp', 'value', 'status')
        Uncached sensor data as a list of record arrays or equivalent (such as a
        :class:`h5py.Dataset`)

    """
    def __init__(self, data):
        self.data = data
        names = unique_in_order([sd.name for sd in data])
        if len(names) != 1:
            raise ConcatenationError('Cannot concatenate sensors with different names: %s' % (names,))
        self.name = names[0]
        dtypes = unique_in_order([sd.dtype for sd in data])
        if len(dtypes) == 1:
            self.dtype = dtypes[0]
        elif np.all([np.issubdtype(dtype, np.str) for dtype in dtypes]):
            # Strings of different lengths have different dtypes (e.g. '|S1' vs '|S10') but can be safely concatenated
            self.dtype = np.dtype('|S%d' % (max([dt.itemsize for dt in dtypes]),))
        else:
            raise ConcatenationError("Cannot concatenate sensor '%s' with different dtypes: %s" % (self.name, dtypes))

    def __getitem__(self, key):
        """Extract timestamp, value and status of each sensor data point."""
        return np.concatenate([sd[key] for sd in self.data])

    def __len__(self):
        """Number of sensor data points."""
        return np.sum([len(sd) for sd in self.data])


def _calc_dummy(cache, name):
    """Dummy virtual sensor that returns NaNs."""
    cache[name] = sensor_data = np.nan * np.ones(len(cache.timestamps))
    return sensor_data

#--------------------------------------------------------------------------------------------------
#--- CLASS :  ConcatenatedSensorCache
#--------------------------------------------------------------------------------------------------

class ConcatenatedSensorCache(SensorCache):
    """Sensor cache that is a concatenation of multiple underlying caches.

    This concatenates a sequence of sensor caches along the time axis and makes
    them appear like a single sensor cache. The combined cache contains a
    superset of all actual and virtual sensors found in the underlying caches
    and replaces any missing sensor data with dummy values.

    Parameters
    ----------
    caches : sequence of :class:`SensorCache` objects
        Sequence of underlying caches to be concatenated
    keep : sequence of bool, optional
        Default (global) time selection specification as boolean mask that will
        be applied to sensor data (this can be disabled on data retrieval)

    """
    def __init__(self, caches, keep=None):
        self.caches = caches
        # Collect the names and types of all actual and virtual sensors in caches, as well as properties
        actual, virtual, self.props = {}, {}, {}
        for cache in caches:
            actual.update(zip(cache.keys(), [sd.dtype for sd in cache.itervalues()]))
            virtual.update(cache.virtual)
            self.props.update(cache.props)
        # Pad out actual sensors on each cache (replace with default sensor values where missing)
        for name, dtype in actual.iteritems():
            if name not in cache:
                cache[name] = dummy_sensor_data(name, dtype)
        # Pad out virtual sensors with default functions (nans)
        for name in virtual:
            if name not in cache.virtual:
                cache.virtual[name] = _calc_dummy
        self.virtual = virtual
        timestamps = [cache.timestamps for cache in caches]
        if np.all([isinstance(ts, LazyIndexer) for ts in timestamps]):
            self.timestamps = ConcatenatedLazyIndexer(timestamps)
        else:
            self.timestamps = np.concatenate([ts[:] for ts in timestamps])
        self._segments = np.cumsum([0] + [len(cache.timestamps) for cache in caches])
        self._set_keep(keep)

    def _set_keep(self, keep=None):
        """Set time selection for sensor values.

        Parameters
        ----------
        keep : array of bool, shape (*T*,), optional
            Boolean selection mask with one entry per timestamp

        """
        if keep is not None:
            # Save top-level / global boolean selection mask and let each cache.keep be a view into this array
            self.keep = keep
            for n, cache in enumerate(self.caches):
                cache._set_keep(keep[self._segments[n]:self._segments[n + 1]])

    def get(self, name, select=False, extract=True, **kwargs):
        """Sensor values interpolated to correlator data timestamps.

        Retrieve raw (uncached) or cached sensor data from each underlying cache
        and concatenate the results along the time axis.

        Parameters
        ----------
        name : string
            Sensor name
        select : {False, True}, optional
            True if preset time selection will be applied to returned data
        extract : {True, False}, optional
            True if sensor data should be extracted from HDF5 file and cached
        kwargs : dict, optional
            Additional parameters are passed to underlying sensor caches

        Returns
        -------
        data : array or :class:`CategoricalData` or :class:`SensorData` object
            If extraction is disabled, this will be a :class:`SensorData` object
            for uncached sensors. If selection is enabled, this will be a 1-D
            array of values, one per selected timestamp. If selection is
            disabled, this will be a 1-D array of values (of the same length as
            the :attr:`timestamps` attribute) for numerical data, and a
            :class:`CategoricalData` object for categorical data.

        Raises
        ------
        KeyError
            If sensor name was not found in cache and did not match virtual template

        """
        # Get array, categorical data or raw sensor data from each cache
        split_data = [cache.get(name, select, extract, **kwargs) for cache in self.caches]
        # If this sensor has already been partially extracted, we are forced to extract it in rest of caches too
        if not extract and not np.all([isinstance(data, SensorData) for data in split_data]):
            split_data = [cache.get(name, select, True, **kwargs) for cache in self.caches]
        if isinstance(split_data[0], SensorData):
            return ConcatenatedSensorData(split_data)
        elif isinstance(split_data[0], CategoricalData):
            # Look up properties associated with this specific sensor
            props = self.props.get(name, {})
            # Look up properties associated with this class of sensor
            for key, val in self.props.iteritems():
                if key[0] == '*' and name.endswith(key[1:]):
                    props.update(val)
            # Any properties passed directly to this method takes precedence
            props.update(kwargs)
            return concatenate_categorical(split_data, **props)
        else:
            return np.concatenate(split_data)

    def __setitem__(self, name, data):
        """Assign data to sensor, splitting it across underlying caches.

        Parameters
        ----------
        name : string
            Sensor name
        data : array or :class:`CategoricalData` or :class:`SensorData` object
            Data to be assigned to sensor

        """
        # Split data into segments and setitem to each cache
        if isinstance(data, CategoricalData):
            split_data = data.partition(self._segments)
            for n, cache in enumerate(self.caches):
                cache[name] = split_data[n]
        else:
            for n, cache in enumerate(self.caches):
                cache[name] = data[self._segments[n]:self._segments[n + 1]]

    def iterkeys(self):
        """Key iterator that iterates through sensor names."""
        # Run through first cache's keys, as they should all be identical
        return self.caches[0].iterkeys()

#--------------------------------------------------------------------------------------------------
#--- CLASS :  ConcatenatedDataSet
#--------------------------------------------------------------------------------------------------

class ConcatenatedDataSet(DataSet):
    """Class that concatenates existing visibility data sets.

    This provides a single DataSet interface to a list of concatenated data sets.
    Where possible, identical targets, subarrays, spectral windows and
    observation sensors are merged. For more information on attributes, see the
    :class:`DataSet` docstring.

    Parameters
    ----------
    datasets : sequence of :class:`DataSet` objects
        List of existing data sets

    """
    def __init__(self, datasets):
        DataSet.__init__(self, '', datasets[0].ref_ant, datasets[0].time_offset)

        # Sort data sets in chronological order via 'decorate-sort-undecorate' (DSU) idiom
        decorated_datasets = [(d.start_time, d) for d in datasets]
        decorated_datasets.sort()
        self.datasets = datasets = [d[-1] for d in decorated_datasets]

        # Merge high-level metadata
        names = unique_in_order([d.name for d in datasets])
        self.name = ','.join([os.path.basename(name) for name in names])
        self.version = ','.join(unique_in_order([d.version for d in datasets]))
        self.observer = ','.join(unique_in_order([d.observer for d in datasets]))
        self.description = ' | '.join(unique_in_order([d.description for d in datasets]))
        self.experiment_id = ','.join(unique_in_order([d.experiment_id for d in datasets]))
        obs_params = unique_in_order(reduce(lambda x, y: x + y, [d.obs_params.keys() for d in datasets]))
        for param in obs_params:
            values = [d.obs_params.get(param, '') for d in datasets]
            self.obs_params[param] = values[0] if len(set(values)) == 1 else values

        dump_periods = unique_in_order([d.dump_period for d in datasets])
        if len(dump_periods) > 1:
            raise ConcatenationError('Data sets cannot be concatenated because of differing dump periods: ' +
                                     ', '.join([('%g' % (dp,)) for dp in dump_periods]))
        self.dump_period = dump_periods[0]
        self._segments = np.cumsum([0] + [len(d.sensor.timestamps) for d in datasets])
        # Keep main time selection mask at top level and ensure that underlying datasets use slice views of main one
        self._set_keep(time_keep=np.ones(self._segments[-1], dtype=np.bool))
        self.start_time = min([d.start_time for d in datasets])
        self.end_time = max([d.end_time for d in datasets])

        self.sensor = ConcatenatedSensorCache([d.sensor for d in datasets], keep=self._time_keep)
        subarray = self.sensor.get('Observation/subarray')
        spw = self.sensor.get('Observation/spw')
        target = self.sensor.get('Observation/target')
        self.subarrays = subarray.unique_values
        self.spectral_windows = spw.unique_values
        self.catalogue.add(target.unique_values)
        self.catalogue.antenna = self.sensor['Antennas/%s/antenna' % (self.ref_ant,)][0]
        split_sub = subarray.partition(self._segments)
        split_spw = spw.partition(self._segments)
        split_target = target.partition(self._segments)
        # Fix index sensors in underlying datasets: scan / compscan runs on and the rest are remapped to merged values
        scan_start, compscan_start = 0, 0
        for n, d in enumerate(datasets):
            d.sensor['Observation/subarray'] = split_sub[n]
            d.sensor['Observation/subarray_index'] = CategoricalData(split_sub[n].indices, split_sub[n].events)
            d.sensor['Observation/spw'] = split_spw[n]
            d.sensor['Observation/spw_index'] = CategoricalData(split_spw[n].indices, split_spw[n].events)
            d.sensor['Observation/target'] = split_target[n]
            d.sensor['Observation/target_index'] = CategoricalData(split_target[n].indices, split_target[n].events)
            scan_index = d.sensor.get('Observation/scan_index')
            scan_index.unique_values += scan_start
            scan_start += len(scan_index.unique_values)
            d.sensor['Observation/scan_index'] = scan_index
            compscan_index = d.sensor.get('Observation/compscan_index')
            compscan_index.unique_values += compscan_start
            compscan_start += len(compscan_index.unique_values)
            d.sensor['Observation/compscan_index'] = compscan_index
        # Apply default selection and initialise all members that depend on selection in the process
        self.select(spw=0, subarray=0)

    def _set_keep(self, time_keep=None, freq_keep=None, corrprod_keep=None):
        """Set time, frequency and/or correlation product selection masks.

        Set the selection masks for those parameters that are present. The time
        mask is split into chunks and applied to the underlying datasets and
        sensor caches, while the frequency and corrprod masks are directly
        applied to the underlying datasets as well.

        Parameters
        ----------
        time_keep : array of bool, shape (*T*,), optional
            Boolean selection mask with one entry per timestamp
        freq_keep : array of bool, shape (*F*,), optional
            Boolean selection mask with one entry per frequency channel
        corrprod_keep : array of bool, shape (*B*,), optional
            Boolean selection mask with one entry per correlation product

        """
        if time_keep is not None:
            self._time_keep = time_keep
            for n, d in enumerate(self.datasets):
                d._set_keep(time_keep=self._time_keep[self._segments[n]:self._segments[n + 1]])
            # Ensure that sensor cache gets updated time selection
            if self.sensor is not None:
                self.sensor._set_keep(self._time_keep)
        if freq_keep is not None:
            self._freq_keep = freq_keep
            for n, d in enumerate(self.datasets):
                d._set_keep(freq_keep=self._freq_keep)
        if corrprod_keep is not None:
            self._corrprod_keep = corrprod_keep
            for n, d in enumerate(self.datasets):
                d._set_keep(corrprod_keep=self._corrprod_keep)

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array indexer of float64, shape
        (*T*,), with one timestamp per integration aligned with the integration
        *midpoint*. To get the data array itself from the indexer `x`, do `x[:]`
        or perform any other form of selection on it.

        """
        return ConcatenatedLazyIndexer([d.timestamps for d in self.datasets])

    @property
    def vis(self):
        """Complex visibility data as a function of time, frequency and baseline.

        The visibility data are returned as an array indexer of complex64, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`. To get the data array
        itself from the indexer `x`, do `x[:]` or perform any other form of
        selection on it.

        """
        return ConcatenatedLazyIndexer([d.vis for d in self.datasets])

    def weights(self, names=None):
        """Visibility weights as a function of time, frequency and baseline.

        Parameters
        ----------
        names : None or string or sequence of strings, optional
            List of names of weights to be multiplied together, as a sequence
            or string of comma-separated names (combine all weights by default)

        Returns
        -------
        weights : :class:`LazyIndexer` object of float32, shape (*T*, *F*, *B*)
            Array indexer with time along the first dimension, frequency along
            the second dimension and correlation product ("baseline") index
            along the third dimension. To get the data array itself from the
            indexer `x`, do `x[:]` or perform any other form of indexing on it.
            Only then will data be loaded into memory.

        """
        return ConcatenatedLazyIndexer([d.weights(names) for d in self.datasets])

    def flags(self, names=None):
        """Visibility flags as a function of time, frequency and baseline.

        Parameters
        ----------
        names : None or string or sequence of strings, optional
            List of names of flags that will be OR'ed together, as a sequence or
            a string of comma-separated names (use all flags by default)

        Returns
        -------
        flags : :class:`LazyIndexer` object of bool, shape (*T*, *F*, *B*)
            Array indexer with time along the first dimension, frequency along
            the second dimension and correlation product ("baseline") index
            along the third dimension. To get the data array itself from the
            indexer `x`, do `x[:]` or perform any other form of indexing on it.
            Only then will data be loaded into memory.

        """
        return ConcatenatedLazyIndexer([d.flags(names) for d in self.datasets])
