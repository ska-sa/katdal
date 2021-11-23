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

"""Class for concatenating visibility data sets."""

import itertools
from functools import reduce

import numpy as np

from .categorical import (CategoricalData, concatenate_categorical,
                          unique_in_order)
from .dataset import DataSet
from .lazy_indexer import LazyIndexer
from .sensordata import (SensorCache, SensorData, SensorGetter,
                         dummy_sensor_getter)


class ConcatenationError(Exception):
    """Sequence of objects could not be concatenated due to incompatibility."""

# -------------------------------------------------------------------------------------------------
# -- CLASS :  ConcatenatedLazyIndexer
# -------------------------------------------------------------------------------------------------


class ConcatenatedLazyIndexer(LazyIndexer):
    """Two-stage deferred indexer that concatenates multiple indexers.

    This indexer concatenates a sequence of indexers along the first (i.e. time)
    axis. The index specification is broken down into chunks along this axis,
    sent to the applicable underlying indexers and the returned data are
    concatenated again before returning it.

    Parameters
    ----------
    indexers : sequence of :class:`LazyIndexer` objects and/or arrays
        Sequence of indexers or raw arrays to be concatenated
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
        # Wrap any raw array in the sequence in a LazyIndexer (slower but more compatible)
        for n, indexer in enumerate(self.indexers):
            self.indexers[n] = indexer if isinstance(indexer, LazyIndexer) else LazyIndexer(indexer)
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
            descr += [f'- Indexer {n:03d}: ' + indexer_descr[0]]
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

        def find_indexer(index):
            return indexer_starts.searchsorted(index, side='right') - 1

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
        shape_tails = {indexer.shape[1:] for indexer in self.indexers}
        if len(shape_tails) != 1:
            raise ConcatenationError("Incompatible shapes among sub-indexers making up indexer '%s':\n%s" %
                                     (self.name, '\n'.join([repr(indexer) for indexer in self.indexers])))
        return tuple([np.sum([len(indexer) for indexer in self.indexers])] + list(shape_tails.pop()))

    @property
    def _initial_dtype(self):
        """Type of data array before transformation."""
        # Each component must have the same dtype, which becomes the overall dtype
        dtypes = {indexer.dtype for indexer in self.indexers}
        if len(dtypes) == 1:
            return dtypes.pop()
        elif np.all([np.issubdtype(dtype, np.string_) for dtype in dtypes]):
            # Strings of different lengths have different dtypes (e.g. '|S1' vs '|S10') but can be safely concatenated
            return np.dtype('|S{}'.format(max([dt.itemsize for dt in dtypes])))
        else:
            raise ConcatenationError(f"Incompatible dtypes among sub-indexers making up indexer '{self.name}':\n"
                                     + '\n'.join([repr(indexer) for indexer in self.indexers]))

# -------------------------------------------------------------------------------------------------
# -- CLASS :  ConcatenatedSensorGetter
# -------------------------------------------------------------------------------------------------


def common_dtype(sensor_data_sequence):
    """The dtype suitable to store all sensor data values in the given sequence.

    This extracts the dtypes of a sequence of sensor data objects and finds the
    minimal dtype to which all of them may be safely cast using NumPy type
    promotion rules (which will typically be the dtype of a concatenation of
    the values).

    Parameters
    ----------
    sensor_data_sequence : sequence of extracted sensor data objects
        These objects may include :class:`numpy.ndarray` and :class:`CategoricalData`

    Returns
    -------
    dtype : :class:`numpy.dtype` object
        The promoted dtype of the sequence, or None if `sensor_data_sequence` is empty

    """
    dtypes = [sd.dtype for sd in sensor_data_sequence]
    # Find resulting dtype through type promotion or give up if nothing is known
    return np.result_type(*dtypes) if dtypes else None


class ConcatenatedSensorGetter(SensorGetter):
    """The concatenation of multiple raw (uncached) sensor data sets.

    This is a convenient container for returning raw (uncached) sensor data sets
    from a :class:`ConcatenatedSensorCache` object. It only accesses the
    underlying data sets when explicitly asked to via the :meth:`get` interface,
    but provides quick access to metadata such as sensor name.

    Parameters
    ----------
    data : sequence of :class:`SensorGetter`
        Uncached sensor data
    """

    def __init__(self, data):
        names = unique_in_order([sd.name for sd in data])
        if len(names) != 1:
            # XXX This is probably not a serious restriction; consider removal.
            # It is a weak verification that we are combining like sensors,
            # but underlying names may legitimately differ for datasets of
            # different minor versions (even within the same version...).
            raise ConcatenationError('Cannot concatenate sensor with different '
                                     f'underlying names: {names}')
        super().__init__(names[0])
        self._data = data

    def get(self):
        parts = [sd.get() for sd in self._data]
        # Filter out empty sensors, because they may have a default dtype that
        # will skew the dtype of the concatenation
        parts = [part for part in parts if part]
        if not parts:
            timestamp = np.array([])
            value = np.array([])
            status = np.array([], dtype='S7')
        else:
            timestamp = np.concatenate([part.timestamp for part in parts])
            value = np.concatenate([part.value for part in parts])
            if all(part.status is not None for part in parts):
                status = np.concatenate([part.status for part in parts])
            else:
                status = None
        return SensorData(self.name, timestamp, value, status)


# -------------------------------------------------------------------------------------------------
# -- CLASS :  ConcatenatedSensorCache
# -------------------------------------------------------------------------------------------------


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
        # Collect all virtual sensors in caches as well as properties.
        virtual, self.props = {}, {}
        for cache in caches:
            virtual.update(cache.virtual)
            self.props.update(cache.props)
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

    def _get(self, name, **kwargs):
        """Extract sensor data from multiple caches (see :meth:`get` for docs).

        This extracts a sequence of sensor data objects, one from each cache.
        For caches which do not contain the sensor it returns `None`.

        """
        # First extract from all caches where the requested sensor is present
        split_data = []
        for cache in self.caches:
            try:
                sensor_data = cache.get(name, **kwargs)
            except KeyError:
                split_data.append(None)
            else:
                split_data.append(sensor_data)
        return split_data

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
            True if sensor data should be extracted from store and cached
        kwargs : dict, optional
            Additional parameters are passed to underlying sensor caches

        Returns
        -------
        data : array or :class:`CategoricalData` or :class:`SensorGetter` object
            If extraction is disabled, this will be a :class:`SensorGetter` object
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
        split_data = self._get(name, select=select, extract=extract, **kwargs)
        if all(sd is None for sd in split_data):
            raise KeyError(f'Key {name} not found in any of the concatenated datasets')
        # If this sensor has already been partially extracted,
        # we are forced to extract it in rest of caches too
        if not extract and not all(sd is None or isinstance(sd, SensorGetter) for sd in split_data):
            extract = True
            split_data = self._get(name, select=select, extract=extract, **kwargs)
        if not extract:
            # Just discard pieces for which the sensor is missing.
            split_data = [sd for sd in split_data if sd is not None]
            return ConcatenatedSensorGetter(split_data)

        props = self._get_props(name, self.props, **kwargs)

        if any(sd is None for sd in split_data):
            # This should not typically happen, and it needs a slow path to
            # figure out the right dummy value. We put the dummy values back
            # into the cache so that this isn't needed next time.
            if select:
                split_data2 = self._get(name, select=False, extract=True, **kwargs)
            else:
                split_data2 = split_data
            split_data2 = [sd for sd in split_data2 if sd is not None]
            dtype = common_dtype(split_data2)
            dummy = dummy_sensor_getter(name, value=props.get('initial_value'), dtype=dtype)
            for i, cache in enumerate(self.caches):
                if split_data[i] is None:
                    cache[name] = self._extract(dummy, cache.timestamps, cache.dump_period, **props)
                    split_data[i] = cache.get(name, select=select, extract=True, **kwargs)

        if any(isinstance(sd, CategoricalData) for sd in split_data):
            return concatenate_categorical(split_data, **props)
        else:
            # Keep arrays as arrays and lists as lists to avoid dtype issues
            if any(isinstance(sd, np.ndarray) for sd in split_data):
                return np.concatenate(split_data)
            else:
                return sum(split_data, [])

    def __setitem__(self, name, data):
        """Assign data to sensor, splitting it across underlying caches.

        Parameters
        ----------
        name : string
            Sensor name
        data : array or :class:`CategoricalData`
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

    def __delitem__(self, name):
        found = False
        for cache in self.caches:
            try:
                del cache[name]
                found = True
            except KeyError:
                pass
        if not found:
            raise KeyError(name)

    def __contains__(self, name):
        return any(name in cache for cache in self.caches)

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        """Key iterator that iterates through sensor names."""
        seen = set()
        for cache in self.caches:
            for key in cache:
                if key not in seen:
                    seen.add(key)
                    yield key

# -------------------------------------------------------------------------------------------------
# -- CLASS :  ConcatenatedDataSet
# -------------------------------------------------------------------------------------------------


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
        self.name = ','.join(unique_in_order([d.name for d in datasets]))
        self.url = ' | '.join(unique_in_order([d.url for d in datasets]))
        self.version = ','.join(unique_in_order([d.version for d in datasets]))
        self.observer = ','.join(unique_in_order([d.observer for d in datasets]))
        self.description = ' | '.join(unique_in_order([d.description for d in datasets]))
        self.experiment_id = ','.join(unique_in_order([d.experiment_id for d in datasets]))
        obs_params = unique_in_order(reduce(lambda x, y: x + y, [list(d.obs_params.keys()) for d in datasets]))
        for param in obs_params:
            values = [d.obs_params.get(param, '') for d in datasets]
            # If all values are the same, extract the unique value from the list; otherwise keep the list
            # The itertools.groupby function should work on any value, even unhashable and unorderable ones
            self.obs_params[param] = values[0] if len([k for k in itertools.groupby(values)]) == 1 else values
        rx_ants = unique_in_order(reduce(lambda x, y: x + y, [list(d.receivers.keys()) for d in datasets]))
        for ant in rx_ants:
            rx = [d.receivers.get(ant, '') for d in datasets]
            self.receivers[ant] = rx[0] if len([k for k in itertools.groupby(rx)]) == 1 else rx

        dump_periods = unique_in_order([d.dump_period for d in datasets])
        if len(dump_periods) > 1:
            raise ConcatenationError('Data sets cannot be concatenated because of differing dump periods: ' +
                                     ', '.join(f'{dp:g}' for dp in dump_periods))
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
        self.catalogue.antenna = self.sensor[f'Antennas/{self.ref_ant}/antenna'][0]
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
            scan_index.unique_values = [index + scan_start for index in scan_index.unique_values]
            scan_start += len(scan_index.unique_values)
            d.sensor['Observation/scan_index'] = scan_index
            compscan_index = d.sensor.get('Observation/compscan_index')
            compscan_index.unique_values = [index + compscan_start for index in compscan_index.unique_values]
            compscan_start += len(compscan_index.unique_values)
            d.sensor['Observation/compscan_index'] = compscan_index
        # Apply default selection and initialise all members that depend on selection in the process
        self.select(spw=0, subarray=0)

    def _set_keep(self, time_keep=None, freq_keep=None, corrprod_keep=None,
                  weights_keep=None, flags_keep=None):
        """Set time, frequency and/or correlation product selection masks.

        Set the selection masks for those parameters that are present. The time
        mask is split into chunks and applied to the underlying datasets and
        sensor caches, while the frequency and corrprod masks are directly
        applied to the underlying datasets as well. Also allow for weights
        and flags selections.

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
        for n, d in enumerate(self.datasets):
            d._set_keep(time_keep=self._time_keep[self._segments[n]:self._segments[n + 1]],
                        freq_keep=self._freq_keep,
                        corrprod_keep=self._corrprod_keep,
                        weights_keep=self._weights_keep,
                        flags_keep=self._flags_keep)

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
        return ConcatenatedLazyIndexer([d.weights for d in self.datasets])

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
        return ConcatenatedLazyIndexer([d.flags for d in self.datasets])

    @property
    def temperature(self):
        """Air temperature in degrees Celsius."""
        return np.concatenate([d.temperature for d in self.datasets])

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        return np.concatenate([d.pressure for d in self.datasets])

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        return np.concatenate([d.humidity for d in self.datasets])

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        return np.concatenate([d.wind_speed for d in self.datasets])

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        return np.concatenate([d.wind_direction for d in self.datasets])
