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

"""Base class for accessing a store of chunks (i.e. N-dimensional arrays)."""

from __future__ import division

import contextlib
import functools
import uuid

import numpy as np
import dask
import dask.array as da
import toolz


class ChunkStoreError(Exception):
    """"Base class for all standard ChunkStore errors."""


class StoreUnavailable(OSError, ChunkStoreError):
    """Could not access underlying storage medium (offline, auth failed, etc)."""


class ChunkNotFound(KeyError, ChunkStoreError):
    """The store was accessible but a chunk with the given name was not found."""


class BadChunk(ValueError, ChunkStoreError):
    """The chunk is malformed, e.g. bad dtype or slices, wrong buffer size."""


def _floor_power_of_two(x):
    """The largest power of two smaller than or equal to `x`."""
    return 2 ** int(np.floor(np.log2(x)))


def generate_chunks(shape, dtype, max_chunk_size, dims_to_split=None,
                    power_of_two=False):
    """Generate dask chunk specification from ndarray parameters.

    Parameters
    ----------
    shape : sequence of int
        Array shape
    dtype : :class:`numpy.dtype` object or equivalent
        Array data type
    max_chunk_size : float or int
        Upper limit on chunk size (if allowed by `dims_to_split`), in bytes
    dims_to_split : sequence of int, optional
        Indices of dimensions that may be split into chunks (default all dims)
    power_of_two : bool, optional
        True if chunk size should be rounded down to a power of two
        (the last chunk size along each dimension will potentially be smaller)

    Returns
    -------
    chunks : tuple of tuple of int
        Dask chunk specification, indicating chunk sizes along each dimension
    """
    if dims_to_split is None:
        dims_to_split = range(len(shape))
    dataset_size = np.prod(shape) * np.dtype(dtype).itemsize
    # The ideal number of chunks to achieve requested chunk size (can be float)
    num_chunks = dataset_size / max_chunk_size
    # Start with the whole array as a single big chunk
    chunks = [(s,) for s in shape]
    # Split the array greedily along each dimension, in order of `dims_to_split`
    for dim in dims_to_split:
        if dim >= len(shape):
            continue
        if num_chunks > shape[dim] / 2:
            # Split the dimension into the maximum number of chunks
            chunk_sizes = (1,) * shape[dim]
        else:
            items = np.arange(shape[dim])
            if power_of_two:
                # Chunk sizes will be (2^P, 2^P, 2^P, ..., 1 <= M <= 2^P)
                chunksize_per_dim = _floor_power_of_two(shape[dim] / num_chunks)
                chunk_indices = toolz.partition_all(chunksize_per_dim, items)
            else:
                # Chunk sizes generally will be (N, N, N, ..., N-1, N-1)
                chunk_indices = np.array_split(items, np.ceil(num_chunks))
            chunk_sizes = tuple([len(chunk) for chunk in chunk_indices])
        chunks[dim] = chunk_sizes
        # Update number of remaining chunks to realise
        num_chunks /= len(chunk_sizes)
    return tuple(chunks)


def _add_offset_to_slices(func, offset):
    """Modify chunk get/put/has to add an offset to its `slices` parameter."""
    offset_slices = tuple(slice(start, None) for start in offset)

    def func_with_offset(array_name, slices, *args, **kwargs):
        """Shift `slices` to start at `offset`."""
        slices = da.core.fuse_slice(offset_slices, slices)
        return func(array_name, slices, *args, **kwargs)

    return func_with_offset


def _scalar_to_chunk(func):
    """Modify chunk get/put/has to turn a scalar return value into a chunk.

    This modifies the given function so that it returns its result as an
    ndarray with the same number of (singleton) dimensions as the corresponding
    chunk to enable assembly into a dask array.
    """

    def func_returning_chunk(array_name, slices, *args, **kwargs):
        """Turn scalar return value into chunk of appropriate dimension."""
        value = func(array_name, slices, *args, **kwargs)
        singleton_shape = len(slices) * (1,)
        return np.full(singleton_shape, value)

    return func_returning_chunk


class ChunkStore(object):
    """Base class for accessing a store of chunks (i.e. N-dimensional arrays).

    A *chunk* is a simple (i.e. unit-stride) slice of an N-dimensional array
    known as its *parent array*. The array is identified by a string name,
    while the chunk within the array is identified by a sequence of slice
    objects which may be used to extract the chunk from the array. The array
    is a :class:`numpy.ndarray` object with an associated *dtype*.

    The basic idea is that the chunk store contains multiple arrays addressed
    by name. The list of available arrays and all array metadata (shape,
    chunks and dtype) are stored elsewhere. The metadata is used to identify
    chunks, while the chunk store takes care of storing and retrieving
    bytestrings of actual chunk data. These are packaged back into NumPy
    arrays for the user. Each array can only be stored once, with a unique
    chunking scheme (i.e. different chunking of the same data is disallowed).

    The naming scheme for arrays and chunks is reasonably generic but has
    some restrictions:

    - Names are treated like paths with components and a standard separator
    - The chunk name is formed by appending a string of indices to the array name
    - It is discouraged to have an array name that is a prefix of another name
    - Each chunk store has its own restrictions on valid characters in names:
      some treat names as URLs while others treat them as filenames. A safe
      choice for name components should be the valid characters for S3 buckets:

      VALID_BUCKET = re.compile(r'^[a-zA-Z0-9.\-_]{1,255}$')

    Parameters
    ----------
    error_map : dict mapping :class:`Exception` to :class:`Exception`, optional
        Dict that maps store-specific errors to standard ChunkStore errors
    """

    def __init__(self, error_map=None):
        if error_map is None:
            error_map = {OSError: StoreUnavailable, KeyError: ChunkNotFound,
                         ValueError: BadChunk}
        self._error_map = error_map

    def get_chunk(self, array_name, slices, dtype):
        """Get chunk from the store.

        Parameters
        ----------
        array_name : string
            Identifier of parent array `x` of chunk
        slices : sequence of unit-stride slice objects
            Identifier of individual chunk, to be extracted as `x[slices]`
        dtype : :class:`numpy.dtype` object or equivalent
            Data type of array `x`

        Returns
        -------
        chunk : :class:`numpy.ndarray` object
            Chunk as ndarray with dtype `dtype` and shape dictated by `slices`

        Raises
        ------
        :exc:`chunkstore.BadChunk`
            If requested `dtype` does not match underlying parent array dtype,
            `slices` has wrong specification or stored buffer has wrong size
        :exc:`chunkstore.StoreUnavailable`
            If interaction with chunk store failed (offline, bad auth, bad config)
        :exc:`chunkstore.ChunkNotFound`
            If requested chunk was not found in store
        """
        raise NotImplementedError

    def put_chunk(self, array_name, slices, chunk):
        """Put chunk into the store.

        Parameters
        ----------
        array_name : string
            Identifier of parent array `x` of chunk
        slices : sequence of unit-stride slice objects
            Identifier of individual chunk, to be extracted as `x[slices]`
        chunk : :class:`numpy.ndarray` object
            Chunk as ndarray with shape commensurate with `slices`

        Raises
        ------
        :exc:`chunkstore.BadChunk`
            If `slices` is wrong or its shape does not match that of `chunk`
        :exc:`chunkstore.StoreUnavailable`
            If interaction with chunk store failed (offline, bad auth, bad config)
        :exc:`chunkstore.ChunkNotFound`
            If `array_name` is incompatible with store
        """
        raise NotImplementedError

    def put_chunk_noraise(self, array_name, slices, chunk):
        """Put chunk into store but return any exceptions instead of raising."""
        try:
            self.put_chunk(array_name, slices, chunk)
        except ChunkStoreError as err:
            return err
        else:
            return None

    def has_chunk(self, array_name, slices, dtype):
        """Check if chunk is in the store.

        Parameters
        ----------
        array_name : string
            Identifier of parent array `x` of chunk
        slices : sequence of unit-stride slice objects
            Identifier of individual chunk, to be extracted as `x[slices]`
        dtype : :class:`numpy.dtype` object or equivalent
            Data type of array `x`

        Returns
        -------
        success : bool
            True if chunk was found in the store, with appropriate size / dtype

        Raises
        ------
        :exc:`chunkstore.BadChunk`
            If `slices` has wrong specification
        :exc:`chunkstore.StoreUnavailable`
            If interaction with chunk store failed (offline, bad auth, bad config)
        """
        try:
            self.get_chunk(array_name, slices, dtype)
        except ChunkNotFound:
            return False
        else:
            return True

    NAME_SEP = '/'
    # Width sufficient to store any dump / channel / corrprod index for MeerKAT
    NAME_INDEX_WIDTH = 5

    @classmethod
    def join(cls, *names):
        """Join components of chunk name with supported separator."""
        return cls.NAME_SEP.join(names)

    @classmethod
    def split(cls, name, maxsplit=-1):
        """Split chunk name into components based on supported separator."""
        return name.split(cls.NAME_SEP, maxsplit)

    @classmethod
    def chunk_metadata(cls, array_name, slices, chunk=None, dtype=None):
        """Turn array name and chunk identifier into chunk name and shape.

        Form the full chunk name from `array_name` and `slices` and extract
        the chunk shape from `slices`, validating it in the process. If `chunk`
        or `dtype` is given, check that `chunk` is commensurate with `slices`
        and that `dtype` contains no objects which would cause nasty segfaults.

        Parameters
        ----------
        array_name : string
            Identifier of parent array `x` of chunk
        slices : sequence of unit-stride slice objects
            Identifier of individual chunk, to be extracted as `x[slices]`
        chunk : :class:`numpy.ndarray` object, optional
            Actual chunk data as ndarray (used to validate shape / dtype)
        dtype : :class:`numpy.dtype` object or equivalent, optional
            Data type of array `x` (used for validation only)

        Returns
        -------
        chunk_name : string
            Full chunk name used to find chunk in underlying storage medium
        shape : tuple of int
            Chunk shape tuple associated with `slices`

        Raises
        ------
        :exc:`chunkstore.BadChunk`
            If `slices` has wrong specification or its shape does not match
            that of `chunk`, or any dtype contains objects

        """
        try:
            shape = tuple(s.stop - s.start for s in slices)
        except (TypeError, AttributeError):
            raise BadChunk('Array {!r}: chunk ID should be a sequence of '
                           'slice objects, not {}'.format(array_name, slices))
        # Verify that all slice strides are unity (i.e. it's a "simple" slice)
        if not all([s.step in (1, None) for s in slices]):
            raise BadChunk('Array {!r}: chunk ID {} contains non-unit strides'
                           .format(array_name, slices))
        # Construct chunk name from array_name + slices
        index = [s.start for s in slices]
        idxstr = '_'.join(["{:0{width}d}".format(i, width=cls.NAME_INDEX_WIDTH)
                           for i in index])
        chunk_name = cls.join(array_name, idxstr)
        if chunk is not None and chunk.shape != shape:
            raise BadChunk('Chunk {!r}: shape {} implied by slices does not '
                           'match actual shape {}'
                           .format(chunk_name, shape, chunk.shape))
        if chunk is not None and chunk.dtype.hasobject:
            raise BadChunk('Chunk {!r}: actual dtype {} cannot contain '
                           'objects'.format(chunk_name, chunk.dtype))
        if dtype is not None and np.dtype(dtype).hasobject:
            raise BadChunk('Chunk {!r}: Requested dtype {} cannot contain '
                           'objects'.format(chunk_name, dtype))
        return chunk_name, shape

    @contextlib.contextmanager
    def _standard_errors(self, chunk_name=None):
        """Catch store-specific errors and turn them into standard errors.

        This uses the internal error map to remap store-specific exceptions
        to standard ChunkStore ones.

        Parameters
        ----------
        chunk_name : string, optional
            Full chunk name, used as prefix to error message if available
        """
        try:
            yield
        except tuple(self._error_map) as e:
            try:
                StandardisedError = self._error_map[type(e)]
            except KeyError:
                # The exception has to be a subclass of one of the error_map
                # keys, so pick the first one found
                FirstBase = next(c for c in self._error_map if isinstance(e, c))
                StandardisedError = self._error_map[FirstBase]
            prefix = 'Chunk {!r}: '.format(chunk_name) if chunk_name else ''
            raise StandardisedError(prefix + str(e))

    def get_dask_array(self, array_name, chunks, dtype, offset=()):
        """Get dask array from the store.

        Parameters
        ----------
        array_name : string
            Identifier of array in chunk store
        chunks : tuple of tuples of ints
            Chunk specification
        dtype : :class:`numpy.dtype` object or equivalent
            Data type of array
        offset : tuple of int, optional
            Offset to add to each dimension when addressing chunks in store

        Returns
        -------
        array : :class:`dask.array.Array` object
            Dask array of given dtype
        """
        getter = functools.partial(self.get_chunk, dtype=dtype)
        if offset:
            getter = _add_offset_to_slices(getter, offset)
        # Use dask utility function that forms the core of da.from_array
        dask_graph = da.core.getem(array_name, chunks, getter)
        return da.Array(dask_graph, array_name, chunks, dtype)

    def put_dask_array(self, array_name, array, offset=()):
        """Put dask array into the store.

        Parameters
        ----------
        array_name : string
            Identifier of array in chunk store
        array : :class:`dask.array.Array` object
            Dask input array
        offset : tuple of int, optional
            Offset to add to each dimension when addressing chunks in store

        Returns
        -------
        success : :class:`dask.array.Array` object
            Dask array of objects indicating success of transfer of each chunk
            (None indicates success, otherwise there is an exception object)
        """
        dask_graph = dask.sharedict.ShareDict()
        dask_graph.update(array.dask)
        # Give better names to these two very similar variables
        in_name = array.name
        out_name = array_name
        # Make out_name unique to avoid clashes and caches
        out_name = 'store-{}-{}-{}'.format(out_name, offset, uuid.uuid4().hex)
        put = _scalar_to_chunk(self.put_chunk_noraise)
        if offset:
            put = _add_offset_to_slices(put, offset)
        # Construct output graph on same chunks as input, but with new name
        graph = da.core.getem(array_name, array.chunks, put, out_name=out_name)
        # Set chunk parameter of put_chunk() to corresponding key in input array
        graph = {k: v + ((in_name,) + k[1:],) for k, v in graph.items()}
        dask_graph.update(graph)
        # The success array has one element per chunk in the input array
        out_chunks = tuple(len(c) * (1,) for c in array.chunks)
        return da.Array(dask_graph, out_name, out_chunks, np.object)

    def has_dask_array(self, array_name, chunks, dtype, offset=()):
        """Check if dask array is in the store.

        Parameters
        ----------
        array_name : string
            Identifier of array in chunk store
        chunks : tuple of tuples of ints
            Chunk specification
        dtype : :class:`numpy.dtype` object or equivalent
            Data type of array
        offset : tuple of int, optional
            Offset to add to each dimension when addressing chunks in store

        Returns
        -------
        success : :class:`dask.array.Array` object
            Dask array of bools indicating presence of each chunk
        """
        has = _scalar_to_chunk(functools.partial(self.has_chunk, dtype=dtype))
        if offset:
            has = _add_offset_to_slices(has, offset)
        # Make out_name unique to avoid clashes and caches
        out_name = 'check-{}-{}-{}'.format(array_name, offset, uuid.uuid4().hex)
        # Use dask utility function that forms the core of da.from_array
        dask_graph = da.core.getem(array_name, chunks, has, out_name=out_name)
        # The success array has one element per chunk in the input array
        out_chunks = tuple(len(c) * (1,) for c in chunks)
        return da.Array(dask_graph, out_name, out_chunks, np.bool_)
