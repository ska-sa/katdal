################################################################################
# Copyright (c) 2017-2023, National Research Foundation (SARAO)
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

import contextlib
import io

import dask
import dask.array as da
import numpy as np


class ChunkStoreError(Exception):
    """"Base class for all standard ChunkStore errors."""


class StoreUnavailable(OSError, ChunkStoreError):
    """Could not access underlying storage medium (offline, auth failed, etc)."""


class ChunkNotFound(KeyError, ChunkStoreError):
    """The store was accessible but a chunk with the given name was not found."""

    def __str__(self):
        """Avoid the implicit repr() of KeyError since we'll have explanatory text."""
        return ChunkStoreError.__str__(self)


class BadChunk(ValueError, ChunkStoreError):
    """The chunk is malformed, e.g. bad dtype, actual shape differs from requested."""


class PlaceholderChunk:
    """Dummy chunk returned to indicate missing data (and chunk identity)."""

    def __init__(self, shape, dtype, name=''):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.name = name

    def __getitem__(self, index):
        # Create an array with a zero-sized dtype, so that it takes no memory
        dummy = np.empty(self.shape, dtype=[])
        new_shape = dummy[index].shape
        return PlaceholderChunk(new_shape, self.dtype, self.name)


def _blocks_ravel(array):
    """Workaround for `array.blocks.ravel()`.

    Return a flat list of mini dask arrays, one per chunk / block in `array`.
    XXX Remove once we depend on dask >= 2021.11.0.
    """
    ndim = len(array.numblocks)
    # Enumerate all block indices and pass them to old .blocks (an IndexCallable)
    block_indices = np.indices(array.numblocks).reshape(ndim, -1).T
    return [array.blocks[tuple(ind)] for ind in block_indices]


def _floor_power_of_two(x):
    """The largest power of two smaller than or equal to `x`."""
    return 2 ** int(np.floor(np.log2(x)))


def generate_chunks(shape, dtype, max_chunk_size, dims_to_split=None,
                    power_of_two=False, max_dim_elements=None):
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
    max_dim_elements : dict, optional
        Maximum number of elements on each dimension (each key is a dimension
        index). Dimensions that are not in `dims_to_split` are ignored.

    Returns
    -------
    chunks : tuple of tuple of int
        Dask chunk specification, indicating chunk sizes along each dimension
    """
    if dims_to_split is None:
        dims_to_split = range(len(shape))
    if max_dim_elements is None:
        max_dim_elements = {}

    dim_elements = list(shape)
    for i in dims_to_split:
        if i in max_dim_elements and max_dim_elements[i] < shape[i]:
            if power_of_two:
                dim_elements[i] = _floor_power_of_two(max_dim_elements[i])
            else:
                dim_elements[i] = max_dim_elements[i]
    # The ideal number of elements per chunk to achieve requested chunk size
    # (can be float).
    max_elements = max_chunk_size / np.dtype(dtype).itemsize
    # Split the array greedily along each dimension, in order of `dims_to_split`
    for dim in dims_to_split:
        cur_elements = int(np.prod(dim_elements))
        if cur_elements <= max_elements:
            break      # We have already split enough to meet the budget
        # Compute number of elements per chunk in this dimension to exactly
        # reach budget.
        trg_elements_real = dim_elements[dim] * max_elements / cur_elements
        if trg_elements_real < 1:
            trg_elements = 1
        elif power_of_two:
            trg_elements = _floor_power_of_two(trg_elements_real)
        else:
            # Try to split into a number of equal-as-possible sized pieces
            pieces = int(np.ceil(shape[dim] / trg_elements_real))
            # Note: np.ceil rather than np.floor here means the max_chunk_size
            # could be breached. It's done like this for backwards
            # compatibility.
            trg_elements = int(np.floor(shape[dim] / pieces))
        dim_elements[dim] = trg_elements

    return da.core.blockdims_from_blockshape(shape, dim_elements)


class _ArrayLikeGetter:
    """Present an array-like interface to chunk getter function."""

    def __init__(self, getter, array_name, chunks, dtype, **kwargs):
        self.getter = getter
        self.array_name = array_name
        self.kwargs = kwargs
        # Basic array-like attributes needed by :func:`dask.array.from_array`
        self.shape = tuple(sum(c) for c in chunks)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, slices, asarray=False, lock=None):
        """Call chunk getter on slices set up by :func:`dask.array.from_array`."""
        return self.getter(self.array_name, slices, self.dtype, **self.kwargs)


def _prune_chunks(chunks, index, offset=()):
    """Prune `chunks` spec, removing chunks completely outside slicing `index`.

    Remove the parts of `chunks` that fall outside the slices in `index`.
    This is done without changing the boundaries and sizes of existing chunks
    (i.e. without rechunking the data). Instead, the `index` and `offset`
    are adjusted to select the same data as before after a minimal set of
    pre-existing chunks are retained.

    This only supports the selection of contiguous regions in `index`.
    In other words, all slices in `index` need to have unit steps.
    """
    chunks = [list(c) for c in chunks]   # Make mutable
    shape = [sum(c) for c in chunks]
    index = list(da.slicing.normalize_index(index, shape))
    if not all(isinstance(idx, slice) and idx.step in (1, None) for idx in index):
        raise IndexError('Only slices with unit step are valid indices in get_dask_array')
    offset = list(offset) if offset else [0] * len(shape)
    for axis in range(len(shape)):
        if index[axis] == slice(None):
            continue
        start, stop, step = index[axis].indices(shape[axis])
        assert step == 1
        # Remove unneeded chunks from the ends
        start_chunk = 0
        while start_chunk < len(chunks[axis]) and chunks[axis][start_chunk] <= start:
            c = chunks[axis][start_chunk]
            offset[axis] += c
            start -= c
            stop -= c
            shape[axis] -= c
            start_chunk += 1
        stop_chunk = len(chunks[axis])
        while stop_chunk > start_chunk and chunks[axis][stop_chunk - 1] <= shape[axis] - stop:
            stop_chunk -= 1
            c = chunks[axis][stop_chunk]
            shape[axis] -= c
        chunks[axis] = chunks[axis][start_chunk:stop_chunk]
        if not chunks[axis]:
            chunks[axis] = (0,)   # Dask doesn't allow empty chunk lists
        index[axis] = slice(start, stop)
    # Go back to tuples
    chunks = tuple(tuple(c) for c in chunks)
    index = tuple(index)
    offset = tuple(offset)
    return chunks, index, offset


def _add_offset_to_slices(func, offset):
    """Modify chunk get/put/has to add an offset to its `slices` parameter."""
    def func_with_offset(array_name, slices, *args, **kwargs):
        """Shift `slices` to start at `offset`."""
        offset_slices = tuple(slice(s.start + i, s.stop + i)
                              for (s, i) in zip(slices, offset))
        return func(array_name, offset_slices, *args, **kwargs)
    return func_with_offset


def _put_map_blocks(chunk, block_info=None, store=None, array_name=None, offset=()):
    """Adapt `put` to `map_blocks` interface."""
    put = store.put_chunk_noraise
    if offset:
        put = _add_offset_to_slices(put, offset)
    # Turn block info into slices specification
    slices = tuple(slice(*loc) for loc in block_info[0]["array-location"])
    success = put(array_name, slices, chunk)
    # Turn scalar output into ndarray with same ndims as chunk
    singleton_shape = chunk.ndim * (1,)
    return np.full(singleton_shape, success)


def npy_header_and_body(chunk):
    """Prepare a chunk for low-level writing.

    Returns the `.npy` header and a view of the chunk corresponding to that
    header. The two should be concatenated (as buffer objects) to form a
    valid `.npy` file.

    This is useful for high-performance code, as it allows a chunk to be
    encoded as a .npy file more efficiently than saving to a
    :class:`io.BytesIO`.
    """
    # Note: don't use ascontiguousarray as it turns 0D into 1D.
    # See https://github.com/numpy/numpy/issues/5300
    chunk = np.asarray(chunk, order='C')
    fp = io.BytesIO()
    # TODO: have a fallback to version 2.0 if the header is too big for 1.0
    header_fields = np.lib.format.header_data_from_array_1_0(chunk)
    np.lib.format.write_array_header_1_0(fp, header_fields)
    header = fp.getvalue()
    return header, chunk


class ChunkStore:
    r"""Base class for accessing a store of chunks (i.e. N-dimensional arrays).

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
      choice for name components should be the valid characters for S3 buckets
      (also including underscores for non-bucket components):

      VALID_BUCKET = re.compile(r'^[a-z0-9][a-z0-9.\-]{2,62}$')

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
        :exc:`TypeError`
            If `slices` is not a sequence of slice(start, stop, 1) objects
        :exc:`chunkstore.BadChunk`
            If requested `dtype` does not match underlying parent array dtype
            or stored buffer has wrong size / shape compared to `slices`
        :exc:`chunkstore.StoreUnavailable`
            If interaction with chunk store failed (offline, bad auth, bad config)
        :exc:`chunkstore.ChunkNotFound`
            If requested chunk was not found in store
        """
        raise NotImplementedError

    def get_chunk_or_default(self, array_name, slices, dtype, default_value=0):
        """Get chunk from the store but return default value if it is missing."""
        try:
            return self.get_chunk(array_name, slices, dtype)
        except ChunkNotFound:
            chunk_name, shape = self.chunk_metadata(array_name, slices)
            return np.full(shape, default_value, dtype)

    def get_chunk_or_placeholder(self, array_name, slices, dtype, dryrun=False):
        """Get chunk from the store but return a :class:`PlaceholderChunk` if it is missing."""
        if not dryrun:
            try:
                return self.get_chunk(array_name, slices, dtype)
            except ChunkNotFound:
                pass
        chunk_name, shape = self.chunk_metadata(array_name, slices)
        return PlaceholderChunk(shape, dtype, chunk_name)

    def create_array(self, array_name):
        """Create a new array if it does not already exist.

        Parameters
        ----------
        array_name : string
            Identifier of array

        Raises
        ------
        :exc:`chunkstore.StoreUnavailable`
            If interaction with chunk store failed (offline, bad auth, bad config)
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
        :exc:`TypeError`
            If `slices` is not a sequence of slice(start, stop, 1) objects
        :exc:`chunkstore.BadChunk`
            If the shape implied by `slices` does not match that of `chunk`
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

    def mark_complete(self, array_name):
        """Write a special object to indicate that `array_name` is finished.

        This operation is idempotent.

        The `array_name` need not correspond to any array written with
        :meth:`put_chunk`. This has no effect on katdal, but a producer can
        call this method to provide a hint to a consumer that no further data
        will be coming for this array. When arrays are arranged in a hierarchy,
        a producer and consumer may agree to write a single completion marker
        at a higher level of the hierarchy rather than one per actual array.

        It is not necessary to call :meth:`create_array` first; the
        implementation will do so if appropriate.

        The presence of this marker can be checked with :meth:`is_complete`.
        """
        raise NotImplementedError

    def is_complete(self, array_name):
        """Check whether :meth:`mark_complete` has been called for this array."""
        raise NotImplementedError

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
    def chunk_id_str(cls, slices):
        """Chunk identifier in string form (e.g. '00012_01024_00000')."""
        return '_'.join("{:0{w}d}".format(s.start, w=cls.NAME_INDEX_WIDTH)
                        for s in slices)

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
        :exc:`TypeError`
            If `slices` is not a sequence of slice(start, stop, 1) objects
        :exc:`chunkstore.BadChunk`
            If the shape implied by `slices` does not match that of `chunk`,
            or any dtype contains objects

        """
        try:
            shape = tuple(s.stop - s.start for s in slices)
        except (TypeError, AttributeError):
            raise TypeError(f'Array {array_name!r}: chunk ID should be a sequence '
                            f'of slice(start, stop) objects, not {slices}')
        # Verify that all slice strides are unity (i.e. it's a "simple" slice)
        if not all([s.step in (1, None) for s in slices]):
            raise TypeError(f'Array {array_name!r}: chunk ID {slices} contains non-unit strides')
        # Construct chunk name from array_name + slices
        chunk_name = cls.join(array_name, cls.chunk_id_str(slices))
        if chunk is not None and chunk.shape != shape:
            raise BadChunk(f'Chunk {chunk_name!r}: shape {shape} implied by slices '
                           f'does not match actual shape {chunk.shape}')
        if chunk is not None and chunk.dtype.hasobject:
            raise BadChunk(f'Chunk {chunk_name!r}: actual dtype {chunk.dtype} '
                           'cannot contain objects')
        if dtype is not None and np.dtype(dtype).hasobject:
            raise BadChunk(f'Chunk {chunk_name!r}: Requested dtype {dtype} '
                           'cannot contain objects')
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
            prefix = f'Chunk {chunk_name!r}: ' if chunk_name else ''
            raise StandardisedError(prefix + str(e)) from e

    def get_dask_array(self, array_name, chunks, dtype, offset=(), index=(), errors=0):
        """Get dask array from the store.

        Handling of missing chunks is determined by the `errors` argument.

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
        index : tuple of unit-step slices, optional
            Index expression that selects a subset of existing chunks. The dask
            array will only ever contain those chunks, making it more efficient
            than a full array followed by standard selection.
        errors : number or 'raise' or 'placeholder' or 'dryrun', optional
            Error handling. If 'raise', exceptions are passed through,
            causing the evaluation to fail.

            If 'placeholder', returns instances of :class:`PlaceholderChunk`
            in place of missing chunks. Note that such an array cannot be used
            as-is, because an ndarray is expected, but it can be used as raw
            material for building new graphs via functions like
            :func:`da.map_blocks`.

            If 'dryrun', pretend that all chunks are missing and return
            :class:`PlaceholderChunk`s instead. This is useful for identifying
            the chunks to be downloaded while avoiding any data transfers.

            If a numeric value, it is used as a default value.

        Returns
        -------
        array : :class:`dask.array.Array` object
            Dask array of given dtype
        """
        getter_kwargs = {}
        if errors in ('placeholder', 'dryrun'):
            getter = self.get_chunk_or_placeholder
            getter_kwargs['dryrun'] = errors == 'dryrun'
        elif errors == 'raise':
            getter = self.get_chunk
        elif isinstance(errors, str):
            raise ValueError(
                "Unexpected value for errors; expected "
                "'placeholder', 'raise', 'dryrun' or a number"
            )
        else:
            getter = self.get_chunk_or_default
            getter_kwargs['default_value'] = errors
        if index:
            # XXX For now, _prune_chunks can't handle non-zero offsets as input...
            # This is not a big deal, since offsets are really meant for put_dask_array
            # and even then not used much, while `index` only applies to get_dask_array.
            assert offset == ()
            chunks, index, offset = _prune_chunks(chunks, index)
        if any(offset):
            getter = _add_offset_to_slices(getter, offset)
        # The final name must differ from array_name, so pick deterministic hash
        token = da.core.tokenize(self, chunks, dtype, index)
        out_name = f'{array_name}-{offset}-{token}'
        getter_shim = _ArrayLikeGetter(getter, array_name, chunks, dtype, **getter_kwargs)
        from_array_kwargs = {}
        # XXX Once we depend on dask>=2021.1.0, just set the keyword argument explicitly
        if dask.utils.has_keyword(da.from_array, 'inline_array'):
            # Dask 2022.04.0 ignores the custom getter unless this is True
            from_array_kwargs['inline_array'] = True
        array = da.from_array(
            getter_shim,
            chunks,
            out_name,
            # Don't cast chunks to ndarray because some of them may be `PlaceholderChunk`s
            asarray=False,
            # Set custom getter anyway to prevent slice fusion during graph optimisation,
            # which ends up calling chunk store with wrong chunks.
            getitem=_ArrayLikeGetter.__getitem__,
            # Provide explicit meta, otherwise dask downloads a chunk just to check type
            meta=np.empty(shape=(0,) * getter_shim.ndim, dtype=getter_shim.dtype),
            **from_array_kwargs,
        )
        return array[index]

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
        return da.map_blocks(
            _put_map_blocks,
            array,
            name=f'store-{array_name}-{offset}',
            dtype=object,
            chunks=array.ndim * (1,),
            meta=np.empty(shape=(0,) * array.ndim, dtype=object),
            # kwargs for _put_map_blocks
            store=self,
            array_name=array_name,
            offset=offset,
        )
