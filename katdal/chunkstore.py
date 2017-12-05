################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
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
    """

    def get(self, array_name, slices, dtype):
        """Get chunk from the store.

        Parameters
        ----------
        array_name : string
            Identifier of parent array `x` of chunk
        slices : sequence of unit-stride slice objects
            Identifier of individual chunk, to be extracted as `x[slices]`
        dtype : :class:`numpy.dtype` object
            Data type of array `x`

        Returns
        -------
        chunk : :class:`numpy.ndarray` object
            Chunk as ndarray with dtype `dtype` and shape dictated by `slices`

        Raises
        ------
        ValueError
            If requested `dtype` does not match underlying parent array dtype
            or `slices` has wrong specification
        OSError
            If requested chunk was not found in store (or connection failed)
        """
        raise NotImplementedError

    def put(self, array_name, slices, chunk):
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
        ValueError
            If `slices` is wrong or its shape does not match that of `chunk`
        OSError
            If connection to store failed (or `array_name` is incompatible)
        """
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
        dtype : :class:`numpy.dtype` object, optional
            Data type of array `x` (used for validation only)

        Returns
        -------
        chunk_name : string
            Full chunk name used to find chunk in underlying storage medium
        shape : tuple of int
            Chunk shape tuple associated with `slices`

        Raises
        ------
        ValueError
            If `slices` has wrong specification or its shape does not match
            that of `chunk`, or any dtype contains objects

        """
        try:
            shape = tuple([s.stop - s.start for s in slices])
        except (TypeError, AttributeError):
            raise ValueError('Array {!r}: chunk ID should be a sequence of '
                             'slice objects, not {}'.format(array_name, slices))
        # Verify that all slice strides are unity (i.e. it's a "simple" slice)
        if not all([s.step in (1, None) for s in slices]):
            raise ValueError('Array {!r}: chunk ID {} contains non-unit strides'
                             .format(array_name, slices))
        # Construct chunk name from array_name + slices
        index = [s.start for s in slices]
        idxstr = '_'.join(["{:0{width}d}".format(i, width=cls.NAME_INDEX_WIDTH)
                           for i in index])
        chunk_name = cls.join(array_name, idxstr)
        if chunk is not None and chunk.shape != shape:
            raise ValueError('Chunk {!r}: shape {} implied by slices does not '
                             'match actual shape {}'
                             .format(chunk_name, shape, chunk.shape))
        if chunk is not None and chunk.dtype.hasobject:
            raise ValueError('Chunk {!r}: actual dtype {} cannot contain '
                             'objects'.format(chunk_name, chunk.dtype))
        if dtype is not None and dtype.hasobject:
            raise ValueError('Chunk {!r}: Requested dtype {} cannot contain '
                             'objects'.format(chunk_name, dtype))
        return chunk_name, shape
