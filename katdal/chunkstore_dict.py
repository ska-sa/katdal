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

"""A store of chunks (i.e. N-dimensional arrays) based on a dict of arrays."""
from __future__ import print_function, division, absolute_import

from .chunkstore import ChunkStore, ChunkNotFound, BadChunk


class DictChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on a dict of arrays.

    This interprets all keyword arguments as NumPy arrays and stores them in
    an `arrays` dict. Each array is identified by its corresponding keyword.
    New arrays cannot be added via :meth:`put` - they all need to be in place
    at store initialisation (or can be added afterwards via direct insertion
    into the `arrays` dict). The `put` method is only useful for in-place
    modification of existing arrays.
    """

    def __init__(self, **kwargs):
        error_map = {KeyError: ChunkNotFound, IndexError: ChunkNotFound}
        super(DictChunkStore, self).__init__(error_map)
        self.arrays = kwargs

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        with self._standard_errors(chunk_name):
            array = self.arrays[array_name]
            # Ensure that chunk is array (otherwise 0-dim array becomes number)
            chunk = array[slices] if slices != () else array
        if chunk.shape != shape or chunk.dtype != dtype:
            raise BadChunk('Chunk {!r}: requested dtype {} and/or shape {} '
                           'differs from expected dtype {} and shape {}'
                           .format(chunk_name, chunk.dtype, chunk.shape,
                                   dtype, shape))
        return chunk

    def create_array(self, array_name):
        if array_name not in self.arrays:
            raise NotImplementedError

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        self.chunk_metadata(array_name, slices, chunk=chunk)
        self.get_chunk(array_name, slices, chunk.dtype)[()] = chunk

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
