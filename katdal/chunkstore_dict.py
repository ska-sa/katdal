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

from .chunkstore import ChunkStore


class DictChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on a dict of arrays.

    This interprets all keyword arguments as NumPy arrays and stores them in
    an `arrays` dict. Each array is identified by its corresponding keyword.
    """

    def __init__(self, **kwargs):
        self.arrays = kwargs

    def get(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        try:
            chunk = self.arrays[array_name][slices]
        except KeyError:
            raise OSError('Array {!r} not found in DictChunkStore which has {}'
                          .format(array_name, self.arrays.keys()))
        if dtype != chunk.dtype:
            raise ValueError('Chunk {!r}: requested dtype {} differs from '
                             'actual dtype {}'
                             .format(chunk_name, dtype, chunk.dtype))
        return chunk

    def put(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put`."""
        self.chunk_metadata(array_name, slices, chunk=chunk)
        self.get(array_name, slices, chunk.dtype)[:] = chunk

    get.__doc__ = ChunkStore.get.__doc__
    put.__doc__ = ChunkStore.put.__doc__
