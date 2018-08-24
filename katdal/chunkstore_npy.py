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

"""A store of chunks (i.e. N-dimensional arrays) based on NPY files."""
from __future__ import print_function, division, absolute_import

import os
import errno

import numpy as np

from .chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk


class NpyFileChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on NPY files.

    Each chunk is stored in a separate binary file in NumPy ``.npy`` format.
    The filename is constructed as

      "<path>/<array>/<idx>.npy"

    where "<path>" is the chunk store directory specified on construction,
    "<array>" is the name of the parent array of the chunk and "<idx>" is
    the index string of each chunk (e.g. "00001_00512").

    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`
    or the relevant NumPy Enhancement Proposal
    `here <http://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.

    Parameters
    ----------
    path : string
        Top-level directory that contains NPY files of chunk store

    Raises
    ------
    :exc:`chunkstore.StoreUnavailable`
        If path does not exist / is not readable
    """

    def __init__(self, path):
        super(NpyFileChunkStore, self).__init__({IOError: ChunkNotFound,
                                                 ValueError: ChunkNotFound})
        if not os.path.isdir(path):
            raise StoreUnavailable('Directory {!r} does not exist'.format(path))
        self.path = path

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        with self._standard_errors(chunk_name):
            chunk = np.load(filename, allow_pickle=False)
        if chunk.shape != shape or chunk.dtype != dtype:
            raise BadChunk('Chunk {!r}: NPY file dtype {} and/or shape {} '
                           'differs from expected dtype {} and shape {}'
                           .format(chunk_name, chunk.dtype, chunk.shape,
                                   dtype, shape))
        return chunk

    def create_array(self, array_name):
        """See the docstring of :meth:`ChunkStore.create_array`."""
        # Ensure any subdirectories are in place
        array_dir = os.path.join(self.path, array_name)
        try:
            os.makedirs(array_dir)
        except OSError as e:
            # Be happy if someone already created the path
            if e.errno != errno.EEXIST:
                raise

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        chunk_name, _ = self.chunk_metadata(array_name, slices, chunk=chunk)
        base_filename = os.path.join(self.path, chunk_name)
        with self._standard_errors(chunk_name):
            # Rename the file when done writing to make put_chunk() atomic
            temp_filename = base_filename + '.writing.npy'
            np.save(temp_filename, chunk, allow_pickle=False)
            os.rename(temp_filename, base_filename + '.npy')

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        chunk_name, _ = self.chunk_metadata(array_name, slices, dtype=dtype)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        return os.path.exists(filename)

    def list_chunk_ids(self, array_name):
        """See the docstring of :meth:`ChunkStore.list_chunk_ids`."""
        array_dir = os.path.join(self.path, array_name)
        # Strip the .npy extension to get the chunk ID string
        try:
            return [fn[:-4] for fn in os.listdir(array_dir) if fn.endswith('.npy')]
        except OSError as e:
            # If the directory is missing, there cannot be any objects
            if e.errno != errno.ENOENT:
                raise
            return []

    def mark_complete(self, array_name):
        """See the docstring of :meth:`ChunkStore.mark_complete`."""
        self.create_array(array_name)
        touch_file = os.path.join(self.path, array_name, 'complete')
        with open(touch_file, 'a'):
            os.utime(touch_file, None)

    def is_complete(self, array_name):
        """See the docstring of :meth:`ChunkStore.is_complete`."""
        touch_file = os.path.join(self.path, array_name, 'complete')
        return os.path.isfile(touch_file)

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
    list_chunk_ids.__doc__ = ChunkStore.list_chunk_ids.__doc__
    mark_complete.__doc__ = ChunkStore.mark_complete.__doc__
    is_complete.__doc__ = ChunkStore.is_complete.__doc__
