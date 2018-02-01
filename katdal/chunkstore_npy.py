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

import os

import numpy as np

from .chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk


def load_npy_header(filename):
    """Load NPY file header only to obtain shape and dtype information."""
    with open(filename, 'r') as fp:
        # Header version: either (1, 0) or (2, 0)
        ver = np.lib.format.read_magic(fp)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fp, ver)
    return shape, fortran_order, dtype


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
        if dtype != chunk.dtype:
            raise BadChunk('Requested dtype {} differs from NPY file dtype {}'
                           .format(dtype, chunk.dtype))
        return chunk

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, chunk=chunk)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        # Ensure any subdirectories are in place
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as e:
            # Be happy if someone already created the path
            if e.errno != os.errno.EEXIST:
                raise
        with self._standard_errors(chunk_name):
            np.save(filename, chunk, allow_pickle=False)

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        try:
            with self._standard_errors(chunk_name):
                npy_shape, _, npy_dtype = load_npy_header(filename)
        except ChunkNotFound:
            return False
        else:
            return npy_dtype == dtype and npy_shape == shape

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
