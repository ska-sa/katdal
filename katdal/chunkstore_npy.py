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

import os

import numpy as np

from .chunkstore import ChunkStore


class NpyFileChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on NPY files.

    Each chunk is stored in a separate binary file in NumPy ``.npy`` format.
    The filename is constructed as

      "<path>/<array>/<idx>.npy"

    where "<path>" is the chunk store directory specified on construction,
    "<array>" is the name of the parent array of the chunk and "<idx>" is
    the index string of each chunk (e.g. "00001_00512").

    For a description of the ``.npy`` format, see the module docstring
    of `numpy.lib.format` or the NumPy Enhancement Proposal
    http://docs.scipy.org/doc/numpy/neps/npy-format.html

    Parameters
    ----------
    path : string
        Top-level directory that contains NPY files of chunk store

    Raises
    ------
    OSError
        If path does not exist / is not readable
    """

    def __init__(self, path):
        if not os.path.isdir(path):
            raise OSError('Directory %r does not exist' % (path,))
        self.path = path

    def get(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get`."""
        chunk_name = ChunkStore.chunk_name(array_name, slices)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        try:
            chunk = np.load(filename, allow_pickle=False)
        except IOError as e:
            raise OSError(e)
        if dtype != chunk.dtype:
            raise ValueError('Requested dtype %s differs from NPY file dtype %s'
                             % (dtype, chunk.dtype))
        return chunk

    def put(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put`."""
        chunk_name = ChunkStore.chunk_name(array_name, slices)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        # Ensure any subdirectories are in place
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as e:
            # Be happy if someone already created the path
            if e.errno != os.errno.EEXIST:
                raise
        try:
            np.save(filename, chunk, allow_pickle=False)
        except IOError as e:
            raise OSError(e)

    get.__doc__ = ChunkStore.get.__doc__
    put.__doc__ = ChunkStore.put.__doc__
