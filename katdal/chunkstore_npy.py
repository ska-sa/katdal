################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
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

import contextlib
import errno
import mmap
import os

import numpy as np

from .chunkstore import (BadChunk, ChunkNotFound, ChunkStore, StoreUnavailable,
                         npy_header_and_body)


def _write_chunk(filename, chunk, direct_write):
    if not direct_write:
        return np.save(filename, chunk, allow_pickle=False)
    header, chunk = npy_header_and_body(chunk)
    size = len(header) + chunk.nbytes
    gran = mmap.ALLOCATIONGRANULARITY
    aligned_size = (size + gran - 1) // gran * gran
    with contextlib.closing(mmap.mmap(-1, aligned_size)) as aligned:
        aligned.write(header)
        aligned.write(chunk)
        aligned.seek(0)
        fd = os.open(filename, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o666)
        try:
            os.write(fd, aligned)
            # We had to round the size up to a page, now correct back to exact size
            os.ftruncate(fd, size)
        finally:
            os.close(fd)


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
    direct_write : bool
        If true, use ``O_DIRECT`` when writing the file. This bypasses the
        OS page cache, which can be useful to avoid filling it up with
        files that won't be read again.

    Raises
    ------
    :exc:`chunkstore.StoreUnavailable`
        If path does not exist / is not readable
    :exc:`chunkstore.StoreUnavailable`
        If `direct_write` was requested but is not available
    """

    def __init__(self, path, direct_write=False):
        super().__init__({IOError: ChunkNotFound, ValueError: ChunkNotFound})
        if not os.path.isdir(path):
            raise StoreUnavailable(f'Directory {path!r} does not exist')
        self.path = path
        self.direct_write = direct_write
        if direct_write and not hasattr(os, 'O_DIRECT'):
            raise StoreUnavailable('direct_write requested but not supported on this OS')

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        filename = os.path.join(self.path, chunk_name) + '.npy'
        with self._standard_errors(chunk_name):
            chunk = np.load(filename, allow_pickle=False)
        if chunk.shape != shape or chunk.dtype != dtype:
            raise BadChunk(f'Chunk {chunk_name!r}: NPY file dtype {chunk.dtype} and/or shape '
                           f'{chunk.shape} differs from expected dtype {dtype} and shape {shape}')
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
            _write_chunk(temp_filename, chunk, self.direct_write)
            os.rename(temp_filename, base_filename + '.npy')

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
    mark_complete.__doc__ = ChunkStore.mark_complete.__doc__
    is_complete.__doc__ = ChunkStore.is_complete.__doc__
