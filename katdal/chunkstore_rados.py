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

"""A store of chunks (i.e. N-dimensional arrays) based on the Ceph RADOS API."""
from __future__ import print_function, division, absolute_import

import numpy as np
try:
    import rados
    _rados_import_error = None
except ImportError as e:
    # librados (i.e. direct Ceph access) is only really available on Linux
    rados = None
    _rados_import_error = e

from .chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk


class RadosChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Ceph RADOS API.

    The full identifier of each chunk (the "chunk name"), which is also the
    object key for the RADOS interface, is given by

     "<array>/<idx>"

    where "<array>" is the name of the parent array of the chunk and "<idx>" is
    the index string of each chunk (e.g. "00001_00512").

    Parameters
    ----------
    ioctx : :class:`rados.Ioctx` object
        RADOS input/output context (ioctx) used to read from / write to Ceph

    Raises
    ------
    ImportError
        If rados is not installed (it's an optional dependency otherwise)
    """

    def __init__(self, ioctx):
        if not rados:
            raise _rados_import_error
        # From now on, ObjectNotFound refers to RADOS objects i.e. chunks
        error_map = {rados.TimedOut: StoreUnavailable,
                     rados.ObjectNotFound: ChunkNotFound}
        super(RadosChunkStore, self).__init__(error_map)
        self.ioctx = ioctx

    @classmethod
    def from_config(cls, config, pool, keyring=None, timeout=5.):
        """Construct RADOS chunk store from config and specified pool.

        Parameters
        ----------
        config : string or dict
            Path to Ceph config file or config dict
        pool : string
            Name of the Ceph pool
        keyring : string, optional
            Path to client keyring file (if not provided by `conf` or override)
        timeout : float, optional
            RADOS client timeout, in seconds (set to None to leave unchanged)

        Raises
        ------
        ImportError
            If rados is not installed (it's an optional dependency otherwise)
        :exc:`chunkstore.StoreUnavailable`
            If connection to Ceph cluster failed or pool is not available
        """
        if not rados:
            raise ImportError('Please install rados for katdal RADOS support')
        try:
            if isinstance(config, dict):
                cluster = rados.Rados(conf=config)
            else:
                cluster = rados.Rados(conffile=config)
            if keyring:
                cluster.conf_set('keyring', keyring)
            if timeout is not None:
                cluster.conf_set('client mount timeout', str(timeout))
            cluster.connect()
            ioctx = cluster.open_ioctx(pool)
        # A missing config file or pool also triggers ObjectNotFound
        except (rados.TimedOut, rados.ObjectNotFound) as e:
            raise StoreUnavailable(str(e))
        return cls(ioctx)

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        dtype = np.dtype(dtype)
        key, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        with self._standard_errors(key):
            # Try to read an extra byte to see if data is more than expected
            data_str = self.ioctx.read(key, expected_bytes + 1)
        actual_bytes = len(data_str)
        if actual_bytes != expected_bytes:
            # Get the actual value via stat() to improve error reporting
            if actual_bytes > expected_bytes:
                with self._standard_errors(key):
                    actual_bytes, _ = self.ioctx.stat(key)
            raise BadChunk('Chunk {!r}: dtype {} and shape {} implies an '
                           'object size of {} bytes, got {} bytes instead'
                           .format(key, dtype, shape, expected_bytes,
                                   actual_bytes))
        return np.ndarray(shape, dtype, data_str)

    def create_array(self, array_name):
        pass

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        key, _ = self.chunk_metadata(array_name, slices, chunk=chunk)
        data_str = chunk.tobytes()
        with self._standard_errors(key):
            self.ioctx.write_full(key, data_str)

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        dtype = np.dtype(dtype)
        key, _ = self.chunk_metadata(array_name, slices, dtype=dtype)
        try:
            with self._standard_errors(key):
                self.ioctx.stat(key)
        except ChunkNotFound:
            return False
        else:
            return True

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
