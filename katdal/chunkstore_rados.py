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

"""A store of chunks (i.e. N-dimensional arrays) based on the Ceph RADOS API."""

import contextlib

import numpy as np
try:
    import rados
except ImportError:
    # librados (i.e. direct Ceph access) is only really available on Linux
    rados = None

from .chunkstore import ChunkStore


@contextlib.contextmanager
def _convert_rados_errors():
    try:
        yield
    except rados.ObjectNotFound as e:
        raise OSError(e)


class RadosChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Ceph RADOS API.

    The full identifier of each chunk (the "chunk name"), which is also the
    object key for the RADOS interface, is given by

     "<array>/<idx>"

    where "<array>" is the name of the parent array of the chunk and "<idx>" is
    the index string of each chunk (e.g. "00001_00512").

    Parameters
    ----------
    conf : string or dict
        Path to the Ceph configuration file or config dict version of that file
    pool : string
        Name of the Ceph pool
    keyring : string, optional
        Path to the client keyring file (if not provided by `conf` or override)

    Raises
    ------
    ImportError
        If rados is not installed (it's an optional dependency otherwise)
    OSError
        If connection to Ceph cluster failed or pool is not available
    """

    def __init__(self, conf, pool, keyring=None):
        if not rados:
            raise ImportError('Please install rados for katdal RADOS support')
        if isinstance(conf, dict):
            cluster = rados.Rados(conf=conf)
        else:
            cluster = rados.Rados(conffile=conf)
        if keyring:
            cluster.conf_set('keyring', keyring)
        with _convert_rados_errors():
            cluster.connect()
            self.ioctx = cluster.open_ioctx(pool)

    def get(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get`."""
        shape = tuple([s.stop - s.start for s in slices])
        key = ChunkStore.chunk_name(array_name, slices)
        num_bytes = np.prod(shape) * dtype.itemsize
        with _convert_rados_errors():
            data_str = self.ioctx.read(key, num_bytes)
        return np.ndarray(shape, dtype, data_str)

    def put(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put`."""
        key = ChunkStore.chunk_name(array_name, slices)
        data_str = chunk.tobytes()
        with _convert_rados_errors():
            self.ioctx.write_full(key, data_str)

    get.__doc__ = ChunkStore.get.__doc__
    put.__doc__ = ChunkStore.put.__doc__
