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

import numpy as np
try:
    import rados
except ImportError:
    # librados (i.e. direct Ceph access) is only really available on Linux
    rados = None

from .chunkstore import ChunkStore


class RadosChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Ceph RADOS API.

    The full identifier of each chunk (the "chunk name"), which is also the
    object key for the RADOS interface, is given by

     "<array>/<idx>"

    where "<array>" is the name of the parent array of the chunk and "<idx>" is
    the index string of each chunk (e.g. "00001_00512").

    Parameters
    ----------
    ceph_conf : string
        Path to the ceph.conf config file used to connect to target Ceph cluster
    ceph_pool : string
        Name of the Ceph pool

    Raises
    ------
    ImportError
        If rados is not installed (it's an optional dependency otherwise)
    IOError
        If requested Ceph pool is not available in cluster
    """

    def __init__(self, ceph_conf, ceph_pool):
        if not rados:
            raise ImportError('Please install rados for katdal RADOS support')
        cluster = rados.Rados(conffile=ceph_conf)
        cluster.connect()
        available_pools = cluster.list_pools()
        if ceph_pool not in available_pools:
            raise IOError("Specified pool %s not available in this cluster (%s)"
                          % (ceph_pool, available_pools))
        self.ioctx = cluster.open_ioctx(ceph_pool)

    def get(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get`."""
        shape = tuple([s.stop - s.start for s in slices])
        key = ChunkStore.chunk_name(array_name, slices)
        num_bytes = np.prod(shape) * dtype.itemsize
        data_str = self.ioctx.read(key, num_bytes)
        return np.ndarray(shape, dtype, data_str)

    def put(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put`."""
        key = ChunkStore.chunk_name(array_name, slices)
        data_str = chunk.tobytes()
        self.ioctx.write_full(key, data_str)

    get.__doc__ = ChunkStore.get.__doc__
    put.__doc__ = ChunkStore.put.__doc__
