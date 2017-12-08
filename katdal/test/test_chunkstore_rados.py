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

"""Tests for :py:mod:`katdal.chunkstore_s3`."""

from nose import SkipTest
from nose.tools import assert_raises

import katdal.chunkstore_rados
from katdal.chunkstore_rados import RadosChunkStore, rados
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


class TestRadosChunkStore(ChunkStoreTestBase):
    """Tests connecting to an actual RADOS service, implying a slower setup."""

    def setup(self):
        # Look for default Ceph installation but expect a special test pool
        config = '/etc/ceph/ceph.conf'
        pool = 'test_katdal'
        try:
            self.store = RadosChunkStore.from_config(config, pool)
        except (ImportError, StoreUnavailable):
            raise SkipTest('Rados not installed or cluster misconfigured / down')

    def array_name(self, path):
        namespace = 'katdal_test_chunkstore_rados'
        return self.store.join(namespace, path)


class TestDudRadosChunkStore(object):
    """Tests that don't need a RADOS connection, only a 'dud' store."""

    def setup(self):
        if not rados:
            raise SkipTest('Rados not installed')

    def test_store_unavailable(self):
        # Pretend that rados is not installed
        katdal.chunkstore_rados.rados = None
        assert_raises(ImportError, RadosChunkStore, None)
        katdal.chunkstore_rados.rados = rados
        assert_raises(StoreUnavailable, RadosChunkStore.from_config, 'x', 'y')
