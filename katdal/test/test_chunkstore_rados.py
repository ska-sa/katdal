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
from __future__ import print_function, division, absolute_import

from nose import SkipTest
from nose.tools import assert_raises

import katdal.chunkstore_rados
from katdal.chunkstore_rados import RadosChunkStore, rados
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


class TestRadosChunkStore(ChunkStoreTestBase):
    """Test Ceph functionality by connecting to an actual RADOS service."""

    @classmethod
    def setup_class(cls):
        # Look for default Ceph installation but expect a special test pool
        config = '/etc/ceph/ceph.conf'
        pool = 'test_katdal'
        try:
            cls.store = RadosChunkStore.from_config(config, pool)
        except (ImportError, StoreUnavailable):
            raise SkipTest('Rados not installed or cluster misconfigured / down')

    def array_name(self, path):
        namespace = 'katdal_test_chunkstore_rados'
        return self.store.join(namespace, path)

    def test_store_unavailable(self):
        # Pretend that rados is not installed (make sure to restore it)
        katdal.chunkstore_rados.rados = None
        katdal.chunkstore_rados._rados_import_error = ImportError()
        try:
            assert_raises(ImportError, RadosChunkStore, None)
        finally:
            katdal.chunkstore_rados.rados = rados
            katdal.chunkstore_rados._rados_import_error = None
        # Missing config file
        assert_raises(StoreUnavailable, RadosChunkStore.from_config, 'x', 'y')
        # Bad config (unknown config variable)
        assert_raises(StoreUnavailable, RadosChunkStore.from_config,
                      {'mon_hos': ''}, 'y')
        # Config OK but host not found (use Test-Net IP from RFC5737)
        assert_raises(StoreUnavailable, RadosChunkStore.from_config,
                      {'mon_host': '192.0.2.1'}, 'y', 'z', timeout=0.1)
