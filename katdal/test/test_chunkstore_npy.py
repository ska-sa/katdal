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

"""Tests for :py:mod:`katdal.chunkstore_npy`."""
from __future__ import print_function, division, absolute_import

import tempfile
import shutil

from nose.tools import assert_raises

from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


class TestNpyFileChunkStore(ChunkStoreTestBase):
    """Test NPY file functionality using a temporary directory."""

    @classmethod
    def setup_class(cls):
        """Create temp dir to store NPY files and build ChunkStore on that."""
        cls.tempdir = tempfile.mkdtemp()
        cls.store = NpyFileChunkStore(cls.tempdir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tempdir)

    def test_store_unavailable(self):
        assert_raises(StoreUnavailable, NpyFileChunkStore, 'hahahahahaha')
