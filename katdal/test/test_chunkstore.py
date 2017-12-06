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

"""Tests for :py:mod:`katdal.chunkstore`."""

import numpy as np
from nose.tools import assert_raises

from katdal.chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound


class TestChunkStore(object):

    def test_get_put(self):
        store = ChunkStore()
        assert_raises(NotImplementedError, store.get, 1, 2, 3)
        assert_raises(NotImplementedError, store.put, 1, 2, 3)

    def test_metadata_validation(self):
        store = ChunkStore()
        # Bad slice specifications
        assert_raises(ValueError, store.chunk_metadata, "x", 3)
        assert_raises(ValueError, store.chunk_metadata, "x", [3, 2])
        assert_raises(ValueError, store.chunk_metadata, "x", slice(10))
        assert_raises(ValueError, store.chunk_metadata, "x", [slice(10)])
        assert_raises(ValueError, store.chunk_metadata, "x", [slice(0, 10, 2)])
        # Chunk mismatch
        assert_raises(ValueError, store.chunk_metadata, "x", [slice(0, 10, 1)],
                      chunk=np.ones(11))
        # Bad dtype
        assert_raises(ValueError, store.chunk_metadata, "x", [slice(0, 10, 1)],
                      chunk=np.array(10 * [{}]))
        assert_raises(ValueError, store.chunk_metadata, "x", [slice(0, 2)],
                      dtype=np.dtype(np.object))

    def test_standard_errors(self):
        error_map = {ZeroDivisionError: StoreUnavailable,
                     KeyError: ChunkNotFound}
        store = ChunkStore(error_map)
        with assert_raises(StoreUnavailable):
            with store._standard_errors():
                1 / 0
        with assert_raises(ChunkNotFound):
            with store._standard_errors():
                {}['ha']
