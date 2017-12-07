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
from numpy.testing import assert_array_equal
from nose.tools import assert_raises

from katdal.chunkstore import (ChunkStore, StoreUnavailable,
                               ChunkNotFound, BadChunk)


class TestChunkStore(object):
    """This tests the base class functionality."""

    def test_put_and_get(self):
        store = ChunkStore()
        assert_raises(NotImplementedError, store.get, 1, 2, 3)
        assert_raises(NotImplementedError, store.put, 1, 2, 3)

    def test_metadata_validation(self):
        store = ChunkStore()
        # Bad slice specifications
        assert_raises(BadChunk, store.chunk_metadata, "x", 3)
        assert_raises(BadChunk, store.chunk_metadata, "x", [3, 2])
        assert_raises(BadChunk, store.chunk_metadata, "x", slice(10))
        assert_raises(BadChunk, store.chunk_metadata, "x", [slice(10)])
        assert_raises(BadChunk, store.chunk_metadata, "x", [slice(0, 10, 2)])
        # Chunk mismatch
        assert_raises(BadChunk, store.chunk_metadata, "x", [slice(0, 10, 1)],
                      chunk=np.ones(11))
        # Bad dtype
        assert_raises(BadChunk, store.chunk_metadata, "x", [slice(0, 10, 1)],
                      chunk=np.array(10 * [{}]))
        assert_raises(BadChunk, store.chunk_metadata, "x", [slice(0, 2)],
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


class ChunkStoreTestBase(object):
    """Standard test performed on all types of ChunkStore.

    Put everything in a single test as setup and teardown can be quite costly.
    """

    def __init__(self):
        # Pick arrays with differently sized dtypes and dimensions
        self.x = np.ones(10, dtype=np.bool)
        self.y = np.arange(96.).reshape(8, 6, 2)

    def array_name(self, name):
        return name

    def test_put_and_get(self):
        # Look for non-existent chunk
        assert_raises(ChunkNotFound, self.store.get, 'haha',
                      (slice(0, 1),), np.dtype(np.float))
        # Check basic put + get on 1-D bool
        s = (slice(3, 5),)
        desired = self.x[s]
        name = self.array_name('x')
        self.store.put(name, s, desired)
        actual = self.store.get(name, s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing x[%s]" % (s,))
        # Stored object has fewer bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get, name, s, self.y.dtype)
        # Check basic put + get on 3-D float
        s = (slice(3, 7), slice(2, 5), slice(1, 2))
        desired = self.y[s]
        name = self.array_name('y')
        self.store.put(name, s, desired)
        actual = self.store.get(name, s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing y[%s]" % (s,))
        # Stored object has more bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get, name, s, self.x.dtype)
