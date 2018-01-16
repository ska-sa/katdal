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

    def test_put_and_get_chunk(self):
        store = ChunkStore()
        assert_raises(NotImplementedError, store.get_chunk, 1, 2, 3)
        assert_raises(NotImplementedError, store.put_chunk, 1, 2, 3)

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
        assert_raises(BadChunk, store.chunk_metadata, "x", (), chunk=np.ones(5))
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
    """Standard tests performed on all types of ChunkStore."""

    # Instance of store instantiated once per class via class-level fixture
    store = None

    def __init__(self):
        # Pick arrays with differently sized dtypes and dimensions
        self.x = np.ones(10, dtype=np.bool)
        self.y = np.arange(96.).reshape(8, 6, 2)
        self.z = np.array(2.)

    def array_name(self, name):
        return name

    def put_and_get_chunk(self, var_name, slices):
        array_name = self.array_name(var_name)
        chunk = getattr(self, var_name)[slices]
        self.store.put_chunk(array_name, slices, chunk)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk,
                           "Error storing {}[{}]".format(var_name, slices))

    def test_chunk_non_existent(self):
        assert_raises(ChunkNotFound, self.store.get_chunk, 'haha',
                      (slice(0, 1),), np.dtype(np.float))

    def test_chunk_bool_1dim_and_too_small(self):
        # Check basic put + get on 1-D bool
        name = self.array_name('x')
        s = (slice(3, 5),)
        self.put_and_get_chunk('x', s)
        # Stored object has fewer bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get_chunk, name, s, self.y.dtype)

    def test_chunk_float_3dim_and_too_large(self):
        # Check basic put + get on 3-D float
        name = self.array_name('y')
        s = (slice(3, 7), slice(2, 5), slice(1, 2))
        self.put_and_get_chunk('y', s)
        # Stored object has more bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get_chunk, name, s, self.x.dtype)

    def test_chunk_zero_size(self):
        # Try a chunk with zero size
        self.put_and_get_chunk('y', (slice(4, 7), slice(3, 3), slice(0, 2)))
        # Try an empty slice on a zero-dimensional array (but why?)
        self.put_and_get_chunk('z', ())
