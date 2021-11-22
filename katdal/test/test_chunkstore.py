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

"""Tests for :py:mod:`katdal.chunkstore`."""

import dask.array as da
import numpy as np
from nose.tools import (assert_equal, assert_false, assert_is_instance,
                        assert_raises, assert_true)
from numpy.testing import assert_array_equal

from katdal.chunkstore import (BadChunk, ChunkNotFound, ChunkStore,
                               PlaceholderChunk, StoreUnavailable,
                               generate_chunks)


class TestGenerateChunks:
    """Test the `generate_chunks` function."""
    def __init__(self):
        self.shape = (10, 8192, 144)
        self.dtype = np.complex64
        self.nbytes = np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def test_basic(self):
        # Basic check
        chunks = generate_chunks(self.shape, self.dtype, 3e6)
        assert_equal(chunks, (10 * (1,), 4 * (2048,), (144,)))
        # Uneven chunks in the final split
        chunks = generate_chunks(self.shape, self.dtype, 1e6)
        assert_equal(chunks, (10 * (1,), 10 * (819,) + (2,), (144,)))

    def test_corner_cases(self):
        # Corner case: don't select any dimensions to split -> one chunk
        chunks = generate_chunks(self.shape, self.dtype, 1e6, ())
        assert_equal(chunks, ((10,), (8192,), (144,)))
        # Corner case: all bytes results in one chunk
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes)
        assert_equal(chunks, ((10,), (8192,), (144,)))
        # Corner case: one byte less than the full size results in a split
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes - 1)
        assert_equal(chunks, ((5, 5), (8192,), (144,)))

    def test_power_of_two(self):
        # Check power_of_two
        chunks = generate_chunks(self.shape, self.dtype, 1e6,
                                 dims_to_split=[1], power_of_two=True)
        assert_equal(chunks, ((10,), 128 * (64,), (144,)))
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes / 16,
                                 dims_to_split=[1], power_of_two=True)
        assert_equal(chunks, ((10,), 16 * (512,), (144,)))
        # Check power_of_two when dimension is not a power-of-two itself
        shape = (10, 32768 - 2048, 144)
        chunks = generate_chunks(shape, self.dtype, self.nbytes / 10,
                                 dims_to_split=(0, 1), power_of_two=True)
        assert_equal(chunks, (10 * (1,), 3 * (8192,) + (6144,), (144,)))
        # Check swapping the order of dims_to_split
        chunks = generate_chunks(shape, self.dtype, self.nbytes / 10,
                                 dims_to_split=(1, 0), power_of_two=True)
        assert_equal(chunks, ((10,), 60 * (512,), (144,)))

    def test_max_dim_elements(self):
        chunks = generate_chunks(self.shape, self.dtype, 150000,
                                 dims_to_split=(0, 1), power_of_two=True,
                                 max_dim_elements={1: 50})
        assert_equal(chunks, ((4, 4, 2), 256 * (32,), (144,)))
        # Case where max_dim_elements forces chunks to be smaller than
        # max_chunk_size.
        chunks = generate_chunks(self.shape, self.dtype, 1e6,
                                 dims_to_split=(0, 1), power_of_two=True,
                                 max_dim_elements={0: 4, 1: 50})
        assert_equal(chunks, ((4, 4, 2), 256 * (32,), (144,)))

    def test_max_dim_elements_ignore(self):
        """Elements not in `dims_to_split` are ignored"""
        chunks = generate_chunks(self.shape, self.dtype, 150000,
                                 dims_to_split=(1, 17), power_of_two=True,
                                 max_dim_elements={0: 2, 1: 50})
        assert_equal(chunks, ((10,), 1024 * (8,), (144,)))


class TestChunkStore:
    """This tests the base class functionality."""

    def test_put_get(self):
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
                     LookupError: ChunkNotFound}
        store = ChunkStore(error_map)
        with assert_raises(StoreUnavailable):
            with store._standard_errors():
                1 / 0
        with assert_raises(ChunkNotFound):
            with store._standard_errors():
                {}['ha']


class ChunkStoreTestBase:
    """Standard tests performed on all types of ChunkStore."""

    # Instance of store instantiated once per class via class-level fixture
    store = None

    def __init__(self):
        # Pick arrays with differently sized dtypes and dimensions
        self.x = np.ones(10, dtype=np.bool)
        self.y = np.arange(96.).reshape(8, 6, 2)
        self.z = np.array(2.)
        self.big_y = np.arange(960.).reshape(8, 60, 2)
        self.big_y2 = np.arange(960).reshape(8, 60, 2)
        self.preloaded_chunks = False

    def array_name(self, name):
        return name

    def put_get_chunk(self, var_name, slices):
        """Put a single chunk into store, get it back and compare."""
        array_name = self.array_name(var_name)
        chunk = getattr(self, var_name)[slices]
        self.store.create_array(array_name)
        self.store.put_chunk(array_name, slices, chunk)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, f"Error storing {var_name}[{slices}]")

    def make_dask_array(self, var_name, slices=()):
        """Turn (part of) an existing ndarray into a dask array."""
        array_name = self.array_name(var_name)
        array = getattr(self, var_name)
        # The chunking is determined by full array to keep things predictable
        chunks = generate_chunks(array.shape, array.dtype, array.nbytes / 10.)
        dask_array = da.from_array(array, chunks)[slices]
        offset = tuple(s.start for s in slices)
        return array_name, dask_array, offset

    def put_dask_array(self, var_name, slices=()):
        """Put (part of) an array into store via dask."""
        array_name, dask_array, offset = self.make_dask_array(var_name, slices)
        self.store.create_array(array_name)
        push = self.store.put_dask_array(array_name, dask_array, offset)
        results = push.compute()
        divisions_per_dim = [len(c) for c in dask_array.chunks]
        assert_array_equal(results, np.full(divisions_per_dim, None))

    def get_dask_array(self, var_name, slices=()):
        """Get (part of) an array from store via dask and compare."""
        array_name, dask_array, offset = self.make_dask_array(var_name, slices)
        pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                         dask_array.dtype, offset)
        array_retrieved = pull.compute()
        array = dask_array.compute()
        assert_array_equal(array_retrieved, array,
                           f'Error retrieving {array_name} / {offset} / {dask_array.chunks}')

    def test_chunk_non_existent(self):
        array_name = self.array_name('haha')
        slices = (slice(0, 1),)
        dtype = np.dtype(np.float)
        args = (array_name, slices, dtype)
        shape = tuple(s.stop - s.start for s in slices)
        assert_raises(ChunkNotFound, self.store.get_chunk, *args)
        zeros = self.store.get_chunk_or_default(*args)
        assert_array_equal(zeros, np.zeros(shape, dtype))
        assert_equal(zeros.dtype, dtype)
        ones = self.store.get_chunk_or_default(*args, default_value=1)
        assert_array_equal(ones, np.ones(shape, dtype))
        placeholder = self.store.get_chunk_or_placeholder(*args)
        assert_is_instance(placeholder, PlaceholderChunk)
        assert_equal(placeholder.shape, shape)
        assert_equal(placeholder.dtype, dtype)

    def test_chunk_bool_1dim_and_too_small(self):
        # Check basic put + get on 1-D bool
        name = self.array_name('x')
        s = (slice(3, 5),)
        self.put_get_chunk('x', s)
        # Stored object has fewer bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get_chunk, name, s, self.y.dtype)

    def test_chunk_float_3dim_and_too_large(self):
        # Check basic put + get on 3-D float
        name = self.array_name('y')
        s = (slice(3, 7), slice(2, 5), slice(1, 2))
        self.put_get_chunk('y', s)
        # Stored object has more bytes than expected (and wrong dtype)
        assert_raises(BadChunk, self.store.get_chunk, name, s, self.x.dtype)

    def test_chunk_zero_size(self):
        # Try a chunk with zero size
        self.put_get_chunk('y', (slice(4, 7), slice(3, 3), slice(0, 2)))
        # Try an empty slice on a zero-dimensional array (but why?)
        self.put_get_chunk('z', ())

    def test_put_chunk_noraise(self):
        name = self.array_name('x')
        self.store.create_array(name)
        result = self.store.put_chunk_noraise(name, (1, 2), [])
        assert_is_instance(result, BadChunk)

    def test_dask_array_basic(self):
        self.put_dask_array('big_y')
        self.get_dask_array('big_y')
        self.get_dask_array('big_y', np.s_[0:3, 0:30, 0:2])

    @staticmethod
    def _placeholder_to_default(array, default):
        """Replace :class:`PlaceholderChunk`s in a dask array with a default value."""
        def map_blocks_func(chunk):
            if isinstance(chunk, PlaceholderChunk):
                return np.full(chunk.shape, default, chunk.dtype)
            else:
                return chunk

        return da.map_blocks(map_blocks_func, array)

    def test_dask_array_put_parts_get_whole(self):
        # Split big array into quarters along existing chunks and reassemble
        self.put_dask_array('big_y2', np.s_[0:3,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[3:8,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[0:3, 30:60, 0:2])
        # Before storing last quarter, check missing chunk handling
        if not self.preloaded_chunks:
            array_name, dask_array, offset = self.make_dask_array('big_y2')
            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors=17)
            array_retrieved = pull.compute()
            assert_equal(array_retrieved.shape, dask_array.shape)
            assert_equal(array_retrieved.dtype, dask_array.dtype)
            assert_array_equal(array_retrieved[np.s_[3:8, 30:60, 0:2]], 17,
                               f'Missing chunk in {array_name} not replaced by default value')

            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors='raise')
            with assert_raises(ChunkNotFound):
                pull.compute()

            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors='placeholder')
            # We can't compute pull directly, because placeholders aren't
            # numpy arrays. So we have to remap them.
            pull = self._placeholder_to_default(pull, 17)
            array_retrieved = pull.compute()
            assert_equal(array_retrieved.shape, dask_array.shape)
            assert_equal(array_retrieved.dtype, dask_array.dtype)
            assert_array_equal(array_retrieved[np.s_[3:8, 30:60, 0:2]], 17,
                               "Missing chunk in {} not replaced by default value"
                               .format(array_name))

        # Now store the last quarter and check that complete array is correct
        self.put_dask_array('big_y2', np.s_[3:8, 30:60, 0:2])
        self.get_dask_array('big_y2')

    def test_get_dask_array_index(self):
        # Load most but not all of the array, to test error handling
        self.put_dask_array('big_y2', np.s_[0:3,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[3:8,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[0:3, 30:60, 0:2])
        array_name, dask_array, offset = self.make_dask_array('big_y2')
        indices = [
            (),
            np.s_[:, :],
            np.s_[0:8, 0:60, 0:2],
            np.s_[..., 0:1],
            np.s_[5:6, 29:31],
            np.s_[5:5, 31:31]
        ]   # TODO: use pytest.mark.parametrize when converted to pytest

        expected = self.big_y2.copy()
        if not self.preloaded_chunks:
            expected[3:8, 30:60, 0:2] = 17
        for index in indices:
            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, index=index, errors=17)
            array_retrieved = pull.compute()
            np.testing.assert_array_equal(array_retrieved, expected[index])
            # Now test placeholders, to ensure that placeholder slicing works
            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, index=index,
                                             errors='placeholder')
            pull = self._placeholder_to_default(pull, 17)
            array_retrieved = pull.compute()
            np.testing.assert_array_equal(array_retrieved, expected[index])

    def _test_mark_complete(self, name):
        try:
            assert_false(self.store.is_complete(name))
        except NotImplementedError:
            pass
        else:
            self.store.mark_complete(name)
            assert_true(self.store.is_complete(name))

    def test_mark_complete_array(self):
        self._test_mark_complete(self.array_name('completetest'))
