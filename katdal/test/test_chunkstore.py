################################################################################
# Copyright (c) 2017-2023, National Research Foundation (SARAO)
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
from numpy.testing import assert_array_equal
import pytest

from katdal.chunkstore import (BadChunk, ChunkNotFound, ChunkStore,
                               PlaceholderChunk, StoreUnavailable,
                               generate_chunks, _prune_chunks, _blocks_ravel)


class TestGenerateChunks:
    """Test the `generate_chunks` function."""
    @classmethod
    def setup_class(cls):
        cls.shape = (10, 8192, 144)
        cls.dtype = np.complex64
        cls.nbytes = np.prod(cls.shape) * np.dtype(cls.dtype).itemsize

    def test_basic(self):
        # Basic check
        chunks = generate_chunks(self.shape, self.dtype, 3e6)
        assert chunks == (10 * (1,), 4 * (2048,), (144,))
        # Uneven chunks in the final split
        chunks = generate_chunks(self.shape, self.dtype, 1e6)
        assert chunks == (10 * (1,), 10 * (819,) + (2,), (144,))

    def test_corner_cases(self):
        # Corner case: don't select any dimensions to split -> one chunk
        chunks = generate_chunks(self.shape, self.dtype, 1e6, ())
        assert chunks == ((10,), (8192,), (144,))
        # Corner case: all bytes results in one chunk
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes)
        assert chunks == ((10,), (8192,), (144,))
        # Corner case: one byte less than the full size results in a split
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes - 1)
        assert chunks == ((5, 5), (8192,), (144,))

    def test_power_of_two(self):
        # Check power_of_two
        chunks = generate_chunks(self.shape, self.dtype, 1e6,
                                 dims_to_split=[1], power_of_two=True)
        assert chunks == ((10,), 128 * (64,), (144,))
        chunks = generate_chunks(self.shape, self.dtype, self.nbytes / 16,
                                 dims_to_split=[1], power_of_two=True)
        assert chunks == ((10,), 16 * (512,), (144,))
        # Check power_of_two when dimension is not a power-of-two itself
        shape = (10, 32768 - 2048, 144)
        chunks = generate_chunks(shape, self.dtype, self.nbytes / 10,
                                 dims_to_split=(0, 1), power_of_two=True)
        assert chunks == (10 * (1,), 3 * (8192,) + (6144,), (144,))
        # Check swapping the order of dims_to_split
        chunks = generate_chunks(shape, self.dtype, self.nbytes / 10,
                                 dims_to_split=(1, 0), power_of_two=True)
        assert chunks == ((10,), 60 * (512,), (144,))

    def test_max_dim_elements(self):
        chunks = generate_chunks(self.shape, self.dtype, 150000,
                                 dims_to_split=(0, 1), power_of_two=True,
                                 max_dim_elements={1: 50})
        assert chunks == ((4, 4, 2), 256 * (32,), (144,))
        # Case where max_dim_elements forces chunks to be smaller than
        # max_chunk_size.
        chunks = generate_chunks(self.shape, self.dtype, 1e6,
                                 dims_to_split=(0, 1), power_of_two=True,
                                 max_dim_elements={0: 4, 1: 50})
        assert chunks == ((4, 4, 2), 256 * (32,), (144,))

    def test_max_dim_elements_ignore(self):
        """Elements not in `dims_to_split` are ignored"""
        chunks = generate_chunks(self.shape, self.dtype, 150000,
                                 dims_to_split=(1, 17), power_of_two=True,
                                 max_dim_elements={0: 2, 1: 50})
        assert chunks == ((10,), 1024 * (8,), (144,))


def test_prune_chunks():
    """Test the `_prune_chunks` internal function."""
    chunks = ((10, 10, 10, 10), (2, 2, 2), (40,))
    # The chunk start-stop boundaries on each axis are:
    # ((0, 10, 20, 30, 40), (0, 2, 4, 6), (0, 40))
    index = np.s_[13:34, :4, 10:]
    new_chunks, new_index, new_offset = _prune_chunks(chunks, index)
    # The new chunk start-stop boundaries on each axis are:
    # ((10, 20, 30, 40), (0, 2, 4), (0, 40))
    assert new_chunks == ((10, 10, 10), (2, 2), (40,))
    assert new_index == np.s_[3:24, 0:4, 10:40]
    assert new_offset == (10, 0, 0)
    with pytest.raises(IndexError):
        _prune_chunks(chunks, np.s_[13:34:2, ::-1, :])


class TestChunkStore:
    """This tests the base class functionality."""

    def test_put_get(self):
        store = ChunkStore()
        with pytest.raises(NotImplementedError):
            store.get_chunk(1, 2, 3)
        with pytest.raises(NotImplementedError):
            store.put_chunk(1, 2, 3)

    def test_metadata_validation(self):
        store = ChunkStore()
        # Bad slice specifications
        with pytest.raises(TypeError):
            store.chunk_metadata("x", 3)
        with pytest.raises(TypeError):
            store.chunk_metadata("x", [3, 2])
        with pytest.raises(TypeError):
            store.chunk_metadata("x", slice(10))
        with pytest.raises(TypeError):
            store.chunk_metadata("x", [slice(10)])
        with pytest.raises(TypeError):
            store.chunk_metadata("x", [slice(0, 10, 2)])
        # Chunk mismatch
        with pytest.raises(BadChunk):
            store.chunk_metadata("x", [slice(0, 10, 1)], chunk=np.ones(11))
        with pytest.raises(BadChunk):
            store.chunk_metadata("x", (), chunk=np.ones(5))
        # Bad dtype
        with pytest.raises(BadChunk):
            store.chunk_metadata("x", [slice(0, 10, 1)], chunk=np.array(10 * [{}]))
        with pytest.raises(BadChunk):
            store.chunk_metadata("x", [slice(0, 2)], dtype=np.dtype(object))

    def test_standard_errors(self):
        error_map = {ZeroDivisionError: StoreUnavailable,
                     LookupError: ChunkNotFound}
        store = ChunkStore(error_map)
        with pytest.raises(StoreUnavailable):
            with store._standard_errors():
                1 / 0
        with pytest.raises(ChunkNotFound):
            with store._standard_errors():
                {}['ha']


def generate_arrays():
    """Generate arrays with differently sized dtypes and dimensions as test data."""
    return dict(
        x=np.ones(10, dtype=bool),
        y=np.arange(96.).reshape(8, 6, 2),
        z=np.array(2.),
        big_y=np.arange(960.).reshape(8, 60, 2),
        big_y2=np.arange(960).reshape(8, 60, 2),
    )


class ChunkStoreTestBase:
    """Standard tests performed on all types of ChunkStore."""

    # Instance of store instantiated once per class via class-level fixture
    store = None
    preloaded_chunks = False
    arrays = {}

    def array_name(self, name):
        return name

    def put_get_chunk(self, var_name, slices):
        """Put a single chunk into store, get it back and compare."""
        array_name = self.array_name(var_name)
        chunk = self.arrays[var_name][slices]
        self.store.create_array(array_name)
        self.store.put_chunk(array_name, slices, chunk)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, f"Error storing {var_name}[{slices}]")

    def make_dask_array(self, var_name, slices=()):
        """Turn (part of) an existing ndarray into a dask array."""
        array_name = self.array_name(var_name)
        array = self.arrays[var_name]
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
        try:
            assert_array_equal(results, np.full(divisions_per_dim, None))
        except AssertionError as exc:
            raise AssertionError(f"Bad put_dask_array: {var_name} {slices} {results.tolist()}") from exc

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
        dtype = np.dtype(float)
        args = (array_name, slices, dtype)
        shape = tuple(s.stop - s.start for s in slices)
        with pytest.raises(ChunkNotFound):
            self.store.get_chunk(*args)
        zeros = self.store.get_chunk_or_default(*args)
        assert_array_equal(zeros, np.zeros(shape, dtype))
        assert zeros.dtype == dtype
        ones = self.store.get_chunk_or_default(*args, default_value=1)
        assert_array_equal(ones, np.ones(shape, dtype))
        placeholder = self.store.get_chunk_or_placeholder(*args)
        assert isinstance(placeholder, PlaceholderChunk)
        assert placeholder.shape == shape
        assert placeholder.dtype == dtype

    def test_chunk_bool_1dim_and_too_small(self):
        # Check basic put + get on 1-D bool
        name = self.array_name('x')
        s = (slice(3, 5),)
        self.put_get_chunk('x', s)
        # Stored object has fewer bytes than expected (and wrong dtype)
        with pytest.raises(BadChunk):
            self.store.get_chunk(name, s, self.arrays['y'].dtype)

    def test_chunk_float_3dim_and_too_large(self):
        # Check basic put + get on 3-D float
        name = self.array_name('y')
        s = (slice(3, 7), slice(2, 5), slice(1, 2))
        self.put_get_chunk('y', s)
        # Stored object has more bytes than expected (and wrong dtype)
        with pytest.raises(BadChunk):
            self.store.get_chunk(name, s, self.arrays['x'].dtype)

    def test_chunk_zero_size(self):
        # Try a chunk with zero size
        self.put_get_chunk('y', (slice(4, 7), slice(3, 3), slice(0, 2)))
        # Try an empty slice on a zero-dimensional array (but why?)
        self.put_get_chunk('z', ())

    def test_put_chunk_noraise(self):
        name = self.array_name('x')
        self.store.create_array(name)
        result = self.store.put_chunk_noraise(name, (slice(1, 2),), np.ones(4))
        assert isinstance(result, BadChunk)

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

        return da.map_blocks(map_blocks_func, array, dtype=array.dtype)

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
            assert array_retrieved.shape == dask_array.shape
            assert array_retrieved.dtype == dask_array.dtype
            assert_array_equal(array_retrieved[np.s_[3:8, 30:60, 0:2]], 17,
                               f'Missing chunk in {array_name} not replaced by default value')

            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors='raise')
            with pytest.raises(ChunkNotFound):
                pull.compute()

            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors='placeholder')
            # We can't compute pull directly, because placeholders aren't
            # numpy arrays. So we have to remap them.
            pull = self._placeholder_to_default(pull, 17)
            array_retrieved = pull.compute()
            assert array_retrieved.shape == dask_array.shape
            assert array_retrieved.dtype == dask_array.dtype
            assert_array_equal(array_retrieved[np.s_[3:8, 30:60, 0:2]], 17,
                               "Missing chunk in {} not replaced by default value"
                               .format(array_name))

            pull = self.store.get_dask_array(array_name, dask_array.chunks,
                                             dask_array.dtype, offset, errors='dryrun')
            # Find the blocks involved in a slice of the original array
            index = np.s_[2:, 10:, 1:]
            placeholder_chunks = da.compute(*_blocks_ravel(pull[index]))
            # XXX Workaround for array.blocks.size (dask >= 2021.11.0)
            assert len(placeholder_chunks) == np.prod(dask_array[index].numblocks)
            assert all(isinstance(c, PlaceholderChunk) for c in placeholder_chunks)
            slices = da.core.slices_from_chunks(dask_array.chunks)
            all_chunks = [self.store.chunk_metadata(array_name, s)[0] for s in slices]
            assert sorted(c.name for c in placeholder_chunks) == all_chunks[4:]

        # Now store the last quarter and check that complete array is correct
        self.put_dask_array('big_y2', np.s_[3:8, 30:60, 0:2])
        self.get_dask_array('big_y2')

    @pytest.mark.parametrize(
        "index",
        [
            (),
            np.s_[:, :],
            np.s_[0:8, 0:60, 0:2],
            np.s_[..., 0:1],
            np.s_[5:6, 29:31],
            np.s_[5:5, 31:31],
        ]
    )
    def test_get_dask_array_index(self, index):
        # Load most but not all of the array, to test error handling
        self.put_dask_array('big_y2', np.s_[0:3,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[3:8,  0:30, 0:2])
        self.put_dask_array('big_y2', np.s_[0:3, 30:60, 0:2])
        array_name, dask_array, offset = self.make_dask_array('big_y2')

        expected = self.arrays['big_y2'].copy()
        if not self.preloaded_chunks:
            expected[3:8, 30:60, 0:2] = 17
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
            assert not self.store.is_complete(name)
        except NotImplementedError:
            pass
        else:
            self.store.mark_complete(name)
            assert self.store.is_complete(name)

    def test_mark_complete_array(self):
        self._test_mark_complete(self.array_name('completetest'))
