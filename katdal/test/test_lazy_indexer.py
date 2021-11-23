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

"""Tests for :py:mod:`katdal.lazy_indexer`."""

from functools import partial
from numbers import Integral

import dask.array as da
import numpy as np
from nose.tools import assert_equal, assert_raises

from katdal.lazy_indexer import (DaskLazyIndexer, _dask_oindex,
                                 _range_to_slice, _simplify_index,
                                 dask_getitem)


def slice_to_range(s, length):
    return range(*s.indices(length))


class TestRangeToSlice:
    """Test the :func:`~katdal.lazy_indexer._range_to_slice` function."""
    @staticmethod
    def _check_slice(start, stop, step):
        s = slice(start, stop, step)
        length = max(start, 0 if stop is None else stop) + 1
        r = slice_to_range(s, length)
        assert_equal(_range_to_slice(r), s)

    def test_basic_slices(self):
        # For testing both `start` and `stop` need to be non-negative
        self._check_slice(0, 10, 1)   # contiguous, ascending
        self._check_slice(0, 10, 2)   # strided, ascending
        self._check_slice(10, 0, -1)  # contiguous, descending
        self._check_slice(10, 0, -2)  # strided, descending
        self._check_slice(10, None, -2)  # strided, descending all the way to 0
        self._check_slice(0, 1, 1)    # single element (treated as ascending)
        self._check_slice(0, 10, 5)   # any two elements (has stop = 2 * step)

    def test_negative_elements(self):
        with assert_raises(ValueError):
            _range_to_slice([-1, -2, -3, -4])

    def test_zero_increments(self):
        with assert_raises(ValueError):
            _range_to_slice([1, 1, 1, 1])

    def test_uneven_increments(self):
        with assert_raises(ValueError):
            _range_to_slice([1, 1, 2, 3, 5, 8, 13])


class TestSimplifyIndex:
    """Test the :func:`~katdal.lazy_indexer._simplify_index` function."""
    def setup(self):
        self.shape = (3, 4, 5)
        self.data = np.arange(np.product(self.shape)).reshape(self.shape)

    def _test_with(self, indices):
        expected = self.data[indices]
        simplified = _simplify_index(indices, self.data.shape)
        actual = self.data[simplified]
        np.testing.assert_array_equal(actual, expected)

    def _test_index_error(self, indices):
        with assert_raises(IndexError):
            simplified = _simplify_index(indices, self.data.shape)
            self.data[simplified]
        with assert_raises(IndexError):
            self.data[indices]

    def test_1d(self):
        self._test_with(np.s_[np.array([False, True, False])])
        self._test_with(np.s_[[1]])

    def test_contiguous(self):
        self._test_with(np.s_[:, np.array([False, True, True, False]), :])
        self._test_with(np.s_[:, [1, 2], :])

    def test_discontiguous_but_regular(self):
        self._test_with(np.s_[:, [False, True, False, True], :])
        self._test_with(np.s_[:, [1, 3], :])

    def test_discontiguous(self):
        self._test_with(np.s_[:, [True, True, False, True], :])
        self._test_with(np.s_[:, [0, 1, 3], :])

    def test_all_false(self):
        self._test_with(np.s_[:, np.array([False, False, False, False]), :])

    def test_all_true(self):
        self._test_with(np.s_[:, np.array([True, True, True, True]), :])

    def test_newaxis(self):
        self._test_with(np.s_[np.newaxis, np.array([True, True, False])])

    def test_ellipsis(self):
        self._test_with(np.s_[..., np.array([True, False, True, False, True])])

    def test_wrong_length(self):
        self._test_index_error(np.s_[:, np.array([True, False]), :])

    def test_too_many_axes(self):
        self._test_index_error(np.s_[0, 0, 0, 0])

    def test_bad_index_dtype(self):
        self._test_index_error(np.s_[:, np.array([1.2, 3.4])])


def ix_(keep, shape):
    r"""Extend numpy.ix\_ to accept slices and single ints as well."""
    # Inspired by Zarr's indexing.py (https://github.com/zarr-developers/zarr)
    keep = [slice_to_range(k, s) if isinstance(k, slice)
            else [k] if isinstance(k, Integral)
            else k
            for k, s in zip(keep, shape)]
    return np.ix_(*keep)


def numpy_oindex(x, keep):
    """Perform outer indexing on a NumPy array (inspired by Zarr).

    This is more onerous, but calls `x.__getitem__` only once.
    """
    # Inspired by Zarr's indexing.py (https://github.com/zarr-developers/zarr)
    # Get rid of ellipsis
    keep = da.slicing.normalize_index(keep, x.shape)
    new_axes = tuple(n for n, k in enumerate(keep) if k is np.newaxis)
    drop_axes = tuple(n for n, k in enumerate(keep) if isinstance(k, Integral))
    # Get rid of newaxis
    keep = tuple(k for k in keep if k is not np.newaxis)
    keep = ix_(keep, x.shape)
    result = x[keep]
    for ax in new_axes:
        result = np.expand_dims(result, ax)
    result = result.squeeze(axis=drop_axes)
    return result


def numpy_oindex_lite(x, keep):
    """Perform outer indexing on a NumPy array (compact version).

    This is more compact, but calls `x.__getitem__` `x.ndim` times.

    It also assumes that `keep` contains no ellipsis to be as pure as possible.
    """
    if not isinstance(keep, tuple):
        keep = (keep,)
    dim = 0
    result = x
    for k in keep:
        cumulative_index = (slice(None),) * dim + (k,)
        result = result[cumulative_index]
        # Handle dropped dimensions
        if not isinstance(k, Integral):
            dim += 1
    return result


UNEVEN = [False, True, True, True, False, False, True, True, False, True]


class TestDaskGetitem:
    """Test the :func:`~katdal.lazy_indexer.dask_getitem` function."""
    def setup(self):
        shape = (10, 20, 30, 40)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(2, 5, 2, 5))

    def _test_with(self, indices, normalised_indices=None):
        npy = numpy_oindex(self.data, indices)
        if normalised_indices is None:
            normalised_indices = indices
        npy_lite = numpy_oindex_lite(self.data, normalised_indices)
        oindex = _dask_oindex(self.data_dask, normalised_indices).compute()
        getitem = dask_getitem(self.data_dask, indices).compute()
        np.testing.assert_array_equal(npy, npy_lite)
        np.testing.assert_array_equal(getitem, npy)
        np.testing.assert_array_equal(oindex, npy)

    def test_misc_indices(self):
        self._test_with(())
        self._test_with(2, (2,))
        self._test_with((2, 3, 4, 5))

    def test_ellipsis(self):
        self._test_with(np.s_[[0], ...], np.s_[[0], :, :, :])
        self._test_with(np.s_[:, [0], ...], np.s_[:, [0], :, :])
        self._test_with(np.s_[[0], ..., [0]], np.s_[[0], :, :, [0]])

    def test_evenly_spaced_ints(self):
        self._test_with(np.s_[:, [0], [0], :])
        self._test_with(np.s_[:, [0], :, [0]])
        self._test_with(np.s_[:, [0], [0, 1, 2, 3], :])
        self._test_with(np.s_[[0], [-1, -2, -3, -4, -5], :, [8, 6, 4, 2, 0]])

    def test_evenly_spaced_booleans(self):
        pick_one = np.zeros(40, dtype=np.bool_)
        pick_one[6] = True
        self._test_with(np.s_[:, [True, False] * 10, pick_one[:30], :])
        self._test_with(np.s_[:, [False, True] * 10, :, pick_one])
        self._test_with(np.s_[4:9, [False, True] * 10,
                              [True, False] * 15, pick_one])

    def test_unevenly_spaced_fancy_indexing(self):
        self._test_with(np.s_[:, [0, 1, 3], [1, 2, 4], :])
        self._test_with(np.s_[UNEVEN, 2 * UNEVEN, 3 * UNEVEN, 4 * UNEVEN])

    def test_repeated_fancy_indexing(self):
        self._test_with(np.s_[:, [1, 1, 1], [6, 6, 6], :])

    def test_slices(self):
        self._test_with(np.s_[0:2, 2:4, 4:6, 6:8])
        self._test_with(np.s_[-8:-6, -4:-2, 3:10:2, -2:])

    def test_single_ints(self):
        self._test_with(np.s_[:, [0], 0, :])
        self._test_with(np.s_[:, [0], :, 0])
        self._test_with(np.s_[:, [0], -1, :])
        self._test_with(np.s_[:, [0], :, -1])
        self._test_with(np.s_[:, 0, [0, 2], [1, 3, 5]])

    def test_newaxis(self):
        self._test_with(np.s_[np.newaxis, :, 2 * UNEVEN, :, 0])
        self._test_with(np.s_[:, 2 * UNEVEN, np.newaxis, 0, :])
        self._test_with(np.s_[0, np.newaxis, 1, np.newaxis, 2, np.newaxis, 3])

    def test_the_lot(self):
        self._test_with(np.s_[..., 0, 2:5, 3 * UNEVEN, np.newaxis, [4, 6]],
                        np.s_[0, 2:5, 3 * UNEVEN, np.newaxis, [4, 6]])


class TestDaskLazyIndexer:
    """Test the :class:`~katdal.lazy_indexer.DaskLazyIndexer` class."""
    def setup(self):
        shape = (10, 20, 30)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(1, 4, 5), name='x')

    def test_str_repr(self):
        def transform1(x):
            return x
        transform2 = lambda x: x  # noqa: E731
        class Transform3:         # noqa: E306
            def __call__(self, x):
                return x
        transform3 = Transform3()
        transform4 = partial(transform1)
        transforms = [transform1, transform2, transform3, transform4]
        indexer = DaskLazyIndexer(self.data_dask, transforms=transforms)
        expected = 'x | transform1 | <lambda> | Transform3 | transform1'
        expected += f' -> {indexer.shape} {indexer.dtype}'
        assert_equal(str(indexer), expected)
        # Simply exercise repr - no need to check result
        repr(indexer)

    def _test_with(self, stage1=(), stage2=()):
        npy1 = numpy_oindex(self.data, stage1)
        npy2 = numpy_oindex(npy1, stage2)
        indexer = DaskLazyIndexer(self.data_dask, stage1)
        np.testing.assert_array_equal(indexer[stage2], npy2)
        # Check nested indexers
        indexer2 = DaskLazyIndexer(indexer, stage2)
        np.testing.assert_array_equal(indexer2[()], npy2)

    def test_stage1_slices(self):
        self._test_with(np.s_[5:, :, 1::2])

    def test_stage2_ints(self):
        self._test_with(np.s_[5:, :, 1::2], np.s_[1, 2, -1])

    def test_stage1_multiple_fancy_indices(self):
        self._test_with(tuple([True] * d for d in self.data.shape))
        self._test_with(tuple([True, False] * (d // 2)
                              for d in self.data.shape))
        self._test_with(np.s_[UNEVEN, 2 * UNEVEN, :24])
        self._test_with(np.s_[:3, [1, 2, 3, 4, 6, 9], [8, 6, 4, 2, 0]])

    def test_stage2_multiple_fancy_indices(self):
        stage1 = tuple([True] * d for d in self.data.shape)
        stage2 = tuple([True] * 4 + [False] * (d - 4) for d in self.data.shape)
        self._test_with(stage1, stage2)
        stage2 = tuple([True, False] * (d // 2) for d in self.data.shape)
        self._test_with(stage1, stage2)
        stage1 = np.s_[UNEVEN, 2 * UNEVEN, :24]
        stage2 = np.s_[:3, [1, 2, 3, 4, 6, 9], [8, 6, 4, 2, 0]]
        self._test_with(stage1, stage2)

    def test_transforms(self):
        # Add transform at initialisation
        indexer = DaskLazyIndexer(self.data_dask, transforms=[lambda x: 0 * x])
        np.testing.assert_array_equal(indexer[:], np.zeros_like(indexer))
        # Check nested indexers
        indexer = DaskLazyIndexer(self.data_dask)
        indexer2 = DaskLazyIndexer(indexer, transforms=[lambda x: 0 * x])
        np.testing.assert_array_equal(indexer[:], self.data)
        np.testing.assert_array_equal(indexer2[:], np.zeros_like(indexer))
