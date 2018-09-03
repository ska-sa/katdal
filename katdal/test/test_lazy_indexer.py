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

"""Tests for :py:mod:`katdal.lazy_indexer`."""
from __future__ import print_function, division, absolute_import

from builtins import object
from numbers import Integral

import numpy as np
import dask.array as da
from nose.tools import assert_raises, assert_equal

from katdal.lazy_indexer import (_range_to_slice, _simplify_index,
                                 _dask_getitem, DaskLazyIndexer)


def slice_to_range(s, l):
    return range(*s.indices(l))


class TestRangeToSlice(object):
    """Test the :func:`~katdal.lazy_indexer._range_to_slice` function."""
    @staticmethod
    def _check_slice(start, stop, step):
        s = slice(start, stop, step)
        length = max(start, stop) + 1
        r = slice_to_range(s, length)
        assert_equal(_range_to_slice(r), s)

    def test_basic_slices(self):
        # For testing both `start` and `stop` need to be non-negative
        self._check_slice(0, 10, 1)   # contiguous, ascending
        self._check_slice(0, 10, 2)   # strided, ascending
        self._check_slice(10, 0, -1)  # contiguous, ascending
        self._check_slice(10, 0, -2)  # strided, descending
        self._check_slice(0, 1, 1)    # single element (treated as ascending)
        self._check_slice(0, 10, 5)   # any two elements (has stop = 2 * end)

    def test_zero_increments(self):
        with assert_raises(ValueError):
            _range_to_slice([1, 1, 1, 1])

    def test_nonuniform_increments(self):
        with assert_raises(ValueError):
            _range_to_slice([1, 1, 2, 3, 5, 8, 13])


class TestSimplifyIndex(object):
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


def ix_(keep, shape):
    """Extend numpy.ix_ to accept slices and single ints as well."""
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


class TestDaskGetitem(object):
    """Test the :func:`~katdal.lazy_indexer._dask_getitem` function."""
    def setup(self):
        shape = (10, 10, 10, 10)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(2, 5, 2, 5))

    def _test_with(self, indices, indices_without_ellipsis=None):
        npy = numpy_oindex(self.data, indices)
        if indices_without_ellipsis is None:
            indices_without_ellipsis = indices
        npy_lite = numpy_oindex_lite(self.data, indices_without_ellipsis)
        getitem = _dask_getitem(self.data_dask, indices).compute()
        np.testing.assert_array_equal(npy, npy_lite)
        np.testing.assert_array_equal(getitem, npy)

    def test_outer_indexing(self):
        self._test_with(())
        self._test_with(2)
        self._test_with((2, 3, 4, 5))
        # ellipsis
        self._test_with(np.s_[[0], ...], np.s_[[0], :, :, :])
        self._test_with(np.s_[:, [0], ...], np.s_[:, [0], :, :])
        # list of ints
        self._test_with(np.s_[:, [0], [0], :])
        self._test_with(np.s_[:, [0], :, [0]])
        self._test_with(np.s_[:, [0], [0, 1], :])
        self._test_with(np.s_[:, [0], :, [0, 1]])
        # booleans
        pick_one = np.zeros(10, dtype=np.bool_)
        pick_one[6] = True
        self._test_with(np.s_[:, [True, False] * 5, pick_one, :])
        self._test_with(np.s_[:, [False, True] * 5, :, pick_one])
        self._test_with(np.s_[4:9, [False, True] * 5,
                              [True, False] * 5, pick_one])
        # slices
        self._test_with(np.s_[0:2, 2:4, 4:6, 6:8])
        self._test_with(np.s_[-8:-6, -4:-2, slice(3, 10, 2), -2:])
        # single ints
        self._test_with(np.s_[:, [0], 0, :])
        self._test_with(np.s_[:, [0], :, 0])
        self._test_with(np.s_[:, [0], -1, :])
        self._test_with(np.s_[:, [0], :, -1])
        self._test_with(np.s_[:, 0, [0, 2], [1, 3, 5]])
        # newaxis
        self._test_with(np.s_[np.newaxis, :, [0], :, 0])
        self._test_with(np.s_[:, [0], np.newaxis, 0, :])
        # the lot
        self._test_with(np.s_[..., 0, 2:5, [True, False] * 5,
                              np.newaxis, [4, 6]],
                        np.s_[0, 2:5, [True, False] * 5, np.newaxis, [4, 6]])


class TestDaskLazyIndexer(object):
    """Test the :class:`~katdal.lazy_indexer.DaskLazyIndexer` class."""
    def setup(self):
        shape = (10, 20, 30)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(1, 4, 5))

    def _test_with(self, stage1=(), stage2=()):
        npy1 = numpy_oindex(self.data, stage1)
        npy2 = numpy_oindex(npy1, stage2)
        indexer = DaskLazyIndexer(self.data_dask, stage1)
        np.testing.assert_array_equal(indexer[stage2], npy2)

    def test_stage1_slices(self):
        self._test_with(np.s_[5:, :, 1::2])

    def test_stage2_ints(self):
        self._test_with(np.s_[5:, :, 1::2], np.s_[1, 2, -1])

    def test_stage1_multiple_boolean_indices(self):
        self._test_with(tuple([True] * d for d in self.data.shape))
        self._test_with(tuple([True, False] * (d // 2)
                              for d in self.data.shape))

    def test_stage2_multiple_boolean_indices(self):
        stage1 = tuple([True] * d for d in self.data.shape)
        stage2 = tuple([True] * 4 + [False] * (d - 4) for d in self.data.shape)
        self._test_with(stage1, stage2)
        stage2 = tuple([True, False] * (d // 2) for d in self.data.shape)
        self._test_with(stage1, stage2)
