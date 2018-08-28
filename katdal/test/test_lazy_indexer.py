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

import numpy as np
import dask.array as da
from nose.tools import assert_raises

from katdal.lazy_indexer import _simplify_index, _dask_getitem, DaskLazyIndexer


class TestSimplifyIndices(object):
    """Test the :func:`~katdal.lazy_indexer._simplify_indices` function."""
    def setup(self):
        self.shape = (3, 4, 5)
        self.data = np.arange(np.product(self.shape)).reshape(self.shape)

    def _test_with(self, indices):
        expected = self.data[indices]
        simplified = _simplify_index(self.data.shape, indices)
        actual = self.data[simplified]
        np.testing.assert_array_equal(actual, expected)

    def _test_index_error(self, indices):
        simplified = _simplify_index(self.data.shape, indices)
        with assert_raises(IndexError):
            self.data[simplified]
        with assert_raises(IndexError):
            self.data[indices]

    def test_1d(self):
        self._test_with(np.s_[np.array([False, True, False])])

    def test_contiguous(self):
        self._test_with(np.s_[:, np.array([False, True, True, False]), :])

    def test_discontiguous(self):
        self._test_with(np.s_[:, np.array([False, True, False, True]), :])

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


class TestDaskGetitem(object):
    """Test the :func:`~katdal.lazy_indexer._dask_getitem` function."""
    def setup(self):
        # Derived from example in NEP 21
        shape = (5, 6, 7, 8)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(1, 2, 1, 2))

    def _check_shape(self, indices, shape):
        us = _dask_getitem(self.data_dask, indices).compute()
        np.testing.assert_equal(us.shape, shape)

    def _compare_indexing(self, indices):
        npy = self.data[indices]
        us = _dask_getitem(self.data_dask, indices).compute()
        np.testing.assert_array_equal(us, npy)

    def _test_with(self, indices, expected_error=None):
        if expected_error is None:
            self._compare_indexing(indices)
        else:
            with assert_raises(expected_error):
                self._compare_indexing(indices)

    def test_legacy_fancy_indexing(self):
        self._test_with(np.s_[[0], ...])
        self._test_with(np.s_[:, [0], ...])
        self._test_with(np.s_[:, [0], [0], :], AssertionError)  # future error
        self._test_with(np.s_[:, [0], :, [0]], AssertionError)  # future error
        self._test_with(np.s_[:, [0], 0, :])
        self._test_with(np.s_[:, [0], :, 0], AssertionError)  # future error
        # Create boolean index with one True value for the last two dimensions
        bindx = np.zeros((7, 8), dtype=np.bool_)
        bindx[0, 0] = True
        self._test_with(np.s_[:, 0, bindx], IndexError)  # dask can't handle it
        self._test_with(np.s_[0, :, bindx], IndexError)  # dask can't handle it
        self._test_with(np.s_[[0], :, bindx], IndexError)  # dask failure
        self._test_with(np.s_[:, [0, 1], bindx], IndexError)  # dask failure

    def test_outer_indexing(self):
        self._check_shape(np.s_[:, [0], [0, 1], :], (5, 1, 2, 8))
        self._check_shape(np.s_[:, [0], :, [0, 1]], (5, 1, 7, 2))
        self._check_shape(np.s_[:, [0], 0, :], (5, 1, 8))
        self._check_shape(np.s_[:, [0], :, 0], (5, 1, 7))
        # Restrict ourselves to more practical 1-D boolean indices
        bindx2 = np.zeros(7, dtype=np.bool_)
        bindx2[0] = True
        bindx3 = np.zeros(8, dtype=np.bool_)
        bindx3[0] = True
        bindx3[3] = True
        self._check_shape(np.s_[:, 0, bindx2, bindx3], (5, 1, 2))
        self._check_shape(np.s_[0, :, bindx2, bindx3], (6, 1, 2))
        self._check_shape(np.s_[[0], :, bindx2, bindx3], (1, 6, 1, 2))
        self._check_shape(np.s_[:, [0, 1], bindx2, bindx3], (5, 2, 1, 2))


class TestDaskLazyIndexer(object):
    """Test the :class:`~katdal.lazy_indexer.DaskLazyIndexer` class."""
    def setup(self):
        shape = (10, 20, 30)
        self.data = np.arange(np.product(shape)).reshape(shape)
        self.data_dask = da.from_array(self.data, chunks=(1, 4, 5))

    def test_stage1_slices(self):
        stage1 = np.s_[5:, :, 1::2]
        indexer = DaskLazyIndexer(self.data_dask, stage1)
        np.testing.assert_array_equal(indexer[:], self.data[stage1])

    def test_stage1_multiple_boolean_indices(self):
        stage1 = tuple([True] * d for d in self.data.shape)
        indexer = DaskLazyIndexer(self.data_dask, stage1)
        np.testing.assert_array_equal(indexer[:], self.data)

    def test_stage2_multiple_boolean_indices(self):
        stage1 = tuple([True] * d for d in self.data.shape)
        indexer = DaskLazyIndexer(self.data_dask, stage1)
        stage2 = tuple([True] * 2 + [False] * (d - 2) for d in self.data.shape)
        # Since this is oindex, direct NumPy indexing won't work - check shape
        np.testing.assert_equal(indexer[stage2].shape,
                                tuple(sum(s) for s in stage2))
