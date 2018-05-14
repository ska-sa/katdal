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

import numpy as np

from nose.tools import assert_raises

from katdal.lazy_indexer import _simplify_index


class TestSimplifyIndices(object):
    """Test the :func:`~katdal.lazy_indexer._simplify_indices function"""
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
