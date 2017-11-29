################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
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

from katdal.chunkstore import ChunkStore, DictOfArraysChunkStore


class TestDictOfArraysChunkStore(object):
    def setup(self):
        self.x = np.arange(10)
        self.y = np.arange(24.).reshape(4, 3, 2)
        self.store = DictOfArraysChunkStore(x=self.x, y=self.y)

    def test_store(self):
        store = ChunkStore()
        assert_raises(NotImplementedError, store.get, 1, 2, 3)
        assert_raises(NotImplementedError, store.put, 1, 2, 3)

    def test_get(self):
        s = (slice(3, 5),)
        actual = self.store.get('x', s, np.dtype(np.int))
        desired = self.x[s]
        assert_array_equal(actual, desired, "Error getting x[%s]" % (s,))
        s = (slice(1, 4), slice(1, 3), slice(1, 2))
        actual = self.store.get('y', s, np.dtype(np.float))
        desired = self.y[s]
        assert_array_equal(actual, desired, "Error getting y[%s]" % (s,))

    def test_put(self):
        s = (slice(3, 5),)
        self.store.put('x', s, np.arange(2))
        actual = self.x[:5]
        desired = np.array([0, 1, 2, 0, 1])
        assert_array_equal(actual, desired, "Error putting x[%s]" % (s,))
        s = (slice(0, 2), slice(0, 3))
        self.store.put('y', s, np.zeros((2, 3, 2), dtype=np.dtype(np.float)))
        actual = self.y[:2, :3, :]
        desired = np.zeros((2, 3, 2))
        assert_array_equal(actual, desired, "Error putting y[%s]" % (s,))
