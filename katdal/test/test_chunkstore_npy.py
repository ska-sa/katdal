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

"""Tests for :py:mod:`katdal.chunkstore_npy`."""

import tempfile
import shutil

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises

from katdal.chunkstore_npy import NpyFileChunkStore


class TestNpyFileChunkStore(object):
    def setup(self):
        """Create temp dir to store NPY files and build ChunkStore on that."""
        self.tempdir = tempfile.mkdtemp()
        self.x = np.arange(10)
        self.y = np.arange(24.).reshape(4, 3, 2)
        self.store = NpyFileChunkStore(self.tempdir)

    def teardown(self):
        shutil.rmtree(self.tempdir)

    def test_store(self):
        assert_raises(OSError, NpyFileChunkStore, 'hahahahaha')

    def test_put_and_get(self):
        s = (slice(3, 5),)
        desired = self.x[s]
        self.store.put('x', s, desired)
        actual = self.store.get('x', s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing x[%s]" % (s,))
        s = (slice(1, 4), slice(1, 3), slice(1, 2))
        desired = self.y[s]
        self.store.put('y', s, desired)
        actual = self.store.get('y', s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing y[%s]" % (s,))
