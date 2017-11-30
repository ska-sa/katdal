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

"""Tests for :py:mod:`katdal.chunkstore_s3`."""

import numpy as np
from numpy.testing import assert_array_equal
from nose import SkipTest
from nose.tools import assert_raises

import katdal.chunkstore_rados
from katdal.chunkstore_rados import RadosChunkStore


class TestRadosChunkStore(object):
    def setup(self):
        self.x = np.arange(10)
        self.y = np.arange(24.).reshape(4, 3, 2)
        # Test configuration on seekat cluster (run this on appropriate machine!)
        conf = '/etc/ceph/ceph.conf'
        pool = 'test_katdal'
        try:
            self.store = RadosChunkStore(conf, pool)
        except (ImportError, OSError):
            raise SkipTest('Rados not installed or cluster misconfigured / down')

    def array_name(self, path):
        namespace = 'katdal_test_chunkstore_rados'
        return self.store.join(namespace, path)

    def test_store(self):
        # Pretend that rados is not installed
        real_rados = katdal.chunkstore_rados.rados
        katdal.chunkstore_rados.rados = None
        assert_raises(ImportError, RadosChunkStore, 'blah', 'blah')
        katdal.chunkstore_rados.rados = real_rados

    def test_put_and_get(self):
        s = (slice(3, 5),)
        desired = self.x[s]
        name = self.array_name('x')
        self.store.put(name, s, desired)
        actual = self.store.get(name, s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing x[%s]" % (s,))
        s = (slice(1, 4), slice(1, 3), slice(1, 2))
        desired = self.y[s]
        name = self.array_name('y')
        self.store.put(name, s, desired)
        actual = self.store.get(name, s, desired.dtype)
        assert_array_equal(actual, desired, "Error storing y[%s]" % (s,))
