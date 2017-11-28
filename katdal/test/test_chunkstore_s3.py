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

import tempfile
import shutil
import subprocess
import requests
import time

import numpy as np
from numpy.testing import assert_array_equal
from nose import SkipTest
from nose.tools import assert_raises

import katdal.chunkstore_s3
from katdal.chunkstore_s3 import S3ChunkStore


class TestS3ChunkStore(object):
    def setup(self):
        """Start Fake S3 service running on temp dir, and ChunkStore on that."""
        self.tempdir = tempfile.mkdtemp()
        host = 'localhost'
        # Choose a random port and hope it's available
        port = np.random.randint(10000, 60000)
        url = "http://%s:%d" % (host, port)
        try:
            self.fakes3 = subprocess.Popen(['fakes3', 'server', '-H', host,
                                            '-p', str(port), '-r', self.tempdir])
        except OSError:
            self.fakes3 = None
            self.teardown()
            raise SkipTest('Could not start fakes3 server (is it installed?)')
        start = time.time()
        # Give up after waiting a few seconds for Fake S3
        while time.time() - start <= 10:
            try:
                response = requests.get(url, timeout=0.5)
            except requests.exceptions.ConnectionError:
                time.sleep(0.3)
            else:
                # In case there is already another service on this port (even S3)
                if response.reason == 'OK':
                    break
                else:
                    self.teardown()
                    raise SkipTest('Unexpected response from fakes3 server')
        else:
            self.teardown()
            raise SkipTest('Could not connect to fakes3 server')
        # Now start up store
        self.x = np.arange(10)
        self.y = np.arange(24.).reshape(4, 3, 2)
        self.store = S3ChunkStore(url)

    def teardown(self):
        if self.fakes3:
            self.fakes3.terminate()
            self.fakes3.wait()
        shutil.rmtree(self.tempdir)

    def array_name(self, path):
        bucket = 'katdal-unittest'
        return self.store.join(bucket, path)

    def test_store(self):
        # Pretend that botocore is not installed
        real_botocore = katdal.chunkstore_s3.botocore
        katdal.chunkstore_s3.botocore = None
        assert_raises(ImportError, S3ChunkStore, 'blah')
        katdal.chunkstore_s3.botocore = real_botocore

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
