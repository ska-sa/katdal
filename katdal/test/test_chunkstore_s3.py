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

"""Tests for :py:mod:`katdal.chunkstore_s3`."""

import tempfile
import shutil
import subprocess
import time
import re

from nose import SkipTest
from nose.tools import assert_raises

from katdal.chunkstore_s3 import S3ChunkStore, botocore
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


class TestS3ChunkStore(ChunkStoreTestBase):
    """Tests connecting to an actual (fake) S3 service, implying a slower setup."""

    def start_fakes3(self, host):
        """Start Fake S3 service as `host` and return its URL."""
        try:
            # Port number is automatically assigned
            self.fakes3 = subprocess.Popen(['fakes3', 'server',
                                            '-r', self.tempdir, '-p', '0',
                                            '-a', host, '-H', host],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, bufsize=1)
        except OSError:
            raise SkipTest('Could not start fakes3 server (is it installed?)')
        start = time.time()
        # Give up after waiting a few seconds for Fake S3
        while time.time() - start <= 5:
            # Look for assigned port number in fakes3 stderr output
            line = self.fakes3.stderr.readline().strip()
            ports_found = re.search(r' port=(\d{4,5})$', line)
            if ports_found:
                port = ports_found.groups()[0]
                return 'http://%s:%s' % (host, port)
        raise SkipTest('Could not connect to fakes3 server')

    def setup(self):
        """Start Fake S3 service running on temp dir, and ChunkStore on that."""
        self.tempdir = tempfile.mkdtemp()
        self.fakes3 = None
        try:
            url = self.start_fakes3('localhost')
            try:
                self.store = S3ChunkStore.from_url(url)
            except ImportError:
                raise SkipTest('S3 botocore dependency not installed')
            except StoreUnavailable:
                # Simplified client setup with dummy authentication keys,
                # useful for Jenkins that doesn't have any S3 credentials
                session = botocore.session.get_session()
                client = session.create_client(service_name='s3',
                                               endpoint_url=url,
                                               aws_access_key_id='blah',
                                               aws_secret_access_key='blah')
                self.store = S3ChunkStore(client)
        except Exception:
            self.teardown()
            raise

    def teardown(self):
        if self.fakes3:
            self.fakes3.terminate()
            self.fakes3.wait()
        shutil.rmtree(self.tempdir)

    def array_name(self, path):
        bucket = 'katdal-unittest'
        return self.store.join(bucket, path)


class TestDudS3ChunkStore(object):
    """Tests that don't need an S3 connection, only a 'dud' store."""

    def setup(self):
        if not botocore:
            raise SkipTest('S3 botocore dependency not installed')

    def test_store_unavailable(self):
        assert_raises(StoreUnavailable, S3ChunkStore.from_url, 'hahahahahaha')
