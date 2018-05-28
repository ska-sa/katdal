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
import re
import threading
import Queue
import os

from nose import SkipTest
from nose.tools import assert_raises

from katdal.chunkstore_s3 import S3ChunkStore, botocore
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


def consume_stderr_find_port(process, queue):
    """Look for assigned port number in fakes3 output while also consuming it."""
    looking_for_port = True
    # Gobble up lines of text from stderr until it is closed when process exits.
    # This is the same as `for line in process.stderr:` but on Python 2 that
    # version deadlocks (see https://stackoverflow.com/a/1085100).
    for line in iter(process.stderr.readline, ''):
        if looking_for_port:
            ports_found = re.search(r' port=([1-9]\d*)$', line.strip())
            if ports_found:
                port_number = ports_found.group(1)
                queue.put(port_number)
                looking_for_port = False


class TestS3ChunkStore(ChunkStoreTestBase):
    """Test S3 functionality against an actual (fake) S3 service."""

    @classmethod
    def start_fakes3(cls, host):
        """Start Fake S3 service as `host` and return its URL."""
        try:
            # Port number is automatically assigned
            cls.fakes3 = subprocess.Popen(['fakes3', 'server',
                                           '-r', cls.tempdir, '-p', '0',
                                           '-a', host, '-H', host],
                                          stdout=cls.devnull,
                                          stderr=subprocess.PIPE)
        except OSError:
            raise SkipTest('Could not start fakes3 server (is it installed?)')
        # The assigned port number is scraped from stderr and returned via queue
        port_queue = Queue.Queue()
        # Ensure that the stderr of fakes3 process is continuously consumed.
        # This pattern is inspired by the "Launch, Interact, Get Output in
        # Real Time, Terminate" section of
        # https://dzone.com/articles/interacting-with-a-long-running-child-process-in-p
        cls.stderr_consumer = threading.Thread(target=consume_stderr_find_port,
                                               args=(cls.fakes3, port_queue))
        cls.stderr_consumer.start()
        # Give up after waiting a few seconds for Fake S3 to announce its port
        try:
            port = port_queue.get(timeout=5)
        except Queue.Empty:
            raise OSError('Could not connect to fakes3 server')
        else:
            return 'http://%s:%s' % (host, port)

    @classmethod
    def setup_class(cls):
        """Start Fake S3 service running on temp dir, and ChunkStore on that."""
        cls.tempdir = tempfile.mkdtemp()
        # XXX Python 3.3+ can use subprocess.DEVNULL instead
        cls.devnull = open(os.devnull, 'wb')
        cls.fakes3 = None
        cls.stderr_consumer = None
        try:
            url = cls.start_fakes3('127.0.0.1')
            try:
                cls.store = S3ChunkStore.from_url(url, timeout=1)
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
                cls.store = S3ChunkStore(client)
        except Exception:
            cls.teardown_class()
            raise

    @classmethod
    def teardown_class(cls):
        if cls.fakes3:
            cls.fakes3.terminate()
            cls.fakes3.wait()
        if cls.stderr_consumer:
            cls.stderr_consumer.join()
        cls.devnull.close()
        shutil.rmtree(cls.tempdir)

    def array_name(self, path):
        bucket = 'katdal-unittest'
        return self.store.join(bucket, path)

    def test_store_unavailable(self):
        # Drastically reduce the default botocore timeout of nearly 7 seconds
        assert_raises(StoreUnavailable, S3ChunkStore.from_url,
                      'http://apparently.invalid/', timeout=0.1)
