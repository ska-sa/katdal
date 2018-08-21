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
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()     # noqa: E402
import tempfile
import shutil
import subprocess
import threading
import os
import time
import socket
import http.server
import urllib.parse
import contextlib

import numpy as np
from nose import SkipTest
from nose.tools import assert_raises, timed
import mock
import requests

from katdal.chunkstore_s3 import S3ChunkStore, _AWSAuth
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


def consume_stdout_check_ready(process, ready):
    """Look for message indicating the server is ready.

    This is used as a workaround for
    https://github.com/minio/minio/issues/6324. Once fixed it would be simpler
    to poll the HTTP port.
    """
    # Gobble up lines of text from stdout until it is closed when process exits.
    # This is the same as `for line in process.stdout:` but on Python 2 that
    # version deadlocks (see https://stackoverflow.com/a/1085100).
    for line in iter(process.stdout.readline, b''):
        if line.strip() == b'Object API (Amazon S3 compatible):':
            ready.set()
    ready.set()  # Avoid blocking the consumer if minio crashed


def gethostbyname_slow(host):
    """Mock DNS lookup that is meant to be slow."""
    time.sleep(30)
    return '127.0.0.1'


@contextlib.contextmanager
def get_free_port(host):
    """Get an unused port number.

    This is a context manager that returns a port, while holding open the
    socket bound to it. This prevents another ephemeral process from
    obtaining the port in the meantime. The target process should bind the
    port with SO_REUSEADDR, after which the context should be exited to close
    the temporary socket.
    """
    with contextlib.closing(socket.socket()) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        yield port


class TestS3ChunkStore(ChunkStoreTestBase):
    """Test S3 functionality against an actual (minio) S3 service."""

    @classmethod
    def start_minio(cls, host):
        """Start Fake S3 service on `host` and return its URL."""
        with get_free_port(host) as port:
            try:
                env = os.environ.copy()
                env['MINIO_BROWSER'] = 'off'
                env['MINIO_ACCESS_KEY'] = cls.credentials[0]
                env['MINIO_SECRET_KEY'] = cls.credentials[1]
                cls.minio = subprocess.Popen(['minio', 'server',
                                              '--address', '{}:{}'.format(host, port),
                                              '-C', os.path.join(cls.tempdir, 'config'),
                                              os.path.join(cls.tempdir, 'data')],
                                             stdout=subprocess.PIPE,
                                             env=env)
            except OSError:
                raise SkipTest('Could not start minio server (is it installed?)')

        ready = threading.Event()
        # Ensure that the stdout of minio process is continuously consumed.
        # This pattern is inspired by the "Launch, Interact, Get Output in
        # Real Time, Terminate" section of
        # https://dzone.com/articles/interacting-with-a-long-running-child-process-in-p
        cls.stdout_consumer = threading.Thread(target=consume_stdout_check_ready,
                                               args=(cls.minio, ready))
        cls.stdout_consumer.start()

        # Wait for minio to be ready to service requests
        if not ready.wait(timeout=30):
            raise OSError('Timed out waiting for minio to be ready')
        return 'http://%s:%s' % (host, port)

    @classmethod
    def from_url(cls, url, authenticate=True, **kwargs):
        """Create the chunk store"""
        if authenticate:
            kwargs['credentials'] = cls.credentials
        return S3ChunkStore.from_url(url, timeout=1, **kwargs)

    @classmethod
    def setup_class(cls):
        """Start minio service running on temp dir, and ChunkStore on that."""
        cls.credentials = ('access*key', 'secret*key')
        cls.tempdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(cls.tempdir, 'config'))
        os.mkdir(os.path.join(cls.tempdir, 'data'))
        # XXX Python 3.3+ can use subprocess.DEVNULL instead
        cls.devnull = open(os.devnull, 'wb')
        cls.minio = None
        cls.stdout_consumer = None
        try:
            cls.url = cls.start_minio('127.0.0.1')
            cls.store = cls.from_url(cls.url)
            # Ensure that pagination is tested
            cls.store.list_max_keys = 3
        except Exception:
            cls.teardown_class()
            raise

    @classmethod
    def teardown_class(cls):
        if cls.minio:
            cls.minio.terminate()
            cls.minio.wait()
        if cls.stdout_consumer:
            cls.stdout_consumer.join()
        cls.devnull.close()
        shutil.rmtree(cls.tempdir)

    def array_name(self, path):
        bucket = 'katdal-unittest'
        return self.store.join(bucket, path)

    def test_public_read(self):
        reader = self.from_url(self.url, authenticate=False)
        # Create a non-public-read array.
        # This test deliberately doesn't use array_name so that it can create
        # several different buckets.
        slices = np.index_exp[0:5]
        x = np.arange(5)
        self.store.create_array('private')
        self.store.put_chunk('private', slices, x)
        with assert_raises(StoreUnavailable):
            reader.get_chunk('private', slices, x.dtype)

        # Now a public-read array
        store = self.from_url(self.url, public_read=True)
        store.create_array('public')
        store.put_chunk('public', slices, x)
        y = reader.get_chunk('public', slices, x.dtype)
        np.testing.assert_array_equal(x, y)

    @timed(0.1 + 0.05)
    def test_store_unavailable_invalid_url(self):
        # Ensure that timeouts work
        assert_raises(StoreUnavailable, S3ChunkStore.from_url,
                      'http://apparently.invalid/',
                      timeout=0.1, extra_timeout=0)

    def test_token_without_https(self):
        # Don't allow users to leak their tokens by accident
        assert_raises(StoreUnavailable, S3ChunkStore.from_url,
                      'http://apparently.invalid/', token='secrettoken')

    @timed(0.1 + 1 + 0.05)
    @mock.patch('socket.gethostbyname', side_effect=gethostbyname_slow)
    def test_store_unavailable_slow_dns(self, mock_dns_lookup):
        # Some pathological DNS setups (sshuttle?) take forever to time out
        assert_raises(StoreUnavailable, S3ChunkStore.from_url,
                      'http://a-valid-domain-is-somehow-harder.kat.ac.za/',
                      timeout=0.1, extra_timeout=1)


class _TokenHTTPProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP proxy that substitutes AWS credentials in place of a bearer token"""
    def __getattr__(self, name):
        if name.startswith('do_'):
            return self.do_all
        else:
            return getattr(super(_TokenHTTPProxyHandler, self), name)

    def do_all(self):
        self.protocol_version = 'HTTP/1.1'
        url = urllib.parse.urljoin(self.server.target, self.path)
        data_len = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(data_len)
        if self.headers.get('Authorization') != 'Bearer mysecret':
            self.send_response(401, 'Unauthorized')
            self.end_headers()
            return

        try:
            with contextlib.closing(
                    self.server.session.request(self.command, url,
                                                headers=self.headers, data=data,
                                                auth=self.server.auth,
                                                allow_redirects=False,
                                                timeout=20)) as resp:
                content = resp.content
                status_code = resp.status_code
                reason = resp.reason
                headers = resp.headers.copy()
        except requests.RequestException as e:
            content = str(e).encode('utf-8')
            status_code = 503
            reason = 'Service unavailable'
            headers = {
                'Content-type': 'text/plain',
                'Content-length': str(len(content))
            }

        self.send_response(status_code, reason)
        for key, value in headers.items():
            if key.lower() not in ['date', 'server', 'transfer-encoding']:
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Print to stdout instead of stderr so that it doesn't spew all over
        # the screen in normal operation.
        print(format % args)


class TestS3ChunkStoreToken(TestS3ChunkStore):
    """Test S3 with token authentication headers."""

    @classmethod
    def setup_class(cls):
        cls.proxy_url = None
        cls.httpd = None
        super(TestS3ChunkStoreToken, cls).setup_class()

    @classmethod
    def teardown_class(cls):
        if cls.httpd is not None:
            cls.httpd.session.close()
            cls.httpd.shutdown()
            cls.httpd = None
            cls.httpd_thread.join()
            cls.httpd_thread = None
        super(TestS3ChunkStoreToken, cls).teardown_class()

    @classmethod
    def from_url(cls, url, authenticate=True, **kwargs):
        """Create the chunk store"""
        if not authenticate:
            return S3ChunkStore.from_url(url, timeout=1, **kwargs)

        if cls.httpd is None:
            proxy_host = '127.0.0.1'
            with get_free_port(proxy_host) as proxy_port:
                httpd = http.server.HTTPServer((proxy_host, proxy_port), _TokenHTTPProxyHandler)
            httpd.target = url
            httpd.session = requests.Session()
            httpd.auth = _AWSAuth(cls.credentials)
            cls.httpd_thread = threading.Thread(target=httpd.serve_forever)
            cls.httpd_thread.start()
            # We delay setting cls.httpd until we've launched serve_forever,
            # because teardown calls httpd.shutdown and that hangs if
            # serve_forever wasn't called.
            cls.httpd = httpd
            cls.proxy_url = 'http://{}:{}'.format(proxy_host, proxy_port)
        elif url != cls.httpd.target:
            raise RuntimeError('Cannot use multiple target URLs with http proxy')
        return S3ChunkStore.from_url(cls.proxy_url, timeout=1, token='mysecret', **kwargs)
