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

from nose import SkipTest
from nose.tools import assert_raises, timed
import mock
import requests

from katdal.chunkstore_s3 import S3ChunkStore, _AWSAuth
from katdal.chunkstore import StoreUnavailable
from katdal.test.test_chunkstore import ChunkStoreTestBase


def gethostbyname_slow(host):
    """Mock DNS lookup that is meant to be slow."""
    time.sleep(30)
    return '127.0.0.1'


def get_free_port(host):
    """Get an unused port number.

    Note that this is racy, because another process could claim the port
    between the time this function returns and the time the port is bound.
    """
    with contextlib.closing(socket.socket()) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        return port


class TestS3ChunkStore(ChunkStoreTestBase):
    """Test S3 functionality against an actual (minio) S3 service."""

    @classmethod
    def start_minio(cls, host):
        """Start Fake S3 service on `host` and return its URL."""
        port = get_free_port(host)
        try:
            env = os.environ.copy()
            env['MINIO_BROWSER'] = 'off'
            env['MINIO_ACCESS_KEY'] = cls.credentials[0]
            env['MINIO_SECRET_KEY'] = cls.credentials[1]
            cls.minio = subprocess.Popen(['minio', 'server', '--quiet',
                                          '--address', '{}:{}'.format(host, port),
                                          '-C', os.path.join(cls.tempdir, 'config'),
                                          os.path.join(cls.tempdir, 'data')],
                                         stdout=cls.devnull,
                                         stderr=cls.devnull,
                                         env=env)
        except OSError:
            raise SkipTest('Could not start minio server (is it installed?)')
        # Wait for minio to start listening on its port.
        for i in range(50):
            with contextlib.closing(socket.socket()) as sock:
                try:
                    sock.connect((host, port))
                except IOError:
                    time.sleep(0.1)
                else:
                    return 'http://%s:%s' % (host, port)
        raise OSError('Could not connect to minio server')

    @classmethod
    def from_url(cls, url):
        """Create the chunk store"""
        return S3ChunkStore.from_url(url, timeout=1, credentials=cls.credentials)

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
        try:
            url = cls.start_minio('127.0.0.1')
            cls.store = cls.from_url(url)
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
        cls.devnull.close()
        shutil.rmtree(cls.tempdir)

    def array_name(self, path):
        bucket = 'katdal-unittest'
        return self.store.join(bucket, path)

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

        with contextlib.closing(self.server.session.request(self.command, url,
                                                            headers=self.headers, data=data,
                                                            auth=self.server.auth,
                                                            allow_redirects=False)) as resp:
            self.send_response(resp.status_code, resp.reason)
            for key, value in resp.headers.items():
                if key.lower() not in ['date', 'server', 'transfer-encoding']:
                    self.send_header(key, value)
            self.end_headers()
            content = resp.content
            self.wfile.write(content)

    def log_message(self, format, *args):
        # Print to stdout instead of stderr so that it doesn't spew all over
        # the screen in normal operation.
        print(format % args)


class TestS3ChunkStoreToken(TestS3ChunkStore):
    """Test S3 with token authentication headers."""

    @classmethod
    def setup_class(cls):
        cls.httpd = None
        cls.httpd_thread = None
        super(TestS3ChunkStoreToken, cls).setup_class()

    @classmethod
    def teardown_class(cls):
        if cls.httpd:
            cls.httpd.session.close()
            cls.httpd.shutdown()
            cls.httpd_thread.join()
        super(TestS3ChunkStoreToken, cls).teardown_class()

    @classmethod
    def from_url(cls, url):
        """Create the chunk store"""
        proxy_host = '127.0.0.1'
        proxy_port = get_free_port(proxy_host)
        cls.httpd = http.server.HTTPServer((proxy_host, proxy_port), _TokenHTTPProxyHandler)
        cls.httpd.target = url
        cls.httpd.session = requests.Session()
        cls.httpd.auth = _AWSAuth(cls.credentials)
        cls.httpd_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.httpd_thread.start()
        proxy_url = 'http://{}:{}'.format(proxy_host, proxy_port)
        return S3ChunkStore.from_url(proxy_url, timeout=1, token='mysecret')
