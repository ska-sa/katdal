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

"""Tests for :py:mod:`katdal.chunkstore_s3`.

The tests require `minio`_ to be installed on the :envvar:`PATH`. If not found,
the test will be skipped.

Versions of minio prior to 2018-08-25T01:56:38Z contain a `race condition`_
that can cause it to crash when queried at the wrong point during startup. If
an older version is detected, the test will be skipped.

.. _minio: https://github.com/minio/minio
.. _race condition: https://github.com/minio/minio/issues/6324
"""
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()     # noqa: E402
import tempfile
import shutil
# Using subprocess32 is important (on 2.7) because it closes non-stdio file
# descriptors in the child. Without that, OS X runs into problems with minio
# failing to bind the socket.
import subprocess32 as subprocess
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
    port with SO_REUSEPORT, after which the context should be exited to close
    the temporary socket.
    """
    with contextlib.closing(socket.socket()) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        yield port


class TestS3ChunkStore(ChunkStoreTestBase):
    """Test S3 functionality against an actual (minio) S3 service."""

    @classmethod
    def start_minio(cls, host):
        """Start Fake S3 service on `host` and return its URL."""

        # Check minio version
        try:
            version_data = subprocess.check_output(['minio', 'version'])
        except OSError as e:
            raise SkipTest('Could not run minio (is it installed): {}'.format(e))
        except subprocess.CalledProcessError:
            raise SkipTest('Failed to get minio version (is it too old)?')

        min_version = u'2018-08-25T01:56:38Z'
        version = None
        version_fields = version_data.decode('utf-8').splitlines()
        for line in version_fields:
            if line.startswith(u'Version: '):
                version = line.split(u' ', 1)[1]
        if version is None:
            raise RuntimeError('Could not parse minio version')
        elif version < min_version:
            raise SkipTest(u'Minio version is {} but {} is required'.format(version, min_version))

        with get_free_port(host) as port:
            try:
                env = os.environ.copy()
                env['MINIO_BROWSER'] = 'off'
                env['MINIO_ACCESS_KEY'] = cls.credentials[0]
                env['MINIO_SECRET_KEY'] = cls.credentials[1]
                cls.minio = subprocess.Popen(['minio', 'server',
                                              '--quiet',
                                              '--address', '{}:{}'.format(host, port),
                                              '-C', os.path.join(cls.tempdir, 'config'),
                                              os.path.join(cls.tempdir, 'data')],
                                             stdout=subprocess.DEVNULL,
                                             env=env)
            except OSError:
                raise SkipTest('Could not start minio server (is it installed?)')

        # Wait for minio to be ready to service requests
        url = 'http://%s:%s' % (host, port)
        health_url = urllib.parse.urljoin(url, '/minio/health/live')
        for i in range(100):
            try:
                with requests.get(health_url) as resp:
                    if resp.status_code == 200:
                        return url
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        raise OSError('Timed out waiting for minio to be ready')

    @classmethod
    def from_url(cls, url, authenticate=True, **kwargs):
        """Create the chunk store"""
        if authenticate:
            kwargs['credentials'] = cls.credentials
        return S3ChunkStore.from_url(url, timeout=10, **kwargs)

    @classmethod
    def setup_class(cls):
        """Start minio service running on temp dir, and ChunkStore on that."""
        cls.credentials = ('access*key', 'secret*key')
        cls.tempdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(cls.tempdir, 'config'))
        os.mkdir(os.path.join(cls.tempdir, 'data'))
        cls.minio = None
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
        # See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Connection
        HOP_HEADERS = {
            'keep-alive', 'transfer-encoding', 'te', 'connection', 'trailer',
            'upgrade', 'proxy-authorization', 'proxy-authenticate'
        }
        self.protocol_version = 'HTTP/1.1'
        url = urllib.parse.urljoin(self.server.target, self.path)
        data_len = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(data_len)
        if self.headers.get('Authorization') != 'Bearer mysecret':
            self.send_response(401, 'Unauthorized')
            self.end_headers()
            return

        # Clear out hop-to-hop headers
        request_headers = dict(self.headers.items())
        for header in self.headers:
            if header.lower() in HOP_HEADERS:
                del request_headers[header]

        try:
            with self.server.session.request(self.command, url,
                                             headers=request_headers, data=data,
                                             auth=self.server.auth,
                                             allow_redirects=False,
                                             timeout=5) as resp:
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
            # The base class automatically sets Date and Server headers
            if key.lower() not in HOP_HEADERS.union({'date', 'server'}):
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Print to stdout instead of stderr so that it doesn't spew all over
        # the screen in normal operation.
        print(format % args)


class _TokenHTTPProxyServer(http.server.HTTPServer):
    """Server for use with :class:`_TokenHTTPProxyHandler`.

    It sets SO_REUSEPORT so that it is compatible with a socket created by
    :func:`get_free_port`, including on OS X.
    """
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # In Python 2.7 it's an old-style class, so super doesn't work
        http.server.HTTPServer.server_bind(self)


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
            return S3ChunkStore.from_url(url, timeout=10, **kwargs)

        if cls.httpd is None:
            proxy_host = '127.0.0.1'
            with get_free_port(proxy_host) as proxy_port:
                httpd = _TokenHTTPProxyServer((proxy_host, proxy_port), _TokenHTTPProxyHandler)
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
        return S3ChunkStore.from_url(cls.proxy_url, timeout=10, token='mysecret', **kwargs)
