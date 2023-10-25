################################################################################
# Copyright (c) 2017-2023, National Research Foundation (SARAO)
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

import contextlib
import http.server
import io
import os
import pathlib
import re
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
import warnings

import jwt
import katsdptelstate
import numpy as np
import requests
from katsdptelstate.rdb_writer import RDBWriter
import pytest
from numpy.testing import assert_array_equal
from urllib3.exceptions import IncompleteRead
from urllib3.util.retry import Retry

from katdal.chunkstore import ChunkNotFound, StoreUnavailable
from katdal.chunkstore_s3 import (
    _CHUNK_EXTENSION,
    _DEFAULT_SERVER_GLITCHES,
    InvalidToken,
    S3ChunkStore,
    S3ServerGlitch,
    _AWSAuth,
    AuthorisationFailed,
    decode_jwt,
    read_array,
    _read_object,
    _normalise_bucket_name,
)
from katdal.datasources import TelstateDataSource, DataSourceNotFound
from katdal.test.s3_utils import MissingProgram, S3Server, S3User
from katdal.test.test_chunkstore import ChunkStoreTestBase, generate_arrays
from katdal.test.test_datasources import (assert_telstate_data_source_equal,
                                          make_fake_data_source)

# Use a standard bucket for most tests to ensure a valid bucket name
# (regex '^[0-9a-z.-]{3,63}$'). While we are at it, go a step further
# and check the underscore-to-dash conversion in `S3ChunkStore.make_url`
# by including an underscore here.
BUCKET = 'katdal_unittest'
# Also authorise this prefix for tests that will make their own buckets
PREFIX = '1234567890'
# Pick quick but different timeouts and retries for unit tests:
#  - The effective connect timeout is 4.0 (initial) + 4.0 (retry #1)
#    + 0.2 (backoff) + 4.0 (retry #2) + 0.4 (backoff) + 4.0 (retry #3) = 16.6 seconds
#  - The effective read timeout is 2.0 + 3 * 2.0 + 0.1 * (0 + 2 + 4) = 8.6 seconds
#  - The effective status timeout is 0.1 * (0 + 2) = 0.2 seconds, or
#    3 * 0.1 + 0.2 = 0.5 second if the suggestions use SUGGESTED_STATUS_DELAY
TIMEOUT = (4.0, 2.0)
RETRY = Retry(connect=3, read=3, status=2, backoff_factor=0.1,
              status_forcelist=_DEFAULT_SERVER_GLITCHES)
SUGGESTED_STATUS_DELAY = 0.1
READ_PAUSE = 0.1
# Dummy private key for ES256 algorithm (taken from PyJWT unit tests)
JWT_PRIVATE_KEY = """
-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg2nninfu2jMHDwAbn
9oERUhRADS6duQaJEadybLaa0YShRANCAAQfMBxRZKUYEdy5/fLdGI2tYj6kTr50
PZPt8jOD23rAR7dhtNpG1ojqopmH0AH5wEXadgk8nLCT4cAPK59Qp9Ek
-----END PRIVATE KEY-----
"""


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


class TestReadArray:
    def _test(self, array, version=None):
        fp = io.BytesIO()
        np.lib.format.write_array(fp, array, version, allow_pickle=False)
        fp.seek(0)
        out = read_array(fp)
        np.testing.assert_equal(array, out)
        # Check that Fortran order was preserved
        assert array.strides == out.strides

    def testSimple(self):
        self._test(np.arange(20))

    def testMultiDim(self):
        self._test(np.arange(20).reshape(4, 5, 1))

    def testFortran(self):
        self._test(np.arange(20).reshape(4, 5, 1).T)

    def testV2(self):
        self._test(np.zeros(100), version=(2, 0))

    def testBadVersion(self):
        data = b'\x93NUMPY\x03\x04'     # Version 3.4
        fp = io.BytesIO(data)
        with pytest.raises(ValueError):
            read_array(fp)

    def testPickled(self):
        array = np.array([str, object])
        fp = io.BytesIO()
        np.save(fp, array)
        fp.seek(0)
        with pytest.raises(ValueError):
            read_array(fp)

    def _truncate_and_fail_to_read(self, *args):
        fp = io.BytesIO()
        np.save(fp, np.arange(20))
        fp.seek(*args)
        fp.truncate()
        fp.seek(0)
        with pytest.raises(IncompleteRead):
            read_array(fp)

    def testShort(self):
        # Chop off everything past first byte (in magic part of bytes)
        self._truncate_and_fail_to_read(1)
        # Chop off everything past byte 20 (in header part of bytes)
        self._truncate_and_fail_to_read(20)
        # Chop off last byte (in array part of bytes)
        self._truncate_and_fail_to_read(-1, 2)


@pytest.mark.parametrize('url,expected',
    [
        ('https://archive/bucket/key/000', 'https://archive/bucket/key/000'),
        ('https://archive/bucket/key/0_0', 'https://archive/bucket/key/0_0'),
        ('https://archive/bu_ket/key/0_0', 'https://archive/bu-ket/key/0_0'),
        ('https://a-chive/bu_ket/key/0-0', 'https://a-chive/bu-ket/key/0-0'),
        ('https://archive/just-a_bucket', 'https://archive/just-a-bucket'),
    ]
)
def test_normalise_bucket_name(url, expected):
    assert _normalise_bucket_name(url) == expected


def encode_jwt(payload):
    """Generate JWT token with ES256-encoded signature."""
    header = {'alg': 'ES256', 'typ': 'JWT'}
    return jwt.encode(payload, JWT_PRIVATE_KEY, algorithm='ES256', headers=header)


class TestTokenUtils:
    """Test token utility and validation functions."""

    def test_jwt_broken_token(self):
        payload = {'exp': 9234567890, 'iss': 'kat', 'prefix': ['123']}
        token = encode_jwt(payload)
        claims = decode_jwt(token)
        assert payload == claims
        # Token has invalid characters
        with pytest.raises(InvalidToken, match=r'does not have exactly two dots'):
            decode_jwt('** bad token **')
        # Token has invalid structure
        with pytest.raises(InvalidToken, match=r'does not have exactly two dots'):
            decode_jwt(token.replace('.', ''))
        # Token header failed to decode
        with pytest.raises(InvalidToken, match=r'Could not decode'):
            decode_jwt(token[1:])
        # Token payload failed to decode
        h, p, s = token.split('.')
        with pytest.raises(InvalidToken, match=r'Could not decode'):
            decode_jwt('.'.join((h, p[:-1], s)))
        # Token signature failed to decode or wrong length
        with pytest.raises(InvalidToken, match=r'too short'):
            decode_jwt(token[:-1])
        with pytest.raises(InvalidToken, match=r'too short'):
            decode_jwt(token[:-2])
        with pytest.raises(InvalidToken, match=r'too long'):
            decode_jwt(token + token[-4:])

    def test_jwt_expired_token(self):
        payload = {'exp': 0, 'iss': 'kat', 'prefix': ['123']}
        with pytest.raises(InvalidToken, match=r'Token expired at 01-Jan-1970'):
            decode_jwt(encode_jwt(payload))
        # Check that expiration time is not-too-large integer
        payload['exp'] = '1.2'
        with pytest.raises(InvalidToken, match=r'Expiration time must be an integer'):
            decode_jwt(encode_jwt(payload))
        payload['exp'] = 12345678901234567890
        with pytest.raises(InvalidToken, match=r'an integer that is not too large'):
            decode_jwt(encode_jwt(payload))
        # Check that it works without expiry date too
        del payload['exp']
        claims = decode_jwt(encode_jwt(payload))
        assert payload == claims


class TestS3ChunkStore(ChunkStoreTestBase):
    """Test S3 functionality against an actual (minio) S3 service."""

    @classmethod
    def start_minio(cls, host):
        """Start Fake S3 service on `host` and return its URL."""
        try:
            host = '127.0.0.1'        # Unlike 'localhost', guarantees IPv4
            with get_free_port(host) as port:
                pass
            # The port is now closed, which makes it available for minio to
            # bind to. While MinIO on Linux is able to bind to the same port
            # as the socket held open by get_free_port, Mac OS is not.
            cls.minio = S3Server(host, port, pathlib.Path(cls.tempdir), S3User(*cls.credentials))
        except MissingProgram as exc:
            pytest.skip(str(exc))
        return cls.minio.url

    @classmethod
    def prepare_store_args(cls, url, **kwargs):
        """Prepare the arguments used to construct `S3ChunkStore`."""
        kwargs.setdefault('timeout', TIMEOUT)
        kwargs.setdefault('retries', RETRY)
        kwargs.setdefault('credentials', cls.credentials)
        return url, kwargs

    @classmethod
    def setup_class(cls):
        """Start minio service running on temp dir, and ChunkStore on that."""
        cls.arrays = generate_arrays()
        cls.credentials = ('access*key', 'secret*key')
        cls.tempdir = tempfile.mkdtemp()
        cls.minio = None
        try:
            cls.s3_url = cls.start_minio('127.0.0.1')
            cls.store_url, cls.store_kwargs = cls.prepare_store_args(cls.s3_url)
            cls.store = S3ChunkStore(cls.store_url, **cls.store_kwargs)
            # Ensure that pagination is tested
            cls.store.list_max_keys = 3
        except Exception:
            cls.teardown_class()
            raise

    @classmethod
    def teardown_class(cls):
        if cls.minio:
            cls.minio.close()
        shutil.rmtree(cls.tempdir)

    def setup_method(self):
        # The server is a class-level fixture (for efficiency), so state can
        # leak between tests. Prevent that by removing any existing objects.
        # It's easier to do that by manipulating the filesystem directly than
        # trying to use the S3 API.
        data_dir = os.path.join(self.tempdir, 'data')
        for entry in os.scandir(data_dir):
            if not entry.name.startswith('.') and entry.is_dir():
                shutil.rmtree(entry.path)
        # Also get rid of the cache of verified buckets
        self.store._verified_buckets.clear()
        self.store.timeout = TIMEOUT
        self.store.retries = RETRY
        print(f"Chunk store: {self.store_url}, S3 server: {self.s3_url}")

    def array_name(self, name):
        """Ensure that bucket is authorised and has valid name."""
        if name.startswith(PREFIX):
            return name
        return self.store.join(BUCKET, name)

    def test_chunk_non_existent(self):
        # An empty bucket will trigger StoreUnavailable so put something in there first
        self.store.mark_complete(self.array_name('crumbs'))
        return super().test_chunk_non_existent()

    def test_public_read(self):
        url, kwargs = self.prepare_store_args(self.s3_url, credentials=None)
        reader = S3ChunkStore(url, **kwargs)
        # Create a non-public-read array.
        # This test deliberately doesn't use array_name so that it can create
        # several different buckets.
        slices = np.index_exp[0:5]
        x = np.arange(5)
        self.store.create_array('private/x')
        self.store.put_chunk('private/x', slices, x)
        with pytest.raises(AuthorisationFailed):
            reader.get_chunk('private/x', slices, x.dtype)

        # Now a public-read array
        url, kwargs = self.prepare_store_args(self.s3_url, public_read=True)
        store = S3ChunkStore(url, **kwargs)
        store.create_array('public/x')
        store.put_chunk('public/x', slices, x)
        y = reader.get_chunk('public/x', slices, x.dtype)
        np.testing.assert_array_equal(x, y)

    # If you connect to localhost on a port that is not listening, you
    # immediately get a "connection refused" response on Linux (but not on macOS)
    @pytest.mark.expected_duration(0.0 if sys.platform == 'linux' else 0.1)
    def test_store_unavailable_unresponsive_server(self):
        host = '127.0.0.1'
        with get_free_port(host) as port:
            url = f'http://{host}:{port}/'
            store = S3ChunkStore(url, timeout=0.1, retries=0)
            with pytest.raises(StoreUnavailable):
                store.is_complete('store_is_not_listening_on_that_port')

    def test_token_without_https(self):
        # Don't allow users to leak their tokens by accident
        with pytest.raises(StoreUnavailable):
            S3ChunkStore('http://apparently.invalid/', token='secrettoken')

    def test_mark_complete_top_level(self):
        self._test_mark_complete(PREFIX + '-completetest')

    def test_rdb_support(self, suggestion=''):
        telstate = katsdptelstate.TelescopeState()
        view, cbid, sn, _, _ = make_fake_data_source(telstate, self.store, (5, 16, 40), PREFIX)
        telstate['capture_block_id'] = cbid
        telstate['stream_name'] = sn
        # Save telstate to temp RDB file since RDBWriter needs a filename and not a handle
        rdb_filename = f'{cbid}_{sn}.rdb'
        temp_filename = os.path.join(self.tempdir, rdb_filename)
        with RDBWriter(temp_filename) as rdbw:
            rdbw.save(telstate)
        # Read the file back in and upload it to S3
        with open(temp_filename, mode='rb') as rdb_file:
            rdb_data = rdb_file.read()
        if suggestion:
            rdb_path = self.store.join(suggestion, cbid, rdb_filename)
        else:
            rdb_path = self.store.join(cbid, rdb_filename)
        rdb_url = self.store.make_url(rdb_path)
        self.store.create_array(cbid)
        self.store.request('PUT', rdb_url, data=rdb_data)
        # Check that data source can be constructed from URL (with auto chunk store)
        source_from_url = TelstateDataSource.from_url(rdb_url, **self.store_kwargs)
        source_direct = TelstateDataSource(view, cbid, sn, self.store)
        assert_telstate_data_source_equal(source_from_url, source_direct)

    def test_missing_or_empty_buckets(self):
        slices = (slice(0, 1),)
        dtype = np.dtype(float)
        # Without create_array the bucket is missing
        with pytest.raises(StoreUnavailable):
            self.store.get_chunk(f'{BUCKET}-missing/x', slices, dtype)
        self.store.create_array(f'{BUCKET}-empty/x')
        # Without put_chunk the bucket is empty
        with pytest.raises(StoreUnavailable):
            self.store.get_chunk(f'{BUCKET}-empty/x', slices, dtype)
        # Check that the standard bucket has not been verified yet
        bucket_url = self.store.make_url(BUCKET)
        assert bucket_url not in self.store._verified_buckets
        # Check that the standard bucket remains verified after initial check
        self.test_chunk_non_existent()
        assert bucket_url in self.store._verified_buckets


class _TokenHTTPProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP proxy that substitutes AWS credentials in place of a bearer token."""

    def __getattr__(self, name):
        """Handle all HTTP requests by the same method since this is a proxy."""
        if name.startswith('do_'):
            return self.do_all
        return self.__getattribute__(name)

    def _reset_connection(self):
        """Send a TCP reset (RST) packet to reset the connection.

        Enable SO_LINGER with a linger interval of 0 s. This drops the
        connection like a hot potato once closed.
        """
        l_onoff = 1  # non-zero value enables linger option in kernel
        l_linger = 0  # timeout interval in seconds
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                   struct.pack('ii', l_onoff, l_linger))

    def do_all(self):
        # See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Connection
        HOP_HEADERS = {
            'keep-alive', 'transfer-encoding', 'te', 'connection', 'trailer',
            'upgrade', 'proxy-authorization', 'proxy-authenticate'
        }
        self.protocol_version = 'HTTP/1.1'
        data_len = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(data_len)
        truncate = False
        pause = 0.0
        reset = False
        glitch_location = 0

        # Extract a proxy suggestion prepended to the path
        suggestion = re.search(r'/please-([^/]+?)(?:-for-([\d\.]+)-seconds)?/',
                               self.path)
        if suggestion:
            # Check when this exact request (including suggestion) was first made
            key = self.requestline
            initial_time = self.server.initial_request_time.setdefault(key, time.time())
            # Remove suggestion from path
            start, end = suggestion.span()
            self.path = self.path[:start] + '/' + self.path[end:]
            command, duration = suggestion.groups()
            duration = float(duration) if duration else np.inf
            # If the suggestion is still active, go ahead with it
            if time.time() < initial_time + duration:
                # Respond with the suggested status code for a while
                respond_with = re.match(r'^respond-with-(\d+)$', command)
                if respond_with:
                    status_code = int(respond_with.group(1))
                    time.sleep(SUGGESTED_STATUS_DELAY)
                    self.send_response(status_code, 'Suggested by unit test')
                    self.end_headers()
                    return
                if command == 'reset-connection':
                    self._reset_connection()
                    return
                # Truncate or pause transmission of the payload after specified bytes
                glitch = re.match(r'^(truncate|pause|reset)-read-after-(\d+)-bytes$', command)
                if glitch:
                    flavour = glitch.group(1)
                    truncate = (flavour == 'truncate')
                    pause = READ_PAUSE if flavour == 'pause' else 0.0
                    reset = (flavour == 'reset')
                    glitch_location = int(glitch.group(2))
                else:
                    raise ValueError(f"Unknown command '{command}' "
                                     f'in proxy suggestion {suggestion}')
            else:
                # We're done with this suggestion since its time ran out
                del self.server.initial_request_time[key]

        # Extract token, validate it and check if path is authorised by it
        auth_header = self.headers.get('Authorization').split()
        if len(auth_header) == 2 and auth_header[0] == 'Bearer':
            token = auth_header[1]
        else:
            token = ''
        try:
            prefixes = decode_jwt(token).get('prefix', [])
        except InvalidToken:
            prefixes = []
        if not any(self.path.lstrip('/').startswith(prefix) for prefix in prefixes):
            self.send_response(401, f'Unauthorized (got: {self.path}, allowed: {prefixes})')
            self.end_headers()
            return

        # Clear out hop-by-hop headers
        request_headers = dict(self.headers.items())
        for header in self.headers:
            if header.lower() in HOP_HEADERS:
                del request_headers[header]

        url = urllib.parse.urljoin(self.server.target, self.path)
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
        # Quit early if there is no data to write to avoid broken pipes (since the client
        # might stop listening if it knows nothing more is coming, like in a PUT response).
        if len(content) == 0:
            return
        # Write the first half of the data
        self.wfile.write(content[:glitch_location])
        if truncate:
            return
        elif reset:
            self._reset_connection()
            return
        elif pause:
            # The wfile object should be an unbuffered _SocketWriter but flush anyway
            self.wfile.flush()
            time.sleep(pause)
        # Write the rest of the data
        self.wfile.write(content[glitch_location:])

    def log_message(self, format, *args):
        # Get time offset from first of these requests (useful for debugging)
        # XXX Could also use args[0] instead of requestline, not sure which is best
        key = self.requestline
        now = time.time()
        # Print 0.0 for a fresh suggestion and -1.0 for a stale / absent suggestion (no key found)
        initial_time = self.server.initial_request_time.get(key, now + 1.0)
        time_offset = now - initial_time
        # Print to stdout instead of stderr so that it doesn't spew all over
        # the screen in normal operation.
        print(f"Token proxy: {self.log_date_time_string()} ({time_offset:.3f}) {format % args}")


class _TokenHTTPProxyServer(http.server.HTTPServer):
    """Server for use with :class:`_TokenHTTPProxyHandler`.

    It sets SO_REUSEPORT so that it is compatible with a socket created by
    :func:`get_free_port`, including on OS X.
    """
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


class TestS3ChunkStoreToken(TestS3ChunkStore):
    """Test S3 with token authentication headers."""

    @classmethod
    def setup_class(cls):
        cls.proxy_url = None
        cls.httpd = None
        super().setup_class()

    @classmethod
    def teardown_class(cls):
        if cls.httpd is not None:
            cls.httpd.session.close()
            cls.httpd.shutdown()
            cls.httpd = None
            cls.httpd_thread.join()
            cls.httpd_thread = None
        super().teardown_class()

    @classmethod
    def prepare_store_args(cls, url, **kwargs):
        """Prepare the arguments used to construct `S3ChunkStore`."""
        if cls.httpd is None:
            proxy_host = '127.0.0.1'
            with get_free_port(proxy_host) as proxy_port:
                httpd = _TokenHTTPProxyServer((proxy_host, proxy_port), _TokenHTTPProxyHandler)
            httpd.target = url
            httpd.session = requests.Session()
            httpd.auth = _AWSAuth(cls.credentials)
            httpd.initial_request_time = {}
            cls.httpd_thread = threading.Thread(target=httpd.serve_forever)
            cls.httpd_thread.start()
            # We delay setting cls.httpd until we've launched serve_forever,
            # because teardown calls httpd.shutdown and that hangs if
            # serve_forever wasn't called.
            cls.httpd = httpd
            cls.proxy_url = f'http://{proxy_host}:{proxy_port}'
        elif url != cls.httpd.target:
            raise RuntimeError('Cannot use multiple target URLs with http proxy')
        # The token authorises the standard bucket and anything starting with PREFIX
        # (as well as suggestions prepended to the path)
        token = encode_jwt({'prefix': [BUCKET.replace('_', '-'), PREFIX, 'please']})
        kwargs.setdefault('token', token)
        return super().prepare_store_args(cls.proxy_url, credentials=None, **kwargs)

    def test_public_read(self):
        # Disable this test defined in the base class because it involves creating
        # buckets, which is not done with tokens but rather with credentials.
        pass

    def test_unauthorised_bucket(self):
        with pytest.raises(InvalidToken):
            self.store.is_complete('unauthorised_bucket')

    def _put_chunk(self, suggestion):
        """Put a chunk into the store and form an array name containing suggestion."""
        var_name = 'x'
        slices = (slice(3, 5),)
        array_name = self.array_name(var_name)
        chunk = self.arrays[var_name][slices]
        self.store.create_array(array_name)
        self.store.put_chunk(array_name, slices, chunk)
        return chunk, slices, self.store.join(array_name, suggestion)

    # 50x STATUSES
    #
    # With the RETRY settings of 2 status retries, backoff factor of 0.1 s
    # and SUGGESTED_STATUS_DELAY of 0.1 s we get the following timeline
    # (indexed by seconds):
    # 0.0 - access chunk for the first time
    # 0.1 - response is 500, immediately try again (retry #1)
    # 0.2 - response is 500, back off for 2 * 0.1 seconds
    # 0.4 - retry #2 (the final attempt) - server should now be fixed
    # 0.4 - success!

    @pytest.mark.expected_duration(0.1)
    def test_server_error(self):
        suggestion = 'please-respond-with-500-for-0.2-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        chunk_name, _ = self.store.chunk_metadata(array_name, slices, dtype=chunk.dtype)
        url = self.store.make_url(chunk_name + _CHUNK_EXTENSION)
        response = self.store.request('GET', url, ignored_errors=(500,), retries=0, stream=True)
        assert response.status_code == 500

    @pytest.mark.expected_duration(0.4)
    def test_recover_from_server_errors(self):
        suggestion = 'please-respond-with-500-for-0.3-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, 'Bad chunk after server error')

    @pytest.mark.expected_duration(0.5)
    def test_persistent_server_errors(self):
        suggestion = 'please-respond-with-502-for-0.7-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        # After 0.4 seconds the client gives up and returns with failure 0.1 s later
        with pytest.raises(ChunkNotFound):
            self.store.get_chunk(array_name, slices, chunk.dtype)

    # TRUNCATED READS
    #
    # With the RETRY settings of 3 read retries and backoff factor of 0.1 s
    # we get the following timeline (indexed by seconds):
    # 0.0 - access chunk for the first time
    # 0.0 - response is 200 but truncated, immediately try again (retry #1)
    # 0.0 - response is 200 but truncated, back off for 2 * 0.1 seconds
    # 0.2 - retry #2, response is 200 but truncated, back off for 4 * 0.1 seconds
    # 0.6 - retry #3 (the final attempt) - server should now be fixed
    # 0.6 - success!

    # The NPY file has a 128-byte header, followed by the array data itself.
    # Check both parts, since they are read somewhat differently (read vs readinto).
    @pytest.mark.parametrize('nbytes', [60, 129])
    @pytest.mark.expected_duration(0.0)
    def test_truncated_read(self, nbytes):
        suggestion = f'please-truncate-read-after-{nbytes}-bytes-for-0.1-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        self.store.retries = Retry(0)
        with pytest.raises(S3ServerGlitch) as excinfo:
            self.store.get_chunk(array_name, slices, chunk.dtype)
        bytes_left = 128 + chunk.nbytes - nbytes
        excinfo.match(fr'IncompleteRead\({nbytes} bytes read, {bytes_left} more expected\)')

    @pytest.mark.parametrize('nbytes', [60, 129])
    @pytest.mark.expected_duration(0.6)
    def test_recover_from_truncated_read(self, nbytes):
        suggestion = f'please-truncate-read-after-{nbytes}-bytes-for-0.4-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, 'Bad chunk after truncated read')

    @pytest.mark.expected_duration(0.6)
    def test_persistent_truncated_reads(self):
        suggestion = 'please-truncate-read-after-60-bytes-for-0.8-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        # After 0.6 seconds the client gives up
        with pytest.raises(ChunkNotFound):
            self.store.get_chunk(array_name, slices, chunk.dtype)

    def test_rdb_support_recover_from_truncated_reads(self):
        super().test_rdb_support('please-truncate-read-after-1000-bytes-for-0.4-seconds')

    def test_rdb_support_persistent_truncated_reads(self):
        with pytest.raises(DataSourceNotFound) as excinfo:
            super().test_rdb_support('please-truncate-read-after-1000-bytes-for-0.8-seconds')
        excinfo.match(r'IncompleteRead\(1000 bytes read')

    @pytest.mark.expected_duration(0.6)
    def test_recover_from_reset_connections(self):
        suggestion = 'please-reset-read-after-129-bytes-for-0.4-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, 'Bad chunk after reset connection')

    @pytest.mark.expected_duration(0.6)
    def test_persistent_reset_connections(self):
        suggestion = 'please-reset-read-after-129-bytes-for-0.8-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        with pytest.raises(ChunkNotFound) as excinfo:
            self.store.get_chunk(array_name, slices, chunk.dtype)
        assert isinstance(excinfo.value, S3ServerGlitch)

    @pytest.mark.expected_duration(0.6)
    def test_persistent_early_reset_connections(self):
        suggestion = 'please-reset-connection-for-0.8-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        with pytest.raises(StoreUnavailable):
            self.store.get_chunk(array_name, slices, chunk.dtype)

    @pytest.mark.parametrize('nbytes', [60, 129])  # check both NPY header and array itself
    @pytest.mark.expected_duration(READ_PAUSE)
    def test_handle_paused_read(self, nbytes):
        suggestion = f'please-pause-read-after-{nbytes}-bytes'
        chunk, slices, array_name = self._put_chunk(suggestion)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, 'Bad chunk after paused read')

    # SOCKET TIMEOUTS
    #
    # With a read timeout of 0.09 seconds and the RETRY settings of 3 read retries and
    # backoff factor of 0.1 s, we get the following timeline (indexed by seconds):
    # 0.00 - access chunk for the first time
    # 0.09 - response is 200 but socket times out after 0.09 seconds, immediately try again
    # 0.18 - retry #1, response is 200 but socket stalls, back off for 2 * 0.1 seconds
    # 0.47 - retry #2, response is 200 but socket stalls, back off for 4 * 0.1 seconds
    # 0.87 - retry #3 (the final attempt) - server should now be fixed
    # 0.87 - success!

    @pytest.mark.parametrize('nbytes', [60, 129])  # check both NPY header and array itself
    @pytest.mark.expected_duration(0.87)
    def test_recover_from_socket_timeout(self, nbytes):
        suggestion = f'please-pause-read-after-{nbytes}-bytes-for-0.8-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        self.store.timeout = (5.0, 0.09)
        chunk_retrieved = self.store.get_chunk(array_name, slices, chunk.dtype)
        assert_array_equal(chunk_retrieved, chunk, 'Bad chunk after socket timeout')

    @pytest.mark.expected_duration(0.96)
    def test_persistent_socket_timeouts(self):
        suggestion = 'please-pause-read-after-129-bytes-for-1.0-seconds'
        chunk, slices, array_name = self._put_chunk(suggestion)
        self.store.timeout = (5.0, 0.09)
        # The final retry starts at 0.87 seconds and the client gives up 0.09 seconds later
        with pytest.raises(ChunkNotFound):
            self.store.get_chunk(array_name, slices, chunk.dtype)

    # XXX Check whether the 'pause' condition (socket timeouts) can be tested
    # with _read_object, which preloads the content due to stream=False.
    @pytest.mark.parametrize('condition', ['truncate', 'reset'])
    @pytest.mark.expected_duration(0.6)
    def test_non_streaming_request_recovery_from_glitches(self, condition):
        data = b'x' * 1000
        cbid = PREFIX
        path = self.store.join(cbid, f'test_recovery_from_{condition}.bin')
        url = self.store.make_url(path)
        self.store.create_array(cbid)
        self.store.request('PUT', url, data=data)
        suggestion = f'please-{condition}-read-after-400-bytes-for-0.52-seconds'
        url = self.store.make_url(self.store.join(suggestion, path))
        retrieved_data = self.store.request('GET', url, process=_read_object)
        assert retrieved_data == data
