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

"""A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API.

It does not support S3 authentication/signatures, relying instead on external
code to provide HTTP authentication.
"""
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
import future.utils
from builtins import object
from future.utils import raise_, bytes_to_native_str
import contextlib
import io
import threading
import queue
import sys
import urllib.parse
import urllib.request
import urllib.error
import hashlib
import base64
import re
import warnings
import copy
import json

import defusedxml.ElementTree
import defusedxml.cElementTree
import numpy as np
import requests
from requests.adapters import HTTPAdapter as _HTTPAdapter
try:
    import botocore.credentials
    import botocore.auth
except ImportError:
    botocore = None

from .chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk
from .sensordata import to_str


_BUCKET_POLICY = {
    "Version": "2012-10-17",
    "Id": "KatdalPolicy",
    "Statement": [
        {
            "Sid": "PublicAccess",
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": ["PLACEHOLDER"],
        }
    ]
}


class _TimeoutHTTPAdapter(_HTTPAdapter):
    """Allow an HTTPAdapter to have a default timeout"""
    def __init__(self, *args, **kwargs):
        self._default_timeout = kwargs.pop('timeout', None)
        super(_TimeoutHTTPAdapter, self).__init__(*args, **kwargs)

    def send(self, request, stream=False, timeout=None, *args, **kwargs):
        if timeout is None:
            timeout = self._default_timeout
        return super(_TimeoutHTTPAdapter, self).send(request, stream, timeout, *args, **kwargs)


class _BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        # Character set from RFC 6750
        if not re.match('^[A-Za-z0-9-._~+/]*$', token):
            raise ValueError('token contains invalid characters')
        self._token = token

    def __call__(self, r):
        r.headers['Authorization'] = 'Bearer ' + self._token
        return r


class _AWSAuth(requests.auth.AuthBase):
    def __init__(self, credentials):
        credentials = botocore.credentials.ReadOnlyCredentials(
            credentials[0], credentials[1], None)
        self._signer = botocore.auth.HmacV1Auth(credentials)

    def __call__(self, r):
        split = urllib.parse.urlsplit(r.url)
        signature = self._signer.get_signature(r.method, split, r.headers)
        r.headers['Authorization'] = 'AWS {}:{}'.format(
            self._signer.credentials.access_key, signature)
        return r


class _Multipart(object):
    """Allow a sequence of bytes-like objects to be used as a request body.

    This is intended to allow a zero-copy upload of bytes-like objects that
    are not contiguous in memory. The requests library treats standard
    iterable classes (list, tuple) specially, which is why a custom class is
    needed.
    """
    def __init__(self, items=()):
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    @property
    def len(self):
        """Total content length (retrieved by requests to set Content-Length)"""
        return sum(memoryview(item).nbytes for item in self.items)


def _raise_for_status(response):
    """Like :meth:`requests.Response.raise_for_status`, but uses ChunkStore exception types."""
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        if response.status_code == 404:
            raise ChunkNotFound(str(error))
        else:
            raise StoreUnavailable(str(error))


class _Pool(object):
    """Thread-safe pool of objects constructed by a factory as needed."""
    def __init__(self, factory):
        self._factory = factory
        self._pool = []
        self._lock = threading.Lock()

    def get(self):
        """Obtain an item from the pool, creating a new one if the pool is empty."""
        with self._lock:
            if not self._pool:
                return self._factory()
            else:
                return self._pool.pop()

    def put(self, item):
        """Return an item to the pool"""
        with self._lock:
            self._pool.append(item)

    @contextlib.contextmanager
    def __call__(self):
        """Context manager interface to get and put an item"""
        item = self.get()
        yield item
        self.put(item)


class S3ChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API.

    This object encapsulates the S3 client / session and its underlying
    connection pool, which allows subsequent get and put calls to share the
    connections.

    The full identifier of each chunk (the "chunk name") is given by

      "<bucket>/<path>/<idx>"

    where "<bucket>" refers to the relevant S3 bucket, "<bucket>/<path>" is
    the name of the parent array of the chunk and "<idx>" is the index string
    of each chunk (e.g. "00001_00512"). The corresponding S3 key string of
    a chunk is "<path>/<idx>.npy" which reflects the fact that the chunk is
    stored as a string representation of an NPY file (complete with header).

    Parameters
    ----------
    session_factory : callable
        A callable called with no arguments that returns an instance of
        :class:`requests.Session`. The returned session is only used by
        one thread at a time.
    url : str
        Base URL for the S3 service. It can be specified as either bytes or
        unicode, and is converted to the native string type with UTF-8.
    public_read : bool
        If set to true, new buckets will be created with a policy that allows
        everyone (including unauthenticated users) to read the data.

    Raises
    ------
    ImportError
        If requests is not installed (it's an optional dependency otherwise)
    """

    def __init__(self, session_factory, url, public_read=False):
        try:
            # Quick smoke test to see if the S3 server is available, by listing
            # buckets. Depending on the server in use, this may return a 403
            # error if we do not have credentials (this occurs for minio, but
            # Ceph RGW just returns an empty list).
            with session_factory() as session:
                with session.get(url) as response:
                    if (response.status_code != 403
                            or 'Authorization' in response.request.headers):
                        _raise_for_status(response)
        except requests.exceptions.RequestException as error:
            raise StoreUnavailable(str(error))

        error_map = {requests.exceptions.RequestException: StoreUnavailable,
                     defusedxml.ElementTree.ParseError: StoreUnavailable}
        super(S3ChunkStore, self).__init__(error_map)
        self._session_pool = _Pool(session_factory)
        self._url = to_str(url)
        self.public_read = public_read

    @classmethod
    def _from_url(cls, url, timeout, token, credentials, public_read):
        """Construct S3 chunk store from endpoint URL (see :meth:`from_url`)."""
        if token is not None:
            parsed = urllib.parse.urlparse(url)
            # The exception for 127.0.0.1 lets the unit test work
            if parsed.scheme != 'https' and parsed.hostname != '127.0.0.1':
                raise StoreUnavailable('auth token may only be used with https')
            auth = _BearerAuth(token)
        elif credentials is not None:
            if not botocore:
                raise StoreUnavailable('passing credentials requires botocore to be installed')
            auth = _AWSAuth(credentials)
        else:
            auth = None

        def session_factory():
            session = requests.Session()
            session.auth = auth
            adapter = _TimeoutHTTPAdapter(max_retries=2, timeout=timeout)
            session.mount(url, adapter)
            return session

        return cls(session_factory, url, public_read)

    @classmethod
    def from_url(cls, url, timeout=300, extra_timeout=10,
                 token=None, credentials=None, public_read=False, **kwargs):
        """Construct S3 chunk store from endpoint URL.

        Parameters
        ----------
        url : string
            Endpoint of S3 service, e.g. 'http://127.0.0.1:9000'
        timeout : int or float, optional
            Read / connect timeout, in seconds (set to None to leave unchanged)
        extra_timeout : int or float, optional
            Additional timeout, useful to terminate e.g. slow DNS lookups
            without masking read / connect errors (ignored if `timeout` is None)
        token : str
            Bearer token to authenticate
        credentials: tuple of str
            AWS access key and secret key to authenticate
        public_read : bool
            If set to true, new buckets will be created with a policy that allows
            everyone (including unauthenticated users) to read the data.
        kwargs : dict
            Extra keyword arguments (unused)

        Raises
        ------
        :exc:`chunkstore.StoreUnavailable`
            If S3 server interaction failed (it's down, no authentication, etc)
        """
        if token is not None and credentials is not None:
            raise ValueError('Cannot specify both token and credentials')

        # XXX This is a poor man's attempt at concurrent.futures functionality
        # (avoiding extra dependency on Python 2, revisit when Python 3 only)
        q = queue.Queue()

        def _from_url(url, timeout, token, credentials, public_read):
            """Construct chunk store and return it (or exception) via queue."""
            try:
                q.put(cls._from_url(url, timeout, token, credentials, public_read))
            except BaseException:
                q.put(sys.exc_info())

        thread = threading.Thread(target=_from_url,
                                  args=(url, timeout, token, credentials, public_read))
        thread.daemon = True
        thread.start()
        if timeout is not None:
            timeout += extra_timeout
        try:
            result = q.get(timeout=timeout)
        except queue.Empty:
            hostname = urllib.parse.urlparse(url).hostname
            raise StoreUnavailable('Timed out, possibly due to DNS lookup '
                                   'of {} stalling'.format(hostname))
        else:
            if isinstance(result, cls):
                return result
            else:
                # Assume result is (exception type, exception value, traceback)
                raise_(result[0], result[1], result[2])

    def _chunk_url(self, chunk_name):
        return urllib.parse.urljoin(self._url, to_str(urllib.parse.quote(chunk_name + '.npy')))

    @contextlib.contextmanager
    def _request(self, chunk_name, method, url, *args, **kwargs):
        """Run a request on a session from the pool, raising HTTP errors"""
        with self._standard_errors(chunk_name), self._session_pool() as session:
            with session.request(method, url, *args, **kwargs) as response:
                _raise_for_status(response)
                yield response

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        url = self._chunk_url(chunk_name)
        with self._request(chunk_name, 'GET', url, stream=True) as response:
            data = response.raw
            chunk = np.lib.format.read_array(data, allow_pickle=False)
        if chunk.shape != shape or chunk.dtype != dtype:
            raise BadChunk('Chunk {!r}: dtype {} and/or shape {} in store '
                           'differs from expected dtype {} and shape {}'
                           .format(chunk_name, chunk.dtype, chunk.shape,
                                   dtype, shape))
        return chunk

    def create_array(self, array_name):
        """See the docstring of :meth:`ChunkStore.create_array`."""
        # The array name is formatted as bucket/array, but we only need to create the bucket
        bucket = array_name.split(self.NAME_SEP)[0]
        url = urllib.parse.urljoin(self._url, to_str(urllib.parse.quote(bucket)))
        with self._standard_errors(), self._session_pool() as session:
            with session.put(url) as response:
                if response.status_code != 409:
                    # 409 indicates the bucket already exists
                    _raise_for_status(response)

        if self.public_read:
            policy_url = urllib.parse.urljoin(url, '?policy')
            policy = copy.deepcopy(_BUCKET_POLICY)
            policy['Statement'][0]['Resource'] = [
                'arn:aws:s3:::{}/*'.format(bucket),
                'arn:aws:s3:::{}'.format(bucket)
            ]
            with self._request(None, 'PUT', policy_url, data=json.dumps(policy)):
                pass

    @classmethod
    def _numpy_header(cls, chunk):
        fp = io.BytesIO()
        header_fields = np.lib.format.header_data_from_array_1_0(chunk)
        np.lib.format.write_array_header_1_0(fp, header_fields)
        return fp.getvalue()

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        chunk_name, _ = self.chunk_metadata(array_name, slices, chunk=chunk)
        url = self._chunk_url(chunk_name)
        # Note: don't use ascontiguousarray as it turns 0D into 1D.
        # See https://github.com/numpy/numpy/issues/5300
        chunk = np.asarray(chunk, order='C')
        npy_header = self._numpy_header(chunk)
        # Compute the MD5 sum to protect the object against corruption in
        # transmission.
        md5_gen = hashlib.md5(npy_header)
        md5_gen.update(chunk)
        md5 = base64.b64encode(md5_gen.digest())
        headers = {'Content-MD5': bytes_to_native_str(md5)}
        if future.utils.PY2:
            # Python 2's httplib doesn't support a sequence of byte-likes.
            data = npy_header + chunk.tobytes()
        else:
            data = _Multipart([npy_header, memoryview(chunk)])
        with self._request(chunk_name, 'PUT', url, headers=headers, data=data):
            pass

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, _ = self.chunk_metadata(array_name, slices, dtype=dtype)
        url = self._chunk_url(chunk_name)
        try:
            with self._request(chunk_name, 'HEAD', url):
                pass
        except ChunkNotFound:
            return False
        else:
            return True

    list_max_keys = 100000

    def list_chunk_ids(self, array_name):
        """See the docstring of :meth:`ChunkStore.list_chunk_ids`."""
        NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
        bucket, prefix = self.split(array_name, 1)
        url = urllib.parse.urljoin(self._url, to_str(urllib.parse.quote(bucket)))
        params = {
            'prefix': prefix,
            'max-keys': self.list_max_keys
        }

        keys = []
        more = True
        while more:
            with self._request(None, 'GET', url, params=params) as response:
                root = defusedxml.cElementTree.fromstring(response.content)
            keys.extend(child.text for child in root.iter(NS + 'Key'))
            truncated = root.find(NS + 'IsTruncated')
            more = (truncated is not None and truncated.text == 'true')
            if more:
                next_marker = root.find(NS + 'NextMarker')
                if next_marker:
                    params['marker'] = next_marker.text
                elif keys:
                    params['marker'] = keys[-1]
                else:
                    warnings.warn('Result had no keys but was marked as truncated')
                    more = False
        # Strip the array name and .npy extension to get the chunk ID string
        return [key[len(prefix) + 1:-4] for key in keys if key.endswith('.npy')]

    def mark_complete(self, array_name):
        """See the docstring of :meth:`ChunkStore.mark_complete`."""
        self.create_array(array_name)
        obj_name = self.join(array_name, 'complete')
        url = urllib.parse.urljoin(self._url, obj_name)
        with self._request(obj_name, 'PUT', url, data=b''):
            pass

    def is_complete(self, array_name):
        """See the docstring of :meth:`ChunkStore.is_complete`."""
        obj_name = self.join(array_name, 'complete')
        url = urllib.parse.urljoin(self._url, obj_name)
        try:
            with self._request(obj_name, 'GET', url):
                pass
        except ChunkNotFound:
            return False
        return True

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
    list_chunk_ids.__doc__ = ChunkStore.list_chunk_ids.__doc__
    mark_complete.__doc__ = ChunkStore.mark_complete.__doc__
    is_complete.__doc__ = ChunkStore.is_complete.__doc__
