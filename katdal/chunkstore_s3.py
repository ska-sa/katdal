################################################################################
# Copyright (c) 2017-2019, National Research Foundation (Square Kilometre Array)
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

"""A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: E402
from builtins import object
import future.utils
from future.utils import bytes_to_native_str, raise_from

import contextlib
import threading
import urllib.parse
import urllib.request
import urllib.error
import hashlib
import base64
import warnings
import copy
import json
import time

import defusedxml.ElementTree
import defusedxml.cElementTree
import numpy as np
import requests
import jwt
try:
    import botocore.credentials
    import botocore.auth
except ImportError:
    botocore = None
from urllib3.util.retry import Retry
from urllib3.response import HTTPResponse
from urllib3.exceptions import MaxRetryError

from .chunkstore import (ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk,
                         npy_header_and_body)
from .sensordata import to_str


# Lifecycle policies unfortunately use XML encoding rather than JSON
# Following path of least resistance we simply .format() this string
# with the number of days for the expiry (and produced a sanitised
# ID at the same time).
_BASE_LIFECYCLE_POLICY = """<?xml version="1.0" encoding="UTF-8"?>
<LifecycleConfiguration><Rule>
<ID>katdal_expiry_{0}_days</ID><Filter></Filter><Status>Enabled</Status>
<Expiration><Days>{0}</Days></Expiration>
</Rule></LifecycleConfiguration>"""

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

# Fake HTTP status code associated with reset connections / truncated data
_TRUNCATED_HTTP_STATUS_CODE = 555
# These HTTP responses typically indicate temporary S3 server / proxy overload,
# which will trigger retries terminating in a missing data response if unsuccessful.
_DEFAULT_SERVER_GLITCHES = (500, 502, 503, 504, _TRUNCATED_HTTP_STATUS_CODE)


class S3ObjectNotFound(ChunkNotFound):
    """An object / bucket was not found in S3 object store."""


class S3ServerGlitch(ChunkNotFound):
    """S3 chunk store responded with an HTTP error deemed to be temporary."""

    def __init__(self, msg, status_code):
        super(S3ServerGlitch, self).__init__(msg)
        self.status_code = status_code


class TruncatedRead(ValueError):
    """HTTP request to S3 chunk store responded with fewer bytes than expected."""


class _DetectTruncation(object):
    """Raise :exc:`TruncatedRead` if wrapped `readable` runs out of data."""

    def __init__(self, readable):
        self._readable = readable

    def __getattr__(self, name):
        """Proxy all attributes to underlying wrapped object."""
        return getattr(self._readable, name)

    def read(self, size, *args, **kwargs):
        """Overload `read` method to detect truncated data source."""
        data = self._readable.read(size, *args, **kwargs)
        if data == b'' and size != 0:
            raise TruncatedRead('Error reading from S3 HTTP response: expected '
                                '{} more byte(s), got EOF'.format(size))
        return data


def read_array(fp):
    """Read a numpy array in npy format from a file descriptor.

    This is the same concept as :func:`numpy.lib.format.read_array`, but
    optimised for the case of reading from :class:`http.client.HTTPResponse`.
    Using the numpy function reads pieces out then copies them into the
    array, while this implementation uses `readinto`. Raise :class:`TruncatedRead`
    if the response runs out of data before the array is complete.

    It does not allow pickled dtypes.
    """
    # Wrap file object in _DetectTruncation since data can run out while
    # within the bowels of NumPy (the alternative is monkey-patching NumPy...)
    fp = _DetectTruncation(fp)
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError('Unsupported .npy version {}'.format(version))
    if dtype.hasobject:
        raise ValueError('Object arrays are not supported')
    count = int(np.product(shape))
    data = np.ndarray(count, dtype=dtype)
    # For HTTPResponse it works to just pass in `data` directly, but the
    # wrapping is added for the benefit of any other implementation that
    # isn't expecting a numpy array
    bytes_read = fp.readinto(memoryview(data.view(np.uint8)))
    if bytes_read != data.nbytes:
        raise TruncatedRead('Error reading from S3 HTTP response: expected {} '
                            'bytes, got {}'.format(data.nbytes, bytes_read))
    if fortran_order:
        data.shape = shape[::-1]
        data = data.transpose()
    else:
        data.shape = shape
    return data


def _read_chunk(response):
    """Efficiently read NumPy array in NPY format from content of HTTP response."""
    data = response.raw
    # Workaround for https://github.com/urllib3/urllib3/issues/1540
    # On Python 2, http.client.HTTPResponse doesn't implement readinto.
    # We also can't use the workaround if the content is encoded (e.g.
    # gzip compressed) because that's decoded in urllib3, not httplib.
    if ('Content-encoding' not in response.headers
            and hasattr(data, '_fp')
            and hasattr(data._fp, 'readinto')):
        chunk = read_array(data._fp)
    else:
        chunk = read_array(data)
    # This shouldn't actually read any data, but will make requests aware that
    # we've consumed all the data and hence it can reuse the connection.
    response.content
    return chunk


class AuthorisationFailed(StoreUnavailable):
    """Authorisation failed, e.g. due to invalid, malformed or expired token."""


class InvalidToken(AuthorisationFailed):
    """Invalid JSON Web Token (JWT)."""

    def __init__(self, token, message):
        # Shorten token string but keep the ends so user can check for truncation
        if len(token) > 17:
            token = '{}[...]{}'.format(token[:6], token[-6:])
        super(InvalidToken, self).__init__("'{}': {}".format(token, message))


def decode_jwt(token):
    """Decode JSON Web Token (JWT) string and extract claims.

    The MeerKAT archive uses JWT bearer tokens for authorisation. Each token is
    a JSON Web Signature (JWS) string with a payload of claims. This function
    extracts the claims as a dict, while also doing basic checks on the token
    (mostly to catch copy-n-paste errors). The signature is decoded but not
    validated, since that would require the server secrets.

    Parameters
    ----------
    token : str
        JWS Compact Serialization as an ASCII string (native string, not bytes)

    Returns
    -------
    claims : dict
        The JWT Claims Set as a dict of key-value pairs

    Raises
    ------
    :exc:`InvalidToken`
        If the token is malformed or truncated, or has expired
    """
    # A valid JWS Compact Serialization has three segments
    try:
        encoded_header, encoded_payload, encoded_signature = token.split('.')
    except ValueError:
        raise InvalidToken(token, "Token does not have exactly two dots ('.') "
                                  "as expected of JWS (maybe it's truncated?)")
    # Remove signature to avoid cryptic PyJWT error message ("Invalid crypto padding")
    token_without_sig = '{}.{}.'.format(encoded_header, encoded_payload)
    try:
        header = jwt.get_unverified_header(token_without_sig)
    except jwt.exceptions.DecodeError as err:
        raise_from(InvalidToken(token, "Could not decode token - maybe it's truncated "
                                       "or corrupted? ({})".format(err)), err)
    # Check signature length for typical MeerKAT signature algorithm
    if header.get('alg') == 'ES256':
        len_sig = len(encoded_signature)
        if len_sig != 86:   # 64 bytes when decoded
            msg = 'Encoded signature has {} bytes instead of 86 - token string ' \
                  'is too {}'.format(len_sig, 'short' if len_sig < 86 else 'long')
            raise InvalidToken(token, msg)
    # Extract token claims and verify everything except signature and audience
    options = {'verify_signature': False, 'verify_aud': False}
    try:
        claims = jwt.decode(token, options=options)
    except jwt.exceptions.DecodeError as err:
        # This covers e.g. bad characters in the signature or non-JSON-dict payload
        raise_from(InvalidToken(token, "Could not decode token - maybe it's truncated "
                                       "or corrupted? ({})".format(err)), err)
    except jwt.exceptions.ExpiredSignatureError as err:
        claims = jwt.decode(token, verify=False)
        exp_time = time.strftime('%d-%b-%Y %H:%M:%S', time.gmtime(claims['exp']))
        raise_from(InvalidToken(token, 'Token expired at {} UTC, please '
                                'obtain a new one'.format(exp_time)), err)
    except jwt.exceptions.InvalidTokenError as err:
        raise_from(InvalidToken(token, str(err)), err)
    return claims


class _BearerAuth(requests.auth.AuthBase):
    """Add bearer token to authorisation request header."""

    def __init__(self, token):
        self._claims = decode_jwt(token)
        if 'prefix' not in self._claims:
            raise InvalidToken(token, "Token has no 'prefix' claim")
        self._token = token

    def __call__(self, r):
        # Check if token authorises URL even before hitting server for better reporting
        path = urllib.parse.urlparse(r.url).path.lstrip('/')
        valid_prefixes = self._claims['prefix']
        if not any(path.startswith(prefix) for prefix in valid_prefixes):
            allowed = ', '.join("'{}*'".format(prefix) for prefix in valid_prefixes)
            raise InvalidToken(self._token, "Token does not grant access to '{}', "
                                            'only to {}'.format(path, allowed))
        r.headers['Authorization'] = 'Bearer ' + self._token
        return r


class _AWSAuth(requests.auth.AuthBase):
    """Add AWS access + secret credentials to authorisation request header."""

    def __init__(self, credentials):
        if not botocore:
            raise AuthorisationFailed('Passing credentials requires botocore to be installed')
        credentials = botocore.credentials.ReadOnlyCredentials(
            credentials[0], credentials[1], None)
        self._signer = botocore.auth.HmacV1Auth(credentials)

    def __call__(self, r):
        split = urllib.parse.urlsplit(r.url)
        signature = self._signer.get_signature(r.method, split, r.headers)
        r.headers['Authorization'] = 'AWS {}:{}'.format(
            self._signer.credentials.access_key, signature)
        return r


def _auth_factory(url, token=None, credentials=None):
    """Turn either JWT token or AWS credentials into a requests auth handler."""
    if token is not None and credentials is not None:
        raise AuthorisationFailed('Cannot specify both token and credentials')
    if token is not None:
        parsed = urllib.parse.urlparse(url)
        # The exception for 127.0.0.1 lets the unit test work
        if parsed.scheme != 'https' and parsed.hostname != '127.0.0.1':
            raise AuthorisationFailed('Token may only be used with https')
        return _BearerAuth(token)
    elif credentials is not None:
        return _AWSAuth(credentials)
    else:
        return None


class _CacheSettingsSession(requests.Session):
    """Session that caches the result of proxy lookup.

    Normally requests spends a lot of time per request just to figure out what
    proxy server to use if any. For our usage, all URLs will be going to the
    same host and hence should always have the same proxy config, so we look
    it up once on the root URL for the chunk store and save the result.

    This has some limitations:
    - Proxy settings can't be changed.
    - Session settings (e.g. certificate-related) must not be changed after the
      first request.
    - All requests should be to the same host.
    - It is not thread-safe.
    """

    def __init__(self, url):
        super(_CacheSettingsSession, self).__init__()
        self._cached_settings = super(_CacheSettingsSession, self).merge_environment_settings(
            url, {}, True, None, None)

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Only cache for a specific combination of input settings (the
        # combination used by get_chunk), rather than trying to cache all
        # variants.
        if (proxies, stream, verify, cert) == ({}, True, None, None):
            if self._cached_settings is None:
                self._cached_settings = \
                    super(_CacheSettingsSession, self).merge_environment_settings(
                        url, proxies, stream, verify, cert)
            return self._cached_settings
        else:
            return super(_CacheSettingsSession, self).merge_environment_settings(
                url, proxies, stream, verify, cert)


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
    url : str
        Endpoint of S3 service, e.g. 'http://127.0.0.1:9000'. It can be
        specified as either bytes or unicode, and is converted to the native
        string type with UTF-8.
    timeout : float or tuple of 2 floats, optional
        Connect / read timeout, in seconds, either a single value for both or
        custom values as (connect, read) tuple (set to None to leave unchanged)
    retries : int or tuple of 2 ints or :class:`urllib3.util.retry.Retry`, optional
        Number of connect / read retries, either a single value for both or
        custom values as (connect, read) tuple, or a `Retry` object for full
        customisation (including status retries).
    token : str, optional
        Bearer token to authenticate
    credentials: tuple of str, optional
        AWS access key and secret key to authenticate
    public_read : bool, optional
        If set to true, new buckets will be created with a policy that allows
        everyone (including unauthenticated users) to read the data.
    expiry_days : int, optional
        If set to a value greater than 0 will set a future expiry time in days
        for any new buckets created.
    kwargs : dict
        Extra keyword arguments (unused)

    Raises
    ------
    :exc:`chunkstore.StoreUnavailable`
        If S3 server interaction failed (it's down, no authentication, etc)
    """

    def __init__(self, url, timeout=(30, 300), retries=2, token=None,
                 credentials=None, public_read=False, expiry_days=0, **kwargs):
        error_map = {requests.exceptions.RequestException: StoreUnavailable,
                     defusedxml.ElementTree.ParseError: StoreUnavailable}
        super(S3ChunkStore, self).__init__(error_map)
        auth = _auth_factory(url, token, credentials)
        if not isinstance(retries, Retry):
            try:
                connect_retries, read_retries = retries
            except TypeError:
                connect_retries = read_retries = retries
            # The backoff factor of 10 provides 5 minutes worth of retries
            # when the S3 server is strained; with 5 retries you get
            # (0 + 2 + 4 + 8 + 16) * 10 = 300 seconds on top of read timeouts.
            retries = Retry(connect=connect_retries, read=read_retries,
                            status=5, backoff_factor=10.,
                            status_forcelist=_DEFAULT_SERVER_GLITCHES)

        def session_factory():
            session = _CacheSettingsSession(url)
            session.auth = auth
            # Don't let requests do status retries as we'll be doing it ourselves
            max_retries = retries.new(status=0, raise_on_status=False)
            adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
            session.mount(url, adapter)
            return session

        self._session_pool = _Pool(session_factory)
        self._url = to_str(url)
        self._retries = retries
        self.timeout = timeout
        self.public_read = public_read
        self.expiry_days = int(expiry_days)

    def _chunk_url(self, chunk_name):
        return urllib.parse.urljoin(self._url, to_str(urllib.parse.quote(chunk_name + '.npy')))

    @contextlib.contextmanager
    def request(self, method, url, chunk_name='', ignored_errors=(), timeout=(), **kwargs):
        """Run a request on a session from the pool and handle error responses.

        This is a context manager like :meth:`requests.Session.request` that
        either yields a successful response or raises the appropriate chunk
        store exception.

        Parameters
        ----------
        method, url : str
            The standard required parameters of :meth:`requests.Session.request`
        chunk_name : str, optional
            Name of chunk, used for error reporting only
        ignored_errors : collection of int, optional
            HTTP status codes that are treated like 200 OK, not raising an error
        timeout : tuple or float, optional
            Override timeout for this request (use the store timeout by default)
        kwargs : optional
            These are passed on to :meth:`requests.Session.request`

        Yields
        ------
        response : `requests.Response`
            HTTP response, considered successful

        Raises
        ------
        AuthorisationFailed
            If the request is not authorised by appropriate token or credentials
        S3ObjectNotFound
            If S3 object request fails because it does not exist
        S3ServerGlitch
            If HTTP error has a retryable status code (i.e. temporary glitch)
        StoreUnavailable
            If a general HTTP error occurred that is not ignored
        """
        kwargs['timeout'] = self.timeout if timeout == () else timeout
        # Use _standard_errors to filter errors emanating from within with-block
        with self._standard_errors(chunk_name), self._session_pool() as session:
            with session.request(method, url, **kwargs) as response:
                # Turn error responses into the appropriate exception, like raise_for_status
                status = response.status_code
                if 400 <= status < 600 and status not in ignored_errors:
                    prefix = 'Chunk {!r}: '.format(chunk_name) if chunk_name else ''
                    msg = '{}Store responded with HTTP error {} ({}) to request: {} {}'.format(
                        prefix, status, response.reason, response.request.method, response.url)
                    if status == 401:
                        raise AuthorisationFailed(msg)
                    elif status in (403, 404):
                        # Ceph RGW returns 403 for missing keys due to our bucket policy
                        # (see https://tracker.ceph.com/issues/38638 for discussion)
                        raise S3ObjectNotFound(msg)
                    elif self._retries.is_retry(method, status):
                        raise S3ServerGlitch(msg, status)
                    else:
                        raise StoreUnavailable(msg)
                try:
                    yield response
                except TruncatedRead as trunc_error:
                    # A truncated read is considered a glitch with custom status
                    prefix = 'Chunk {!r}: '.format(chunk_name) if chunk_name else ''
                    glitch_error = S3ServerGlitch(prefix + str(trunc_error),
                                                  _TRUNCATED_HTTP_STATUS_CODE)
                    raise_from(glitch_error, trunc_error)

    def complete_request(self, method, url, chunk_name='',
                         process=lambda response: None, **kwargs):
        """Send HTTP request to S3 server, process response and retry if needed.

        This retries temporary HTTP errors, including reset connections while
        processing a successful response.

        Parameters
        ----------
        method, url : str
            The standard required parameters of :meth:`requests.Session.request`
        chunk_name : str, optional
            Name of chunk, used for error reporting only
        process : function, signature ``result = process(response)``, optional
            Function that will process response (the default does nothing)
        kwargs : optional
            Passed on to :meth:`request` and :meth:`requests.Session.request`

        Returns
        -------
        result : object
            The output of the `process` function applied to a successful response

        Raises
        ------
        AuthorisationFailed
            If the request is not authorised by appropriate token or credentials
        S3ObjectNotFound
            If S3 object request fails because it does not exist
        S3ServerGlitch
            If S3 object request fails because server is temporarily overloaded
        StoreUnavailable
            If a general HTTP error occurred that is not ignored
        """
        retries = self._retries.new()
        while True:
            try:
                with self.request(method, url, chunk_name, **kwargs) as response:
                    result = process(response)
            except S3ServerGlitch as e:
                # Retry based on status of response until we run out of retries
                response = HTTPResponse(status=e.status_code)
                try:
                    retries = retries.increment(method, url, response)
                except MaxRetryError:
                    # Raise the final straw that broke the retry camel's back
                    raise_from(e, None)
                else:
                    retries.sleep(response)
            else:
                return result

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        url = self._chunk_url(chunk_name)
        # Our hacky optimisation to speed up response reading doesn't
        # work with non-identity encodings.
        headers = {'Accept-Encoding': 'identity'}
        chunk = self.complete_request('GET', url, chunk_name, _read_chunk,
                                      headers=headers, stream=True)
        if chunk.shape != shape or chunk.dtype != dtype:
            raise BadChunk('Chunk {!r}: dtype {} and/or shape {} in store '
                           'differs from expected dtype {} and shape {}'
                           .format(chunk_name, chunk.dtype, chunk.shape,
                                   dtype, shape))
        return chunk

    def create_array(self, array_name):
        """See the docstring of :meth:`ChunkStore.create_array`."""
        # Array name is formatted as bucket/array but we only need to create bucket
        bucket, _ = self.split(array_name, 1)
        self._create_bucket(bucket)

    def _create_bucket(self, bucket):
        url = urllib.parse.urljoin(self._url, to_str(urllib.parse.quote(bucket)))
        # Make bucket (409 indicates the bucket already exists, which is OK)
        self.complete_request('PUT', url, ignored_errors=(409,))

        if self.public_read:
            policy_url = urllib.parse.urljoin(url, '?policy')
            policy = copy.deepcopy(_BUCKET_POLICY)
            policy['Statement'][0]['Resource'] = [
                'arn:aws:s3:::{}/*'.format(bucket),
                'arn:aws:s3:::{}'.format(bucket)
            ]
            self.complete_request('PUT', policy_url, data=json.dumps(policy))

        if self.expiry_days > 0:
            xml_payload = _BASE_LIFECYCLE_POLICY.format(self.expiry_days)
            b64_md5 = base64.b64encode(hashlib.md5(xml_payload.encode('utf-8')).digest()).decode('utf-8')
            lifecycle_headers = {'Content-Type': 'text/xml', 'Content-MD5': b64_md5}
            self.complete_request('PUT', url, params='lifecycle',
                                  data=xml_payload, headers=lifecycle_headers)

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        chunk_name, _ = self.chunk_metadata(array_name, slices, chunk=chunk)
        url = self._chunk_url(chunk_name)
        npy_header, chunk = npy_header_and_body(chunk)
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
        self.complete_request('PUT', url, chunk_name, headers=headers, data=data)

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, _ = self.chunk_metadata(array_name, slices, dtype=dtype)
        url = self._chunk_url(chunk_name)
        try:
            self.complete_request('HEAD', url, chunk_name)
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

        def _read_xml(response):
            return defusedxml.cElementTree.fromstring(response.content)

        keys = []
        more = True
        while more:
            root = self.complete_request('GET', url, process=_read_xml, params=params)
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
        bucket = self.split(array_name, 1)[0]
        self._create_bucket(bucket)
        obj_name = self.join(array_name, 'complete')
        url = urllib.parse.urljoin(self._url, obj_name)
        self.complete_request('PUT', url, obj_name, data=b'')

    def is_complete(self, array_name):
        """See the docstring of :meth:`ChunkStore.is_complete`."""
        obj_name = self.join(array_name, 'complete')
        url = urllib.parse.urljoin(self._url, obj_name)
        try:
            self.complete_request('GET', url, obj_name)
        except ChunkNotFound:
            return False
        return True

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
    list_chunk_ids.__doc__ = ChunkStore.list_chunk_ids.__doc__
    mark_complete.__doc__ = ChunkStore.mark_complete.__doc__
    is_complete.__doc__ = ChunkStore.is_complete.__doc__
