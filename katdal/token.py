################################################################################
# Copyright (c) 2019, National Research Foundation (Square Kilometre Array)
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

"""Utilities to validate and inspect bearer tokens of the S3 chunk store."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: E402
from future.utils import bytes_to_native_str, raise_from

import base64
import json
import re
import time


BASE64URL_PADDING = {0: '', 2: '==', 3: '='}


class InvalidToken(ValueError):
    """Invalid JSON Web Token (JWT)."""


def encode_base64url_without_padding(b):
    """Encode bytes `b` to base64url string without padding (see RFC 7515)."""
    # Strip off padding as described in Appendix C of RFC 7515
    return bytes_to_native_str(base64.urlsafe_b64encode(b).rstrip(b'='))


def decode_base64url_without_padding(s):
    """Decode base64url-encoded string `s` without padding (RFC 7515) to bytes."""
    # Restore padding as described in Appendix C of RFC 7515
    try:
        s += BASE64URL_PADDING[len(s) % 4]
    except KeyError:
        raise ValueError('Invalid base64url-encoded string {}: number of data '
                         'characters ({}) cannot be 1 more than a multiple of 4'
                         .format(s, len(s)))
    # Use the standard base64 decoder with padding (but with base64url alphabet)
    try:
        return base64.urlsafe_b64decode(s)
    except TypeError as err:
        # In Python 2, base64 raises a TypeError if string is incorrectly padded
        raise_from(ValueError(err), err)


def encode_json_base64url(d):
    """Encode Python object `d` to JSON-serialised base64url-encoded string."""
    return encode_base64url_without_padding(json.dumps(d).encode())


def decode_json_base64url(s):
    """Decode JSON-serialised base64url-encoded string `s` (ie a JWS segment)."""
    try:
        b = decode_base64url_without_padding(s)
    except ValueError as err:
        raise_from(ValueError('Token segment {} has invalid base64url encoding'
                              .format(s)), err)
    try:
        return json.loads(b)
    except ValueError as err:
        raise_from(ValueError('Decoded token segment {} is not valid JSON'
                              .format(b)), err)


def encode_jwt(header, payload, signature):
    """Encode JSON Web Token (JWT) components to a base64url-encoded token string.

    Parameters
    ----------
    header : dict
        JSON Object Signing and Encryption (JOSE) header
    payload : dict
        JWT Claims Set
    signature : bytes
        Digital signature or MAC (calculated externally)

    Returns
    -------
    token : str
        JWT as a JWS Compact Serialization string
    """
    return '.'.join((encode_json_base64url(header),
                     encode_json_base64url(payload),
                     encode_base64url_without_padding(signature)))


def decode_jwt(token):
    """Decode JSON Web Token (JWT) string and extract claims.

    The MeerKAT archive uses JWT bearer tokens for authorisation. Each token is
    a JSON Web Signature (JWS) string with a payload of claims. This function
    extracts the claims as a dict, while also doing basic validation on the
    token. The signature is decoded but not validated, since that would require
    the server secrets.

    Parameters
    ----------
    token : str
        JWS Compact Serialization as an ASCII string

    Returns
    -------
    claims : dict
        The JWT Claims Set as a dict of key-value pairs

    Raises
    ------
    :exc:`InvalidToken`
        If token is malformed or the wrong type, or has expired
    """
    # A valid JWS Compact Serialization has multiple base64url-encoded segments
    # without padding separated by periods ('.') - see RFC 7515, RFC 4648
    if not re.match('^[A-Za-z0-9-_.]*$', token):
        raise InvalidToken("Token {} contains invalid characters not in base64url "
                           "set (or .) - are you sure it's a token?".format(token))
    # A valid JWS Compact Serialization has three segments
    try:
        encoded_header, encoded_payload, encoded_signature = token.split('.')
    except ValueError:
        raise InvalidToken("Token does not have JWS structure (maybe it's truncated?)")
    # Obtain JSON Object Signing and Encryption (JOSE) header
    try:
        header = decode_json_base64url(encoded_header)
    except ValueError as err:
        raise_from(InvalidToken('Could not decode token header {}'
                                .format(encoded_header)), err)
    # Obtain payload consisting of JWT claims
    try:
        claims = decode_json_base64url(encoded_payload)
    except ValueError as err:
        raise_from(InvalidToken('Could not decode token payload {}'
                                .format(encoded_payload)), err)
    # Check that signature is at least encoded properly (maybe truncated otherwise)
    try:
        decode_base64url_without_padding(encoded_signature)
    except ValueError as err:
        raise_from(InvalidToken('Could not decode token signature {} (maybe token '
                                'is truncated?)'.format(encoded_signature)), err)
    # A valid JWT must have 'alg' parameter, and be of type 'JWT' in our application
    if 'alg' not in header or header.get('typ') != 'JWT':
        raise InvalidToken('Token is not valid JWT')
    # Check whether the JWT has expired
    if 'exp' in claims:
        if claims['exp'] <= time.time():
            exp_time = time.strftime('%d-%b-%Y %H:%M:%S', time.gmtime(claims['exp']))
            raise InvalidToken('Token has expired on {} UTC'.format(exp_time))
    return claims
