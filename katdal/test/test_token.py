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

"""Tests for :py:mod:`katdal.token`."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()     # noqa: E402

from nose.tools import assert_raises, assert_equal

from katdal.token import (
    encode_base64url_without_padding, decode_base64url_without_padding,
    encode_json_base64url, decode_json_base64url, encode_jwt, decode_jwt,
    InvalidToken)


# Simulate an ES256 signature that is always 64 bytes long
SIGNATURE = 64 * b'*'


class TestTokenUtils(object):
    """Test token utility and validation functions."""

    def test_base64url_without_padding(self):
        b1 = b'\xff\xfe0123'
        s = encode_base64url_without_padding(b1)
        b2 = decode_base64url_without_padding(s)
        assert_equal(b1, b2)
        # Invalid string length
        with assert_raises(ValueError):
            decode_base64url_without_padding('12345')
        # String not UTF-8
        with assert_raises(ValueError):
            decode_base64url_without_padding('123\xff')
        # Character not in base64url alphabet
        with assert_raises(ValueError):
            decode_base64url_without_padding('123^')

    def test_json_base64url(self):
        d1 = {'typ': 'JWT', 'iat': 123456789.3, 'ok': False}
        s = encode_json_base64url(d1)
        d2 = decode_json_base64url(s)
        assert_equal(d1, d2)
        # Truncated JSON string
        with assert_raises(ValueError):
            decode_json_base64url(s[:-1])
        # JSON string not UTF-8
        with assert_raises(ValueError):
            decode_json_base64url(s[1:])

    def test_jwt(self):
        header = {'alg': 'ES256', 'typ': 'JWT'}
        payload = {'exp': 9234567890, 'iss': 'kat', 'prefix': ['123']}
        token = encode_jwt(header, payload, SIGNATURE)
        claims = decode_jwt(token)
        assert_equal(payload, claims)
        # Token has invalid characters
        with assert_raises(InvalidToken):
            decode_jwt('** bad token **')
        # Token has invalid structure
        with assert_raises(InvalidToken):
            decode_jwt(token.replace('.', ''))
        # Token header failed to decode
        with assert_raises(InvalidToken):
            decode_jwt(token[1:])
        # Token payload failed to decode
        with assert_raises(InvalidToken):
            h, p, s = token.split('.')
            decode_jwt('.'.join((h, p[:-1], s)))
        # Token signature failed to decode or wrong length
        with assert_raises(InvalidToken):
            decode_jwt(token[:-1])
        with assert_raises(InvalidToken):
            decode_jwt(token[:-2])
        with assert_raises(InvalidToken):
            decode_jwt(token + token[-4:])

    def test_jwt_invalid_header(self):
        header = {'typ': 'JWT'}
        payload = {'exp': 0, 'iss': 'kat', 'prefix': ['123']}
        token = encode_jwt(header, payload, SIGNATURE)
        with assert_raises(InvalidToken):
            decode_jwt(token)
        header = {'alg': 'ES256', 'typ': 'JWS'}
        payload = {'exp': 0, 'iss': 'kat', 'prefix': ['123']}
        token = encode_jwt(header, payload, SIGNATURE)
        with assert_raises(InvalidToken):
            decode_jwt(token)

    def test_jwt_expired(self):
        header = {'alg': 'ES256', 'typ': 'JWT'}
        payload = {'exp': 0, 'iss': 'kat', 'prefix': ['123']}
        token = encode_jwt(header, payload, SIGNATURE)
        with assert_raises(InvalidToken):
            decode_jwt(token)
