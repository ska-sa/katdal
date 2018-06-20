"""Authentication with the MeerKAT archive.

This provides integration with OpenID Connect authentication to authenticate to
a proxy in front of an S3 service, rather than using S3 secret key
authentication. It does not hardcode any MeerKAT-specific addresses, so could
be used elsewhere. The only requirement is that the proxy provides a
X-Katdal-Authentication-Provider header with the URL of the OpenID provider, and
that the provider supports out-of-band transfer of the code.

When authentication is required, it gives an URL to the user. The user opens
the link in a browser (possibly on a separate machine) and obtains an
authorization code, which they paste back into the terminal running katdal.
"""

from __future__ import print_function
import json
import requests
import threading
import urlparse
import logging

import oic
from oic import rndstr
from oic.oic import Client
from oic.oic.message import (RegistrationResponse, AuthorizationResponse,
                             AccessTokenResponse, TokenErrorResponse)
from oic.utils.authn.client import CLIENT_AUTHN_METHOD


logger = logging.getLogger('main')


class Auth(requests.auth.AuthBase):
    """Authenticator for requests that handles 401 responses.

    It takes care of the OpenID Connect interactions, including automatic
    refreshing of tokens.

    It assumes that all requests are for the same security realm. Thus, it
    should *not* be used across different servers with different authentication
    requirements.
    """
    def __init__(self):
        self._client = Client(client_authn_method=CLIENT_AUTHN_METHOD)
        self._access_token = None
        self._refresh_token = None
        self._state = rndstr()
        self._lock = threading.Lock()

    def _clear_access_token(self):
        with self._lock:
            self._access_token = None

    def _ensure_registered(self, provider):
        if 'redirect_uris' in self._client.registration_response:
            return
        self._client.provider_config(provider)
        client_info = {
            "client_id": "katdal",
            "client_secret": "notused",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"]
        }
        client_reg = RegistrationResponse(**client_info)
        self._client.store_registration_info(client_reg)
        self._registered = True

    def _update_access_token(self, response):
        try:
            provider = response.headers.get('X-Katdal-Authentication-Provider')
            if provider is None:
                provider = urlparse.urljoin(response.request.url, '/')
            self._ensure_registered(provider)
            access = None

            if self._refresh_token is not None:
                logger.debug('Using refresh token to get new access token')
                access = self._client.do_access_token_refresh(state=self._state, request_args={
                    "refresh_token": self._refresh_token})
                if isinstance(access, TokenErrorResponse):
                    logger.debug('Refresh token failed: %s',
                                 access.get('error_description', 'unknown error'))

            while not isinstance(access, AccessTokenResponse):
                nonce = rndstr()
                auth_req = self._client.construct_AuthorizationRequest(request_args=dict(
                    client_id=self._client.client_id,
                    response_type="code",
                    scope=["openid"],
                    nonce=nonce,
                    redirect_uri=self._client.registration_response["redirect_uris"][0],
                    state=self._state))
                login_url = auth_req.request(self._client.authorization_endpoint)
                print('Authentication is required for {}'.format(response.request.url))
                print('Please visit the URL below:')
                print()
                print(login_url)
                print()
                code = raw_input('Enter the code: ')
                auth_json = json.dumps({"state": self._state, "nonce": nonce, "code": code})
                self._client.parse_response(AuthorizationResponse, info=auth_json,
                                            sformat="json")
                access = self._client.do_access_token_request(state=self._state,
                                                              request_args={"code": code})
                if isinstance(access, TokenErrorResponse):
                    print(access.get('error_description', 'unknown error'))
                    print()
            self._access_token = access.get('access_token')
            self._refresh_token = access.get('refresh_token')
        except oic.exception.CommunicationError:
            logger.warning('Internal authentication error', exc_info=True)

    def access_token(self, require=False, expire=None, response=None):
        """Obtains a current access token.

        If `require` is false, an access token is returned only if one is
        already known. If it is true, then `request` must be given and it
        will attempt to obtain a new token if there isn't one.

        If `expire` is given, it will invalid the current access token if it
        matches `expire`.

        If there is a problem with authentication, it will log a warning and
        return ``None``.
        """
        with self._lock:
            if self._access_token == expire:
                self._access_token = None
            if self._access_token is None and require:
                self._update_access_token(response)
            return self._access_token

    def response_hook(self, response, **kwargs):
        if response.status_code != 401:
            return response
        # The header has a complex grammar (see RFC 7235), so this is just a
        # sanity check to ensure that bearer tokens are acceptable.
        www_auth = response.headers.get('WWW-Authenticate', '')
        if 'bearer' not in www_auth.lower():
            return response

        # Check if we already tried a token, in which case it must be invalid
        old_token = None
        if 'Authorization' in response.request.headers:
            old_token = response.request.headers['Authorization'].split(' ')[1]
        token = self.access_token(require=True, expire=old_token, response=response)
        if token is None:
            return response

        # Consume and close the original response, and prepare a duplicate.
        # This is based on code in requests/auth.py.
        response.content
        response.close()
        req = response.request.copy()
        requests.cookies.extract_cookies_to_jar(req._cookies, response.request, response.raw)
        req.prepare_cookies(req._cookies)

        if token is not None:
            req.headers['Authorization'] = 'Bearer ' + token
        new_response = response.connection.send(req, **kwargs)
        new_response.history.append(response)
        new_response.request = req
        return new_response

    def __call__(self, request):
        token = self.access_token()
        if token is not None:
            request.headers['Authorization'] = 'Bearer ' + token
        request.register_hook('response', self.response_hook)
        return request
