import botocore.session
import botocore.auth
from collections import namedtuple
from .ska_connect import login
import botocore.exceptions
import requests

def wrap_session (session_func):
    
    def auth_session(*args, **kwargs):
        session = session_func(*args,**kwargs)

        def wrap_client(client_func):

            def authenticated_client(*args,**kwargs):
                endpoint_url = ""
                if (client_func.func_code.co_varnames.index('endpoint_url') <= len(args)):
                    endpoint_url = args[client_func.func_code.co_varnames.index('endpoint_url') - 1]
                else:
                    endpoint_url = kwargs['endpoint_url']
                #Should create a config with hosts that have an auth landing page
                if ("stgr1" in endpoint_url or
                    "archive-gw-1" in endpoint_url or
                    "10.98.52.16" in endpoint_url):

                    endpoint_url += "/auth.html"

                try:
                    cookie_string = login(endpoint_url)
                except requests.exceptions.ConnectionError as e:
                    raise botocore.exceptions.ConnectionError(str(e))
                if cookie_string:
                    botocore.client.prepare_request_dict = authed(botocore.client.prepare_request_dict, cookie_string)
                return client_func(*args, **kwargs)

            return authenticated_client

        session.create_client = wrap_client(session.create_client)

        return session

    return auth_session 
            

def authed (http_func, cookie_string):
    """Wrapper to attempt to automatically authenticate the wrapped function and fail 
    seamlessly into an unauthenticated function call. 
    
    Parameters
    ---------
    http_func : The function we are trying to authenticate
    cookie_string : The authentication cookies to add to requests
    """

    def authenticated_function(*args,**kwargs):
        request_dict = http_func (*args, **kwargs)
        args[0]["headers"].setdefault("Cookie","")
        args[0]["headers"]["Cookie"] += cookie_string
        args[0]["headers"]['User-Agent'] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36" 

    return authenticated_function

def inject_aws_credentials (auth_func):
    """Wrapper to inject the read only credentials for the s3 gateway to MeerKAT CEPH if the user does not have their own credentials"""
    
    def credentialed_function (*args, **kwargs):
        if args[0].credentials == None:
            ReadOnlyCredentials = namedtuple('ReadOnlyCredentials',
                                            ['access_key', 'secret_key', 'token'])
            args[0].credentials = ReadOnlyCredentials(u'5I12GEIC7ISATT97KYL5',u'5PMtfYVemu78lgv6cHzH6EeIhelyzkrSnF0RYrmp',None)

        return auth_func(*args,**kwargs)

    return credentialed_function
    
botocore.session.get_session = wrap_session(botocore.session.get_session)
botocore.auth.SigV4Auth.add_auth = inject_aws_credentials(botocore.auth.SigV4Auth.add_auth)


