import boto
import boto.s3.connection
from .ska_connect import login 

def wrap_connect(connect_func):
    def authenticated_connect (*args, **kwargs):
        endpoint = "%s:%i"%(kwargs['host'],kwargs['port'])
        #Should create a config with hosts that have an auth landing  page
        if (kwargs['host'] == "stgr1.sdp.mkat.chpc.kat.ac.za" or
            kwargs['host'] == "stgr1" or
            kwargs['host'] == "10.98.52.16" or
            kwargs['host'] == "archive-gw-1.kat.ac.za" or
            kwargs['host'] == "archive-gw-1"):

            endpoint += "/auth.html"

        cookie_string = login(endpoint)
        s3 = connect_func(*args, **kwargs)
        if cookie_string:
            s3.make_request = authed(s3.make_request, cookie_string)
        return s3
    return authenticated_connect

def authed (http_func, cookie_string):
    """Wrapper to attempt to automatically authenticate the wrapped function and fail 
    seamlessly into an unauthenticated function call. 
    
    Parameters
    ---------
    http_func : The function we are trying to authenticate
    cookie_string : The authentication cookies to add to requests
    """
    def authenticated_function(*args,**kwargs):
        if "headers" in http_func.func_code.co_varnames and http_func.func_code.co_varnames.index('headers') <= len(args):
            index = http_func.func_code.co_varnames.index('headers') - 1
            args[index].setdefault("Cookie","")
            args[index]["Cookie"] += cookie_string
        else:
            if not kwargs:
                kwargs = {}
            kwargs.setdefault("headers",{})
            if not kwargs["headers"]:
                kwargs["headers"] = {}
            kwargs["headers"].setdefault("Cookie","")
            kwargs["headers"]["Cookie"] += cookie_string
        try:
            res = http_func (*args, **kwargs)
        except:
            print ("Cannot run %s with authentication header, cookies may have expired, try logging in again."%http_func)
            kwargs.pop("headers")
            res = http_func (*args, **kwargs)
        return res
    return authenticated_function

boto.connect_s3 = wrap_connect(boto.connect_s3)
