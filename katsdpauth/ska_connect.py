from getpass import getpass
import requests
from requests.exceptions import InvalidSchema
from bs4 import BeautifulSoup
from urlparse import urlparse
import os
import pkg_resources

def botocore_authed (http_func, cookie_string):
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

    return authenticated_function

def login(endpoint, username="", password=""):
    cookies = get_auth_cookie(endpoint,
        username=username,
        password=password)

    return "; ".join([str(x)+"="+str(y) for x,y in cookies.items()])

def wrap_functions(cls, cookie_string):
     for func in inspect.getmembers(cls, inspect.isroutine):
        print "Wrapping %s"%func[0]
        cls.__setattr__(func[0],authed(func[1], cookie_string))
     return cls

def get_auth_cookie (auth_url="https://kat-archive.kat.ac.za", idp='SARAO Google', username=None, password=None):
    print("Accessing %s"%auth_url)

    s = requests.session()

    try:
        r = s.get(auth_url)
    except InvalidSchema:
        r = s.get("http://" + auth_url)

    exit = False
    cookies = {}

    while r.status_code == 200 and "Since your browser does not support JavaScript" in r.text and not exit:

        #Grab form for no javascript
        form_data = BeautifulSoup(r.content, "html5lib").find('form').find_all('input')
        action = BeautifulSoup(r.content, "html5lib").find('form').get('action')
        request_data = {}
        for u in form_data:
            if u.has_attr('value'):
                request_data[u['name']] = u['value']
        
        cert_file=pkg_resources.resource_filename('katsdpauth','certs/kat-archive_kat_ac_za_interm.crt')
        r = s.post(action,request_data,verify=cert_file)

        idp = BeautifulSoup(r.content, "html5lib").find('a',{"id":"zocial-skasa_google"}, href=True) #Get idp link
        parsed_uri = urlparse(r.url)
        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

        action = '/'.join([domain.strip('/'),idp['href'].strip('/')])
        r = s.get(action)  #Go to SKA SA google log in

        #Grab form for no javascript
        form_data = BeautifulSoup(r.content, "html5lib").find('form').find_all('input')
        action = BeautifulSoup(r.content, "html5lib").find('form').get('action')

        request_data = {}
        for u in form_data:
            if u.has_attr('value') and u.has_attr('name'):
                request_data[u['name']] = u['value']

        r = s.post(action,request_data)

        #Login details
        if not username:
            username = raw_input ("Enter your SKA SA email : ")
        if not password:
            password = getpass ("Enter your SKA SA password : ")

        #Grab google login form
        form_data = BeautifulSoup(r.content, "html5lib").find('form').find_all('input')

        request_data = {}
        for u in form_data:
            if u.has_attr('value'):
                request_data[u['name']] = u['value']

        #Insert login details
        request_data["Email"] = username
        request_data["Passwd"] = password

        r = s.post("https://accounts.google.com/signin/challenge/sl/password", request_data)

        #Grab no javascript form
        form_data = BeautifulSoup(r.content, "html5lib").find('form').find_all('input')
        action = BeautifulSoup(r.content, "html5lib").find('form').get('action')

        request_data = {}
        for u in form_data:
            if u.has_attr('value') and u.has_attr('name'):
                request_data[u['name']] = u['value']

        r = s.post(action,request_data)

        #Grab no javascript form
        form_data = BeautifulSoup(r.content, "html5lib").find('form').find_all('input')
        action = BeautifulSoup(r.content, "html5lib").find('form').get('action')

        request_data = {}
        for u in form_data:
            if u.has_attr('value') and u.has_attr('name'):
                request_data[u['name']] = u['value']

        r = s.post(action,request_data)

        cookies = requests.utils.dict_from_cookiejar(s.cookies)

        r = s.get(auth_url)

        if r.status_code == 200 and "Since your browser does not support JavaScript" in r.text:
            cont = raw_input("Could not verify your credentials, try again? Y/N :")
            if cont.lower() == 'n':
                exit = True

        username = None
        password = None

    return cookies

def authenticate(username, password):
    cookies = get_auth_cookie (username=username, password=password)
    if cookies["mellon-cookie"] != 'cookietest':
        return True
    else:
        return False

