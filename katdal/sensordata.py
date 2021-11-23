################################################################################
# Copyright (c) 2011-2021, National Research Foundation (SARAO)
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

"""Container that stores cached (interpolated) and uncached (raw) sensor data."""

import logging
import re
import threading

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import katpoint
import katsdptelstate
import numpy as np
import requests

from .categorical import (ComparableArrayWrapper, infer_dtype,
                          sensor_to_categorical)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorGetter
# -------------------------------------------------------------------------------------------------


class SensorData:
    """Raw (uninterpolated) sensor values.

    This is a simple struct that holds timestamps, values, and optionally
    status.

    Parameters
    ----------
    name : string
        Sensor name
    timestamp : np.ndarray
        Array of timestamps
    value : np.ndarray
        Array of values (wrapped in :class:`ComparableArrayWrapper` if necessary)
    status : np.ndarray, optional
        Array of sensor statuses
    """

    def __init__(self, name, timestamp, value, status=None):
        assert value.shape == timestamp.shape
        assert status is None or status.shape == timestamp.shape
        self.name = name
        self.timestamp = timestamp
        self.value = value
        self.status = status

    def __bool__(self):
        """True if sensor has at least one data point."""
        return len(self.timestamp) > 0


class SensorGetter:
    """Raw (uninterpolated) sensor data placeholder.

    This is an abstract lazy interface that provides a :class:`SensorData`
    object on request but does not store values itself. Subclasses must
    implement :meth:`get` to retrieve values from underlying storage. They
    should *not* cache the results.

    Where possible, object-valued sensors (including sensors with ndarrays as
    values) will have values wrapped by :class:`ComparableArrayWrapper`.

    Parameters
    ----------
    name : string
        Sensor name
    """

    def __init__(self, name):
        self.name = name

    def get(self):
        """Retrieve the values from underlying storage.

        Returns
        -------
        values : :class:`SensorData`
            Underlying data
        """
        raise NotImplementedError

    def __repr__(self):
        """Short human-friendly string representation of sensor data object."""
        class_name = self.__class__.__name__
        return f"<katdal.{class_name} '{self.name}' at {id(self):#x}>"


class SimpleSensorGetter(SensorGetter):
    """Raw sensor data held in memory.

    This is a simple wrapper for :class:`SensorData` that implements the
    :class:`SensorGetter` interface.
    """

    def __init__(self, name, timestamp, value, status=None):
        super().__init__(name)
        self._data = SensorData(name, timestamp, value, status)

    def get(self):
        return self._data


class RecordSensorGetter(SensorGetter):
    """Raw (uninterpolated) sensor data in record array form.

    This is a wrapper for uninterpolated sensor data which resembles a record
    array with fields 'timestamp', 'value' and optionally 'status'. This is
    also the typical format of HDF5 datasets used to store sensor data.

    Technically, the data is interpreted as a NumPy "structured" array, which
    is a simpler version of a recarray that only provides item-style access to
    fields and not attribute-style access.

    Object-valued sensors are not treated specially in this class, as it is
    assumed that any wrapping already occurred in the construction of the
    recarray-like `data` input and will be reflected in its dtype. The original
    HDF5 sensor datasets also did not contain any objects as they only support
    standard KATCP types, so there was no need for wrapping there.

    Parameters
    ----------
    data : recarray-like, with fields 'timestamp', 'value' and optionally 'status'
        Uninterpolated sensor data as structured array or equivalent (such as
        an :class:`h5py.Dataset`)
    name : string or None, optional
        Sensor name (assumed to be data.name by default, if it exists)

    """

    def __init__(self, data, name=None):
        name = name if name is not None else getattr(data, 'name', '')
        super().__init__(name)
        self._data = data

    def get(self):
        """Extract timestamp, value and status of each sensor data point.

        Values are passed through :func:`to_str`.
        """
        timestamp = np.asarray(self._data['timestamp'])
        value = to_str(np.asarray(self._data['value']))
        try:
            status = self._data['status']
        except ValueError:
            status = None
        return SensorData(self.name, timestamp, value, status)

    def __repr__(self):
        """Short human-friendly string representation of sensor data object."""
        return "<katdal.{} '{}' len={} type={} at {:#x}>".format(
               self.__class__.__name__, self.name,
               len(self._data), self._data['value'].dtype, id(self))


def to_str(value):
    """Convert string-likes to the native string type.

    Bytes are decoded to str, with surrogateencoding error handler.

    Tuples, lists, dicts and numpy arrays are processed recursively, with the
    exception that numpy structured types with string or object fields won't
    be handled.
    """
    if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
        return np.char.decode(value, 'utf-8', 'surrogateescape')
    elif isinstance(value, bytes):
        return value.decode('utf-8', 'surrogateescape')

    # We use type(value) so that subclasses are reconstructed correctly
    if isinstance(value, (list, tuple)):
        return type(value)(to_str(item) for item in value)
    elif isinstance(value, dict):
        return type(value)((to_str(key), to_str(val)) for key, val in value.items())
    elif isinstance(value, np.ndarray) and value.dtype == 'O':
        return np.vectorize(to_str, otypes='O')(value)
    else:
        return value


def telstate_decode(raw, no_decode=()):
    """Load a katsdptelstate-encoded value that might be wrapped in np.void or
    np.ndarray.

    The np.void/np.ndarray wrapping is needed to pass variable-length binary
    strings through h5py.

    If the value is a string and is in no_decode, it is returned verbatim.
    This is for backwards compatibility with older files that didn't use
    any encoding at all.

    The return value is also passed through :func:`to_str`.
    """
    if isinstance(raw, (np.void, np.ndarray)):
        return to_str(katsdptelstate.decode_value(raw.tostring()))
    raw_str = to_str(raw)
    if raw_str in no_decode:
        return raw_str
    else:
        return to_str(katsdptelstate.decode_value(raw_str.encode()))


def _h5_telstate_unpack(s):
    """Unpack a telstate value from its encoded representation."""
    try:
        # Since 2016-05-09 the HDF5 TelescopeState contains encoded values
        return telstate_decode(s)
    except katsdptelstate.DecodeError:
        try:
            # Before 2016-05-09 the telstate values were str() representations
            # This cannot be unpacked in general but works for numbers at least
            return np.safe_eval(s)
        except (ValueError, SyntaxError):
            # When unsure, return the string itself (correct for string sensors)
            return s


class H5TelstateSensorGetter(RecordSensorGetter):
    """Raw (uninterpolated) sensor data in HDF5 TelescopeState recarray form.

    This wraps the telstate sensors stored in recent HDF5 files. It differs
    in two ways from the normal HDF5 sensors: no 'status' field and values
    encoded by katsdptelstate.

    TODO: This is a temporary fix to get at missing sensors in telstate and
    should be replaced by a proper wrapping of any telstate object.

    Object-valued sensors (including sensors with ndarrays as values) will have
    its values wrapped by :class:`ComparableArrayWrapper`.

    Parameters
    ----------
    data : recarray-like, with fields ('timestamp', 'value')
        Uninterpolated sensor data as structured array or equivalent (such as
        an :class:`h5py.Dataset`)
    name : string or None, optional
        Sensor name (assumed to be data.name by default, if it exists)

    """

    def __init__(self, data, name=None):
        super().__init__(data, name)

    def get(self):
        """Extract timestamp and value of each sensor data point."""
        timestamp = np.asarray(self._data['timestamp'])
        # Unpack everything first, otherwise old files will be a mess
        values = [_h5_telstate_unpack(s) for s in self._data['value']]
        # Figure out dtype and wrap any objects
        dtype = infer_dtype(values)
        if dtype == np.object:
            values = [ComparableArrayWrapper(value) for value in values]
        return SensorData(self.name, timestamp, to_str(np.asarray(values)))


class TelstateToStr:
    """Wrap an existing telescope state and pass return values through :meth:`to_str`"""
    def __init__(self, telstate):
        if isinstance(telstate, TelstateToStr):
            self._telstate = telstate._telstate
        else:
            self._telstate = telstate

    @property
    def wrapped(self):
        return self._telstate

    def view(self, name, add_separator=True, exclusive=False):
        return TelstateToStr(self._telstate.view(name, add_separator, exclusive))

    def root(self):
        return TelstateToStr(self._telstate.root())

    def __getattr__(self, key):
        # __getattr__ can be used for item access or to get attributes of _telstate
        if hasattr(self._telstate.__class__, key):
            return getattr(self._telstate, key)
        else:
            return to_str(getattr(self._telstate, key))

    def __contains__(self, key):
        # Needed because __getattr__ won't pick it up from child
        return key in self._telstate

    def __dir__(self):
        # Include public attributes of _telstate that are reachable via __getattr__
        basic = dir(super())
        extra = [d for d in dir(self._telstate)
                 if d not in basic and not d.startswith('_')]
        return basic + extra

    def __getitem__(self, key):
        return to_str(self._telstate[key])

    def get_message(self, channel=None):
        return to_str(self._telstate.get_message(channel))

    def get(self, key, default=None, return_encoded=False):
        value = self._telstate.get(key, default, return_encoded)
        if not return_encoded:
            value = to_str(value)
        return value

    def get_range(self, key, st=None, et=None,
                  include_previous=None, include_end=False, return_encoded=False):
        value = self._telstate.get_range(key, st, et, include_previous, include_end, return_encoded)
        if not return_encoded:
            value = to_str(value)
        return value

    def get_indexed(self, key, sub_key, default=None, return_encoded=False):
        value = self._telstate.get_indexed(
            key, sub_key, default=default, return_encoded=return_encoded)
        if not return_encoded:
            value = to_str(value)
        return value


class TelstateSensorGetter(SensorGetter):
    """Raw (uninterpolated) sensor data stored in original TelescopeState.

    This wraps sensor data stored in a TelescopeState object. The data is
    only read out on item access.

    Object-valued sensors (including sensors with ndarrays as values) will have
    their values wrapped by :class:`ComparableArrayWrapper`.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState` object
        Telescope state object
    name : string
        Sensor name, also used as telstate key

    Raises
    ------
    KeyError
        If sensor name is not found in telstate or it is an attribute instead
    """

    def __init__(self, telstate, name):
        self._telstate = TelstateToStr(telstate)
        key_type = telstate.key_type(name)
        if key_type is None:
            raise KeyError('No sensor named %r in telstate (key not found)' %
                           (name,))
        if key_type != katsdptelstate.KeyType.MUTABLE:
            raise KeyError("No sensor named %r in telstate (it's %s)" %
                           (name, key_type.name))
        super().__init__(name)

    def __bool__(self):
        """True if sensor has at least one data point (already checked in init)."""
        return True

    def get(self):
        values, times = zip(*self._telstate.get_range(self.name, st=0))
        dtype = infer_dtype(values)
        if dtype == np.object:
            values = [ComparableArrayWrapper(v) for v in values]
        return SensorData(self.name, np.asarray(times), np.asarray(values))


# -------------------------------------------------------------------------------------------------
# -- Utility functions
# -------------------------------------------------------------------------------------------------


def get_sensor_from_katstore(store, name, start_time, end_time):
    """Get raw sensor data from katstore (CAM's central sensor database).

    Parameters
    ----------
    store : string
        Hostname / endpoint of katstore webserver speaking katstore64 API
    name : string
        Sensor name (the normalised / escaped version with underscores)
    start_time, end_time : float
        Time range for sensor records as UTC seconds since Unix epoch

    Returns
    -------
    data : :class:`RecordSensorGetter` object
        Retrieved sensor data with 'timestamp', 'value' and 'status' fields

    Raises
    ------
    ConnectionError
        If this cannot connect to the katstore server
    RuntimeError
        If connection succeeded but interaction with katstore64 API failed
    KeyError
        If the sensor was not found in the store or it has no data in time range
    """
    # The sensor name won't be in sensor store if it contains invalid characters
    if not str.isidentifier(name):
        raise KeyError(f"Sensor name '{name}' is not valid Python identifier")
    with requests.Session() as session:
        url = f"http://{store}/katstore/api/query"
        params = {'sensor': name, 'start_time': start_time, 'end_time': end_time,
                  'limit': 1000000, 'include_value_time': 'True'}
        try:
            response = session.get(url, params=params)
        except requests.exceptions.ConnectionError as exc:
            err = ConnectionError(f"Could not connect to sensor store '{store}'")
            raise err from exc
        with response:
            try:
                response.raise_for_status()
                sensor_data = response.json()['data']
                samples = [(rec['value_time'], rec['value'], rec['status'])
                           for rec in sensor_data if rec['sensor'] == name]
            except (ValueError, IndexError, TypeError, KeyError,
                    requests.exceptions.RequestException) as exc:
                err = RuntimeError("Could not retrieve samples from '%s' (%d: %s)" %
                                   (url, response.status_code, response.reason))
                raise err from exc
        if not samples:
            raise KeyError(f"Sensor store has no data for sensor '{name}'")
        samples = np.rec.fromrecords(samples, names='timestamp,value,status')
        return RecordSensorGetter(samples, name)


def dummy_sensor_getter(name, value=None, dtype=np.float64, timestamp=0.0):
    """Create a SensorGetter object with a single default value based on type.

    This creates a dummy :class:`SimpleSensorGetter` object based on a default
    value or a type, for use when no sensor data are available, but filler data
    is required (e.g. when concatenating sensors from different datasets and
    one dataset lacks the sensor). The dummy dataset contains a single data
    point with the filler value and a configurable timestamp (defaulting to
    way back). If the filler value is an object it will be wrapped in a
    :class:`ComparableArrayWrapper` to match the behaviour of other
    :class:`SensorGetter` objects.

    Parameters
    ----------
    name : string
        Sensor name
    value : object, optional
        Filler value (default is None, meaning `dtype` will be used instead)
    dtype : :class:`numpy.dtype` object or equivalent, optional
        Desired sensor data type, used if no explicit value is given
    timestamp : float, optional
        Time when dummy value occurred (default is way back)

    Returns
    -------
    data : :class:`SimpleSensorGetter` object, shape (1,)
        Dummy sensor data object with 'timestamp' and 'value' fields

    """
    if value is None:
        if np.issubdtype(dtype, np.floating):
            value = np.dtype(dtype).type(np.nan)
        elif np.issubdtype(dtype, np.integer):
            value = np.dtype(dtype).type(-1)
        elif np.issubdtype(dtype, np.string_):
            # Order is important here, because np.str is a subtype of np.bool,
            # but not the other way around...
            value = ''
        elif np.issubdtype(dtype, np.bool_):
            value = False
    else:
        dtype = infer_dtype([value])
    if dtype == np.object:
        value = ComparableArrayWrapper(value)
    return SimpleSensorGetter(name, np.array([timestamp]), np.array([value]))


def remove_duplicates_and_invalid_values(sensor):
    """Remove duplicate timestamps and invalid values from sensor data.

    This sorts the 'timestamp' field of the sensor record array and removes any
    duplicate values, updating the corresponding 'value' and 'status' fields as
    well. If more than one timestamp has the same value, the value and status
    of the last of these timestamps are selected. If the values differ for the
    same timestamp, a warning is logged (and the last one is still picked).

    In addition, if there is a 'status' field, get rid of data with a status
    other than 'nominal', 'warn' or 'error', which indicates that the sensor
    could not be read and the corresponding value will therefore be invalid.
    Afterwards, remove the 'status' field from the data as this is the only
    place it plays a role.

    Parameters
    ----------
    sensor : :class:`SensorData` object, length *N*
        Raw sensor dataset.

    Returns
    -------
    clean_sensor : :class:`SensorData` object, length *M*
        Sensor data with duplicate timestamps and invalid values removed
        (*M* <= *N*), and only 'timestamp' and 'value' attributes left.

    """
    x = sensor.timestamp
    y = sensor.value
    z = sensor.status
    # Sort x via mergesort, as it is usually already sorted and stability is important
    sort_ind = np.argsort(x, kind='mergesort')
    x = x[sort_ind]
    y = y[sort_ind]
    if z is not None:
        z = z[sort_ind]
    # Array contains True where an x value is unique or the last of a run of identical x values
    last_of_run = np.asarray(list(np.diff(x) != 0) + [True])
    # Discard the False values, as they represent duplicates - simultaneously keep last of each run of duplicates
    unique_ind = last_of_run.nonzero()[0]
    # Determine the index of the x value chosen to represent each original x value (used to pick y values too)
    replacement = unique_ind[len(unique_ind) - np.cumsum(last_of_run[::-1])[::-1]]
    # All duplicates should have the same y and z values - complain otherwise, but continue
    y_differs = [n for (r, n) in zip(replacement, range(len(y))) if r != n and y[r] != y[n]]
    if y_differs:
        logger.debug("Sensor %r has duplicate timestamps with different values",
                     sensor.name)
        for ind in y_differs:
            logger.debug("At %s, sensor %r has values of %s and %s - "
                         "keeping last one", katpoint.Timestamp(x[ind]).local(),
                         sensor.name, y[ind], y[replacement[ind]])
    if z is not None:
        z_differs = [n for (r, n) in zip(replacement, range(len(z))) if z[r] != z[n]]
        if z_differs:
            logger.debug("Sensor %r has duplicate timestamps with different statuses",
                         sensor.name)
            for ind in z_differs:
                logger.debug("At %s, sensor %r has statuses of %r and %r - "
                             "keeping last one", katpoint.Timestamp(x[ind]).local(),
                             sensor.name, z[ind], z[replacement[ind]])
    # Remove entries where 'status' implies invalid values, if 'status' is present
    if z is not None:
        # Explicitly cast status to string type, as k7_augment produced sensors with integer statuses
        status = z[unique_ind].astype('|S7')
        unique_ind = unique_ind[(status == b'nominal') | (status == b'warn') |
                                (status == b'error')]
    # Strip 'status' / z field from final output as its job is done
    return SensorData(sensor.name, x[unique_ind], y[unique_ind])

# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorCache
# -------------------------------------------------------------------------------------------------


class SensorCache(MutableMapping):
    """Container for sensor data providing name lookup, interpolation and caching.

    *Sensor data* is defined as a one-dimensional time series of values. The
    values may be numerical or non-numerical (*categorical*), and the timestamps
    are monotonically increasing but not necessarily regularly spaced.

    A *sensor cache* stores sensor data with dictionary-like lookup based on
    the sensor name. Since the extraction of sensor data from e.g. HDF5 files
    may be costly, the data is first represented in uncached (raw) form as
    :class:`SensorGetter` objects, which typically wrap the underlying sensor
    HDF5 datasets. After extraction, the sensor data are stored either as
    a NumPy array (for numerical data) or as a :class:`CategoricalData` object
    (for non-numerical data).

    The sensor cache stores a timestamp array (or indexer) onto which the sensor
    data will be interpolated, together with a boolean selection mask that
    selects a subset of the interpolated data as the final output. Interpolation
    is linear for numerical data and zeroth-order for non-numerical data. Both
    extraction and selection may be enabled or disabled through the appropriate
    use of the two main interfaces that retrieve sensor data:

    * The __getitem__ interface (i.e. `cache[sensor]`) presents a simple
      high-level interface to the end user that always extracts the sensor data
      and selects the requested subset from it. In addition, the return type is
      always a NumPy array.

    * The get() interface (i.e. `cache.get(sensor)`) is an advanced interface
      for library builders that provides full control of the extraction process
      via *sensor properties*. It does not apply selection by default, as this
      is more convenient for library routines.

    In addition, the sensor cache may contain *virtual sensors* which calculate
    their values based on the values of other sensors. They are identified by
    pattern templates that potentially match multiple sensor names.

    Parameters
    ----------
    cache : mapping from string to :class:`SensorGetter` objects
        Initial sensor cache mapping sensor names to raw (uncached) sensor data
    timestamps : array of float
        Correlator data timestamps onto which sensor values will be interpolated,
        as UTC seconds since Unix epoch
    dump_period : float
        Dump period, in seconds
    keep : int or slice or sequence of int or sequence of bool, optional
        Default time selection specification that will be applied to sensor data
        (this can be disabled on data retrieval)
    props : dict, optional
        Default properties that govern how sensor data are interpreted and
        interpolated (this can be overridden on data retrieval). Can use ``*``
        as a wildcard anywhere in the key.
    virtual : dict mapping string to function, optional
        Virtual sensors, specified as a pattern matching the virtual sensor name
        and a corresponding function that will create the sensor (together with
        any associated virtual sensors)
    aliases : dict mapping string to string, optional
        Alternate names for sensors, as a dictionary mapping each alias to the
        original sensor name suffix. This will create additional sensors with
        the aliased names and the data of the original sensors.
    store : string, optional
        Hostname / endpoint of katstore webserver to access additional sensors
    """

    def __init__(self, cache, timestamps, dump_period, keep=slice(None),
                 props=None, virtual={}, aliases={}, store=None):
        super().__init__()
        # This needs to be an RLock because instantiating a virtual sensor
        # may require further sensor lookups (hopefully without a loop, which
        # would really cause problems).
        self._lock = threading.RLock()
        # Store internals of the cache in a regular dict
        self._raw = dict(cache)
        self.timestamps = timestamps
        self.dump_period = dump_period
        self.keep = keep
        self.props = props if props is not None else {}
        # Add virtual sensor templates
        self.virtual = virtual
        # Add sensor aliases
        for alias, original in aliases.items():
            self.add_aliases(alias, original)
        self.store = store

    def __str__(self):
        """Verbose human-friendly string representation of sensor cache object."""
        with self._lock:
            names = sorted([key for key in self.keys()])
            maxlen = max([len(name) for name in names])
            objects = [self.get(name, extract=False) for name in names]
        obj_reprs = [(f"<numpy.ndarray shape={obj.shape} type={obj.dtype} at {id(obj):#x}>"
                      if isinstance(obj, np.ndarray) else repr(obj)) for obj in objects]
        actual = [f'{name!s:{maxlen}} : {obj_repr}'
                  for name, obj_repr in zip(names, obj_reprs)]
        virtual = [f'{pat!s:{maxlen}} : <function {func.__module__}.{func.__name__}>'
                   for pat, func in self.virtual.items()]
        return '\n'.join(['Actual sensors', '--------------'] + actual +
                         ['\nVirtual sensors', '---------------'] + virtual)

    def __repr__(self):
        """Short human-friendly string representation of sensor cache object."""
        with self._lock:
            sensors = [self.get(name, extract=False) for name in self.keys()]
        return "<katdal.{} sensors={} cached={} virtual={} at {:#x}>".format(
               self.__class__.__name__, len(sensors),
               np.sum([not isinstance(s, SensorGetter) for s in sensors]),
               len(self.virtual), id(self))

    def __getitem__(self, name):
        """Sensor values interpolated to correlator data timestamps.

        Time selection is enforced as this is a user-facing sensor data
        extraction method, and end users expect their selections to apply.
        This has the added benefit of a consistent array return type.

        Parameters
        ----------
        name : string
            Sensor name

        Returns
        -------
        sensor_data : array
           Interpolated sensor data as 1-D array, one value per selected timestamp

        Raises
        ------
        KeyError
            If sensor name was not found in cache

        """
        return self.get(name, select=True)

    def _set_keep(self, keep=None):
        """Set time selection for sensor values."""
        if keep is not None:
            self.keep = keep

    @staticmethod
    def _get_props(name, prop_map, **kwargs):
        """Retrieve properties for a sensor.

        Sensor names in `prop_map` may contain ``*`` wildcard characters. All
        matching entries are merged, and `prop_map` is updated in place with
        the merged result.

        Parameters
        ----------
        name : str
            Sensor name
        prop_map : dict
            Maps sensor names to mappings of properties.
        kwargs
            Extra properties to apply, overriding those in `prop_map`.
        """
        # Look up properties associated with this specific sensor
        props = prop_map.setdefault(name, {})
        # Look up properties associated with this class of sensor
        for key, val in prop_map.items():
            if '*' in key:
                regex = '.*'.join(re.escape(part) for part in key.split('*'))
                if re.match('^' + regex + '$', name):
                    props.update(val)
        # Any properties passed directly to this method takes precedence
        props.update(kwargs)
        return props

    @staticmethod
    def _extract(sensor_getter, timestamps, dump_period, **props):
        sensor_data = sensor_getter.get()
        # Clean up sensor data if non-empty
        if sensor_data:
            time_offset = props.get('time_offset', 0)
            sensor_data.timestamp += time_offset
            # Sort sensor events in chronological order and discard duplicates and unreadable sensor values
            sensor_data = remove_duplicates_and_invalid_values(sensor_data)
        if not sensor_data:
            sensor_data = dummy_sensor_getter(sensor_data.name, value=props.get('initial_value'),
                                              dtype=sensor_data.value.dtype).get()
            logger.warning("No usable data found for sensor '%s' - replaced with dummy data (%r)" %
                           (sensor_data.name, sensor_data.value[0]))
        # Determine if sensor produces categorical or numerical data
        # (float data are non-categorical, by default)
        categ = props.get('categorical', not np.issubdtype(sensor_data.value.dtype, np.floating))
        props['categorical'] = categ
        if categ:
            sensor_data = sensor_to_categorical(sensor_data.timestamp, sensor_data.value,
                                                timestamps, dump_period, **props)
        else:
            # Interpolate numerical data onto data timestamps
            sensor_timestamps = sensor_data.timestamp
            # Warn if sensor data will be extrapolated to start or end
            # of data set with potentially bogus results
            if len(sensor_timestamps) > 1:
                if sensor_timestamps[0] > timestamps[0]:
                    logger.warning("First data point for sensor '%s' only arrives %g seconds into data set" %
                                   (sensor_data.name, sensor_timestamps[0] - timestamps[0]))
                if sensor_timestamps[-1] < timestamps[-1]:
                    logger.warning("Last data point for sensor '%s' arrives %g seconds "
                                   "before end of data set" %
                                   (sensor_data.name, timestamps[-1] - sensor_timestamps[-1]))
            sensor_data = np.interp(timestamps, sensor_timestamps, sensor_data.value)
        return sensor_data

    def add_aliases(self, alias, original):
        """Add alternate names / aliases for sensors.

        Search for sensors with names ending in the `original` suffix and form
        a corresponding alternate name by replacing `original` with `alias`.
        The new aliased sensors will re-use the data of the original sensors.

        Parameters
        ----------
        alias : string
            The new sensor name suffix that replaces `original`
        original : string
            Sensors with names that end in this will get aliases

        """
        for name, data in list(self._raw.items()):
            if name.endswith(original):
                self._raw[name.replace(original, alias)] = data

    def get(self, name, select=False, extract=True, **kwargs):
        """Sensor values interpolated to correlator data timestamps.

        Time selection is disabled by default, as this is a more advanced data
        extraction method typically called by library routines that want to
        operate on the full array of sensor values. For additional allowed
        parameters when extracting categorical data, see the docstring for
        :func:`sensor_to_categorical`.

        Parameters
        ----------
        name : string
            Sensor name
        select : {False, True}, optional
            True if preset time selection will be applied to interpolated data
        extract : {True, False}, optional
            True if sensor data should be extracted, interpolated and cached
        categorical : {None, True, False}, optional
            Interpret sensor data as categorical or numerical (by default, data
            of type float is numerical and of any other type is categorical)
        kwargs : dict, optional
            Additional parameters are passed to :func:`sensor_to_categorical`

        Returns
        -------
        data : array or :class:`CategoricalData` or :class:`SensorGetter` object
            If extraction is disabled, this will be a :class:`SensorGetter` object
            for uncached sensors. If selection is enabled, this will be a 1-D
            array of values, one per selected timestamp. If selection is
            disabled, this will be a 1-D array of values (of the same length as
            the :attr:`timestamps` attribute) for numerical data, and a
            :class:`CategoricalData` object for categorical data.

        Raises
        ------
        ValueError
            If select=True and extract=False, as select requires interpolation
        KeyError
            If sensor name was not found in cache and did not match virtual template

        """
        if select and not extract:
            raise ValueError('Cannot apply selection on raw sensor data')
        with self._lock:
            try:
                # First try to load the actual sensor data from cache
                sensor_data = self._raw[name]
            except KeyError:
                # Otherwise, iterate through virtual sensor templates and look for a match
                for pattern, create_sensor in self.virtual.items():
                    # Expand variable names enclosed in braces to the relevant regular expression
                    # (match anything but slashes, which are preferred delimiters in virtual sensor names)
                    pattern = re.sub(r'(\{[a-zA-Z_]\w*\})',
                                     lambda m: '(?P<{}>[^/]+)'.format(m.group(0)[1:-1]), pattern)
                    match = re.match(pattern, name)
                    if match:
                        # Call sensor creation function with extracted variables from sensor name
                        sensor_data = create_sensor(self, name, **match.groupdict())
                        break
                else:
                    if self.store:
                        # Katstore samples sensors at least once every 10 minutes
                        # Go that far back to support sporadic discrete sensors
                        start_time = self.timestamps[0] - self.dump_period - 600
                        end_time = self.timestamps[-1] + self.dump_period + 60
                        sensor_data = get_sensor_from_katstore(
                            self.store, name, start_time, end_time)
                    else:
                        raise KeyError(f"Unknown sensor '{name}' (does not match actual name or "
                                       "virtual template and no sensor store provided)")
            # If this is the first time this sensor is accessed, extract its data and store it in cache, if enabled
            if isinstance(sensor_data, SensorGetter) and extract:
                props = self._get_props(name, self.props, **kwargs)
                # If this is the first time any sensor is accessed, obtain all data timestamps via indexer
                self.timestamps = self.timestamps[:] if not isinstance(self.timestamps, np.ndarray) else self.timestamps
                sensor_data = self._extract(sensor_data, self.timestamps, self.dump_period, **props)
                self._raw[name] = sensor_data
        return sensor_data[self.keep] if select else sensor_data

    def get_with_fallback(self, sensor_type, names):
        """Sensor values interpolated to correlator data timestamps.

        Get data for a type of sensor that may have one of several names.
        Try each name in turn until something works, or crash sensibly.

        Parameters
        ----------
        sensor_type : string
            Name of sensor class / type, used for informational purposes only
        names : sequence of strings
            Sensor names to try until one of them provides data

        Returns
        -------
        sensor_data : array
           Interpolated sensor data as 1-D array, one value per selected timestamp

        Raises
        ------
        KeyError
            If none of the sensor names were found in the cache

        """
        for name in names:
            try:
                return self.get(name, select=True)
            except KeyError:
                logger.debug('Could not find %s sensor with name %r, trying next option', sensor_type, name)
        raise KeyError(f'Could not find any {sensor_type} sensor, tried {names}')

    # MutableMapping abstract methods

    def __setitem__(self, key, item):
        with self._lock:
            self._raw[key] = item

    def __delitem__(self, key):
        with self._lock:
            del self._raw[key]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    def __contains__(self, key):
        # __contains__ is implemented by MutableMapping via __getitem__, but
        # that does unnecessary extraction. This approach is cheaper but only
        # reflects keys that have been explicitly created or cached.
        with self._lock:
            return key in self._raw
