################################################################################
# Copyright (c) 2011-2018, National Research Foundation (Square Kilometre Array)
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
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
import future.utils
from builtins import zip
from builtins import range
from builtins import object
import logging
import functools
import re
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import katpoint

from .categorical import (ComparableArrayWrapper, infer_dtype,
                          sensor_to_categorical)

logger = logging.getLogger(__name__)

# Optionally depend on scikits.fitting for higher-order sensor data interpolation
try:
    from scikits.fitting import PiecewisePolynomial1DFit
except ImportError:
    PiecewisePolynomial1DFit = None

# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorData
# -------------------------------------------------------------------------------------------------


class SensorData(object):
    """Raw (uninterpolated) sensor data placeholder.

    This is basically a placeholder for uninterpolated sensor data resembling
    a structured array with fields 'timestamp', 'value' and optionally 'status'.

    Its main advantage is that it exposes the sensor data dtype (from the
    'value' field, if available) as a top-level attribute, making it compatible
    with NumPy arrays and :class:`CategoricalData` objects when used together
    in a sensor cache. It also exposes the sensor name, if available.

    The idea is that the raw sensor data is not initially cached in this object,
    making it a light-weight wrapper focussing on sensor metadata. All data
    access should be via __getitem__ of the appropriate field.

    Where possible, object-valued sensors (including sensors with ndarrays as
    values) will have values wrapped by :class:`ComparableArrayWrapper`.

    Parameters
    ----------
    name : string
        Sensor name
    dtype : :class:`numpy.dtype` object or equivalent or None
        Sensor value type as NumPy dtype (None if not available yet)

    """

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __getitem__(self, key):
        """Extract timestamp and value (and status) of each sensor data point.

        Parameters
        ----------
        key : {'timestamp', 'value', 'status'}
            Name of field to access ('status' is optional and raises ValueError
            if not supported)

        Returns
        -------
        field : :class:`numpy.ndarray` object, shape (N,)
            Requested field as 1-D array of appropriate dtype (float,
            self.dtype or string, respectively, for timestamp / value / status)

        Raises
        ------
        ValueError
            If `key` is unsupported field name

        """
        raise NotImplementedError

    def __bool__(self):
        """True if sensor has at least one data point."""
        raise NotImplementedError

    def __repr__(self):
        """Short human-friendly string representation of sensor data object."""
        return "<katdal.%s '%s' type=%s at 0x%x>" % \
               (self.__class__.__name__, self.name, self.dtype, id(self))


class RecordSensorData(SensorData):
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
        dtype = data.dtype.fields['value'][0]
        super(RecordSensorData, self).__init__(name, dtype)
        self._data = data

    def __getitem__(self, key):
        """Extract timestamp, value and status of each sensor data point.

        Values are passed through :func:`to_str`.
        """
        values = np.asarray(self._data[key])
        if key == 'value':
            values = to_str(values)
        return values

    def __bool__(self):
        """True if sensor has at least one data point."""
        return len(self._data) > 0

    def __repr__(self):
        """Short human-friendly string representation of sensor data object."""
        return "<katdal.%s '%s' len=%d type=%s at 0x%x>" % \
               (self.__class__.__name__, self.name,
                len(self._data), self.dtype, id(self))


def to_str(value):
    """Convert string-likes to the native string type.

    On Python 3, bytes are decoded to str, with surrogateencoding error
    handler. On Python 2, unicode is encoded to str, with UTF-8 encoding.

    Tuples, lists, dicts and numpy arrays are processed recursively, with the
    exception that numpy structured types with string or object fields won't
    be handled.
    """
    if future.utils.PY3:
        if isinstance(value, np.ndarray) and value.dtype.kind == 'S':
            return np.char.decode(value, 'utf-8', 'surrogateescape')
        elif isinstance(value, bytes):
            return value.decode('utf-8', 'surrogateescape')
    else:
        if isinstance(value, np.ndarray) and value.dtype.kind == 'U':
            return np.char.encode(value, 'utf-8')
        elif isinstance(value, unicode):
            return value.encode('utf-8')

    # We use type(value) so that subclasses are reconstructed correctly
    if isinstance(value, (list, tuple)):
        return type(value)(to_str(item) for item in value)
    elif isinstance(value, dict):
        return type(value)((to_str(key), to_str(val)) for key, val in value.items())
    elif isinstance(value, np.ndarray) and value.dtype == 'O':
        return np.vectorize(to_str, otypes='O')(value)
    else:
        return value


# https://stackoverflow.com/questions/11305790
if future.utils.PY3:
    _pickle_loads = functools.partial(pickle.loads, encoding='latin1')
else:
    _pickle_loads = pickle.loads


def pickle_loads(raw, no_unpickle=()):
    """Load a pickle that might be wrapped in np.void or np.ndarray.

    The np.void wrapping is needed to pass variable-length binary strings
    through h5py. The pickle module handles it transparently, but cPickle does
    not.

    If the value is a string and is in no_unpickle, it is returned verbatim.
    This is for backwards compatibility with older files that didn't use
    pickles.

    The return value is also passed through :func:`to_str`.
    """
    if isinstance(raw, (np.void, np.ndarray)):
        return to_str(_pickle_loads(raw.tostring()))
    elif raw not in no_unpickle:
        return to_str(_pickle_loads(raw))
    else:
        return to_str(raw)


def _h5_telstate_unpack(s):
    """Unpack a telstate value from its string representation."""
    try:
        # Since 2016-05-09 the HDF5 TelescopeState contains pickled values
        return pickle_loads(s)
    except (pickle.UnpicklingError, ValueError, EOFError):
        try:
            # Before 2016-05-09 the telstate values were str() representations
            # This cannot be unpacked in general but works for numbers at least
            return np.safe_eval(s)
        except (ValueError, SyntaxError):
            # When unsure, return the string itself (correct for string sensors)
            return s


class H5TelstateSensorData(RecordSensorData):
    """Raw (uninterpolated) sensor data in HDF5 TelescopeState recarray form.

    This wraps the telstate sensors stored in recent HDF5 files. It differs
    in two ways from the normal HDF5 sensors: no 'status' field and values
    stored as pickles.

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
        super(H5TelstateSensorData, self).__init__(data, name)
        # The dtype is not immediately available - need to unpickle data first
        self.dtype = None

    def __getitem__(self, key):
        """Extract timestamp and value of each sensor data point."""
        if key == 'timestamp':
            return np.asarray(self._data[key])
        elif key == 'value':
            # Unpack everything first, otherwise old files will be a mess
            values = [_h5_telstate_unpack(s) for s in self._data[key]]
            # Figure out dtype and wrap any objects
            self.dtype = infer_dtype(values)
            if self.dtype == np.object:
                values = [ComparableArrayWrapper(value) for value in values]
            return values
        else:
            raise ValueError("Sensor %r data has no key '%s'" % (self.name, key))


class TelstateToStr(object):
    """Wrap an existing telescope state and pass return values through :meth:`to_str`"""
    def __init__(self, telstate):
        if isinstance(telstate, TelstateToStr):
            self._telstate = telstate._telstate
        else:
            self._telstate = telstate

    def view(self, name, add_separator=True, exclusive=False):
        return TelstateToStr(self._telstate.view(name, add_separator, exclusive))

    def root(self):
        return TelstateToStr(self._telstate.root())

    def keys(self, filter='*'):
        return to_str(self._telstate.keys(filter))

    @property
    def prefixes(self):
        return to_str(self._telstate.prefixes)

    def __getattr__(self, key):
        # __getattr__ can be used for item access or to get a property of the
        # class.
        if hasattr(self._telstate.__class__, key):
            return getattr(self._telstate, key)
        else:
            return to_str(getattr(self._telstate, key))

    def __contains__(self, key):
        # Needed because __getattr__ won't pick it up from child
        return key in self._telstate

    def __getitem__(self, key):
        return to_str(self._telstate[key])

    def get_message(self, channel=None):
        return to_str(self._telstate.get_message(channel))

    def get(self, key, default=None, return_pickle=False):
        value = self._telstate.get(key, default, return_pickle)
        if not return_pickle:
            value = to_str(value)
        return value

    def get_range(self, key, st=None, et=None,
                  include_previous=None, include_end=False, return_pickle=False):
        value = self._telstate.get_range(key, st, et, include_previous, include_end, return_pickle)
        if not return_pickle:
            value = to_str(value)
        return value


class TelstateSensorData(SensorData):
    """Raw (uninterpolated) sensor data stored in original TelescopeState.

    This wraps sensor data stored in a TelescopeState object. The data is
    only read out on item access.

    Object-valued sensors (including sensors with ndarrays as values) will have
    its values wrapped by :class:`ComparableArrayWrapper`.

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

    Notes
    -----
    The sensor data is cached on the object after any item access to ensure
    that requesting timestamps and then values only loads the data once instead
    of twice. The caching should be fine as a SensorData object is typically
    replaced by either a CategoricalData object or a NumPy array as part of
    sensor extraction, right after the caching occurs.

    """

    def __init__(self, telstate, name):
        self._telstate = TelstateToStr(telstate)
        # This cache simplifies separate 'timestamp' / 'value' access pattern
        self._values = self._times = None
        if name not in telstate:
            raise KeyError('No sensor named %r in telstate (key not found)' %
                           (name,))
        if telstate.is_immutable(name):
            raise KeyError("No sensor named %r in telstate (it's an attribute)" %
                           (name,))
        # The dtype is not immediately available - need to unpickle data first
        super(TelstateSensorData, self).__init__(name, None)

    def __bool__(self):
        """True if sensor has at least one data point (already checked in init)."""
        return True

    def _cache_data(self):
        if not self._times:
            value_times = self._telstate.get_range(self.name, st=0)
            self._values = [v for v, t in value_times]
            self.dtype = infer_dtype(self._values)
            if self.dtype == np.object:
                self._values = [ComparableArrayWrapper(v) for v in self._values]
            self._times = [t for v, t in value_times]

    def __getitem__(self, key):
        """Extract timestamp and value of each sensor data point."""
        if key == 'timestamp':
            self._cache_data()
            return np.array(self._times)
        elif key == 'value':
            self._cache_data()
            return np.array(self._values)
        else:
            raise ValueError("Sensor %r data has no key '%s'" % (self.name, key))


# -------------------------------------------------------------------------------------------------
# -- Utility functions
# -------------------------------------------------------------------------------------------------


def _safe_linear_interp(xi, yi, x):
    """Linearly interpolate (xi, yi) values to x positions, safely.

    Given a set of N ``(x, y)`` points, provided in the `xi` and `yi` arrays,
    this will calculate ``y``-coordinate values for a set of M ``x``-coordinates
    provided in the `x` array, using linear interpolation.

    It is safe in the sense that if `xi` and `yi` only contain a single point
    it will revert to zeroth-order interpolation. In addition, data will not
    be extrapolated linearly past the edges of `xi`, but the closest value
    will be used instead (i.e. also zeroth-order interpolation).

    Parameters
    ----------
    xi : array, shape (N,)
        Array of fixed x-coordinates, sorted in ascending order and with no
        duplicate values
    yi : array, shape (N,)
        Corresponding array of fixed y-coordinates
    x : float or array, shape (M,)
        Array of x-coordinates at which to do interpolation of y-values

    Returns
    -------
    y : float or array, shape (M,)
        Array of interpolated y-values

    Notes
    -----
    This is mostly lifted from scikits.fitting.poly as it is the only part of
    the package that is typically required. This weens katdal off SciPy too.

    """
    # Do zeroth-order interpolation for a single fixed (x, y) coordinate
    if len(xi) == 1:
        # The simplest way to handle x of e.g. 3, np.array(3) and [1, 2, 3]
        return yi[0] * np.ones_like(x)
    # Find lowest xi value >= x (end of segment containing x)
    end = np.atleast_1d(xi.searchsorted(x))
    # Associate any x found outside xi range with closest segment (first or last one)
    end[end == 0] += 1
    end[end == len(xi)] -= 1
    start = end - 1
    # Ensure that output y has same shape as input x
    # (especially, let scalar input result in scalar output)
    start, end = np.reshape(start, np.shape(x)), np.reshape(end, np.shape(x))
    # Set up weight such that xi[start] => 0 and xi[end] => 1
    end_weight = (x - xi[start]) / (xi[end] - xi[start])
    # Do zeroth-order interpolation beyond the range of xi
    end_weight = np.clip(end_weight, 0.0, 1.0)
    return (1.0 - end_weight) * yi[start] + end_weight * yi[end]


def dummy_sensor_data(name, value=None, dtype=np.float64, timestamp=0.0):
    """Create a SensorData object with a single default value based on type.

    This creates a dummy :class:`RecordSensorData` object based on a default
    value or a type, for use when no sensor data are available, but filler data
    is required (e.g. when concatenating sensors from different datasets and
    one dataset lacks the sensor). The dummy dataset contains a single data
    point with the filler value and a configurable timestamp (defaulting to
    way back). If the filler value is an object it will be wrapped in a
    :class:`ComparableArrayWrapper` to match the behaviour of other
    :class:`SensorData` objects.

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
    data : :class:`RecordSensorData` object, shape (1,)
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
    data = np.array([(timestamp, value)],
                    dtype=[('timestamp', np.float64), ('value', dtype)])
    return RecordSensorData(data, name)


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
        Raw sensor dataset, which acts like a record array with fields
        'timestamp', 'value' and optionally 'status'

    Returns
    -------
    clean_sensor : :class:`RecordSensorData` object, length *M*
        Sensor data with duplicate timestamps and invalid values removed
        (*M* <= *N*), and only 'timestamp' and 'value' fields left

    """
    x = np.atleast_1d(sensor['timestamp'])
    y = np.atleast_1d(sensor['value'])
    try:
        z = np.atleast_1d(sensor['status'])
    except ValueError:
        z = None
    # Sort x via mergesort, as it is usually already sorted and stability is important
    sort_ind = np.argsort(x, kind='mergesort')
    x, y = x[sort_ind], y[sort_ind]
    # Array contains True where an x value is unique or the last of a run of identical x values
    last_of_run = np.asarray(list(np.diff(x) != 0) + [True])
    # Discard the False values, as they represent duplicates - simultaneously keep last of each run of duplicates
    unique_ind = last_of_run.nonzero()[0]
    # Determine the index of the x value chosen to represent each original x value (used to pick y values too)
    replacement = unique_ind[len(unique_ind) - np.cumsum(last_of_run[::-1])[::-1]]
    # All duplicates should have the same y and z values - complain otherwise, but continue
    y_differs = [n for (r, n) in zip(replacement, range(len(y))) if y[r] != y[n]]
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
    data = np.array(list(zip(x[unique_ind], y[unique_ind])),
                    dtype=[('timestamp', x.dtype), ('value', y.dtype)])
    return RecordSensorData(data, sensor.name)

# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorCache
# -------------------------------------------------------------------------------------------------


class SensorCache(dict):
    """Container for sensor data providing name lookup, interpolation and caching.

    *Sensor data* is defined as a one-dimensional time series of values. The
    values may be numerical or non-numerical (*categorical*), and the timestamps
    are monotonically increasing but not necessarily regularly spaced.

    A *sensor cache* stores sensor data with dictionary-like lookup based on
    the sensor name. Since the extraction of sensor data from e.g. HDF5 files
    may be costly, the data is first represented in uncached (raw) form as
    :class:`SensorData` objects, which typically wrap the underlying sensor
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
    cache : mapping from string to :class:`SensorData` objects
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
        interpolated (this can be overridden on data retrieval)
    virtual : dict mapping string to function, optional
        Virtual sensors, specified as a pattern matching the virtual sensor name
        and a corresponding function that will create the sensor (together with
        any associated virtual sensors)
    aliases : dict mapping string to string, optional
        Alternate names for sensors, as a dictionary mapping each alias to the
        original sensor name suffix. This will create additional sensors with
        the aliased names and the data of the original sensors.

    """

    def __init__(self, cache, timestamps, dump_period, keep=slice(None), props=None, virtual={}, aliases={}):
        # Initialise cache via dict constructor
        super(SensorCache, self).__init__(cache)
        self.timestamps = timestamps
        self.dump_period = dump_period
        self.keep = keep
        self.props = props if props is not None else {}
        # Add virtual sensor templates
        self.virtual = virtual
        # Add sensor aliases
        for alias, original in aliases.items():
            self.add_aliases(alias, original)

    def __str__(self):
        """Verbose human-friendly string representation of sensor cache object."""
        names = sorted([key for key in self.keys()])
        maxlen = max([len(name) for name in names])
        objects = [self.get(name, extract=False) for name in names]
        obj_reprs = [(("<numpy.ndarray shape=%s type=%s at 0x%x>" % (obj.shape, obj.dtype, id(obj)))
                     if isinstance(obj, np.ndarray) else repr(obj)) for obj in objects]
        actual = ['%s : %s' % (str(name).ljust(maxlen), obj_repr) for name, obj_repr in zip(names, obj_reprs)]
        virtual = ['%s : <function %s.%s>' % (str(pat).ljust(maxlen), func.__module__, func.__name__)
                   for pat, func in self.virtual.items()]
        return '\n'.join(['Actual sensors', '--------------'] + actual +
                         ['\nVirtual sensors', '---------------'] + virtual)

    def __repr__(self):
        """Short human-friendly string representation of sensor cache object."""
        sensors = [self.get(name, extract=False) for name in self.keys()]
        return "<katdal.%s sensors=%d cached=%d virtual=%d at 0x%x>" % \
               (self.__class__.__name__, len(sensors),
                np.sum([not isinstance(s, SensorData) for s in sensors]),
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
        for name, data in list(self.items()):
            if name.endswith(original):
                self[name.replace(original, alias)] = data

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
        interp_degree : int, optional
            Polynomial degree for interpolation of numerical data (default = 1)
        kwargs : dict, optional
            Additional parameters are passed to :func:`sensor_to_categorical`

        Returns
        -------
        data : array or :class:`CategoricalData` or :class:`SensorData` object
            If extraction is disabled, this will be a :class:`SensorData` object
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
        try:
            # First try to load the actual sensor data from cache (remember to call base class here!)
            sensor_data = super(SensorCache, self).__getitem__(name)
        except KeyError:
            # Otherwise, iterate through virtual sensor templates and look for a match
            for pattern, create_sensor in self.virtual.items():
                # Expand variable names enclosed in braces to the relevant regular expression
                pattern = re.sub('({[a-zA-Z_]\w*})', lambda m: '(?P<' + m.group(0)[1:-1] + '>[^//]+)', pattern)
                match = re.match(pattern, name)
                if match:
                    # Call sensor creation function with extracted variables from sensor name
                    sensor_data = create_sensor(self, name, **match.groupdict())
                    break
            else:
                raise KeyError("Unknown sensor '%s' (does not match actual name or virtual template)" % (name,))
        # If this is the first time this sensor is accessed, extract its data and store it in cache, if enabled
        if isinstance(sensor_data, SensorData) and extract:
            # Look up properties associated with this specific sensor
            self.props[name] = props = self.props.get(name, {})
            # Look up properties associated with this class of sensor
            for key, val in self.props.items():
                if key[0] == '*' and name.endswith(key[1:]):
                    props.update(val)
            # Any properties passed directly to this method takes precedence
            props.update(kwargs)
            # Clean up sensor data if non-empty
            if sensor_data:
                # Sort sensor events in chronological order and discard duplicates and unreadable sensor values
                sensor_data = remove_duplicates_and_invalid_values(sensor_data)
            if not sensor_data:
                sensor_data = dummy_sensor_data(name, value=props.get('initial_value'), dtype=sensor_data.dtype)
                logger.warning("No usable data found for sensor '%s' - replaced with dummy data (%r)" %
                               (name, sensor_data['value'][0]))
            # If this is the first time any sensor is accessed, obtain all data timestamps via indexer
            self.timestamps = self.timestamps[:] if not isinstance(self.timestamps, np.ndarray) else self.timestamps
            # Determine if sensor produces categorical or numerical data (by default, float data are non-categorical)
            categ = props.get('categorical', not np.issubdtype(sensor_data.dtype, np.floating))
            props['categorical'] = categ
            if categ:
                sensor_data = sensor_to_categorical(sensor_data['timestamp'], sensor_data['value'],
                                                    self.timestamps, self.dump_period, **props)
            else:
                # Interpolate numerical data onto data timestamps (fallback option is linear interpolation)
                props['interp_degree'] = interp_degree = props.get('interp_degree', 1)
                sensor_timestamps = sensor_data['timestamp']
                # Warn if sensor data will be extrapolated to start or end of data set with potentially bogus results
                if interp_degree > 0 and len(sensor_timestamps) > 1:
                    if sensor_timestamps[0] > self.timestamps[0]:
                        logger.warning(("First data point for sensor '%s' only arrives %g seconds into data set" %
                                       (name, sensor_timestamps[0] - self.timestamps[0])) +
                                       " - extrapolation may lead to ridiculous values")
                    if sensor_timestamps[-1] < self.timestamps[-1]:
                        logger.warning(("Last data point for sensor '%s' arrives %g seconds before end of data set" %
                                       (name, self.timestamps[-1] - sensor_timestamps[-1])) +
                                       " - extrapolation may lead to ridiculous values")
                if PiecewisePolynomial1DFit is not None:
                    interp = PiecewisePolynomial1DFit(max_degree=interp_degree)
                    interp.fit(sensor_timestamps, sensor_data['value'])
                    sensor_data = interp(self.timestamps)
                else:
                    if interp_degree != 1:
                        logger.warning('Requested sensor interpolation with polynomial degree ' + str(interp_degree) +
                                       ' but scikits.fitting not installed - falling back to linear interpolation')
                    sensor_data = _safe_linear_interp(sensor_timestamps, sensor_data['value'], self.timestamps)
            self[name] = sensor_data
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
                logger.debug('Could not find %s sensor with name %r, trying next option' % (sensor_type, name))
        raise KeyError('Could not find any %s sensor, tried %s' % (sensor_type, names))
