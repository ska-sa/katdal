################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
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

"""Container that stores cached and uncached (raw) sensor data."""

import logging
import re
import cPickle as pickle

import numpy as np
import katpoint

from .categorical import sensor_to_categorical

logger = logging.getLogger(__name__)

# This is needed for tab completion, but is ignored if no IPython is installed
try:
    # IPython 0.11 and above
    from IPython.core.error import TryNext
except ImportError:
    try:
        # IPython 0.10 and below
        from IPython.ipapi import TryNext
    except ImportError:
        pass
# The same goes for readline
try:
    import readline
except ImportError:
    readline = None

# Optionally depend on scikits.fitting for higher-order sensor data interpolation
try:
    from scikits.fitting import PiecewisePolynomial1DFit
except ImportError:
    PiecewisePolynomial1DFit = None

# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorData
# -------------------------------------------------------------------------------------------------


class SensorData(object):
    """Raw (uncached) sensor data in record array form.

    This is a convenient container for uncached sensor data which resembles
    a record array with fields ('timestamp', 'value', 'status'). This is also
    the typical format of HDF5 datasets used to store sensor data. Its main
    advantage is that it exposes the sensor data dtype (from the 'value' field)
    as a top-level attribute, making it compatible with NumPy arrays and
    :class:`CategoricalData` objects when used together in a sensor cache. It
    also exposes the sensor name, if available.

    Parameters
    ----------
    data : recarray-like, with fields ('timestamp', 'value', 'status')
        Uncached sensor data as record array or equivalent (such as a
        :class:`h5py.Dataset`)
    name : string or None, optional
        Sensor name (assumed to be data.name by default, if it exists)

    Attributes
    ----------
    dtype : :class:`numpy.dtype` object
        Type of sensor data, as a NumPy dtype

    """
    def __init__(self, data, name=None):
        self.data = data
        self.name = name if name is not None else getattr(data, 'name', '')
        self.dtype = self.data.dtype.fields['value'][0]

    def __getitem__(self, key):
        """Extract timestamp, value and status of each sensor data point."""
        return np.asarray(self.data[key])

    def __len__(self):
        """Number of sensor data points."""
        return len(self.data)

    def __repr__(self):
        """Short human-friendly string representation of sensor data object."""
        return "<katdal.%s '%s' len=%d type='%s' at 0x%x>" % \
               (self.__class__.__name__, self.name, len(self), self.dtype, id(self))


def _telstate_unpack(s):
    """This unpacks a telstate value from its string representation."""
    try:
        # Since 2016-05-09 the HDF5 TelescopeState contains pickled values
        return pickle.loads(s)
    except (pickle.UnpicklingError, ValueError, EOFError):
        try:
            # Before 2016-05-09 the telstate values were str() representations
            # This cannot be unpacked in general but works for numbers at least
            return np.safe_eval(s)
        except (ValueError, SyntaxError):
            # When unsure, return the string itself (correct for string sensors)
            return s


class TelstateSensorData(SensorData):
    """Raw (uncached) sensor data in TelescopeState record array form.

    This wraps the telstate sensors stored in recent HDF5 files. It differs
    in two ways from the normal HDF5 sensors: no 'status' column and values
    stored as pickles.

    TODO: This is a temporary fix to get at missing sensors in telstate and
    should be replaced by a proper wrapping of any telstate object.

    Parameters
    ----------
    data : recarray-like, with fields ('timestamp', 'value')
        Uncached sensor data as record array or equivalent (such as a
        :class:`h5py.Dataset`)
    name : string or None, optional
        Sensor name (assumed to be data.name by default, if it exists)

    Attributes
    ----------
    dtype : :class:`numpy.dtype` object
        Type of sensor data, as a NumPy dtype

    """
    def __init__(self, data, name=None):
        super(TelstateSensorData, self).__init__(data, name)
        # Unpickle first value to derive dtype (should be simple for sensors)
        self.dtype = np.array([_telstate_unpack(data['value'][0])]).dtype

    def __getitem__(self, key):
        """Extract timestamp, value and status of each sensor data point."""
        if key == 'value':
            # Unpickle sensor data upon request
            return np.array([_telstate_unpack(s) for s in self.data[key]])
        elif key == 'status':
            # Fake the missing status column
            return np.repeat('nominal', len(self))
        else:
            # WARNING: if key is an index or slice, values will still be pickled
            # It is safer to extract data explicitly into recarray in __init__
            return np.asarray(self.data[key])

# -------------------------------------------------------------------------------------------------
# -- Utility functions
# -------------------------------------------------------------------------------------------------


def _safe_linear_interp(xi, yi, x):
    """Linearly interpolate (xi, yi) values to x positions, safely.

    Given a set of N ``(x, y)`` points, provided in the *xi* and *yi* arrays,
    this will calculate ``y``-coordinate values for a set of M ``x``-coordinates
    provided in the *x* array, using linear interpolation.

    It is safe in the sense that if *xi* and *yi* only contain a single point
    it will revert to zeroth-order interpolation. In addition, data will not
    be extrapolated linearly past the edges of *xi*, but the closest value
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

    This creates a dummy :class:`SensorData` object based on a default value
    or a type, for use when no sensor data are available, but filler data are
    required (e.g. when concatenating sensors from different datasets and one
    dataset lacks the sensor). The dummy dataset contains a single data point
    with the filler value and a configurable timestamp (defaulting to way back).

    Parameters
    ----------
    name : string
        Sensor name
    value : object, optional
        Filler value (default is None, meaning *dtype* will be used instead)
    dtype : :class:`numpy.dtype` object or equivalent, optional
        Desired sensor data type, used if no explicit value is given
    timestamp : float, optional
        Time when dummy value occurred (default is way back)

    Returns
    -------
    data : :class:`SensorData` object
        Dummy sensor data object

    """
    if value is None:
        if np.issubdtype(dtype, np.float):
            value = np.nan
        elif np.issubdtype(dtype, np.int):
            value = -1
        elif np.issubdtype(dtype, np.str):
            # Order is important here, because np.str is a subtype of np.bool, but not the other way around...
            value = ''
        elif np.issubdtype(dtype, np.bool):
            value = False
    else:
        dtype = np.array(value).dtype
    data = np.rec.fromarrays([[timestamp], [value], ['nominal']],
                             dtype=[('timestamp', np.float64), ('value', dtype), ('status', '|S7')])
    return SensorData(data, name)


def remove_duplicates(sensor):
    """Remove duplicate timestamp values from sensor data.

    This sorts the 'timestamp' field of the sensor record array and removes any
    duplicate values, updating the corresponding 'value' and 'status' fields as
    well. If more than one timestamp have the same value, the value and status
    of the last of these timestamps are selected. If the values differ for the
    same timestamp, a warning is logged (and the last one is still picked).

    Parameters
    ----------
    sensor : :class:`SensorData` object, length N
        Sensor dataset, which acts like a record array with fields 'timestamp',
        'value' and 'status'

    Returns
    -------
    unique_sensor : :class:`SensorData` object, length M
        Sensor data with duplicate timestamps removed (M <= N)

    """
    x = np.atleast_1d(sensor['timestamp'])
    y = np.atleast_1d(sensor['value'])
    z = np.atleast_1d(sensor['status'])
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
    if not np.all(y[replacement] == y) or not np.all(z[replacement] == z):
        logger.debug("Sensor '%s' has duplicate timestamps with different values or statuses" % (sensor.name,))
        for ind in (y[replacement] != y).nonzero()[0]:
            logger.debug("At %s, sensor '%s' has values of %s and %s - keeping last one" %
                         (katpoint.Timestamp(x[ind]).local(), sensor.name, y[ind], y[replacement][ind]))
        for ind in (z[replacement] != z).nonzero()[0]:
            logger.debug("At %s, sensor '%s' has statuses of '%s' and '%s' - keeping last one" %
                         (katpoint.Timestamp(x[ind]).local(), sensor.name, z[ind], z[replacement][ind]))
    return SensorData(np.rec.fromarrays([x[unique_ind], y[unique_ind], z[unique_ind]],
                                        dtype=[('timestamp', x.dtype), ('value', y.dtype), ('status', z.dtype)]),
                      sensor.name)

# -------------------------------------------------------------------------------------------------
# -- CLASS :  SensorCache
# -------------------------------------------------------------------------------------------------


class SensorCache(dict):
    """Container for sensor data providing name lookup, interpolation and caching.

    *Sensor data* is defined as a one-dimensional time series of values. The
    values may be numerical or non-numerical (*categorical*), and the timestamps
    are monotonically increasing but not necessarily regularly spaced.

    A *sensor cache* stores sensor data with dictionary-like lookup based on the
    sensor name. Since the extraction of sensor data from e.g. HDF5 files may be
    costly, the data are first represented in uncached (raw) form as
    :class:`SensorData` objects, which typically wrap the underlying sensor
    HDF5 datasets. After extraction, the sensor data are stored either as a NumPy
    array (for numerical data) or as a :class:`CategoricalData` object (for
    non-numerical data).

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
        for alias, original in aliases.iteritems():
            for name, data in self.iteritems():
                if name.endswith(original):
                    self[name.replace(original, alias)] = data

    def __str__(self):
        """Verbose human-friendly string representation of sensor cache object."""
        names = sorted([key for key in self.iterkeys()])
        maxlen = max([len(name) for name in names])
        objects = [self.get(name, extract=False) for name in names]
        obj_reprs = [(("<numpy.ndarray shape=%s type='%s' at 0x%x>" % (obj.shape, obj.dtype, id(obj)))
                     if isinstance(obj, np.ndarray) else repr(obj)) for obj in objects]
        actual = ['%s : %s' % (str(name).ljust(maxlen), obj_repr) for name, obj_repr in zip(names, obj_reprs)]
        virtual = ['%s : <function %s.%s>' % (str(pat).ljust(maxlen), func.__module__, func.__name__)
                   for pat, func in self.virtual.iteritems()]
        return '\n'.join(['Actual sensors', '--------------'] + actual +
                         ['\nVirtual sensors', '---------------'] + virtual)

    def __repr__(self):
        """Short human-friendly string representation of sensor cache object."""
        sensor_data = [self.get(name, extract=False) for name in self.iterkeys()]
        return "<katdal.%s sensors=%d cached=%d at 0x%x>" % \
               (self.__class__.__name__, len(sensor_data),
                np.sum([not isinstance(data, SensorData) for data in sensor_data]), id(self))

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

    def itervalues(self):
        """Custom value iterator that avoids extracting sensor data."""
        return iter([self.get(key, extract=False) for key in self.iterkeys()])

    def iteritems(self):
        """Custom item iterator that avoids extracting sensor data."""
        return iter([(key, self.get(key, extract=False)) for key in self.iterkeys()])

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
            True if preset time selection will be applied to returned data
        extract : {True, False}, optional
            True if sensor data should be extracted from HDF5 file and cached
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
        KeyError
            If sensor name was not found in cache and did not match virtual template

        """
        try:
            # First try to load the actual sensor data from cache (remember to call base class here!)
            sensor_data = super(SensorCache, self).__getitem__(name)
        except KeyError:
            # Otherwise, iterate through virtual sensor templates and look for a match
            for pattern, create_sensor in self.virtual.iteritems():
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
            for key, val in self.props.iteritems():
                if key[0] == '*' and name.endswith(key[1:]):
                    props.update(val)
            # Any properties passed directly to this method takes precedence
            props.update(kwargs)
            # Clean up sensor data if non-empty
            if len(sensor_data) > 0:
                # Sort sensor events in chronological order and discard duplicates and unreadable sensor values
                sensor_data = remove_duplicates(sensor_data)
                # Explicitly cast status to string type, as k7_augment produced sensors with integer statuses
                sensor_data.data = np.atleast_1d(sensor_data[sensor_data['status'].astype('|S7') != 'failure'])
            if len(sensor_data) == 0:
                sensor_data = dummy_sensor_data(name, value=props.get('initial_value'), dtype=sensor_data.dtype)
                logger.warning("No usable data found for sensor '%s' - replaced with dummy data (%r)" %
                               (name, sensor_data['value'][0]))
            # If this is the first time any sensor is accessed, obtain all data timestamps via indexer
            self.timestamps = self.timestamps[:] if not isinstance(self.timestamps, np.ndarray) else self.timestamps
            # Determine if sensor produces categorical or numerical data (by default, float data are non-categorical)
            categ = props.get('categorical', not np.issubdtype(sensor_data.dtype, np.float))
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


# -------------------------------------------------------------------------------------------------
# -- FUNCTION :  _sensor_completer
# -------------------------------------------------------------------------------------------------

dict_lookup_match = re.compile(r"""(?:.*\=)?(.*)\[(?P<quote>['|"])(?!.*(?P=quote))(.*)$""")


def _sensor_completer(context, event):
    """Custom IPython completer for sensor name lookups in sensor cache.

    This is inspired by Darren Dale's custom dict-like completer for h5py.

    """
    # Parse command line as (ignored = )base['start_of_name
    base, start_of_name = dict_lookup_match.split(event.line)[1:4:2]

    # Avoid calling any functions during eval()...
    if '(' in base:
        raise TryNext

    # Obtain sensor cache object from user namespace
    try:
        cache = eval(base, context.user_ns)
    except (NameError, AttributeError):
        try:
            # IPython version < 1.0
            cache = eval(base, context.shell.user_ns)
        except (NameError, AttributeError):
            raise TryNext

    # Only continue if this object is actually a sensor cache
    if not isinstance(cache, SensorCache):
        raise TryNext

    if readline:
        # Remove space and plus from delimiter list, so completion works past spaces and pluses in names
        readline.set_completer_delims(readline.get_completer_delims().replace(' ', '').replace('+', ''))

    return [name for name in cache.iterkeys() if name[:len(start_of_name)] == start_of_name]
