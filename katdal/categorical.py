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

"""Container for categorical (i.e. non-numerical) sensor data and related tools."""

import collections

import numpy as np
from dask.base import tokenize


class ComparableArrayWrapper:
    """Wrapper that improves comparison of array objects.

    This wrapper class has two main benefits:

      - It prevents sensor values that are NumPy ndarrays themselves
        (or array-like objects such as tuples and lists) from dissolving
        and losing their identity when they are assembled into an array.

      - It ensures that array-valued sensor values become properly comparable
        (avoiding array-valued booleans resulting from standard comparisons).

    The former is needed because :class:`SensorGetter` is treated as a structured
    array even if it contains object values. The latter is needed because the
    equality operator crops up in hard-to-reach places like inside list.index().

    Parameters
    ----------
    value : object
        The sensor value to be wrapped

    """

    def __init__(self, value):
        self.unwrapped = value

    def __repr__(self):
        """Short human-friendly string representation of wrapper object."""
        class_name = self.__class__.__name__
        return f"<katdal.{class_name} {{ {self.unwrapped!r} }} at {id(self):#x}>"

    def __str__(self):
        """Longer human-friendly string representation of wrapped object."""
        return str(self.unwrapped)

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, ComparableArrayWrapper):
            other = other.unwrapped
        if isinstance(self.unwrapped, np.ndarray) or isinstance(other, np.ndarray):
            return np.array_equal(self.unwrapped, other)
        else:
            return self.unwrapped == other

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self == other

    def __lt__(self, other):
        """Less-than comparison operator."""
        if isinstance(other, ComparableArrayWrapper):
            other = other.unwrapped
        return self.unwrapped < other

    def __gt__(self, other):
        """Greather-than comparison operator."""
        if isinstance(other, ComparableArrayWrapper):
            other = other.unwrapped
        return self.unwrapped > other

    def __le__(self, other):
        """Less-than-or-equal comparison operator."""
        if isinstance(other, ComparableArrayWrapper):
            other = other.unwrapped
        return self.unwrapped <= other

    def __ge__(self, other):
        """Greater-than-or-equal comparison operator."""
        if isinstance(other, ComparableArrayWrapper):
            other = other.unwrapped
        return self.unwrapped >= other

    def __hash__(self):
        """If the underlying object is hashable, the wrapper is too."""
        return hash(self.unwrapped)

    @staticmethod
    def unwrap(v):
        """Unwrap value if needed."""
        return v.unwrapped if isinstance(v, ComparableArrayWrapper) else v


def infer_dtype(values):
    """Figure out dtype of sequence of sensor values.

    The common dtype is determined by explicit NumPy promotion. If the values
    are array-like themselves, treat them as opaque objects to simplify
    sensor processing. If the sequence is empty, the dtype is unknown and
    set to None. In addition, short-circuit to an actual dtype for objects
    with this attribute to simplify calling this on a mixed collection of
    sensor data.

    Parameters
    ----------
    values : sequence, or object with dtype
        Sequence of sensor values (typically a list), or a sensor data object
        with a dtype attribute (like ndarray or :class:`SensorGetter`)

    Returns
    -------
    dtype : :class:`numpy.dtype` object or None
        Inferred dtype, or None if `values` is an empty sequence

    Notes
    -----
    This is almost, but not quite, entirely like :func:`numpy.result_type`.
    The differences are that this accepts generic objects in the sequence,
    treats ndarrays as objects regardless of their underlying dtype, supports
    a dtype of None and short-circuits the check if the sequence itself is an
    object with a dtype. And this accepts the sequence as the first parameter
    as opposed to being unpacked across the argument list.

    """
    # If values already has a dtype (because it is an ndarray, SensorGetter,
    # CategoricalData, etc), return that instead
    if hasattr(values, 'dtype'):
        return values.dtype
    if not values:
        return None
    # Put all values into array to perform explicit NumPy type promotion
    test_data = np.array(values)
    # Beware array-valued sensors; treat their values as opaque objects
    # This forces sensor values to be 1-D at all times (an invariant)
    return test_data.dtype if test_data.ndim == 1 else np.dtype(object)


def unique_in_order(elements, return_inverse=False):
    """Extract unique elements from `elements` while preserving original order.

    Parameters
    ----------
    elements : sequence
        Sequence of equality-comparable objects
    return_inverse : {False, True}, optional
        If True, also return sequence of indices that can be used to reconstruct
        original `elements` sequence via `[unique_elements[i] for i in inverse]`

    Returns
    -------
    unique_elements : list
        List of unique objects, in original order they were found in `elements`
    inverse : array of int, optional
        If `return_inverse` is True, sequence of indices that can be used to
        reconstruct original sequence

    """
    # In Python 3, each iteration over a np.ndarray creates new objects. This
    # can lead to problems if there are NaNs, because NaN != NaN, so we rely
    # on the behaviour of dict that first checks object identity. We thus
    # force to list at the start to get consistent object identities.
    elements = list(elements)
    unique_elements, inverse = [], []
    try:
        # Surprisingly, a zero generator like itertools.repeat does not buy you anything
        lookup = collections.OrderedDict(zip(elements, len(elements) * [0]))
    except TypeError:
        # Fall back to slower lookup using dask's tokenizer
        lookup = {}
        for element in elements:
            token = tokenize(ComparableArrayWrapper.unwrap(element))
            try:
                index = lookup[token]
            except KeyError:
                index = len(unique_elements)
                lookup[token] = index
                unique_elements.append(element)
            if return_inverse:
                inverse.append(index)
    else:
        for index, element in enumerate(lookup):
            lookup[element] = index
        unique_elements = list(lookup.keys())
        if return_inverse:
            inverse = [lookup[element] for element in elements]
    # Force inverse to int dtype in case it is an empty array (float otherwise)
    return (unique_elements, np.array(inverse, dtype=np.int)) \
        if return_inverse else unique_elements


# -------------------------------------------------------------------------------------------------
# -- CLASS :  CategoricalData
# -------------------------------------------------------------------------------------------------


class CategoricalData:
    """Container for categorical (i.e. non-numerical) sensor data.

    This container allows simple manipulation and interpolation of a time series
    of non-numerical data represented as discrete events. The data is stored as
    a list of sensor values and two integer arrays:

    * `unique_values` stores one copy of each unique object in the data series
    * `events` stores the time indices (dumps) where each event occurs
    * `indices` stores indices linking each event to the `unique_values` list

    The __getitem__ interface (i.e. `data[dump]`) returns the data associated
    with the last event before the requested dump(s), in effect doing a
    zeroth-order interpolation of the data at each event. Events can be added
    and removed and realigned, and the container can be split along the time
    axis, amongst other functionality.

    Parameters
    ----------
    sensor_values : sequence, length *N*
        Sequence of sensor values (of any type, preferably not None [see Notes])
    events : sequence of non-negative ints, length *N* + 1
        Corresponding monotonic sequence of dump indices where each sensor value
        came into effect. The last event is one past the last dump where the
        final sensor value applied, and therefore equal to the total number of
        dumps for which sensor values were specified.

    Attributes
    ----------
    unique_values : list, length *M*
        List of unique sensor values in order they were found in `sensor_values`
        with any :class:`ComparableArrayWrapper` objects unwrapped
    indices : array of int, shape (*N*,)
        Array of indices into `unique_values`, one per sensor event
    dtype : :class:`numpy.dtype` object
        Sensor data type as NumPy dtype (found on demand from `unique_values`)

    Notes
    -----
    Any object values wrapped in a :class:`ComparableArrayWrapper` will be
    unwrapped before adding it to `unique_values`. When adding, removing and
    comparing values to this container, any object values will be wrapped again
    temporarily to ensure proper comparisons.

    It is discouraged to have a sensor value of None as this value is given
    a special meaning in methods such as :meth:`CategoricalData.add` and
    :meth:`sensor_to_categorical`. On the other hand, it is the most sensible
    dummy object value and any Nones entering through this initialiser will
    probably not cause any issues.

    It is better to make `unique_values` a list instead of an array because an
    array assimilates objects such as tuples, lists and other arrays. The
    alternative is an array of :class:`ComparableArrayWrapper` objects but
    these then need to be unpacked at some later stage which is also tricky.

    """

    def __init__(self, sensor_values, events):
        values, self.indices = unique_in_order(sensor_values, return_inverse=True)
        self.unique_values = [ComparableArrayWrapper.unwrap(v) for v in values]
        self.events = np.asarray(events)

    @property
    def _comparable_values(self):
        """Comparable version of unique values, wrapping any objects."""
        return [ComparableArrayWrapper(value) for value in self.unique_values]

    def _lookup(self, dumps):
        """Look up relevant indices occurring at specified `dumps`.

        Parameters
        ----------
        dumps : int or sequence of int
            Specified dumps

        Returns
        -------
        indices : int or array of int
            Corresponding sensor value indices at specified dumps

        """
        preceding_events = self.events.searchsorted(dumps, side='right') - 1
        if np.any(preceding_events < 0) or np.any(preceding_events >= len(self.indices)):
            raise IndexError('Some dumps in (%s) are outside event range: %d <= dumps < %d' %
                             (dumps, self.events[0], self.events[-1]))
        return self.indices[preceding_events]

    def __getitem__(self, key):
        """Look up sensor value at selected dumps.

        Be aware that the dtype of the returned array may differ from that of
        the :class:`CategoricalData` object if the sensor values are
        multi-dimensional arrays themselves, in effect falling back to the
        underlying dtype. For large multi-dimensional sensor values this
        method may also cause memory issues as it will duplicate these arrays
        into the final output array.

        Parameters
        ----------
        key : int or slice or sequence of int or sequence of bool
            Index or slice specification selecting certain dumps

        Returns
        -------
        val : object or array of objects
            Sensor values at selected dumps, either single value or array of them

        """
        if isinstance(key, slice):
            # Convert slice notation to the corresponding sequence of dump indices
            key = list(range(*key.indices(self.events[-1])))
        # Convert sequence of bools (one per dump) to sequence of indices where key is True
        elif np.asarray(key).dtype == np.bool and len(np.asarray(key)) == self.events[-1]:
            key = np.nonzero(key)[0]
        indices = self._lookup(key)
        # Interpret indices as either a sequence of ints or a single int
        try:
            values = [self.unique_values[index] for index in indices]
        except TypeError:
            return self.unique_values[indices]
        # Handle empty selections specially to ensure proper dtype and shape
        if not values:
            all_possible_values = np.array(self.unique_values)
            dtype = all_possible_values.dtype
            shape = all_possible_values.shape
            return np.empty((0,) + shape[1:], dtype)
        return np.array(values)

    def __repr__(self):
        """Short human-friendly string representation of categorical data object."""
        return "<katdal.CategoricalData events={} values={} type={} at {:#x}>".format(
               len(self.indices), len(self.unique_values), self.dtype, id(self))

    def __str__(self):
        """Long human-friendly string representation of categorical data object."""
        index_width = len(str(self.events[-1] - 1))
        return '\n'.join([('%*d - %*d: %s' % (index_width, segm.start, index_width, segm.stop - 1, val))
                          for segm, val in self.segments()])

    def __len__(self):
        """Length operator indicates number of events produced by sensor."""
        return len(self.indices)

    def _bool_per_dump(self, bool_per_value):
        """Turn list of bools per unique value into an array of bools per dump."""
        bool_per_event = np.atleast_1d(np.array(bool_per_value)[self.indices])
        bool_per_dump = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            bool_per_dump[start:end] = bool_per_event[n]
        return bool_per_dump

    def __eq__(self, other):
        """Equality comparison operator."""
        return self._bool_per_dump([value == other for value in self._comparable_values])

    def __ne__(self, other):
        """Inequality comparison operator."""
        return self._bool_per_dump([value != other for value in self._comparable_values])

    def __lt__(self, other):
        """Less-than comparison operator."""
        return self._bool_per_dump([value < other for value in self._comparable_values])

    def __gt__(self, other):
        """Greather-than comparison operator."""
        return self._bool_per_dump([value > other for value in self._comparable_values])

    def __le__(self, other):
        """Less-than-or-equal comparison operator."""
        return self._bool_per_dump([value <= other for value in self._comparable_values])

    def __ge__(self, other):
        """Greater-than-or-equal comparison operator."""
        return self._bool_per_dump([value >= other for value in self._comparable_values])

    @property
    def dtype(self):
        """Sensor value type."""
        return infer_dtype(self.unique_values)

    def segments(self):
        """Generator that iterates through events and returns segment and value.

        Yields
        ------
        segment : slice object
            The slice representing range of dump indices of the current segment
        value : object
            Sensor value associated with segment

        """
        for start, end, ind in zip(self.events[:-1], self.events[1:], self.indices):
            yield slice(start, end), self.unique_values[ind]

    def add(self, event, value=None):
        """Add or override sensor event.

        This adds a new event to the container, with a new value or a duplicate
        of the existing value at that dump. If the new event coincides with an
        existing one, it overrides the value at that dump.

        Parameters
        ----------
        event : int
            Dump of event to add or override
        value : object, optional
            New value for event (duplicate current value at this dump by default)

        """
        # If value has not been seen before, add it to unique_values (and create new index for it)
        if value is not None:
            try:
                value_index = self._comparable_values.index(value)
            except ValueError:
                value_index = len(self.unique_values)
                self.unique_values += [value]
        else:
            value_index = self._lookup(event)
        # If new event coincides with existing event, simply change value of that event, else insert new event
        event_index = self.events.searchsorted(event)
        before, after = event_index, (event_index + 1 if self.events[event_index] == event else event_index)
        self.indices = np.r_[self.indices[:before], [value_index], self.indices[after:]]
        self.events = np.r_[self.events[:before], [event], self.events[after:]]

    def remove(self, value):
        """Remove sensor value, remapping indices and merging segments in process.

        If the sensor value does not exist, do nothing.

        Parameters
        ----------
        value : object
            Sensor value to remove from container

        """
        try:
            index = self._comparable_values.index(value)
        except ValueError:
            pass
        else:
            keep = (self.indices != index)
            remap = np.arange(len(self.unique_values))
            remap[index:] -= 1
            self.indices = remap[self.indices[keep]]
            self.events = np.r_[self.events[:-1][keep], self.events[-1]]
            del self.unique_values[index]

    def add_unmatched(self, segments, match_dist=1):
        """Add duplicate events for segment starts that don't match sensor events.

        Given a sequence of segments, this matches each segment start to the
        nearest sensor event dump (within `match_dist`). Any unmatched segment
        starts are added as duplicate sensor events (or ignored if they fall
        outside the sensor event range).

        Parameters
        ----------
        segments : sequence of int
            Monotonically increasing sequence of segment starts, including an
            extra element at the end that is one past the end of the last segment
        match_dist : int, optional
            Maximum distance in dumps that signify a match between events

        """
        # Identify unmatched segment starts
        segments = np.asarray(segments)
        unmatched = segments[np.abs(self.events[np.newaxis, :] - segments[:, np.newaxis]).min(axis=1) > match_dist]
        # Add these dumps as duplicate events, ignoring those that are out of bounds
        for segm in unmatched:
            try:
                self.add(segm)
            except IndexError:
                pass

    def align(self, segments):
        """Align sensor events with segment starts, possibly discarding events.

        Given a sequence of segments, this moves each sensor event dump onto the
        nearest segment start. If more than one event ends up in the same segment,
        only keep the last event, discarding the rest.

        The end result is that the sensor event dumps become a subset of the
        segment starts and there cannot be more sensor events than segments.

        Parameters
        ----------
        segments : sequence of int
            Monotonically increasing sequence of segment starts, including an
            extra element at the end that is one past the end of the last segment

        """
        # For each event, pick the segment with the closest start to it and then shift event onto segment start
        segments_with_event = np.abs(self.events[np.newaxis, :] - segments[:, np.newaxis]).argmin(axis=0)
        events = segments[segments_with_event]
        # When multiple sensor events are associated with the same segment, only keep the final one
        final = np.nonzero(np.diff(events) > 0)[0]
        subset, self.indices = np.unique(self.indices[final], return_inverse=True)
        self.unique_values = [self.unique_values[index] for index in subset]
        self.events = np.r_[events[final], events[-1]]

    def partition(self, segments):
        """Partition dataset into multiple sets along time axis.

        Given a sequence of segments, split the container into a sequence of
        containers, one per segment. Each container contains only the events
        occurring within its corresponding segment, with event dumps relative to
        the start of the segment, and the containers share the same unique
        values.

        Parameters
        ----------
        segments : sequence of int
            Monotonically increasing sequence of segment starts, including an
            extra element at the end that is one past the end of the last segment

        Returns
        -------
        split_data : sequence of :class:`CategoricalData` objects
            Resulting multiple datasets in chronological order

        """
        # Ignore last element in event list, as it is not a real event but a placeholder for dataset length
        events = self.events[:-1]
        # Find segment starts in event sequence, associating dumps before first event with it, ditto for ones past last
        initial_indices = self.indices[(events.searchsorted(segments[:-1], side='right') - 1).clip(0, len(events) - 1)]
        split_data = []
        for start, end, initial_index in zip(segments[:-1], segments[1:], initial_indices):
            segment_events = (events >= start) & (events < end)
            # Bypass the normal CategoricalData initialiser to ensure that each cat_data has the same unique_values
            cat_data = CategoricalData([], [])
            cat_data.unique_values = self.unique_values
            cat_data.indices = self.indices[segment_events]
            cat_data.events = events[segment_events] - start
            # Insert initial event if it is not there, and pad events with data segment length
            if len(cat_data.events) == 0 or cat_data.events[0] != 0:
                cat_data.indices = np.r_[initial_index, cat_data.indices]
                cat_data.events = np.r_[0, cat_data.events, end - start]
            else:
                cat_data.events = np.r_[cat_data.events, end - start]
            split_data.append(cat_data)
        return split_data

    def remove_repeats(self):
        """Remove repeated events of the same value."""
        changes = np.nonzero([1] + np.diff(self.indices).tolist())[0]
        self.indices = self.indices[changes]
        self.events = np.r_[self.events[changes], self.events[-1]]

# -------------------------------------------------------------------------------------------------
# -- Utility functions
# -------------------------------------------------------------------------------------------------


def concatenate_categorical(split_data, **kwargs):
    """Concatenate multiple categorical datasets into one along time axis.

    Join a sequence of categorical datasets together, by forming a common set of
    unique values, remapping events to these and incrementing the event dumps of
    each dataset to start off where the previous dataset ended.

    Parameters
    ----------
    split_data : sequence of :class:`CategoricalData` objects
        Sequence of containers to concatenate

    Returns
    -------
    data : :class:`CategoricalData` object
        Concatenated dataset

    """
    if len(split_data) == 1:
        return split_data[0]
    # Synthesise segment starts from the time length of each dataset
    segments = np.cumsum([0] + [cat_data.events[-1] for cat_data in split_data])
    data = CategoricalData([], [])
    # Combine all unique values in the order they are found in datasets
    split_values = [cat_data._comparable_values for cat_data in split_data]
    inverse_splits = np.cumsum([0] + [len(vals) for vals in split_values])
    values, inverse = unique_in_order(sum(split_values, []), return_inverse=True)
    data.unique_values = [ComparableArrayWrapper.unwrap(v) for v in values]
    indices, events = [], []
    for n, cat_data in enumerate(split_data):
        # Remap indices to new unique_values array
        lookup = np.array(inverse[inverse_splits[n]:inverse_splits[n + 1]])
        indices.append(lookup[cat_data.indices])
        # Offset events by the start of each segment
        events.append(cat_data.events[:-1] + segments[n])
    # Add overall time length as the final event
    events.append([segments[-1]])
    data.indices = np.concatenate(indices)
    data.events = np.concatenate(events)
    if not kwargs.get('allow_repeats', False):
        data.remove_repeats()
    return data


def _single_event_per_dump(events, greedy):
    """Ensure that each dump is associated with a single sensor event.

    This generates a sequence of cleaned-up sensor events (represented by
    indices into the original `events` sequence), which ensures that each dump
    is associated with a single event. When there are multiple events inside
    a dump, pick the final one. In addition, some sensor values designated as
    "greedy" will override non-greedy ones and grab a dump even if it is not
    the final value. In this scenario, move the final (non-greedy) event to the
    next dump by modifying its dump index in the `events` parameter. The
    generator returns up to *N* events but not the special terminal event.

    Parameters
    ----------
    events : mutable sequence of non-negative ints, length *N* + 1
        Monotonic sequence of dump indices associated with each sensor event.
        The last event is one past the last dump (i.e. the total number of
        dumps). Be aware that this parameter is mutated by the function.
    greedy : sequence of bool, length *N*
        Flags indicating whether the sensor value at a given event is "greedy"

    Yields
    ------
    event_index : non-negative int
        Index into `events` sequence of the next cleaned up event (does not
        yield the one-past-last-dump terminal event)

    """
    # The previous winning event is the dominant event in the previous dump
    previous_winning_event = 0
    previous_dump = 0
    # This generates consecutive event indices with associated dump indices
    for current_event, current_dump in enumerate(events):
        # At the start of a new dump, process the events of the previous dump
        if current_dump > previous_dump:
            # This previous event segment is assumed to straddle dump boundary
            assert current_event >= 1, "First sensor event not at dump 0"
            event_at_dump_start = current_event - 1
            # The normal victory condition is to be the final event in the dump
            if not greedy[previous_winning_event]:
                previous_winning_event = event_at_dump_start
            winning_dump = events[previous_winning_event]
            # Only yield winning event in immediate past to avoid duplicates
            if previous_dump <= winning_dump < current_dump:
                yield previous_winning_event
            # If winning event was greedy and final event was non-greedy,
            # push final event to the start of next dump and yield if it is
            # the only event in that dump (otherwise it has to fight it out...)
            if event_at_dump_start != previous_winning_event:
                # NB: This modifies `events`! It simplifies bookkeeping.
                events[event_at_dump_start] += 1
                if current_dump > events[event_at_dump_start]:
                    yield event_at_dump_start
                previous_winning_event = event_at_dump_start
            previous_dump = current_dump
        # While within the same dump, pick the latest greedy event as winner
        # Also, avoid indexing greedy with final one-past-last event
        if (current_event < len(greedy)) and greedy[current_event]:
            previous_winning_event = current_event


def sensor_to_categorical(sensor_timestamps, sensor_values, dump_midtimes,
                          dump_period, transform=None, initial_value=None,
                          greedy_values=None, allow_repeats=False, **kwargs):
    """Align categorical sensor events with dumps and clean up spurious events.

    This converts timestamped sensor data into a categorical dataset by
    comparing the sensor timestamps to a series of dump timestamps and assigning
    each sensor event to the dump in which it occurred. When multiple sensor
    events happen in the same dump, only the last one is kept. The first dump is
    guaranteed to have a valid value by either using the supplied `initial_value`
    or extrapolating the first proper value back in time. The sensor data may
    be transformed before events that repeat values are potentially discarded.
    Finally, events with values marked as "greedy" take precedence over normal
    events when both occur within the same dump (either changing from or to the
    greedy value, or if the greedy value occurs completely within a dump).

    XXX Future improvements include picking the event with the longest duration
    within a dump as opposed to the final event, and "snapping" event boundaries
    to dump boundaries with a given tolerance (e.g. 5-10% of dump period).

    Parameters
    ----------
    sensor_timestamps : sequence of float, length *M*
        Sequence of sensor timestamps (typically UTC seconds since Unix epoch)
    sensor_values : sequence, length *M*
        Corresponding sequence of sensor values [potentially wrapped]
    dump_midtimes : sequence of float, length *N*
        Sequence of dump midtimes (same reference as sensor timestamps)
    dump_period : float
        Duration of each dump, in seconds
    transform : callable or None, optional
        Transform [unwrapped] sensor values before fixing initial value,
        mapping dumps to events and discarding repeats
    initial_value : object or None, optional
        Sensor value [transformed, unwrapped] to use for dump = 0 up to first
        proper event (force first proper event to start at dump = 0 by default)
    greedy_values : sequence or None, optional
        List of [transformed, unwrapped] sensor values considered "greedy"
    allow_repeats : {False, True}, optional
        If False, discard sensor events that do not change [transformed] value

    Returns
    -------
    data : :class:`CategoricalData` object
        Constructed categorical dataset [unwraps any wrapped values]

    """
    sensor_timestamps = np.atleast_1d(sensor_timestamps)
    sensor_values = np.atleast_1d(sensor_values)
    dump_endtimes = dump_midtimes + 0.5 * dump_period
    num_dumps = len(dump_endtimes)
    # Insert an extra prior dump to collect sensor values before first dump
    dump_endtimes = np.r_[dump_endtimes[0] - dump_period, dump_endtimes]
    # Check if sensor values are objects wrapped in ComparableArrayWrappers
    wrapped_values = len(sensor_values) and isinstance(sensor_values[0],
                                                       ComparableArrayWrapper)
    # Convert sensor event times to dump indices by picking the dump during
    # which each sensor event occurred (more precisely: closest dump centroid)
    events = dump_endtimes.searchsorted(sensor_timestamps) - 1
    # Get rid of excess events before the first dump and after the last dump
    # The dump index of prior events is -1 and of later events is `num_dumps`
    first_proper_event = events.searchsorted(-1, side='right')
    # Shift the final prior event (if any) to the start of the first dump
    if first_proper_event > 0:
        first_proper_event -= 1
        events[first_proper_event] = 0
    one_past_last_event = events.searchsorted(num_dumps)
    within_dumps = slice(first_proper_event, one_past_last_event)
    sensor_values = sensor_values[within_dumps]
    events = events[within_dumps]
    # Apply optional transform to sensor values
    if transform is not None:
        if wrapped_values:
            orig_transform = transform
            def transform(value):   # noqa: E306
                """Unwrap wrapped value, transform and rewrap."""
                return ComparableArrayWrapper(orig_transform(value.unwrapped))
        sensor_values = np.array([transform(y) for y in sensor_values])
    # Force first dump to have valid sensor value
    # (insert initial value or let the first proper value apply from the start)
    if events[0] != 0 and initial_value is not None:
        if wrapped_values:
            initial_value = ComparableArrayWrapper(initial_value)
        sensor_values = np.r_[[initial_value], sensor_values]
        events = np.r_[0, events]
    events[0] = 0
    # Clean up dump->event mapping, taking into account greedy values
    greedy_values = () if greedy_values is None else greedy_values
    greedy = [value in greedy_values for value in sensor_values]
    # Add one-past-last-dump terminator (will be removed again by `cleaned_up`)
    events = np.r_[events, num_dumps]
    # NB: `events` is mutated by `_single_event_per_dump`
    cleaned_up = list(_single_event_per_dump(events, greedy))
    sensor_values = sensor_values[cleaned_up]
    events = events[cleaned_up]
    # Discard sensor events that do not change the (transformed) sensor value
    # (i.e. that repeat the previous value)
    if not allow_repeats:
        changes_value = [n for n in range(len(sensor_values)) if n == 0 or
                         sensor_values[n] != sensor_values[n - 1]]
        sensor_values = sensor_values[changes_value]
        events = events[changes_value]
    # Last event is fixed at one-past-last-dump to indicate end of last segment
    return CategoricalData(sensor_values, np.r_[events, num_dumps])
