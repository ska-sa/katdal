"""Container for categorical (i.e. non-numerical) sensor data and related tools."""

import numpy as np

def unique_in_order(elements, return_inverse=False):
    """Extract unique elements from *elements* while preserving original order.

    Parameters
    ----------
    elements : sequence
        Sequence of sortable objects (i.e. objects supporting the < operator)
    return_inverse : {False, True}, optional
        If True, also return sequence of indices that can be used to reconstruct
        original `elements` sequence via `unique_elements[inverse]`

    Returns
    -------
    unique_elements : array
        Array of unique objects, in original order they were found in `elements`
    inverse : array of int, optional
        If `return_inverse` is True, sequence of indices that can be used to
        reconstruct original sequence

    """
    elements = np.atleast_1d(elements)
    # Ensure that objects support the appropriate comparison operators for sorting (i.e. <)
    if (elements.dtype == np.object) and not hasattr(elements[0], '__lt__') and not hasattr(elements[0], '__cmp__'):
        raise TypeError("Cannot identify unique objects of %s as it has no __lt__ or __cmp__ methods" %
                        (elements[0].__class__,))
    # Find unique elements based on sorting
    unique_elements, indices = np.unique(elements, return_inverse=True)
    # Shuffle unique values back into their original order as they were found in elements
    indices_list = indices.tolist()
    original_order = np.argsort([indices_list.index(n) for n in range(len(unique_elements))])
    return (unique_elements[original_order], original_order.argsort()[indices]) \
           if return_inverse else unique_elements[original_order]

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CategoricalData
#--------------------------------------------------------------------------------------------------

class CategoricalData(object):
    """Container for categorical (i.e. non-numerical) sensor data.

    This container allows simple manipulation and interpolation of a time series
    of non-numerical data represented as discrete events. The data is stored in
    three arrays:

    * `events` stores the time indices (dumps) where each event occurs
    * `unique_values` stores one copy of each unique object in the data series
    * `indices` stores indices linking each event to the `unique_values` array

    The __getitem__ interface (i.e. `data[dump]`) returns the data associated
    with the last event before the requested dump(s), in effect doing a
    zeroth-order interpolation of the data at each event. Events can be added
    and removed and realigned, and the container can be split along the time
    axis, amongst other functionality.

    Parameters
    ----------
    sensor_values : sequence, length *N*
        Sequence of sensor values (of any type)
    events : sequence of non-negative ints, length *N* + 1
        Corresponding monotonic sequence of dump indices where each sensor value
        came into effect. The last event is one past the last dump where the
        final sensor value applied, and therefore equal to the total number of
        dumps for which sensor values were specified.

    Attributes
    ----------
    unique_values : array, shape (*M*,)
        Array of unique sensor values, in order they were found in `sensor_values`
    indices : array of int, shape (*N*,)
        Array of indices into `unique_values`, one per sensor event

    """
    def __init__(self, sensor_values, events):
        self.unique_values, self.indices = unique_in_order(sensor_values, return_inverse=True)
        self.events = np.asarray(events)

    def _lookup(self, dumps):
        """Look up relevant indices occurring at specified *dumps*.

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

        Parameters
        ----------
        key : int or slice or sequence of int or sequence of bool
            Index or slice specification selecting certain dumps

        Returns
        -------
        val : array
            Sensor values at selected dumps

        """
        if isinstance(key, slice):
            # Convert slice notation to the corresponding sequence of dump indices
            key = range(*key.indices(self.events[-1]))
        # Convert sequence of bools (one per dump) to sequence of indices where key is True
        elif np.asarray(key).dtype == np.bool and len(np.asarray(key)) == self.events[-1]:
            key = np.nonzero(key)[0]
        return self.unique_values[self._lookup(key)]

    def __repr__(self):
        """Short human-friendly string representation of categorical data object."""
        return "<katfile.CategoricalData events=%d values=%d type='%s' at 0x%x>" % \
               (len(self.indices), len(self.unique_values), self.unique_values.dtype, id(self))

    def __str__(self):
        """Long human-friendly string representation of categorical data object."""
        index_width = len(str(self.events[-1] - 1))
        return '\n'.join([('%*d - %*d: %s' % (index_width, segm.start, index_width, segm.stop - 1, val))
                          for segm, val in self.segments()])

    def __len__(self):
        """Length operator indicates number of events produced by sensor."""
        return len(self.indices)

    def __eq__(self, other):
        """Equality comparison operator."""
        index_truths = np.atleast_1d((self.unique_values == other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    def __ne__(self, other):
        """Inequality comparison operator."""
        index_truths = np.atleast_1d((self.unique_values != other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    def __lt__(self, other):
        """Less-than comparison operator."""
        index_truths = np.atleast_1d((self.unique_values < other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    def __gt__(self, other):
        """Greather-than comparison operator."""
        index_truths = np.atleast_1d((self.unique_values > other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    def __le__(self, other):
        """Less-than-or-equal comparison operator."""
        index_truths = np.atleast_1d((self.unique_values <= other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    def __ge__(self, other):
        """Greater-than-or-equal comparison operator."""
        index_truths = np.atleast_1d((self.unique_values >= other)[self.indices])
        result = np.empty(self.events[-1], dtype=np.bool)
        for n, (start, end) in enumerate(zip(self.events[:-1], self.events[1:])):
            result[start:end] = index_truths[n]
        return result

    @property
    def dtype(self):
        """Sensor value type."""
        return self.unique_values.dtype

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

        This adds a new event to the container, with a new value or re-using the
        existing value at that dump. If the new event coincides with an existing
        one, it overrides the value at that dump.

        Parameters
        ----------
        event : int
            Dump of event to add or override
        value : object, optional
            New value for event (re-use the current value at this dump by default)

        """
        # If value has not been seen before, add it to unique_values (and create new index for it)
        value_index = np.nonzero(self.unique_values == value)[0] if value is not None else [self._lookup(event)]
        if len(value_index) == 0:
            value_index = len(self.unique_values)
            self.unique_values = np.r_[self.unique_values, [value]]
        else:
            value_index = value_index[0]
        # If new event coincides with existing event, simply change value of that event, else insert new event
        event_index = self.events.searchsorted(event)
        before, after = event_index, (event_index + 1 if self.events[event_index] == event else event_index)
        self.indices = np.r_[self.indices[:before], [value_index], self.indices[after:]]
        self.events = np.r_[self.events[:before], [event], self.events[after:]]

    def remove(self, value):
        """Remove sensor value, remapping indices and merging segments in process.

        Parameters
        ----------
        value : object
            Sensor value to remove from container

        """
        index = np.nonzero(self.unique_values == value)[0]
        if len(index) > 0:
            index = index[0]
            keep = (self.indices != index)
            remap = np.arange(len(self.unique_values))
            remap[index:] -= 1
            self.indices = remap[self.indices[keep]]
            self.events = np.r_[self.events[keep], self.events[-1]]
            self.unique_values = np.r_[self.unique_values[:index], self.unique_values[index + 1:]]

    def align(self, segments):
        """Align sensor events with segment starts, possibly discarding some values.

        Given a sequence of segments, this moves each sensor event dump onto the
        nearest segment start. If more than one event ends up in the same segment,
        only keep the last event, discarding the rest.

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
        final = (np.diff(events) > 0)
        subset, self.indices = np.unique(self.indices[final], return_inverse=True)
        self.unique_values = self.unique_values[subset]
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

#--------------------------------------------------------------------------------------------------
#--- Utility functions
#--------------------------------------------------------------------------------------------------

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
    # Synthesise segment starts from the time length of each dataset
    segments = np.cumsum([0] + [cat_data.events[-1] for cat_data in split_data])
    data = CategoricalData([], [])
    # Combine all unique values in the order they are found in datasets
    data.unique_values = unique_in_order(np.concatenate([cat_data.unique_values for cat_data in split_data]))
    indices, events = [], []
    for n, cat_data in enumerate(split_data):
        # Remap indices to new unique_values array
        lookup = np.array([data.unique_values.tolist().index(value) for value in cat_data.unique_values])
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


def sensor_to_categorical(sensor_timestamps, sensor_values, dump_midtimes, dump_period,
                          greedy_values=None, initial_value=None, transform=None, allow_repeats=False, **kwargs):
    """Align categorical sensor events with dumps and clean up spurious events.

    This converts timestamped sensor data into a categorical dataset by
    comparing the sensor timestamps to a series of dump timestamps and assigning
    each sensor event to the dump in which it occurred. When multiple sensor
    events happen in the same dump, only the last one is kept. The first dump is
    guaranteed to have a valid value by either using the supplied `initial_value`
    or extrapolating the first proper value back in time. The sensor data may
    be transformed before events that repeat values are potentially discarded.
    Finally, events with values marked as "greedy" swallow the last dump of their
    preceding events, where the sensor value is changing into the greedy value.
    A greedy value therefore takes precedence over another value when both occur
    within the same dump (either changing from or to the greedy value).

    Parameters
    ----------
    sensor_timestamps : sequence of float, length *M*
        Sequence of sensor timestamps (typically UTC seconds since epoch)
    sensor_values : sequence, length *M*
        Corresponding sequence of sensor values
    dump_midtimes : sequence of float, length *N*
        Sequence of dump midtimes (same reference as sensor timestamps)
    dump_period : float
        Duration of each dump, in seconds
    greedy_values : sequence or None, optional
        List of (transformed) sensor values considered "greedy"
    initial_value : object or None, optional
        Sensor value to use for dump = 0 up to the first proper event
        (the default is to force the first proper event to start at dump = 0)
    transform : callable or None, optional
        Transform sensor values before discarding repeats and applying greed
    allow_repeats : {False, True}, optional
        If False, discard sensor events that do not change (transformed) value

    Returns
    -------
    data : :class:`CategoricalData` object
        Constructed categorical dataset

    """
    sensor_timestamps = np.atleast_1d(sensor_timestamps)
    sensor_values = np.atleast_1d(sensor_values)
    dump_midtimes = np.atleast_1d(dump_midtimes)
    # Convert sensor event times to dump indices (pick the dump during which each sensor event occurred)
    # The last event is fixed at one-past-the-last-dump, to indicate the end of the last segment
    events = np.r_[dump_midtimes.searchsorted(sensor_timestamps - 0.5 * dump_period), len(dump_midtimes)]
    # Cull any empty segments (i.e. when multiple sensor events occur within a single dump, only keep last one)
    # This also gets rid of excess events before the first dump and after the last dump
    non_empty = (np.diff(events) > 0)
    sensor_values, events = sensor_values[non_empty], np.r_[events[non_empty], len(dump_midtimes)]
    # Force first dump to have valid sensor value (use initial value or first proper value advanced to start)
    if events[0] != 0 and initial_value is not None:
        sensor_values, events = np.r_[[initial_value], sensor_values], np.r_[0, events]
    events[0] = 0
    # Apply optional transform to sensor values
    sensor_values = np.array([transform(y) for y in sensor_values]) if transform is not None else sensor_values
    # Discard sensor events that do not change the (transformed) sensor value (i.e. that repeat the previous value)
    if not allow_repeats:
        changes = [n for n in range(len(sensor_values)) if (n == 0) or (sensor_values[n] != sensor_values[n - 1])]
        sensor_values, events = sensor_values[changes], np.r_[events[changes], len(dump_midtimes)]
    # Extend segments with "greedy" values to include first dump of next segment where sensor value is changing
    if greedy_values is not None:
        greedy_to_nongreedy = [n for n in range(1, len(sensor_values))
                               if sensor_values[n - 1] in greedy_values and sensor_values[n] not in greedy_values]
        events[greedy_to_nongreedy] += 1
    return CategoricalData(sensor_values, events)
