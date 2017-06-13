"""Tests for :py:mod:`katdal.categorical`."""

from nose.tools import assert_equal, assert_true
import numpy as np

from katdal.categorical import _single_event_per_dump


def test_dump_to_event_parsing():
    values = np.array([c for c in 'ABCDEFGH'])
    events = np.array([0, 0, 1, 3, 3, 4, 4, 6, 8])
    greedy = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    cleaned = [i for i in _single_event_per_dump(events, greedy)]
    new_values = values[cleaned]
    new_events = np.r_[events[cleaned], events[-1]]
    print cleaned, new_values, new_events
    assert_equal(cleaned, [0, 2, 4, 6, 7], 'Dump -> event parser failed')
    assert_equal(''.join(new_values), 'ACEGH', 'Dump -> event parser failed')
    events_equal = np.array_equal(new_events, [0, 1, 3, 5, 6, 8])
    assert_true(events_equal, 'Dump -> event parser failed')
