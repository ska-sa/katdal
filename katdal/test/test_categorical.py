"""Tests for :py:mod:`katdal.categorical`."""

import numpy as np
from numpy.testing import assert_equal

from katdal.categorical import _single_event_per_dump


def test_dump_to_event_parsing():
    values = np.array(list('ABCDEFGH'))
    events = np.array([0, 0, 1, 3, 3, 4, 4, 6, 8])
    greedy = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    cleaned = list(_single_event_per_dump(events, greedy))
    new_values = values[cleaned]
    new_events = events[cleaned]
    assert_equal(cleaned, [0, 2, 4, 6, 7], 'Dump -> event parser failed')
    assert_equal(new_values, list('ACEGH'), 'Dump -> event parser failed')
    assert_equal(new_events, [0, 1, 3, 5, 6], 'Dump -> event parser failed')
