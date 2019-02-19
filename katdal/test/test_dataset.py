###############################################################################
# Copyright (c) 2018-2019, National Research Foundation (Square Kilometre Array)
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
###############################################################################

"""Tests for :py:mod:`katdal.dataset`."""

from __future__ import print_function, division, absolute_import

from nose.tools import assert_equal

from katdal.dataset import _selection_to_list


def test_selection_to_list():
    # Empty
    assert_equal(_selection_to_list(''), [])
    assert_equal(_selection_to_list([]), [])
    # Names
    assert_equal(_selection_to_list('a,b,c'), ['a', 'b', 'c'])
    assert_equal(_selection_to_list('a, b,c'), ['a', 'b', 'c'])
    assert_equal(_selection_to_list(['a', 'b', 'c']), ['a', 'b', 'c'])
    assert_equal(_selection_to_list(('a', 'b', 'c')), ['a', 'b', 'c'])
    assert_equal(_selection_to_list('a'), ['a'])
    # Objects
    assert_equal(_selection_to_list([1, 2, 3]), [1, 2, 3])
    assert_equal(_selection_to_list(1), [1])
    # Groups
    assert_equal(_selection_to_list('all', all=['a', 'b']), ['a', 'b'])
