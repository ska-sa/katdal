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

"""Tests for :py:mod:`katdal.dataset_utils`."""

from nose.tools import assert_equal

from katdal.dataset_utils import parse_url_or_path, selection_to_list


def test_parse_url_or_path():
    # Normal URLs and empty strings pass right through
    assert_equal(parse_url_or_path('https://archive/file').geturl(),
                 'https://archive/file')
    assert_equal(parse_url_or_path('').geturl(), '')
    # Relative paths are turned into absolute paths and gain a 'file' scheme
    relative_file_url = parse_url_or_path('dir/filename.rdb')
    assert_equal(relative_file_url.scheme, 'file')
    parts = relative_file_url.path.rpartition('dir/filename.rdb')
    assert len(parts[0]) > 0
    assert_equal(parts[1], 'dir/filename.rdb')
    assert len(parts[2]) == 0
    # Absolute paths remain the same (just gaining a 'file' scheme)
    absolute_file_url = parse_url_or_path('/dir/filename.rdb')
    assert_equal(absolute_file_url.scheme, 'file')
    assert_equal(absolute_file_url.path, '/dir/filename.rdb')


def test_selection_to_list():
    # Empty
    assert_equal(selection_to_list(''), [])
    assert_equal(selection_to_list([]), [])
    # Names
    assert_equal(selection_to_list('a,b,c'), ['a', 'b', 'c'])
    assert_equal(selection_to_list('a, b,c'), ['a', 'b', 'c'])
    assert_equal(selection_to_list(['a', 'b', 'c']), ['a', 'b', 'c'])
    assert_equal(selection_to_list(('a', 'b', 'c')), ['a', 'b', 'c'])
    assert_equal(selection_to_list('a'), ['a'])
    # Objects
    assert_equal(selection_to_list([1, 2, 3]), [1, 2, 3])
    assert_equal(selection_to_list(1), [1])
    # Groups
    assert_equal(selection_to_list('all', all=['a', 'b']), ['a', 'b'])
