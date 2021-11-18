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

from katdal.categorical import parse_categorical_table, tabulate_categorical
from katdal.dataset_utils import (align_scans, parse_url_or_path,
                                  selection_to_list)


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


ALIGN_SCANS_TEST_CASES = [(
    # extract_scan_alignment.py 1629113328_sdp_l0.full.rdb
    # BEFORE
    """
    Dumps  Label   Target  Scan
    ------------------------------
    0      "       A       stop
    4      track           slew
    6                      track
    14     raster          scan
    183
    """,
    # AFTER
    """
    Dumps  Label   Target  Scan
    ------------------------------
    0      "       A       stop
    4      track           slew
    6                      track
    14     raster          scan
    183
    """), (
    # extract_scan_alignment.py 1615463457_sdp_l0.full.rdb --dumps=70
    # BEFORE
    """
    Dumps  Label                     Target  Scan
    ------------------------------------------------
    0      interferometric_pointing  A       slew
    54                                       track
    63                               B
    64                                       slew
    69                                       track
    70
    """,
    # AFTER
    """
    Dumps  Label                     Target  Scan
    ------------------------------------------------
    0      interferometric_pointing  A       slew
    54                                       track
    64                               B       slew
    69                                       track
    70
    """), (
    # OPS-1631 / SPR1-1177
    # extract_scan_alignment.py 1612141271_sdp_l0.full.rdb --dumps 666
    # BEFORE
    """
    Dumps  Label  Target  Scan
    -----------------------------
    0      "      A       track
    5      track
    306    track
    307           B
    308                   slew
    353                   track
    652    track
    653           C
    654                   slew
    666
    """,
    # AFTER
    """
    Dumps  Label  Target  Scan
    -----------------------------
    0      "      A       track
    5      track          track
    306    track  B       track
    308                   slew
    353                   track
    652    track  C       track
    654                   slew
    666
    """), (
    # OPS-1631 / SPR1-1177
    # extract_scan_alignment.py 1622455035_sdp_l0.full.rdb --dumps 1000
    # BEFORE
    """
    Dumps  Label  Target  Scan
    -----------------------------
    0      "      A       slew
    2      track          track
    483    track
    484           B
    485                   slew
    513                   track
    992    track
    993           C
    994                   slew
    1000
    """,
    # AFTER
    """
    Dumps  Label  Target  Scan
    -----------------------------
    0      "      A       slew
    2      track          track
    483    track  B       track
    485                   slew
    513                   track
    992    track  C       track
    994                   slew
    1000
    """)
]


def test_align_scans():
    names = 'Scan Label Target'.split()
    for before, expected in ALIGN_SCANS_TEST_CASES:
        indent = len(expected) - len(expected.lstrip()) - 1
        sensors_in = parse_categorical_table(before)
        sensors_out = align_scans(*(sensors_in[name] for name in names))
        sensors_out = dict(zip(names, sensors_out))
        actual = tabulate_categorical(sensors_out, list(sensors_in.keys()), indent)
        assert_equal(actual, expected)
