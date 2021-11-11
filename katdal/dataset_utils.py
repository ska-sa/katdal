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

"""Utility functions for data sets."""

import copy
import logging
import pathlib
import urllib.parse

import katpoint
from katpoint import is_iterable

logger = logging.getLogger(__name__)


def parse_url_or_path(url_or_path):
    """Parse URL into components, converting path to absolute file URL.

    Parameters
    ----------
    url_or_path : string
        URL, or filesystem path if there is no scheme

    Returns
    -------
    url_parts : :class:`urllib.parse.ParseResult`
        Components of the parsed URL ('file' scheme will have an absolute path)
    """
    url_parts = urllib.parse.urlparse(url_or_path)
    # Assume filesystem path if there is no scheme (unless url is empty string)
    if not url_parts.scheme and url_parts.path:
        # A file:// URL expects an absolute path (local paths can't be located)
        absolute_path = str(pathlib.Path(url_parts.path).absolute())
        # Note to self: namedtuple._replace is not a private method, despite the underscore!
        url_parts = url_parts._replace(scheme='file', path=absolute_path)
    return url_parts


def robust_target(description):
    """Robust build of :class:`katpoint.Target` object from description string."""
    if not description:
        return katpoint.Target('Nothing, special')
    try:
        return katpoint.Target(description)
    except ValueError:
        logger.warning("Invalid target description '%s' - replaced with dummy target", description)
        return katpoint.Target('Nothing, special')


def selection_to_list(names, **groups):
    """Normalise string of comma-separated names or sequence of names / objects.

    Parameters
    ----------
    names : string / object or sequence of strings / objects
        A string of comma-separated names or a sequence of names / objects
    groups : dict, optional
        Each extra keyword is the name of a predefined list of names / objects

    Returns
    -------
    list : list of strings / objects
        List of names / objects
    """
    if isinstance(names, str):
        if not names:
            return []
        elif names in groups:
            return list(groups[names])
        else:
            return [name.strip() for name in names.split(',')]
    elif is_iterable(names):
        return list(names)
    else:
        return [names]


def is_deselection(selectors):
    """If all the selectors have a tilde ~ , then this is treated as a
     deselect and we are going to invert the selection.
     TODO: For version 1 release the deselector interface should just have a leading ~
     """
    for selector in selectors:
        if selector[0] != '~':
            return False
    return True


def align_scans(scan, label, target):
    """Align scan, compound scan and target boundaries.

    Parameters
    ----------
    scan, label, target : :class:`~katdal.categorical.CategoricalData`
        Sensors for scan activities, compound scan labels and targets

    Returns
    -------
    scan, label, target : :class:`~katdal.categorical.CategoricalData`
        Aligned sensors
    """
    # First copy the sensors to avoid modifying the originals
    scan = copy.deepcopy(scan)
    label = copy.deepcopy(label)
    target = copy.deepcopy(target)
    # If the antenna starts slewing on the second dump, incorporate the
    # first dump into the slew too. This scenario typically occurs when the
    # first target is only set after the first dump is received.
    # The workaround avoids putting the first dump in a scan by itself,
    # typically with an irrelevant target.
    if len(scan) > 1 and scan.events[1] == 1 and scan[1] == 'slew':
        scan.events, scan.indices = scan.events[1:], scan.indices[1:]
        scan.events[0] = 0
    # Discard empty labels (typically found in raster scans, where first
    # scan has proper label and rest are empty) However, if all labels are
    # empty, keep them, otherwise whole data set will be one pathological
    # compscan...
    if len(label.unique_values) > 1:
        label.remove('')
    # Create duplicate scan events where labels are set during a scan
    # (i.e. not at start of scan)
    # ASSUMPTION: Number of scans >= number of labels
    # (i.e. each label should introduce a new scan)
    scan.add_unmatched(label.events)
    # Move proper label events onto the nearest scan start
    # ASSUMPTION: Number of labels <= number of scans
    # (i.e. only a single label allowed per scan)
    label.align(scan.events)
    # If one or more scans at start of data set have no corresponding label,
    # add a default label for them
    if label.events[0] > 0:
        label.add(0, '')
    # Move target events onto the nearest scan start
    # ASSUMPTION: Number of targets <= number of scans
    # (i.e. only a single target allowed per scan)
    target.align(scan.events)
    # Remove repeats introduced by scan alignment (e.g. when sequence of
    # targets [A, B, A] becomes [A, A] if B and second A are in same scan)
    target.remove_repeats()
    # Remove initial target if antennas start in mode STOP
    # (typically left over from previous capture block)
    for segment, scan_state in scan.segments():
        # Keep going until first non-STOP scan or a new target is set
        if scan_state == 'stop' and target[segment.start] is target[0]:
            continue
        # Only remove initial target event if we move to a different target
        if target[segment.start] is not target[0]:
            # Only lose 1 event because target sensor doesn't allow repeats
            target.events = target.events[1:]
            target.indices = target.indices[1:]
            target.events[0] = 0
            # Remove initial target from target.unique_values if not used
            target.align(target.events)
        break
    return scan, label, target
