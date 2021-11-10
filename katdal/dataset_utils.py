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

import pathlib
import urllib.parse
import logging

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


def _robust_target(description):
    """Robust build of :class:`katpoint.Target` object from description string."""
    if not description:
        return katpoint.Target('Nothing, special')
    try:
        return katpoint.Target(description)
    except ValueError:
        logger.warning("Invalid target description '%s' - replaced with dummy target", description)
        return katpoint.Target('Nothing, special')


def _selection_to_list(names, **groups):
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


def _is_deselection(selectors):
    """If all the selectors have a tilde ~ , then this is treated as a
     deselect and we are going to invert the selection.
     TODO: For version 1 release the deselector interface should just have a leading ~
     """
    for selector in selectors:
        if selector[0] != '~':
            return False
    return True
