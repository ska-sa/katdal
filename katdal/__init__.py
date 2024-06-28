################################################################################
# Copyright (c) 2013-2021, National Research Foundation (SARAO)
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

"""Data access library for data sets in the MeerKAT Visibility Format (MVF)."""

import logging as _logging
import urllib.parse

from .concatdata import ConcatenatedDataSet
from .dataset import DataSet, WrongVersion  # noqa: F401
from .datasources import open_data_source
from .h5datav1 import H5DataV1
from .h5datav2 import H5DataV2
from .h5datav3 import H5DataV3
from .lazy_indexer import LazyTransform, dask_getitem  # noqa: F401
from .spectral_window import SpectralWindow  # noqa: F401
from .visdatav4 import VisibilityDataV4


# Setup library logger and add a print-like handler used when no logging is configured
class _NoConfigFilter(_logging.Filter):
    """Filter which only allows event if top-level logging is not configured."""

    def filter(self, record):
        return 1 if not _logging.root.handlers else 0


_no_config_handler = _logging.StreamHandler()
_no_config_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT))
_no_config_handler.addFilter(_NoConfigFilter())
logger = _logging.getLogger(__name__)
logger.addHandler(_no_config_handler)

# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])
# END VERSION CHECK

# -----------------------------------------------------------------------------
# -- Top-level functions passed on to the appropriate format handler
# -----------------------------------------------------------------------------

formats = [H5DataV3, H5DataV2, H5DataV1]


def _file_action(action, filename, *args, **kwargs):
    """Perform action on data file using the appropriate format class.

    Parameters
    ----------
    action : string
        Name of method to call on format class
    filename : string
        Data file name
    args, kwargs : extra parameters to method (optional)

    Returns
    -------
    result : object
        Result of action

    """
    for format in formats:
        try:
            result = getattr(format, action)(filename, *args, **kwargs)
            break
        except WrongVersion:
            continue
    else:
        raise WrongVersion(f"File '{filename}' has unknown data file format or version")
    return result


def open(filename, ref_ant='', time_offset=0.0, **kwargs):
    """Open data file(s) with loader of the appropriate version.

    Parameters
    ----------
    filename : string or sequence of strings
        Data file name or list of file names
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds
    kwargs : dict, optional
        Extra keyword arguments are passed on to underlying accessor class:

        mode (string, optional)
            [H5DataV*] File opening mode (e.g. 'r+' to open file in write mode)
        quicklook (bool)
            [H5DataV2] True if synthesised timestamps should be used to
            partition data set even if real timestamps are irregular, thereby
            avoiding the slow loading of real timestamps at the cost of
            slightly inaccurate label borders

        See the documentation of :class:`VisibilityDataV4` for the keywords
        it accepts.

    Returns
    -------
    data : :class:`DataSet` object
        Object providing :class:`DataSet` interface to file(s)

    """
    if isinstance(filename, str):
        filenames = [filename]
    else:
        unexpected = set(kwargs.get('preselect', {})) - {'channels'}
        if unexpected:
            raise IndexError(f'Unsupported preselect key(s) for ConcatenatedDataSet: {unexpected}')
        filenames = filename
    datasets = []
    for f in filenames:
        # V4 RDB file or live telstate with optional URL-style query string
        parsed = urllib.parse.urlsplit(f)
        if parsed.path.endswith('.rdb') or parsed.scheme != '':
            dataset = VisibilityDataV4(open_data_source(f, **kwargs),
                                       ref_ant, time_offset, **kwargs)
        else:
            if 'preselect' in kwargs:
                raise TypeError('preselect is not supported for this format')
            dataset = _file_action('__call__', f, ref_ant, time_offset, **kwargs)
        datasets.append(dataset)
    return datasets[0] if isinstance(filename, str) else ConcatenatedDataSet(datasets)


def get_ants(filename):
    """Quick look function to get the list of antennas in a data file.

    Parameters
    ----------
    filename : string
        Data file name

    Returns
    -------
    antennas : list of :class:`katpoint.Antenna` objects

    """
    return _file_action('_get_ants', filename)


def get_targets(filename):
    """Quick look function to get the list of targets in a data file.

    Parameters
    ----------
    filename : string
        Data file name

    Returns
    -------
    targets : :class:`katpoint.Catalogue` object
        All targets in file

    """
    return _file_action('_get_targets', filename)
