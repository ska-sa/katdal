################################################################################
# Copyright (c) 2017-2018, National Research Foundation (Square Kilometre Array)
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

"""Various sources of correlator data and metadata."""

import urlparse

import katsdptelstate
import numpy as np

from .dataset import AttrsSensors
from .sensordata import TelstateSensorData


class DataSource(object):
    """A generic data source presenting both correlator data and metadata.

    Parameters
    ----------
    metadata : :class:`AttrsSensors` object
        Metadata attributes and sensors
    timestamps : array-like of float, length *T*
        Timestamps at centroids of visibilities in UTC seconds since Unix epoch
    data : :class:`VisFlagsWeights` object, optional
        Correlator data (visibilities, flags and weights)

    """
    def __init__(self, metadata, timestamps, data=None):
        self.metadata = metadata
        self.timestamps = timestamps
        self.data = data

    @property
    def name(self):
        name = self.metadata.name
        if self.data and self.data.name != name:
            name += ' | ' + self.data.name
        return name


class TelstateDataSource(DataSource):
    """A data source based on :class:`katsdptelstate.TelescopeState`."""
    def __init__(self, telstate, source_name='telstate'):
        self.telstate = telstate
        attrs = {}
        sensors = {}
        for key in telstate.keys():
            if telstate.is_immutable(key):
                attrs[key] = telstate[key]
            else:
                sensors[key] = TelstateSensorData(telstate, key)
        metadata = AttrsSensors(attrs, sensors, name=source_name)
        DataSource.__init__(self, metadata, None)


def open_data_source(url):
    """Construct the data source described by the given URL."""
    url_parts = urlparse.urlparse(url)
    if url_parts.scheme == 'telstate+redis':
        telstate = katsdptelstate.TelescopeState(url_parts.netloc)
        return TelstateDataSource(telstate, url)
    else:
        raise ValueError("Unsupported data source '%s'" % (url,))
