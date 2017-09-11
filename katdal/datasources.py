################################################################################
# Copyright (c) 2011-2017, National Research Foundation (Square Kilometre Array)
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

from .dataset import AttrsSensors, VisFlagsWeights
from .sensordata import TelstateSensorData


class DataSource(AttrsSensors, VisFlagsWeights):
    """A generic data source presenting both correlator data and metadata."""
    def __init__(self, attrs=None, sensors=None, timestamps=None,
                 vis=None, flags=None, weights=None, name='container'):
        AttrsSensors.__init__(self, attrs, sensors, name)
        self.timestamps = timestamps
        VisFlagsWeights.__init__(self, vis, flags, weights, name)


class TelstateDataSource(DataSource):
    """A data source based on :class:`katsdptelstate.TelescopeState`."""
    def __init__(self, telstate, name='telstate'):
        self.telstate = telstate
        attrs = {}
        sensors = {}
        for key in telstate.keys():
            if telstate.is_immutable(key):
                attrs[key] = telstate[key]
            else:
                sensors[key] = TelstateSensorData(telstate, key)
        DataSource.__init__(self, attrs, sensors, name=name)


def open_data_source(url):
    """Construct the data source described by the given URL."""
    url_parts = urlparse.urlparse(url)
    if url_parts.scheme == 'redis':
        telstate = katsdptelstate.TelescopeState(url_parts.netloc)
        return TelstateDataSource(telstate, url)
