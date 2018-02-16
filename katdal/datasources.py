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
import os
import logging

import katsdptelstate
import redis
import numpy as np

from .sensordata import TelstateSensorData
from .chunkstore_s3 import S3ChunkStore


logger = logging.getLogger(__name__)


class DataSourceNotFound(Exception):
    """File associated with DataSource not found or server not responding."""


class AttrsSensors(object):
    """Metadata in the form of attributes and sensors.

    Parameters
    ----------
    attrs : mapping from string to object
        Metadata attributes
    sensors : mapping from string to :class:`SensorData` objects
        Metadata sensor cache mapping sensor names to raw sensor data
    name : string, optional
        Identifier that describes the origin of the metadata (backend-specific)

    """
    def __init__(self, attrs, sensors, name='custom'):
        self.attrs = attrs
        self.sensors = sensors
        self.name = name


class VisFlagsWeights(object):
    """Correlator data in the form of visibilities, flags and weights.

    Parameters
    ----------
    vis : array-like of complex64, shape (*T*, *F*, *B*)
        Complex visibility data as a function of time, frequency and baseline
    flags : array-like of uint8, shape (*T*, *F*, *B*)
        Flags as a function of time, frequency and baseline
    weights : array-like of float32, shape (*T*, *F*, *B*)
        Visibility weights as a function of time, frequency and baseline
    name : string, optional
        Identifier that describes the origin of the data (backend-specific)

    """
    def __init__(self, vis, flags, weights, name='custom'):
        if not (vis.shape == flags.shape == weights.shape):
            raise ValueError("Shapes of vis %s, flags %s and weights %s differ"
                             % (vis.shape, flags.shape, weights.shape))
        self.vis = vis
        self.flags = flags
        self.weights = weights
        self.name = name

    @property
    def shape(self):
        return self.vis.shape


class ChunkStoreVisFlagsWeights(VisFlagsWeights):
    """Correlator data stored in a chunk store.

    Parameters
    ----------
    store : :class:`ChunkStore` object
        Chunk store
    base_name : string
        Name of dataset in store, as array name prefix (akin to a filename)
    chunk_info : dict mapping array name to info dict
        Dict specifying dtype, shape and chunks per array
    """
    def __init__(self, store, base_name, chunk_info):
        self.store = store
        da = {}
        for array, info in chunk_info.iteritems():
            array_name = store.join(base_name, array)
            da[array] = store.get_dask_array(array_name, info['chunks'],
                                             info['dtype'])
        vis = da['correlator_data']
        flags = da['flags']
        # Combine low-resolution weights and high-resolution weights_channel
        weights = da['weights'] * da['weights_channel'][..., np.newaxis]
        VisFlagsWeights.__init__(self, vis, flags, weights, base_name)


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


def view_capture_stream(telstate, capture_block_id=None, stream_name=None):
    """Create telstate view based on capture block ID and stream name.

    This figures out the appropriate capture block ID (the latest one with
    obs_params) and L0 stream name (the first one with chunk_info), or use
    the provided ones. It then constructs a view on `telstate` with at least
    the prefixes

      - <capture_block_id>_<stream_name>
      - <capture_block_id>
      - <stream_name>

    Parameters
    ----------
    telstate: :class:`katsdptelstate.TelescopeState` object
        Original telescope state
    capture_block_id : string, optional
        Specify capture block ID explicitly (detected otherwise)
    stream_name : string, optional
        Specify L0 stream name explicitly (detected otherwise)

    Returns
    -------
    telstate: :class:`katsdptelstate.TelescopeState` object
        Telstate with a view that incorporates capture block, stream and combo

    Raises
    ------
    ValueError
        If no capture block or L0 stream could be detected (with no override)
    """
    # Detect the capture block
    if not capture_block_id:
        capture_blocks = []
        if 'sdp_capture_block_id' in telstate:
            for value_time in telstate.get_range('sdp_capture_block_id', st=0):
                cbid = str(value_time[0])
                if 'obs_params' in telstate.view(cbid, exclusive=True):
                    capture_blocks.append(cbid)
        if not capture_blocks:
            raise ValueError('No capture block IDs found in telstate - '
                             'please specify it manually')
        # Pick the latest capture block
        capture_block_id = capture_blocks[-1]
        if len(capture_blocks) > 1:
            logger.warning('Telstate has more than one capture block - %s - '
                           'picking the latest', capture_blocks)
    # Detect the captured stream
    if not stream_name:
        streams = []
        for stream, config in telstate['sdp_config']['outputs'].items():
            if config['type'] != 'sdp.l0':
                continue
            capture_stream = telstate.SEPARATOR.join((capture_block_id, stream))
            if 'chunk_info' in telstate.view(capture_stream, exclusive=True):
                streams.append(stream)
        if not streams:
            raise ValueError('No captured streams found in telstate - '
                             'please specify the stream manually')
        stream_name = streams[0]
        if len(streams) > 1:
            logger.warning('Telstate has more than one captured stream - %s - '
                           'picking the first', streams)
    logger.info('Found capture block %s and stream %s',
                capture_block_id, stream_name)
    capture_stream = telstate.SEPARATOR.join((capture_block_id, stream_name))
    telstate = telstate.view(stream_name)
    telstate = telstate.view(capture_block_id)
    telstate = telstate.view(capture_stream)
    return telstate


class TelstateDataSource(DataSource):
    """A data source based on :class:`katsdptelstate.TelescopeState`.

    It is assumed that the provided `telstate` already has the appropriate
    views to find observation, stream and chunk store information. It typically
    needs the following prefixes:

      - <capture block ID>_<L0 stream>
      - <capture block ID>
      - <L0 stream>

    Parameters
    ----------
    telstate: :class:`katsdptelstate.TelescopeState` object
        Telescope state with appropriate views
    source_name : string, optional
        Name of telstate source (used for metadata name)
    """
    def __init__(self, telstate, source_name='telstate'):
        self.telstate = telstate
        # Collect sensors
        sensors = {}
        for key in telstate.keys():
            if not telstate.is_immutable(key):
                sensors[key] = TelstateSensorData(telstate, key)
        metadata = AttrsSensors(telstate, sensors, name=source_name)
        try:
            chunk_name = telstate['chunk_name']
            chunk_info = telstate['chunk_info']
            s3_endpoint_url = telstate['s3_endpoint_url']
        except KeyError:
            # Metadata without data
            DataSource.__init__(self, metadata, None)
        else:
            # Extract VisFlagsWeights and timestamps from telstate
            store = S3ChunkStore.from_url(s3_endpoint_url)
            ts_name = store.join(chunk_name, 'timestamps')
            ts_chunks = chunk_info['timestamps']['chunks']
            ts_dtype = chunk_info['timestamps']['dtype']
            timestamps = store.get_dask_array(ts_name, ts_chunks, ts_dtype)
            # Make timestamps explicit, mutable (to be removed from store soon)
            timestamps = timestamps.compute().copy()
            data = ChunkStoreVisFlagsWeights(store, chunk_name, chunk_info)
            DataSource.__init__(self, metadata, timestamps, data)

    @classmethod
    def from_url(cls, url):
        """Construct TelstateDataSource from URL (RDB file / REDIS server)."""
        url_parts = urlparse.urlparse(url, scheme='file')
        kwargs = dict(urlparse.parse_qsl(url_parts.query))
        # Extract Redis database number if provided
        db = int(kwargs.pop('db', '0'))
        source_name = url_parts.geturl()
        if url_parts.scheme == 'file':
            # RDB dump file
            telstate = katsdptelstate.TelescopeState()
            try:
                telstate.load_from_file(url_parts.path)
            except OSError as err:
                raise DataSourceNotFound(str(err))
            return cls(view_capture_stream(telstate, **kwargs), source_name)
        elif url_parts.scheme == 'redis':
            # Redis server
            try:
                telstate = katsdptelstate.TelescopeState(url_parts.netloc, db)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                raise DataSourceNotFound(str(e))
            return cls(view_capture_stream(telstate, **kwargs), source_name)


def open_data_source(url):
    """Construct the data source described by the given URL."""
    try:
        return TelstateDataSource.from_url(url)
    except DataSourceNotFound as err:
        # Amend the error message for the case of an IP address without scheme
        url_parts = urlparse.urlparse(url, scheme='file')
        if url_parts.scheme == 'file' and not os.path.isfile(url_parts.path):
            raise DataSourceNotFound(
                '{} (add a URL scheme if {!r} is not meant to be a file)'
                .format(err, url_parts.path))
