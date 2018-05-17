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
import numpy as np
import dask.array as da

from .sensordata import TelstateSensorData
from .chunkstore_s3 import S3ChunkStore
from .chunkstore_npy import NpyFileChunkStore


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


def _has_chunk_to_flags(has_chunk, block_id, full_chunks):
    """Turn a has_chunk bool into chunk of flags with correct data_lost bit."""
    shape = tuple(chk[idx] for chk, idx in zip(full_chunks, block_id))
    return np.full(shape, 0 if has_chunk else 8, dtype=np.uint8)


def _multi_or_3d(*args):
    """Do bitwise 'or' of two or more 3-D arrays (without modifying them)."""
    args = np.atleast_3d(*args)
    out = args[0] | args[1]
    for arg in args[2:]:
        out |= arg
    return out


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
        darray = {}
        extra_flags = []
        for array, info in chunk_info.iteritems():
            array_name = store.join(base_name, array)
            chunk_args = (array_name, info['chunks'], info['dtype'])
            darray[array] = store.get_dask_array(*chunk_args)
            # Find all missing chunks in array and convert to 'data_lost' flags
            has_array = store.has_dask_array(*chunk_args)
            chunks_lost = da.map_blocks(_has_chunk_to_flags, has_array,
                                        token='missing-chunks-' + array_name,
                                        chunks=info['chunks'], dtype=np.uint8,
                                        full_chunks=info['chunks'])
            extra_flags.append(chunks_lost)
            extra_flags.append('ijk'[:chunks_lost.ndim])
        vis = darray['correlator_data']
        # Combine original L0 flags with extras (missing chunks per array)
        flags = da.atop(_multi_or_3d, 'ijk', darray['flags'], 'ijk',
                        *extra_flags, token=store.join(base_name, 'flags_raw'),
                        dtype=np.uint8)
        # Combine low-resolution weights and high-resolution weights_channel
        weights = darray['weights'] * darray['weights_channel'][..., np.newaxis]
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

    This figures out the appropriate capture block ID and L0 stream name from
    a capture-stream specific telstate, or uses the provided ones. It then
    constructs a view on `telstate` with at least the prefixes

      - <capture_block_id>_<stream_name>
      - <capture_block_id>
      - <stream_name>

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState` object
        Original telescope state
    capture_block_id : string, optional
        Specify capture block ID explicitly (detected otherwise)
    stream_name : string, optional
        Specify L0 stream name explicitly (detected otherwise)

    Returns
    -------
    telstate : :class:`katsdptelstate.TelescopeState` object
        Telstate with a view that incorporates capture block, stream and combo

    Raises
    ------
    ValueError
        If no capture block or L0 stream could be detected (with no override)
    """
    # Detect the capture block
    if not capture_block_id:
        try:
            capture_block_id = str(telstate['capture_block_id'])
        except KeyError:
            raise ValueError('No capture block ID found in telstate - '
                             'please specify it manually')
    # Detect the captured stream
    if not stream_name:
        try:
            stream_name = str(telstate['stream_name'])
        except KeyError:
            raise ValueError('No captured stream found in telstate - '
                             'please specify the stream manually')
    # Check the stream type
    telstate = telstate.view(stream_name)
    stream_type = telstate.get('stream_type', 'unknown')
    expected_type = 'sdp.vis'
    if stream_type != expected_type:
        raise ValueError("Found stream {!r} but it has the wrong type {!r},"
                         " expected {!r}".format(stream_name, stream_type,
                                                 expected_type))
    logger.info('Using capture block %r and stream %r',
                capture_block_id, stream_name)
    telstate = telstate.view(capture_block_id)
    capture_stream = telstate.SEPARATOR.join((capture_block_id, stream_name))
    telstate = telstate.view(capture_stream)
    return telstate


def _shorten_key(telstate, key):
    """Shorten telstate key by subtracting the first prefix that fits.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState` object
        Telescope state
    key : string
        Telescope state key

    Returns
    -------
    short_key : string
        Suffix of `key` after subtracting first matching prefix, or empty
        string if `key` does not start with any of the prefixes (or exactly
        matches a prefix, which is also considered pathological)

    """
    for prefix in telstate.prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return ''


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
    telstate : :class:`katsdptelstate.TelescopeState` object
        Telescope state with appropriate views
    chunk_store : :class:`katdal.ChunkStore` object, optional
        Chunk store for visibility data (the default is no data - metadata only)
    source_name : string, optional
        Name of telstate source (used for metadata name)
    """
    def __init__(self, telstate, chunk_store=None, source_name='telstate'):
        self.telstate = telstate
        # Collect sensors
        sensors = {}
        for key in telstate.keys():
            if not telstate.is_immutable(key):
                sensor_name = _shorten_key(telstate, key)
                if sensor_name:
                    sensors[sensor_name] = TelstateSensorData(telstate, key)
        metadata = AttrsSensors(telstate, sensors, name=source_name)
        try:
            t0 = telstate['sync_time'] + telstate['first_timestamp']
            int_time = telstate['int_time']
            chunk_name = telstate['chunk_name']
            chunk_info = telstate['chunk_info']
        except KeyError:
            # Metadata without data or timestamps
            DataSource.__init__(self, metadata, None)
        else:
            # Extract timestamps from telstate
            n_dumps = chunk_info['correlator_data']['shape'][0]
            timestamps = t0 + np.arange(n_dumps) * int_time
            data = ChunkStoreVisFlagsWeights(
                chunk_store, chunk_name, chunk_info) if chunk_store else None
            # Metadata and timestamps with or without data
            DataSource.__init__(self, metadata, timestamps, data)

    @classmethod
    def from_url(cls, url, chunk_store='auto'):
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
            telstate = view_capture_stream(telstate, **kwargs)
            # Look for adjacent data directory (presumably containing NPY files)
            if chunk_store == 'auto':
                rdb_path = os.path.abspath(url_parts.path)
                store_path = os.path.dirname(os.path.dirname(rdb_path))
                if not store_path:
                    store_path = os.path.curdir
                try:
                    data_path = os.path.join(store_path, telstate['chunk_name'])
                except KeyError:
                    chunk_store = None
                else:
                    if os.path.isdir(data_path):
                        chunk_store = NpyFileChunkStore(store_path)
                    else:
                        chunk_store = S3ChunkStore.from_url(telstate['s3_endpoint_url'])
            return cls(telstate, chunk_store, source_name)
        elif url_parts.scheme == 'redis':
            # Redis server
            try:
                telstate = katsdptelstate.TelescopeState(url_parts.netloc, db)
            except katsdptelstate.ConnectionError as e:
                raise DataSourceNotFound(str(e))
            telstate = view_capture_stream(telstate, **kwargs)
            if chunk_store == 'auto':
                chunk_store = S3ChunkStore.from_url(telstate['s3_endpoint_url'])
            return cls(telstate, chunk_store, source_name)


def open_data_source(url, *args, **kwargs):
    """Construct the data source described by the given URL."""
    try:
        return TelstateDataSource.from_url(url, *args, **kwargs)
    except DataSourceNotFound as err:
        # Amend the error message for the case of an IP address without scheme
        url_parts = urlparse.urlparse(url, scheme='file')
        if url_parts.scheme == 'file' and not os.path.isfile(url_parts.path):
            raise DataSourceNotFound(
                '{} (add a URL scheme if {!r} is not meant to be a file)'
                .format(err, url_parts.path))
