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
import itertools
from collections import defaultdict

import katsdptelstate
import numpy as np
import dask.array as da
from dask.array.rechunk import intersect_chunks

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


def _apply_data_lost(orig_flags, lost, block_id):
    mark = lost.get(block_id)
    if not mark:
        return orig_flags    # Common case - no data lost
    flags = orig_flags.copy()
    for idx in mark:
        flags[idx] |= 8
    return flags


class ChunkStoreVisFlagsWeights(VisFlagsWeights):
    """Correlator data stored in a chunk store.

    Parameters
    ----------
    store : :class:`ChunkStore` object
        Chunk store
    chunk_info : dict mapping array name to info dict
        Dict specifying prefix, dtype, shape and chunks per array
    """
    def __init__(self, store, chunk_info):
        self.store = store
        darray = {}
        has_arrays = []
        for array, info in chunk_info.items():
            array_name = store.join(info['prefix'], array)
            chunk_args = (array_name, info['chunks'], info['dtype'])
            darray[array] = store.get_dask_array(*chunk_args)
            # Find all missing chunks in array and convert to 'data_lost' flags
            has_arrays.append((store.has_array(array_name, info['chunks'], info['dtype']),
                               info['chunks']))
        vis = darray['correlator_data']
        base_name = chunk_info['correlator_data']['prefix']
        flags_raw_name = store.join(chunk_info['flags']['prefix'], 'flags_raw')
        # Combine original flags with data_lost indicating where values were lost from
        # other arrays.
        lost = defaultdict(list)  # Maps chunk index to list of index expressions to mark as lost
        for has_array, chunks in has_arrays:
            # array may have fewer dimensions than flags
            # (specifically, for weights_channel).
            if has_array.ndim < darray['flags'].ndim:
                chunks += tuple((x,) for x in darray['flags'].shape[has_array.ndim:])
            intersections = intersect_chunks(darray['flags'].chunks, chunks)
            for has, pieces in itertools.izip(has_array.flat, intersections):
                if not has:
                    for piece in pieces:
                        chunk_idx, slices = zip(*piece)
                        lost[chunk_idx].append(slices)
        flags = da.map_blocks(_apply_data_lost, darray['flags'], dtype=np.uint8,
                              name=flags_raw_name, lost=lost)
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


def view_capture_stream(telstate, capture_block_id=None, stream_name=None,
                        **kwargs):
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
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other methods and ignored

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


def _ensure_prefix_is_set(chunk_info, telstate):
    """Augment `chunk_info` with chunk name prefix if not set."""
    for info in chunk_info.values():
        if 'prefix' not in info:
            info['prefix'] = telstate['chunk_name']
    return chunk_info


def _infer_chunk_store(url_parts, telstate, npy_store_path=None,
                       s3_endpoint_url=None, **kwargs):
    """Construct chunk store automatically from dataset URL and telstate.

    Parameters
    ----------
    url_parts : :class:`urlparse.ParseResult` object
        Parsed dataset URL
    telstate : :class:`katsdptelstate.TelescopeState` object
        Telescope state
    npy_store_path : string, optional
        Top-level directory of NpyFileChunkStore (overrides the default)
    s3_endpoint_url : string, optional
        Endpoint of S3 service, e.g. 'http://127.0.0.1:9000' (overrides default)
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other methods and ignored

    Returns
    -------
    store : :class:`katdal.ChunkStore` object
        Chunk store for visibility data

    Raises
    ------
    KeyError
        If telstate lacks critical keys
    :exc:`katdal.chunkstore.StoreUnavailable`
        If the chunk store could not be constructed
    """
    # Use overrides if provided, regardless of URL and telstate (NPY first)
    if npy_store_path:
        return NpyFileChunkStore(npy_store_path)
    if s3_endpoint_url:
        return S3ChunkStore.from_url(s3_endpoint_url, **kwargs)
    # NPY chunk store is an option if the dataset is an RDB file
    if url_parts.scheme == 'file':
        # Look for adjacent data directory (presumably containing NPY files)
        rdb_path = os.path.abspath(url_parts.path)
        store_path = os.path.dirname(os.path.dirname(rdb_path))
        chunk_info = telstate['chunk_info']
        chunk_info = _ensure_prefix_is_set(chunk_info, telstate)
        vis_prefix = chunk_info['correlator_data']['prefix']
        data_path = os.path.join(store_path, vis_prefix)
        if os.path.isdir(data_path):
            return NpyFileChunkStore(store_path)
    return S3ChunkStore.from_url(telstate['s3_endpoint_url'], **kwargs)


def _upgrade_flags(chunk_info, telstate):
    """Look for associated flag streams and override chunk_info to use them."""
    try:
        archived_streams = telstate['sdp_archived_streams']
        capture_block_id = str(telstate['capture_block_id'])
        stream_name = str(telstate['stream_name'])
    except KeyError as e:
        logger.debug('No additional flag capture streams found: %s', e)
        return chunk_info
    for s in archived_streams:
        telstate_s = telstate.root().view(s)
        if telstate_s['stream_type'] != 'sdp.flags' or \
           stream_name not in telstate_s['src_streams']:
            continue
        # Look for chunk metadata in appropriate capture_stream telstate view
        flags_cs = telstate.SEPARATOR.join((capture_block_id, s))
        telstate_cs = telstate_s.view(flags_cs)
        flags_info = telstate_cs['chunk_info']
        flags_info = _ensure_prefix_is_set(flags_info, telstate_cs)
        chunk_info.update(flags_info)
    return chunk_info


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
    timestamps : array of float, optional
        Visibility timestamps, overriding (or fixing) the ones found in telstate
    source_name : string, optional
        Name of telstate source (used for metadata name)

    Raises
    ------
    KeyError
        If telstate lacks critical keys
    """
    def __init__(self, telstate, chunk_store=None, timestamps=None,
                 source_name='telstate'):
        self.telstate = telstate
        # Collect sensors
        sensors = {}
        for key in telstate.keys():
            if not telstate.is_immutable(key):
                sensor_name = _shorten_key(telstate, key)
                if sensor_name:
                    sensors[sensor_name] = TelstateSensorData(telstate, key)
        metadata = AttrsSensors(telstate, sensors, name=source_name)
        if timestamps is None:
            # Synthesise timestamps from the relevant telstate bits
            t0 = telstate['sync_time'] + telstate['first_timestamp']
            int_time = telstate['int_time']
            chunk_info = telstate['chunk_info']
            n_dumps = chunk_info['correlator_data']['shape'][0]
            timestamps = t0 + np.arange(n_dumps) * int_time
        if chunk_store is None:
            data = None
        else:
            chunk_info = telstate['chunk_info']
            chunk_info = _ensure_prefix_is_set(chunk_info, telstate)
            chunk_info = _upgrade_flags(chunk_info, telstate)
            data = ChunkStoreVisFlagsWeights(chunk_store, chunk_info)
        # Metadata and timestamps with or without data
        DataSource.__init__(self, metadata, timestamps, data)

    @classmethod
    def from_url(cls, url, chunk_store='auto', **kwargs):
        """Construct TelstateDataSource from URL (RDB file / REDIS server).

        Parameters
        ----------
        url : string
            URL serving as entry point to dataset (typically RDB file or REDIS)
        chunk_store : :class:`katdal.ChunkStore` object, optional
            Chunk store for visibility data (obtained automatically by default,
            or set to None for metadata-only dataset)
        kwargs : dict, optional
            Extra keyword arguments passed to telstate view and chunk store init
        """
        url_parts = urlparse.urlparse(url, scheme='file')
        # Merge key-value pairs from URL query with keyword arguments
        # of function (the latter takes precedence)
        url_kwargs = dict(urlparse.parse_qsl(url_parts.query))
        url_kwargs.update(kwargs)
        kwargs = url_kwargs
        # Extract Redis database number if provided
        db = int(kwargs.pop('db', '0'))
        if url_parts.scheme == 'file':
            # RDB dump file
            telstate = katsdptelstate.TelescopeState()
            try:
                telstate.load_from_file(url_parts.path)
            except OSError as err:
                raise DataSourceNotFound(str(err))
        elif url_parts.scheme == 'redis':
            # Redis server
            try:
                telstate = katsdptelstate.TelescopeState(url_parts.netloc, db)
            except katsdptelstate.ConnectionError as e:
                raise DataSourceNotFound(str(e))
        telstate = view_capture_stream(telstate, **kwargs)
        if chunk_store == 'auto':
            chunk_store = _infer_chunk_store(url_parts, telstate, **kwargs)
        return cls(telstate, chunk_store, source_name=url_parts.geturl())


def open_data_source(url, **kwargs):
    """Construct the data source described by the given URL."""
    try:
        return TelstateDataSource.from_url(url, **kwargs)
    except DataSourceNotFound as err:
        # Amend the error message for the case of an IP address without scheme
        url_parts = urlparse.urlparse(url, scheme='file')
        if url_parts.scheme == 'file' and not os.path.isfile(url_parts.path):
            raise DataSourceNotFound(
                '{} (add a URL scheme if {!r} is not meant to be a file)'
                .format(err, url_parts.path))
