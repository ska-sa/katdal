################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
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

import io
import logging
import os.path
import urllib.parse

import katsdptelstate
import numpy as np

from .chunkstore import ChunkStoreError
from .chunkstore_npy import NpyFileChunkStore
from .chunkstore_s3 import S3ChunkStore
from .dataset import parse_url_or_path
from .sensordata import TelstateSensorGetter, TelstateToStr
from .vis_flags_weights import ChunkStoreVisFlagsWeights

logger = logging.getLogger(__name__)


class DataSourceNotFound(Exception):
    """File associated with DataSource not found or server not responding."""


class AttrsSensors:
    """Metadata in the form of attributes and sensors.

    Parameters
    ----------
    attrs : mapping from string to object
        Metadata attributes
    sensors : mapping from string to :class:`SensorGetter` objects
        Metadata sensor cache mapping sensor names to raw sensor data

    """
    def __init__(self, attrs, sensors):
        self.attrs = attrs
        self.sensors = sensors


class DataSource:
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
        self.name = ''
        self.url = ''


def view_capture_stream(telstate, capture_block_id, stream_name):
    """Create telstate view based on given capture block ID and stream name.

    It constructs a view on `telstate` with at least the prefixes

      - <capture_block_id>_<stream_name>
      - <capture_block_id>
      - <stream_name>

    Additionally if there is a <stream_name>_inherit key, that stream is
    added too (recursively).

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState` object
        Original telescope state
    capture_block_id : string
        Capture block ID
    stream_name : string
        Stream name

    Returns
    -------
    telstate : :class:`~katdal.sensordata.TelescopeState` object
        Telstate with a view that incorporates capture block, stream and combo
    """
    streams = [stream_name]
    while True:
        inherit = telstate.view(streams[-1], exclusive=True).get('inherit')
        if inherit is None:
            break
        streams.append(inherit)
    streams.reverse()

    for stream in streams:
        telstate = telstate.view(stream)
    telstate = telstate.view(capture_block_id)
    for stream in streams:
        capture_stream = telstate.join(capture_block_id, stream)
        telstate = telstate.view(capture_stream)
    return telstate


def view_l0_capture_stream(telstate, capture_block_id=None, stream_name=None,
                           **kwargs):
    """Create telstate view based on auto-determined capture block ID and stream name.

    This figures out the appropriate capture block ID and L0 stream name from
    a capture-stream specific telstate, or uses the provided ones. It then
    calls :meth:`view_capture_capture` to generate a view.

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
    telstate : :class:`~katdal.sensordata.TelstateToStr` object
        Telstate with a view that incorporates capture block, stream and combo
    capture_block_id : string
        Actual capture block ID used
    stream_name : string
        Actual L0 stream name used

    Raises
    ------
    ValueError
        If no capture block or L0 stream could be detected (with no override)
    """
    # Detect the capture block
    telstate = TelstateToStr(telstate)
    if not capture_block_id:
        try:
            capture_block_id = telstate['capture_block_id']
        except KeyError as e:
            raise ValueError('No capture block ID found in telstate - '
                             'please specify it manually') from e
    # Detect the captured stream
    if not stream_name:
        try:
            stream_name = telstate['stream_name']
        except KeyError as e:
            raise ValueError('No captured stream found in telstate - '
                             'please specify the stream manually') from e
    # Build the view
    telstate = view_capture_stream(telstate, capture_block_id, stream_name)
    # Check the stream type
    stream_type = telstate.get('stream_type', 'unknown')
    expected_type = 'sdp.vis'
    if stream_type != expected_type:
        raise ValueError(f"Found stream {stream_name!r} but it has the wrong type "
                         f"{stream_type!r}, expected {expected_type!r}")
    logger.info('Using capture block %r and stream %r',
                capture_block_id, stream_name)
    return telstate, capture_block_id, stream_name


def _shorten_key(telstate, key):
    """Shorten telstate key by subtracting the first prefix that fits.

    Parameters
    ----------
    telstate : :class:`katdal.sensordata.TelstateToStr` object
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


def _upgrade_chunk_info(chunk_info, improved_chunk_info):
    """Replace chunk info items with better ones while preserving number of dumps."""
    for key, improved_info in improved_chunk_info.items():
        original_info = chunk_info.get(key, improved_info)
        if improved_info['shape'][1:] != original_info['shape'][1:]:
            raise ValueError(f"Original '{key}' array has shape {original_info['shape']} "
                             f"while improved version has shape {improved_info['shape']}")
        chunk_info[key] = improved_info
    return chunk_info


def _align_chunk_info(chunk_info):
    """Inject phantom chunks to ensure all arrays have same number of dumps"""
    max_dumps = max(info['shape'][0] for info in chunk_info.values())
    for key, info in chunk_info.items():
        shape = info['shape']
        n_dumps = shape[0]
        if n_dumps < max_dumps:
            info['shape'] = (max_dumps,) + shape[1:]
            # We could just add a single new chunk, but that could cause an
            # inconveniently large chunk if there is a big difference between
            # n_dumps and max_dumps.
            time_chunks = info['chunks'][0] + (max_dumps - n_dumps) * (1,)
            info['chunks'] = (time_chunks,) + info['chunks'][1:]
            logger.debug('Adding %d phantom dumps to array %s', max_dumps - n_dumps, key)
    return chunk_info


def infer_chunk_store(url_parts, telstate, npy_store_path=None,
                      s3_endpoint_url=None, array='correlator_data', **kwargs):
    """Construct chunk store automatically from dataset URL and telstate.

    Parameters
    ----------
    url_parts : :class:`urlparse.ParseResult` object
        Parsed dataset URL
    telstate : :class:`~katdal.sensordata.TelstateToStr` object
        Telescope state
    npy_store_path : string, optional
        Top-level directory of NpyFileChunkStore (overrides the default)
    s3_endpoint_url : string, optional
        Endpoint of S3 service, e.g. 'http://127.0.0.1:9000' (overrides default)
    array : string, optional
        Array within the bucket from which to determine the prefix
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
        return S3ChunkStore(s3_endpoint_url, **kwargs)
    # NPY chunk store is an option if the dataset is an RDB file
    if url_parts.scheme == 'file':
        # Look for adjacent data directory (presumably containing NPY files)
        rdb_path = os.path.abspath(url_parts.path)
        store_path = os.path.dirname(os.path.dirname(rdb_path))
        chunk_info = telstate['chunk_info']
        chunk_info = _ensure_prefix_is_set(chunk_info, telstate)
        vis_prefix = chunk_info[array]['prefix']
        data_path = os.path.join(store_path, vis_prefix)
        if os.path.isdir(data_path):
            return NpyFileChunkStore(store_path)
    return S3ChunkStore(telstate['s3_endpoint_url'], **kwargs)


def _upgrade_flags(chunk_info, telstate, capture_block_id, stream_name):
    """Look for associated flag streams and override chunk_info to use them."""
    try:
        archived_streams = telstate['sdp_archived_streams']
    except KeyError as e:
        logger.debug('No additional flag capture streams found: %s', e)
        return chunk_info
    for s in archived_streams:
        telstate_cs = view_capture_stream(telstate, capture_block_id, s)
        if telstate_cs.get('stream_type') != 'sdp.flags' or \
           stream_name not in telstate_cs['src_streams']:
            continue
        # Look for chunk metadata in appropriate capture_stream telstate view
        logger.info('Upgrading flags to use %s instead of %s', s, stream_name)
        flags_info = telstate_cs['chunk_info']
        flags_info = _ensure_prefix_is_set(flags_info, telstate_cs)
        chunk_info = _upgrade_chunk_info(chunk_info, flags_info)
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
    capture_block_id : string
        Capture block ID
    stream_name : string
        Name of the L0 stream
    chunk_store : :class:`katdal.ChunkStore` object, optional
        Chunk store for visibility data (the default is no data - metadata only)
    timestamps : array of float, optional
        Visibility timestamps, overriding (or fixing) the ones found in telstate
    url : string, optional
        Location of the telstate source
    upgrade_flags : bool, optional
        Look for associated flag streams and use them if True (default)
    van_vleck : {'off', 'autocorr'}, optional
        Type of Van Vleck (quantisation) correction to perform
    preselect : dict, optional
        Subset of data to select. The keys in the dictionary correspond to the
        keyword arguments of :meth:`.DataSet.select`, but with restrictions:

        - Only ``channels`` and ``dumps`` can be specified.
        - The values must be slices with unit step.
    kwargs : dict, optional
        Extra keyword arguments, typically meant for other methods and ignored

    Raises
    ------
    KeyError
        If telstate lacks critical keys
    IndexError
        If `preselect` does not meet the criteria above.
    """
    def __init__(self, telstate, capture_block_id, stream_name,
                 chunk_store=None, timestamps=None, url='',
                 upgrade_flags=True, van_vleck='off', preselect=None, **kwargs):
        if preselect is None:
            preselect = {}
        unexpected = set(preselect.keys()) - {'channels', 'dumps'}
        if unexpected:
            raise IndexError("'preselect' can only specify 'channels' and 'dumps'")
        for key, idx in preselect.items():
            if not isinstance(idx, slice) or idx.step not in {None, 1}:
                raise IndexError(f'{key} must be a slice with unit step')

        self.telstate = TelstateToStr(telstate)
        # Collect sensors
        sensors = {}
        for key in telstate.keys():
            if telstate.key_type(key) == katsdptelstate.KeyType.MUTABLE:
                sensor_name = _shorten_key(telstate, key)
                if sensor_name:
                    sensors[sensor_name] = TelstateSensorGetter(telstate, key)
        metadata = AttrsSensors(telstate, sensors)
        if chunk_store is not None or timestamps is None:
            chunk_info = telstate['chunk_info']
            chunk_info = _ensure_prefix_is_set(chunk_info, telstate)
            if upgrade_flags:
                chunk_info = _upgrade_flags(chunk_info, telstate, capture_block_id, stream_name)
            chunk_info = _align_chunk_info(chunk_info)

        if chunk_store is None:
            data = None
        else:
            need_weights_power_scale = telstate.get('need_weights_power_scale', False)
            if preselect:
                index = (preselect.get('dumps', np.s_[:]), preselect.get('channels', np.s_[:]))
            else:
                index = ()
            data = ChunkStoreVisFlagsWeights(chunk_store, chunk_info,
                                             corrprods=telstate['bls_ordering'],
                                             stored_weights_are_scaled=not need_weights_power_scale,
                                             van_vleck=van_vleck,
                                             index=index)

        if timestamps is None:
            # Synthesise timestamps from the relevant telstate bits
            t0 = telstate['sync_time'] + telstate['first_timestamp']
            int_time = telstate['int_time']
            n_dumps = chunk_info['correlator_data']['shape'][0]
            timestamps = t0 + np.arange(n_dumps) * int_time
        if 'dumps' in preselect:
            timestamps = timestamps[preselect['dumps']]
        # Metadata and timestamps with or without data
        DataSource.__init__(self, metadata, timestamps, data)
        self.capture_block_id = capture_block_id
        self.stream_name = stream_name
        self.url = url
        self.name = f'{capture_block_id}_{stream_name}'

    @classmethod
    def from_url(cls, url, chunk_store='auto', **kwargs):
        """Construct TelstateDataSource from URL or RDB filename.

        The following URL styles are supported:

          - Local RDB filename (no scheme): '1556574656/1556574656_sdp_l0.rdb'
          - Archive: 'https://archive/1556574656/1556574656_sdp_l0.rdb?token=<>'
          - Redis server: 'redis://cal5.sdp.mkat.karoo.kat.ac.za:31852'

        Parameters
        ----------
        url : string
            URL or RDB filename serving as entry point to data set
        chunk_store : :class:`katdal.ChunkStore` object, optional
            Chunk store for visibility data (obtained automatically by default,
            or set to None for metadata-only data set)
        kwargs : dict, optional
            Extra keyword arguments passed to init, telstate view, chunk store init
        """
        url_parts = parse_url_or_path(url)
        # Merge key-value pairs from URL query with keyword arguments
        # of function (the latter takes precedence)
        url_kwargs = dict(urllib.parse.parse_qsl(url_parts.query))
        url_kwargs.update(kwargs)
        kwargs = url_kwargs
        # Extract Redis database number if provided
        db = int(kwargs.pop('db', '0'))
        if url_parts.scheme == 'file':
            # RDB dump file
            telstate = katsdptelstate.TelescopeState()
            try:
                telstate.load_from_file(url_parts.path)
            except (OSError, katsdptelstate.RdbParseError) as e:
                raise DataSourceNotFound(str(e)) from e
        elif url_parts.scheme == 'redis':
            # Redis server
            try:
                telstate = katsdptelstate.TelescopeState(url_parts.netloc, db)
            except katsdptelstate.ConnectionError as e:
                raise DataSourceNotFound(str(e)) from e
        elif url_parts.scheme in {'http', 'https'}:
            # Treat URL prefix as an S3 object store (with auth info in kwargs)
            store_url = urllib.parse.urljoin(url, '..')
            # Strip off parameters, query strings and fragments to get basic URL
            rdb_url = urllib.parse.urlunparse(
                (url_parts.scheme, url_parts.netloc, url_parts.path, '', '', ''))
            telstate = katsdptelstate.TelescopeState()
            try:
                rdb_store = S3ChunkStore(store_url, **kwargs)
                with rdb_store.request('GET', rdb_url) as response:
                    telstate.load_from_file(io.BytesIO(response.content))
            except ChunkStoreError as e:
                raise DataSourceNotFound(str(e)) from e
            # If the RDB file is opened via archive URL, use that URL and
            # corresponding S3 credentials or token to access the chunk store
            if chunk_store == 'auto' and not kwargs.get('s3_endpoint_url'):
                chunk_store = rdb_store
        else:
            raise DataSourceNotFound(f"Unknown URL scheme '{url_parts.scheme}' - "
                                     'telstate expects file, redis, or http(s)')
        telstate, capture_block_id, stream_name = view_l0_capture_stream(telstate, **kwargs)
        if chunk_store == 'auto':
            chunk_store = infer_chunk_store(url_parts, telstate, **kwargs)
        # Remove these from kwargs since they have already been extracted by view_l0_capture_stream
        kwargs.pop('capture_block_id', None)
        kwargs.pop('stream_name', None)
        return cls(telstate, capture_block_id, stream_name, chunk_store,
                   url=url_parts.geturl(), **kwargs)


def open_data_source(url, **kwargs):
    """Construct the data source described by the given URL."""
    try:
        return TelstateDataSource.from_url(url, **kwargs)
    except DataSourceNotFound as e:
        # Amend the error message for the case of an IP address without scheme
        url_parts = urllib.parse.urlparse(url)
        if not url_parts.scheme and not os.path.isfile(url_parts.path):
            raise DataSourceNotFound(f'{e} (add a URL scheme if {url_parts.path!r} '
                                     'is not meant to be a file)') from e
        raise
