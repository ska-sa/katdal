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
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import object
import urllib.parse
import os
import logging
import itertools
from collections import defaultdict, Mapping

import katsdptelstate
import numpy as np
import dask.array as da
from dask.array.rechunk import intersect_chunks
import numba

from .sensordata import TelstateSensorData, TelstateToStr
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


def corrprod_to_autocorr(corrprods):
    """Find the autocorrelation indices of correlation products.

    Parameters
    ----------
    corrprods : sequence of 2-tuples or ndarray
        Input labels of the correlation products

    Returns
    -------
    auto_indices : np.ndarray
        The indices in corrprods that correspond to auto-correlations
    index1, index2 : np.ndarray
        Lists of the same length as corrprods, containing the indices within
        `auto_indices` referring to the first and second corresponding
        autocorrelations.

    Raises
    ------
    KeyError
        If any of the autocorrelations are missing
    """
    auto_indices = []
    auto_lookup = {}
    for i, baseline in enumerate(corrprods):
        if baseline[0] == baseline[1]:
            auto_lookup[baseline[0]] = len(auto_indices)
            auto_indices.append(i)
    index1 = [auto_lookup[a] for (a, b) in corrprods]
    index2 = [auto_lookup[b] for (a, b) in corrprods]
    return np.array(auto_indices), np.array(index1), np.array(index2)


@numba.jit(nopython=True, nogil=True)
def weight_power_scale(block, auto_indices, index1, index2, out=None, tmp=None):
    """Compute weight scale from visibility data.

    This function is designed to be usable with
    :func:`dask.array.map_blocks`.

    Parameters
    ----------
    block : np.ndarray
        Chunk of visibility data, with dimensions time, frequency, baseline
        (or any two dimensions then baseline). It must contain all the
        baselines of a stream.
    auto_indices, index1, index2 : np.ndarray
        Arrays returned by :func:`corrprod_to_autocrr`
    out : np.ndarray, optional
        If specified, the output array, with same shape as `block` and dtype ``np.float32``
    tmp : np.ndarray, optional
        If specified, an array to use as internal scratch space (useful to
        reuse the space across multiple calls). It must have the same shape
        as `auto_indices` and dtype ``np.float32``.
    """
    auto_scale = np.empty(len(auto_indices), np.float32) if tmp is None else tmp
    power_scale = np.empty(block.shape, np.float32) if out is None else out
    bad_weight = np.float32(2.0**-32)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            for k in range(len(auto_indices)):
                # Have to force to an array here, because standard Python
                # division raises ZeroDivisionError on divide by zero but we
                # want to get IEEE semantics (Inf/NaN).
                auto_scale[k] = 1 / np.array(block[i, j, auto_indices[k]].real)
            for k in range(block.shape[2]):
                p = auto_scale[index1[k]] * auto_scale[index2[k]]
                # If either or both of the autocorrelations has zero power then
                # there is likely something wrong with the system. Set the
                # weight to very close to zero (not actually zero, since that
                # can cause divide-by-zero problems downstream).
                if not np.isfinite(p):
                    p = bad_weight
                power_scale[i, j, k] = p
    return power_scale


class ChunkStoreVisFlagsWeights(VisFlagsWeights):
    """Correlator data stored in a chunk store.

    Parameters
    ----------
    store : :class:`ChunkStore` object
        Chunk store
    chunk_info : dict mapping array name to info dict
        Dict specifying prefix, dtype, shape and chunks per array
    corrprods : sequence of 2-tuples of input labels
        Correlation products. If given, the weights for baseline (inp1, inp2)
        will be divided by the square root of the product of the corresponding
        autocorrelations vis[inp1,inp1] and vis[inp2,inp2].
    """
    def __init__(self, store, chunk_info, corrprods):
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
            for has, pieces in zip(has_array.flat, intersections):
                if not has:
                    for piece in pieces:
                        chunk_idx, slices = zip(*piece)
                        lost[chunk_idx].append(slices)
        flags = da.map_blocks(_apply_data_lost, darray['flags'], dtype=np.uint8,
                              name=flags_raw_name, lost=lost)
        # Combine low-resolution weights and high-resolution weights_channel
        weights = darray['weights'] * darray['weights_channel'][..., np.newaxis]
        # Scale weights according to power
        if corrprods is not None:
            assert len(corrprods) == vis.shape[2]
            # Ensure that we have only a single chunk on the baseline axis.
            if len(vis.chunks[2]) > 1:
                vis = vis.rechunk({2: vis.shape[2]})
            auto_indices, index1, index2 = corrprod_to_autocorr(corrprods)
            power_scale = da.map_blocks(weight_power_scale, vis, dtype=np.float32,
                                        auto_indices=auto_indices, index1=index1, index2=index2)
            weights *= power_scale

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
        capture_stream = telstate.SEPARATOR.join((capture_block_id, stream))
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
        except KeyError:
            raise ValueError('No capture block ID found in telstate - '
                             'please specify it manually')
    # Detect the captured stream
    if not stream_name:
        try:
            stream_name = telstate['stream_name']
        except KeyError:
            raise ValueError('No captured stream found in telstate - '
                             'please specify the stream manually')
    # Build the view
    telstate = view_capture_stream(telstate, capture_block_id, stream_name)
    # Check the stream type
    stream_type = telstate.get('stream_type', 'unknown')
    expected_type = 'sdp.vis'
    if stream_type != expected_type:
        raise ValueError("Found stream {!r} but it has the wrong type {!r},"
                         " expected {!r}".format(stream_name, stream_type,
                                                 expected_type))
    logger.info('Using capture block %r and stream %r',
                capture_block_id, stream_name)
    return telstate


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
        n_original_dumps = original_info['shape'][0]
        n_improved_dumps = improved_info['shape'][0]
        if n_improved_dumps != n_original_dumps:
            logger.debug("Original '%s' array has %d dumps while improved "
                         "version has %d - forcing it to %d dumps", key,
                         n_original_dumps, n_improved_dumps, n_original_dumps)
            chunks = improved_info['chunks']
            time_chunks = chunks[0]
            # Extend/truncate improved version to match original number of dumps
            if n_improved_dumps < n_original_dumps:
                time_chunks += (n_original_dumps - n_improved_dumps) * (1,)
            else:
                for n, total in enumerate(np.cumsum((0,) + time_chunks)):
                    if total >= n_original_dumps:
                        time_chunks = time_chunks[:n]
                        break
            improved_info['chunks'] = (time_chunks,) + chunks[1:]
            shape = improved_info['shape']
            improved_info['shape'] = (sum(time_chunks),) + shape[1:]
        if improved_info['shape'] != original_info['shape']:
            raise ValueError("Original '{}' array has shape {} while improved"
                             "version has shape {}, even after fixing dumps"
                             .format(key, original_info['shape'],
                                     improved_info['shape']))
        chunk_info[key] = improved_info
    return chunk_info


def _infer_chunk_store(url_parts, telstate, npy_store_path=None,
                       s3_endpoint_url=None, **kwargs):
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
        capture_block_id = telstate['capture_block_id']
        stream_name = telstate['stream_name']
    except KeyError as e:
        logger.debug('No additional flag capture streams found: %s', e)
        return chunk_info
    for s in archived_streams:
        telstate_cs = view_capture_stream(telstate, capture_block_id, s)
        if telstate_cs['stream_type'] != 'sdp.flags' or \
           stream_name not in telstate_cs['src_streams']:
            continue
        # Look for chunk metadata in appropriate capture_stream telstate view
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
        self.telstate = TelstateToStr(telstate)
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
            if telstate.get('need_weights_power_scale', False):
                corrprods = telstate['bls_ordering']
            else:
                corrprods = None
            chunk_info = _ensure_prefix_is_set(chunk_info, telstate)
            chunk_info = _upgrade_flags(chunk_info, telstate)
            data = ChunkStoreVisFlagsWeights(chunk_store, chunk_info, corrprods)
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
        url_parts = urllib.parse.urlparse(url, scheme='file')
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
            except OSError as err:
                raise DataSourceNotFound(str(err))
        elif url_parts.scheme == 'redis':
            # Redis server
            try:
                telstate = katsdptelstate.TelescopeState(url_parts.netloc, db)
            except katsdptelstate.ConnectionError as e:
                raise DataSourceNotFound(str(e))
        telstate = view_l0_capture_stream(telstate, **kwargs)
        if chunk_store == 'auto':
            chunk_store = _infer_chunk_store(url_parts, telstate, **kwargs)
        return cls(telstate, chunk_store, source_name=url_parts.geturl())


def open_data_source(url, **kwargs):
    """Construct the data source described by the given URL."""
    try:
        return TelstateDataSource.from_url(url, **kwargs)
    except DataSourceNotFound as err:
        # Amend the error message for the case of an IP address without scheme
        url_parts = urllib.parse.urlparse(url, scheme='file')
        if url_parts.scheme == 'file' and not os.path.isfile(url_parts.path):
            raise DataSourceNotFound(
                '{} (add a URL scheme if {!r} is not meant to be a file)'
                .format(err, url_parts.path))
