#!/usr/bin/env python

"""Rechunk an existing MVF dataset"""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: 402
from builtins import object

import sys
import os
import re
import argparse
import multiprocessing
import urllib.parse

from katsdptelstate.rdb_writer import RDBWriter
import numpy as np
import dask
import dask.array as da

from katdal.chunkstore import ChunkStoreError
from katdal.chunkstore_s3 import S3ChunkStore
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import TelstateDataSource, view_capture_stream, infer_chunk_store
from katdal.flags import DATA_LOST
from katdal.applycal import from_block_function    # TODO: get from dask once available there


class RechunkSpec(object):
    def __init__(self, arg):
        match = re.match(r'^([A-Za-z0-9_]+)/([A-Za-z0-9_]+):(\d+),(\d+)', arg)
        if not match:
            raise ValueError('Could not parse {!r}'.format(arg))
        self.stream = match.group(1)
        self.array = match.group(2)
        self.time = int(match.group(3))
        self.freq = int(match.group(4))
        if self.time <= 0 or self.freq <= 0:
            raise ValueError('Chunk sizes must be positive')


class Array(object):
    def __init__(self, stream_name, array_name, store, chunk_info):
        self.stream_name = stream_name
        self.array_name = array_name
        self.chunk_info = chunk_info
        self.store = store
        full_name = store.join(chunk_info['prefix'], array_name)
        shape = chunk_info['shape']
        chunks = chunk_info['chunks']
        dtype = chunk_info['dtype']
        self.data = store.get_dask_array(full_name, chunks, dtype)
        self.has_data = store.has_array(full_name, chunks, dtype)
        self.lost_flags = from_block_function(
            self._make_lost, shape=shape, chunks=chunks, dtype=np.uint8,
            name='lost-flags-{}-{}'.format(self.stream_name, self.array_name))

    def _make_lost(self, block_info):
        loc = block_info['array-location']
        shape = [l[1] - l[0] for l in loc]
        if self.has_data[block_info['chunk-location']]:
            return np.zeros(shape, np.uint8)
        else:
            return np.full(shape, DATA_LOST, np.uint8)


def get_chunk_store(source, telstate, array):
    """A wrapper around katdal.datasources.infer_chunk_store.

    It has a simpler interface, taking an URL rather than url_parts and kwargs.
    """
    url_parts = urllib.parse.urlparse(source, scheme='file')
    kwargs = dict(urllib.parse.parse_qsl(url_parts.query))
    return infer_chunk_store(url_parts, telstate, array=array, **kwargs)


def comma_list(value):
    return ','.split(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rechunk a single capture block. For each array within each stream, '
        'a new chunking scheme may be specified. A chunking scheme is '
        'specified as the number of dumps and channels per chunk.')
    parser.add_argument('--workers', type=int, default=8*multiprocessing.cpu_count(),
                        help='Number of dask workers I/O [%(default)s]')
    parser.add_argument('--streams', type=comma_list, metavar='STREAM,STREAM',
                        help='Streams to copy [all]')
    parser.add_argument('--s3-endpoint-url', help='URL where rechunked data will be uploaded')
    parser.add_argument('--new-prefix', help='Replacement for capture block ID in output bucket names')
    parser.add_argument('source', help='Input .rdb file')
    parser.add_argument('dest', help='Output directory')
    parser.add_argument('spec', nargs='*', default=[], type=RechunkSpec,
                        metavar='STREAM/ARRAY:TIME,FREQ', help='New chunk specification')
    args = parser.parse_args()
    return args


def get_streams(telstate, streams):
    """Determine streams to copy based on what the user asked for"""
    archived_streams = telstate.get('sdp_archived_streams', [])
    if not archived_streams:
        raise RuntimeError('Source dataset does not contain any streams')
    if streams is None:
        streams = archived_streams
    else:
        for stream in streams:
            if stream not in archived_streams:
                raise RuntimeError('Stream {!r} is not known (should be one of {})'
                                   .format(stream, ', '.join(archived_streams)))

    return streams


def main():
    args = parse_args()
    dask.config.set(num_workers=args.workers)

    # Lightweight open with no data - just to create telstate and identify the CBID
    ds = TelstateDataSource.from_url(args.source, upgrade_flags=False, chunk_store=None)
    # View the CBID, but not any specific stream
    cbid = ds.capture_block_id
    telstate = ds.telstate.root().view(cbid)
    streams = get_streams(telstate, args.streams)

    # Find all arrays in the selected streams, and also ensure we're not
    # trying to write things back on top of an existing dataset.
    arrays = {}
    for stream_name in streams:
        sts = view_capture_stream(telstate, cbid, stream_name)
        try:
            chunk_info = sts['chunk_info']
        except KeyError as exc:
            raise RuntimeError('Could not get chunk info for {!r}: {}'.format(stream_name, exc))
        for array_name, array_info in chunk_info.items():
            if args.new_prefix is not None:
                array_info['prefix'] = args.new_prefix + '-' + stream_name.replace('_', '-')
            prefix = array_info['prefix']
            path = os.path.join(args.dest, prefix)
            if os.path.exists(path):
                raise RuntimeError('Directory {!r} already exists'.format(path))
            store = get_chunk_store(args.source, sts, array_name)
            # Older files have dtype as an object that can't be encoded in msgpack
            dtype = np.dtype(array_info['dtype'])
            array_info['dtype'] = np.lib.format.dtype_to_descr(dtype)
            arrays[(stream_name, array_name)] = Array(stream_name, array_name, store, array_info)

    # Apply DATA_LOST bits to the flags arrays. This is a less efficient approach than
    # datasources.py, but much simpler.
    for stream_name in streams:
        flags_array = arrays.get((stream_name, 'flags'))
        if not flags_array:
            continue
        sources = [stream_name]
        sts = view_capture_stream(telstate, cbid, stream_name)
        sources += sts['src_streams']
        for src_stream in sources:
            if src_stream not in streams:
                continue
            src_ts = view_capture_stream(telstate, cbid, src_stream)
            for array_name in src_ts['chunk_info']:
                if array_name == 'flags' and src_stream != stream_name:
                    # Upgraded flags completely replace the source stream's
                    # flags, rather than augmenting them. Thus, data lost in
                    # the source stream has no effect.
                    continue
                lost_flags = arrays[(src_stream, array_name)].lost_flags
                lost_flags = lost_flags.rechunk(flags_array.data.chunks[:lost_flags.ndim])
                # weights_channel doesn't have a baseline axis
                while lost_flags.ndim < flags_array.data.ndim:
                    lost_flags = lost_flags[..., np.newaxis]
                lost_flags = da.broadcast_to(lost_flags, flags_array.data.shape,
                                             chunks=flags_array.data.chunks)
                flags_array.data |= lost_flags

    # Apply the rechunking specs
    for spec in args.spec:
        key = (spec.stream, spec.array)
        if key not in arrays:
            raise RuntimeError('{}/{} is not a known array'.format(spec.stream, spec.array))
        arrays[key].data = arrays[key].data.rechunk({0: spec.time, 1: spec.freq})

    # Write out the new data
    dest_store = NpyFileChunkStore(args.dest)
    stores = []
    for array in arrays.values():
        full_name = dest_store.join(array.chunk_info['prefix'], array.array_name)
        dest_store.create_array(full_name)
        stores.append(dest_store.put_dask_array(full_name, array.data))
        array.chunk_info['chunks'] = array.data.chunks
    stores = da.compute(*stores)
    # put_dask_array returns an array with an exception object per chunk
    for result_set in stores:
        for result in result_set.flat:
            if result is not None:
                raise result

    # Fix up chunk_info for new chunking
    for stream_name in streams:
        sts = view_capture_stream(telstate, cbid, stream_name)
        chunk_info = sts['chunk_info']
        for array_name in chunk_info.keys():
            chunk_info[array_name] = arrays[(stream_name, array_name)].chunk_info
        sts.wrapped.delete('chunk_info')
        sts.wrapped['chunk_info'] = chunk_info
        # s3_endpoint_url is for the old version of the data
        sts.wrapped.delete('s3_endpoint_url')
        if args.s3_endpoint_url is not None:
            sts.wrapped['s3_endpoint_url'] = args.s3_endpoint_url

    # Write updated RDB file
    url_parts = urllib.parse.urlparse(args.source, scheme='file')
    dest_file = os.path.join(args.dest, args.new_prefix or cbid, os.path.basename(url_parts.path))
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    writer = RDBWriter(client=telstate.backend)
    writer.save(dest_file)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ChunkStoreError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
