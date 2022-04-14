#! /usr/bin/env python3

################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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

#
# Make a local copy of an MVF4 dataset, optionally filtering it.
#
# Ludwig Schwardt
# 19 October 2021
#

import argparse
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import katsdptelstate
from katsdptelstate.rdb_writer import RDBWriter
import katdal
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import view_capture_stream
from katdal.lazy_indexer import dask_getitem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Dataset URL (or input RDB file path)')
    parser.add_argument('dest', type=Path, help='Output directory')
    parser.add_argument('--corrprods',
                        help='Select correlation products (kwarg to katdal.DataSet.select). '
                        'Keeps all corrprods by default.')
    parser.add_argument('--workers', type=int, default=8 * dask.system.CPU_COUNT,
                        help='Number of dask workers for parallel I/O [%(default)s]')
    args = parser.parse_args()
    return args


def extra_flag_streams(telstate, capture_block_id, stream_name):
    """Look for associated flag streams and return corresponding telstate views."""
    # This is a simplified version of katdal.datasources._upgrade_flags
    telstate_extra_flags = []
    for s in telstate.get('sdp_archived_streams', []):
        telstate_cs = view_capture_stream(telstate.root(), capture_block_id, s)
        if telstate_cs.get('stream_type') == 'sdp.flags' and \
           stream_name in telstate_cs['src_streams']:
            telstate_extra_flags.append(telstate_cs)
    return telstate_extra_flags


def stream_graphs(telstate, store, corrprod_mask, out_telstate, out_store):
    """Prepare Dask graphs to copy all chunked arrays of a capture stream.

    This returns a list of Dask graphs and also modifies `out_telstate` and
    `out_store`.
    """
    out_n_baselines = corrprod_mask.sum()
    out_chunk_info = {}
    graphs = []
    for array, info in telstate['chunk_info'].items():
        array_name = store.join(info['prefix'], array)
        darray = store.get_dask_array(array_name, info['chunks'], info['dtype'])
        # Filter the correlation products if array has them
        if darray.ndim == 3:
            indices = (slice(None), slice(None), corrprod_mask)
            # Try to turn fancy indexing into slices (works for autocorrs)
            darray = dask_getitem(darray, indices)
            info['chunks'] = info['chunks'][:2] + ((out_n_baselines,),)
            info['shape'] = info['shape'][:2] + (out_n_baselines,)
        out_store.create_array(array_name)
        graphs.append(out_store.put_dask_array(array_name, darray))
        out_chunk_info[array] = info
    out_telstate[telstate.prefixes[0] + 'chunk_info'] = out_chunk_info
    return graphs


def main():
    args = parse_args()

    d = katdal.open(args.source)
    # XXX Simplify this once corrprods can accept slices as advertised
    kwargs = {}
    if args.corrprods is not None:
        kwargs['corrprods'] = args.corrprods
    d.select(**kwargs)

    # Convenience variables
    cbid = d.source.capture_block_id
    stream = d.source.stream_name
    telstate = d.source.telstate
    # XXX Replace private member with public corrprod index member when it exists
    corrprod_mask = d._corrprod_keep
    rdb_filename = PurePosixPath(urlparse(args.source).path).name

    telstate_overrides = katsdptelstate.TelescopeState()
    # Override bls_ordering in telstate (in stream namespace) to match dataset selection
    telstate_overrides.view(stream)['bls_ordering'] = d.corr_products
    telstate_overrides.view(stream)['n_bls'] = len(d.corr_products)
    os.makedirs(args.dest / cbid, exist_ok=True)
    out_store = NpyFileChunkStore(args.dest)
    # Iterate over all stream views, setting up Dask graph for each chunked array
    graphs = []
    for view in [telstate] + extra_flag_streams(telstate, cbid, stream):
        graphs.extend(stream_graphs(view, d.source.data.store, corrprod_mask,
                                    telstate_overrides, out_store))

    # Save original telstate + overrides to new RDB file (without duplicate keys)
    unmodified_keys = set(telstate.keys()) - set(telstate_overrides.keys())
    with RDBWriter(args.dest / cbid / rdb_filename) as rdbw:
        rdbw.save(telstate.backend, unmodified_keys)
        rdbw.save(telstate_overrides.backend)
    # Transfer chunks to final resting place, filtering them along the way
    with ProgressBar():
        errors = da.compute(*graphs, num_workers=args.workers)
    # put_dask_array returns an array with an exception object per chunk
    for array_errors in errors:
        for chunk_error in array_errors.flat:
            if chunk_error is not None:
                raise chunk_error


if __name__ == '__main__':
    main()
