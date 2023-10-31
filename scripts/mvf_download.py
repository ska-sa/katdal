#! /usr/bin/env python3

################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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
# Download an MVF4 dataset using rclone.
#
# Ludwig Schwardt
# 16 May 2023
#

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import dask
import katdal
from katdal.chunkstore import _blocks_ravel
from katdal.datasources import view_capture_stream
from katdal.lazy_indexer import dask_getitem
from packaging import version

# This version is good for file-less config, enabling --config "" and --files-from -
MINIMUM_RCLONE_VERSION = version.Version('1.56')


def parse_args(args=None, namespace=None):
    """Parse script arguments into script-specific ones and ones meant for rclone."""
    parser = argparse.ArgumentParser(
        usage='%(prog)s [-h] [--select JSON] [--workers N] '
              'source dest [rclone options]',
        description='Download MVFv4 dataset (or a subset of chunks) using rclone.',
        epilog='Any extra script options are passed to rclone.',
    )
    parser.add_argument('source', help='Dataset URL')
    parser.add_argument('dest', type=Path, help='Output directory')
    parser.add_argument('--select', type=json.loads, default={},
                        help='Kwargs for DataSet.select as a JSON object')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of rclone threads for parallel I/O [%(default)s]')
    mvf_download_args, rclone_args = parser.parse_known_args(args, namespace)
    rclone_args = [
        '--transfers', str(mvf_download_args.workers),
        '--checkers', str(mvf_download_args.workers + 4)
    ] + rclone_args
    return mvf_download_args, rclone_args


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


def stream_graphs(telstate, store, keep):
    """Prepare Dask graphs to copy all chunked arrays of a capture stream.

    This returns a list of Dask graphs and also modifies `out_telstate` and
    `out_store`.
    """
    all_chunks = defaultdict(list)
    for array, info in telstate['chunk_info'].items():
        darray = store.get_dask_array(
            array, info['chunks'], info['dtype'], errors='dryrun'
        )
        kept_blocks = _blocks_ravel(dask_getitem(darray, keep[:darray.ndim]))
        chunks = sorted(chunk.name + '.npy' for chunk in dask.compute(*kept_blocks))
        all_chunks[info['prefix']].extend(chunks)
    return all_chunks


def has_recent_rclone():
    """Check that rclone is installed and has an appropriate version."""
    try:
        result = subprocess.run(['rclone', 'version'], capture_output=True, check=True)
    except FileNotFoundError:
        print('The rclone tool was not found. Please install at least version '
              f'{MINIMUM_RCLONE_VERSION} (see rclone.org) or check the path.')
    else:
        installed_version = version.parse(result.stdout.split()[1].decode())
        if installed_version >= MINIMUM_RCLONE_VERSION:
            return True
        print(f'Found rclone {installed_version} but the script needs version '
              f'{MINIMUM_RCLONE_VERSION}. See rclone.org for installation options.')
    return False


def rclone_fit_output_to_terminal(args):
    """Reduce rclone output to a single line if it won't fit on terminal."""
    new_args = args.copy()
    # Find the last instances of --transfers and --checkers flags
    reversed_args = list(reversed(new_args))
    n_transfers = int(reversed_args[reversed_args.index('--transfers') - 1])
    n_checkers = int(reversed_args[reversed_args.index('--checkers') - 1])
    if n_transfers + n_checkers + 6 > shutil.get_terminal_size().lines:
        new_args.append('--stats-one-line')
    return new_args


def rclone_copy(endpoint, bucket, dest, args, token=None, files=None):
    """Run 'rclone copy' with appropriate arguments."""
    env = os.environ.copy()
    # Ignore config file as we will configure rclone with environment variables instead
    env['RCLONE_CONFIG'] = ''
    env['RCLONE_CONFIG_ARCHIVE_TYPE'] = 's3'
    env['RCLONE_CONFIG_ARCHIVE_ENDPOINT'] = endpoint
    rclone_args = [
        'rclone', 'copy', f'archive:{bucket}', dest,
        '--s3-provider', 'Ceph',
        '--fast-list',
        '--checksum',
        '--progress',
    ]
    if token:
        rclone_args.extend(['--header', f'Authorization: Bearer {token}'])
    run_kwargs = dict(check=True, env=env)
    if files is not None:
        rclone_args.extend(['--files-from', '-'])
        run_kwargs.update(input='\n'.join(files), text=True)
    # User-supplied arguments can override any of the above args
    rclone_args.extend(args)
    rclone_args = rclone_fit_output_to_terminal(rclone_args)
    subprocess.run(rclone_args, **run_kwargs)  # pylint: disable=subprocess-run-check


def main():
    """Main routine of mvf_download script."""
    args, rclone_args = parse_args()
    if not has_recent_rclone():
        return False
    url_parts = urlparse(args.source)
    *_, cbid, rdb_filename = PurePosixPath(url_parts.path).parts
    endpoint = urlunparse((url_parts.scheme, url_parts.netloc, '', '', '', ''))
    token = dict(parse_qsl(url_parts.query)).get('token')
    meta_path = args.dest / cbid
    print(f"\nDownloading metadata bucket ({cbid}) to {meta_path.absolute()} ...")
    rclone_copy(endpoint, cbid, meta_path, rclone_args, token)

    query_params = {'s3_endpoint_url': endpoint}
    if token:
        query_params['token'] = token
    query = urlencode(query_params)
    rdb_path = (meta_path / rdb_filename).absolute()
    local_rdb = urlunparse(('file', '', str(rdb_path), '', query, ''))
    print(f"Opening local RDB file: {local_rdb}")
    d = katdal.open(local_rdb)
    cbid = d.source.capture_block_id
    stream = d.source.stream_name
    telstate = d.source.telstate
    store = d.source.data.store
    d.select(**args.select)
    # Iterate over all stream views, collecting chunk names for each chunked array
    chunks = {}
    for view in [telstate] + extra_flag_streams(telstate, cbid, stream):
        chunks.update(stream_graphs(view, store, d.vis.keep))
    for bucket, files in chunks.items():
        bucket_path = args.dest / bucket
        n_chunks = len(files)
        if not args.select:
            n_chunks = f'all {n_chunks}'
            files = None
        print(f"\nDownloading {n_chunks} chunks from data bucket {bucket} "
              f"to {bucket_path.absolute()} ...")
        rclone_copy(endpoint, bucket, bucket_path, rclone_args, token, files)
    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)
