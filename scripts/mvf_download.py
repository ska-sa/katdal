#! /usr/bin/env python3

################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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
from katdal.lazy_indexer import dask_getitem
from packaging import version

# This version is good for file-less config, enabling --config "" and --files-from -
MINIMUM_RCLONE_VERSION = version.Version('1.56')
DESCRIPTION = """
Download MVFv4 dataset (or a subset of chunks) from S3 to disk using rclone.

You need rclone (https://rclone.org/downloads/) if it is not on your system.
It is a single executable file that you could download to your user account.
Just ensure that it is on your PATH; no need to configure it any further.

Run the script like this:

  mvf_download.py https://archive/1698676533/1698676533_sdp_l0.full.rdb?token=<> dest

Data will appear in three subdirectories in the specified output directory as

  dest/1698676533/...
  dest/1698676533-sdp-l0/...
  dest/1698676533-sdp-l1-flags/...

Open the local dataset like this:

  d = katdal.open("dest/1698676533/1698676533_sdp_l0.full.rdb")

If the script crashes or you terminate it, you can just run it again and
it will carry on, fixing any half-downloaded chunks along the way. If it
completes, you can be sure that all your data is safely downloaded.

BONUS: you can even copy just parts of the data (e.g. the tracks and not the
slews). This works as long as your selection picks out a subset of the chunks
but leaves the chunks themselves intact. It is well suited for time-based
selections.

Because MeerKAT data is chunked first in time and then in frequency, but not
in correlation product, this won't help to select a subset of antennas or
baselines or autocorrelations, as that would require breaking up chunks into
smaller chunks. For that, consider using the mvf_copy.py script instead, which
is also useful if you want to copy a subset of data from disk to disk.

Note that you have to pass a JSON object (which resembles a Python dict) as a
string to the --select argument. The "dict" contains keyword arguments meant
for the DataSet.select() method. It's important to note that the strings in
the dict need double quotes (") while the entire string has to be encapsulated
in single quotes ('). Some examples:

  mvf_download.py url directory --select='{"scans": "track"}'
  mvf_download.py url directory --select='{"scans": 1}'
  mvf_download.py url directory --select='{"scans": [0, 1, 2]}'
  mvf_download.py url directory --select='{"targets": "J1939-6342"}'

The chunks that are not copied will appear as "lost" data in the downloaded
dataset, but that is fine. If you apply the same selection, you won't see it.
"""


def parse_args(args=None, namespace=None):
    """Parse script arguments into script-specific ones and ones meant for rclone."""
    parser = argparse.ArgumentParser(
        usage='%(prog)s [-h] [--select JSON] [--workers N] '
              'source dest [rclone options]',
        description=DESCRIPTION,
        epilog='Any extra script options are passed to rclone.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('source', help='Dataset URL (including token if needed)')
    parser.add_argument('dest', type=Path, help='Output directory')
    parser.add_argument('--select', type=json.loads, default={},
                        help='Kwargs for katdal.DataSet.select as a JSON object')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of rclone threads for parallel I/O [%(default)s]')
    mvf_download_args, rclone_args = parser.parse_known_args(args, namespace)
    rclone_args = [
        '--transfers', str(mvf_download_args.workers),
        '--checkers', str(mvf_download_args.workers + 4)
    ] + rclone_args
    return mvf_download_args, rclone_args


def chunk_names(vfw, keep):
    """Names of chunks covered by selection `keep` in all storage arrays in `vfw`."""
    all_chunks = defaultdict(list)
    for array, info in vfw.chunk_info.items():
        darray = vfw.store.get_dask_array(
            array,
            info['chunks'],
            info['dtype'],
            index=vfw.preselect_index,
            errors='dryrun',
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
    # Find last instances of --transfers and --checkers flags (guaranteed one of each)
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfers', action='append', type=int)
    parser.add_argument('--checkers', action='append', type=int)
    n, _ = parser.parse_known_args([str(arg) for arg in new_args])
    if n.transfers[-1] + n.checkers[-1] + 6 > shutil.get_terminal_size().lines:
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
    d.select(**args.select)
    # Collect names of chunks covered by selection in each chunked storage array
    chunks = chunk_names(d.source.data, d.vis.keep)
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
