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
# Make a local copy of an MVF4 dataset using rclone.
#
# Ludwig Schwardt
# 16 May 2023
#

import argparse
import os
import subprocess

import dask

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, urlunparse, parse_qsl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Dataset URL')
    parser.add_argument('dest', type=Path, help='Output directory')
    parser.add_argument('--workers', type=int, default=8 * dask.system.CPU_COUNT,
                        help='Number of rclone workers for parallel I/O [%(default)s]')
    args = parser.parse_args()
    return args


def rclone_copy(bucket, dest, endpoint, token=None, workers=4):
    env = os.environ.copy()
    # Ignore config file as we will configure rclone with environment variables instead
    env['RCLONE_CONFIG'] = ''
    env['RCLONE_S3_PROVIDER'] = 'Ceph'
    env['RCLONE_PROGRESS'] = '1'
    env['RCLONE_CACHE_WORKERS'] = str(workers)
    env['RCLONE_CONFIG_ARCHIVE_TYPE'] = 's3'
    env['RCLONE_CONFIG_ARCHIVE_ENDPOINT'] = endpoint
    if token:
         env['RCLONE_HEADER'] = f'Authorization: Bearer {token}'
    rclone_args = ['rclone', 'copy', f'archive:{bucket}', dest]
    subprocess.run(rclone_args, check=True, env=env)


def main():
    args = parse_args()
    url_parts = urlparse(args.source)
    cbid = PurePosixPath(url_parts.path).parts[1]
    endpoint = urlunparse((url_parts.scheme, url_parts.netloc, '', '', '', ''))
    token = dict(parse_qsl(url_parts.query)).get('token')
    dest = args.dest / cbid
    print(f"\nCopying metadata bucket ({cbid}) to {dest.absolute()} ...")
    rclone_copy(cbid, dest, endpoint, token, args.workers)


if __name__ == '__main__':
    main()
