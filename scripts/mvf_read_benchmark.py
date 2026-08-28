#!/usr/bin/env python

################################################################################
# Copyright (c) 2018-2021,2023, National Research Foundation (SARAO)
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

import argparse
import logging
import time

import dask
import numpy as np

import katdal
from katdal.lazy_indexer import DaskLazyIndexer

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--time', type=int, default=10, help='Number of times to read per batch')
parser.add_argument('--channels', type=int, help='Number of channels to read')
parser.add_argument('--dumps', type=int, help='Number of times to read')
parser.add_argument('--joint', action='store_true', help='Load vis, weights, flags together')
parser.add_argument('--applycal', help='Calibration solutions to apply')
parser.add_argument('--workers', type=int, help='Number of dask workers')
parser.add_argument('--blperchunk', type=int, help='Adjust chunks to specified number of baselines per chunk')
parser.add_argument('--chperchunk', type=int, help='Adjust chunks to specified number of channels per chunk')

args = parser.parse_args()

logging.basicConfig(level='INFO', format='%(asctime)s [%(levelname)s] %(message)s')
if args.workers is not None:
    dask.config.set(num_workers=args.workers)
logging.info('Starting')
kwargs = {}
if args.applycal is not None:
    kwargs['applycal'] = args.applycal
f = katdal.open(args.filename, **kwargs)
logging.info('File loaded, shape %s', f.shape)
if args.channels:
    f.select(channels=np.s_[:args.channels])
if args.dumps:
    f.select(dumps=np.s_[:args.dumps])
# Trigger creation of the dask graphs, population of sensor cache for applycal etc
_ = (f.vis[0, 0, 0], f.weights[0, 0, 0], f.flags[0, 0, 0])
nblc = min(args.blperchunk if args.blperchunk else f.vis.dataset.chunksize[0], 
           f.vis.dataset.shape[0])
nch = min(args.chperchunk if args.chperchunk else f.vis.dataset.chunksize[1],
          f.vis.dataset.shape[1])
cs = f.vis.dataset.chunksize
cs = tuple([nblc, nch, cs[2]])
f.vis.dataset.rechunk(cs)
f.weights.dataset.rechunk(cs)
f.flags.dataset.rechunk(cs)
csMB = np.prod(tuple([*cs[0:2], args.time])) * (f.vis.dataset.nbytes // f.vis.dataset.size) / 1024.0**2 
visshpGB = f.vis.dataset.nbytes / 1024.0**3
logging.info(f'Selected visibility chunk size {csMB:.2f} MiB of '
             f'total selection size {visshpGB:.2f} GiB')
logging.info('Selection complete')
chunk_sizeB = f.vis.dataset.nbytes // f.vis.dataset.size + \
              f.weights.dataset.nbytes // f.weights.dataset.size + \
              f.flags.dataset.nbytes // f.weights.dataset.size

start = time.time()
last_time = start
for st in range(0, f.shape[0], args.time):
    et = st + args.time
    if args.joint:
        vis, weights, flags = DaskLazyIndexer.get([f.vis, f.weights, f.flags], np.s_[st:et])
    else:
        vis = f.vis[st:et]
        weights = f.weights[st:et]
        flags = f.flags[st:et]
    current_time = time.time()
    elapsed = current_time - last_time
    last_time = current_time
    size = np.prod(vis.shape) * chunk_sizeB / 1024.**2
    logging.info('Loaded %d dumps (%.3f MiB/s)', vis.shape[0], size / elapsed)
size = np.prod(f.shape) * chunk_sizeB / 1024.**2
elapsed = time.time() - start
logging.info('Loaded %d bytes in %.3f s (%.3f MiB/s)', size, elapsed, size / elapsed)
