#!/usr/bin/env python

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
logging.info('Selection complete')
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
    size = np.product(vis.shape) * 10
    logging.info('Loaded %d dumps (%.3f MB/s)', vis.shape[0], size / elapsed / 1e6)
size = np.product(f.shape) * 10
elapsed = time.time() - start
logging.info('Loaded %d bytes in %.3f s (%.3f MB/s)', size, elapsed, size / elapsed / 1e6)
