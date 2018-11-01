#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
from builtins import range
import argparse
import logging
import time

import katdal
from katdal.lazy_indexer import DaskLazyIndexer
import dask.array as da
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--time', type=int, default=10, help='Number of times to read per batch')
parser.add_argument('--channels', type=int, help='Number of channels to read')
parser.add_argument('--dumps', type=int, help='Number of times to read')
parser.add_argument('--joint', action='store_true', help='Load vis, weights, flags together')
args = parser.parse_args()

logging.basicConfig(level='INFO', format='%(asctime)s [%(levelname)s] %(message)s')
logging.info('Starting')
f = katdal.open(args.filename)
logging.info('File loaded, shape %s', f.shape)
if args.channels:
    f.select(channels=np.s_[:args.channels])
if args.dumps:
    f.select(dumps=np.s_[:args.dumps])
start = time.time()
for st in range(0, f.shape[0], args.time):
    et = st + args.time
    if args.joint:
        vis, weights, flags = DaskLazyIndexer.get([f.vis, f.weights, f.flags], np.s_[st:et])
    else:
        vis = f.vis[st:et]
        weights = f.weights[st:et]
        flags = f.flags[st:et]
    logging.info('Loaded %d dumps', vis.shape[0])
size = np.product(f.shape) * 10
elapsed = time.time() - start
logging.info('Loaded %d bytes in %.3f s (%.3f MB/s)', size, elapsed, size / elapsed / 1e6)
