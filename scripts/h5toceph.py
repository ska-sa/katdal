#!/usr/bin/env python

"""Simple script to take an HDF5 file and chunk it into objects,
   and populate a specified (or managed) Redis instance to hold
   telescope state and other metadata.

   The primary use is for testing the coming katdal access layer
   that will wrap up such an object+Redis dataset and provide
   the usual standard access.

   Objects are stored in chunks of all baselines for obj_chunk_size 
   frequencies. The chunking is chosen to produce objects of order 1MB.
   The schema used is as follows:

        <obj_basename>_<ts_index>_<chunk_offset>

   obj_basename: by default the obs start time in integer seconds
   ts_index: integer timestamp index of this object
   chunk_offset: integer start index for this frequency chunk

   The following useful object parameters are stored in telstate:
        obj_basename: as above
        obj_chunk_size: as above
        obj_size: size per object (needed for reads)
        obj_count: estimated object count (set before writing in this case)
        obj_pool: the name of the CEPH pool used    
        obj_ceph_conf: copy of the ceph.conf used to connect to the target ceph cluster
"""

import h5py
import struct
import redis
import katsdptelstate
import rados
import logging
import katsdpservices
import sys
import time
import numpy as np
import shlex
import subprocess
import cPickle
import os
logging.basicConfig()

logger = logging.getLogger('h5toceph')
logger.setLevel(logging.INFO)


def gen_redis_proto(*args):
    proto = ''
    proto += '*' + str(len(args)) + '\r\n'
    for arg in args:
        proto += '$' + str(len(arg)) + '\r\n'
        proto += str(arg) + '\r\n'
    return proto

def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, metavar='FILE', help='h5 file to process.')
    parser.add_argument('--ceph_conf', type=str, default="/etc/ceph/ceph.conf", metavar='CEPHCONF', help='CEPH configuration file used for cluster connect.')
    parser.add_argument('--ts_limit', type=int, default=0, help='Number of timestamps to process. Default is all.')
    parser.add_argument('--pool', type=str, default=None, metavar='POOL', help='CEPH pool to use for object storage.')
    parser.add_argument('--basename', type=str, default=None, metavar='BASENAME', help='Basename to use for object naming. Default is to use file start time.')
    parser.add_argument('--redis', type=str, default=None, help='Redis host to connect to as Telescope State. Default is to start a new local instance.')
    parser.add_argument('--redis_port', type=int, default=6379, help='Port to use when connecting to Redis instance (or creating a new one).')
    parser.add_argument('--redis-only', dest='redis_only', action='store_true', help='Only (re)build Redis DB - no object creation')
    parser.add_argument('--obj_size', type=int, default=20, help='Target obj size as a power of 2. Default: 2**20 (1 MB)')
    args = parser.parse_args()
    if args.file is None:
        parser.error('argument --file is required')
    if args.pool is None:
        parser.error('argument --pool is required')
    if args.basename is None:
        args.basename = args.file.split(".")[0]
    return args

def redis_bulk_str(r_str, host, port):
    bulk_cmd = "redis-cli --pipe -h {} -p {}".format(host, port)
    bulk_redis = subprocess.Popen(shlex.split(bulk_cmd), stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    (retout, reterr) = bulk_redis.communicate(input=r_str)
    if bulk_redis.returncode:
        logger.error("Failed on bulk key insert. Retcode: {}, Stderr: {}, Stdout: {}".format(bulk_redis.returncode, retout, reterr))
        sys.exit()
    logger.debug("Bulk insert r_str of len {} completed: {}".format(len(r_str), retout))


def w_onc(completion): pass
def w_ons(completion): pass

def write_ts(ioctx, ts_index,data,obj_base,freq_chunk):
    total_write = 0
    write_objs = 0
    chans = data.shape[0]
    completions = []
    for x in np.arange(0,chans,freq_chunk):
        to_write = data[x:x+freq_chunk].dumps()
        total_write += len(to_write)
        write_objs += 1
        completions.append(ioctx.aio_write_full('{}_{}_{}'.format(obj_base, ts_index, x),to_write, w_onc, w_ons))
    for c in completions:
        c.wait_for_safe_and_cb()
    return (total_write, write_objs)

def get_freq_chunk(data_shape, target_obj_size=20):
    """ Get a frequency chunking that results in slices into this array
        being as close to 1MB as possible. Baselines are always grouped
        into a single object, so outliers in baseline number may produce
        objects much smaller or larger than 1MB"""
    bytes_per_baseline = data_shape[2] * data_shape[3] * 4
    logger.info("Bytes per baseline: {}".format(bytes_per_baseline))
    channels = data_shape[1]
    for chunk_power in range(int(np.log2(channels)) + 1):
        chunk_size = 2**chunk_power
        if (chunk_size * bytes_per_baseline) >= 2**target_obj_size: break
    real_obj_size = len(np.zeros((chunk_size, data_shape[2], data_shape[3]), dtype=np.float32).dumps())
     # figure out what the real written size of each dumps'ed object will be
    logger.info("Using chunk size {} giving obj size {}".format(chunk_size, real_obj_size))
    return (chunk_size, real_obj_size)

def main():
    args = parse_args()
    try:
        h5_file = h5py.File(args.file)
    except Exception as e:
        logger.error("Failed to open specified HDF5 file. {}".format(e))
        sys.exit()
    try:
        data = h5_file['Data/correlator_data']
    except KeyError:
        logger.error("This does not appear to be a valid MeerKAT HDF5 file")
        sys.exit()

    if args.redis_only:
        logger.warning("Building Redis DB only - no data will be written...")

    if args.redis is None:
        logger.info("Launching local Redis instance")
        launch_cmd = "/usr/bin/redis-server --port {}".format(args.redis_port)
        local_redis = subprocess.Popen(shlex.split(launch_cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        time.sleep(3)
        if local_redis.poll():
            logger.error("Failed to launch local Redis instance, terminating. {}".format(local_redis.communicate()))
            sys.exit()
        logger.info("Local Redis instance launched successfully")
        redis_host = 'localhost'
    else:
        redis_host = args.redis
    ts = katsdptelstate.TelescopeState(endpoint='{}:{}'.format(redis_host, args.redis_port))
    logger.info("Connected to Redis on {}:{}. DB has {} existing keys".format(redis_host, args.redis_port, len(ts.keys())))

    im_count = 0    
    r_str = ""
    for attr in h5_file['TelescopeState'].attrs:
        r_str += gen_redis_proto("SET", attr, h5_file['TelescopeState'].attrs[attr])
        im_count += 1
    redis_bulk_str(r_str, redis_host, args.redis_port)

    d_count = 0
    for dset in h5_file['TelescopeState'].keys():
        i_count = 0
        st = time.time()
        r_str = ""
        d_val = h5_file['TelescopeState'][dset].value
         # much quicker to read it first and then iterate
        for (timestamp, pval) in d_val:
            packed_ts = struct.pack('>d', float(timestamp))
            r_str += gen_redis_proto("ZADD", dset, "0", packed_ts + pval)
            i_count += 1
        bss = time.time()
        redis_bulk_str(r_str, redis_host, args.redis_port)  
        logger.info("Added {} items in {}s to key {}. Bulk insert time: {}".format(i_count, time.time()-st, dset, (time.time() - bss)))
        d_count += 1
    logger.info("Added {} ranged keys to TelescopeState".format(d_count))

    if args.redis_only:
        if args.redis is None:
            logger.warning("Terminating locally launched redis instance")
            local_redis.terminate()
        sys.exit(0)
    cluster = rados.Rados(conffile=args.ceph_conf)
    cluster.connect()
    available_pools = cluster.list_pools()
    if args.pool not in available_pools:
        logger.error("Specified pool {} not available in this cluster ({})".format(args.pool, available_pools))
        sys.exit()
    ioctx = cluster.open_ioctx(args.pool)
    pool_stats = ioctx.get_stats()
    logger.info("Connected to pool {}. Currently holds {} objects totalling {} GB".format(args.pool, pool_stats['num_objects'], pool_stats['num_bytes']/2**30))
    
    (freq_chunk, obj_size) = get_freq_chunk(data.shape, args.obj_size)
    if args.ts_limit > 0: ts_limit = args.ts_limit
    else: ts_limit = data.shape[0]
    obj_count_per_ts = data.shape[1] // freq_chunk
    obj_count = ts_limit * obj_count_per_ts

    ts.add("obj_basename", args.basename)
    ts.add("obj_chunk_size", freq_chunk)
    ts.add("obj_size", obj_size)
    ts.add("obj_count", obj_count)
    ts.add("obj_pool", args.pool)
    f = open(args.ceph_conf,"r")
    ts.add("obj_ceph_conf", f.readlines())
    f.close()
    logger.info("Inserted obj schema metadata into telstate")

    logger.info("Processing {} timestamps into {} objects of size {} with basename {}".format(ts_limit, obj_count, obj_size, args.basename))
    ts_index = 0
    for ts_slice in data:
        st = time.time()
        (bytes_written, objs_written) = write_ts(ioctx, ts_index, ts_slice, args.basename, freq_chunk)
        et = time.time() - st
        if objs_written == obj_count_per_ts and bytes_written == (obj_size * obj_count_per_ts):
            logger.info("Stored ts index {} in {}s ({} objects totalling {}MBps)".format(ts_index, et, obj_count_per_ts, obj_count_per_ts * obj_size / (1024*1024) / et))
        else:
            logger.error("Failed to full write ts index {}. Wrote {}/{} objects and {}/{} bytes".format(ts_index, objs_written, obj_count_per_ts, bytes_written, obj_size * obj_count_per_ts))
        ts_index += 1
        if ts_index >= ts_limit:
            logger.info("Reached specified ts limit ({}).".format(ts_limit))
            break
    logger.info("Staging complete...")
    if args.redis is None:
        raw_input("You have started a local Redis server. Hit enter to kill this and cleanup.")
        local_redis.terminate()

if __name__ == '__main__':
    main()
