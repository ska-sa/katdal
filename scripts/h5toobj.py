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

        <obj_basename>_<dump_index>_<chunk_offset>

   obj_basename: by default the obs start time in integer seconds
   dump_index: integer dump index of this object
   chunk_offset: integer start index for this frequency chunk

   The following useful object parameters are stored in telstate:
        obj_basename: as above
        obj_chunk_size: as above
        obj_size: size per object (needed for reads)
        obj_count: estimated object count (set before writing in this case)
        obj_pool: the name of the CEPH pool used
        obj_ceph_conf: copy of ceph.conf used to connect to target ceph cluster
"""

import struct
import logging
import sys
import time
import shlex
import subprocess

import numpy as np
import h5py
import katsdptelstate
import katsdpservices

try:
    import rados
except ImportError:
    # librados (i.e. direct Ceph access) is only really available on Linux
    rados = None


logging.basicConfig()

logger = logging.getLogger('h5toobj')
logger.setLevel(logging.INFO)


def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('file', type=str, default=None, nargs=1,
                        metavar='FILE', help='h5 file to process.')
    parser.add_argument('--ceph-conf', type=str, default="/etc/ceph/ceph.conf",
                        metavar='CEPHCONF',
                        help='CEPH configuration file used for cluster connect.')
    parser.add_argument('--max-dumps', type=int, default=0,
                        help='Number of dumps to process. Default is all.')
    parser.add_argument('--pool', type=str, default=None, metavar='POOL',
                        help='CEPH pool to use for object storage.')
    parser.add_argument('--basename', type=str, default=None, metavar='BASENAME',
                        help='Basename to use for object naming. '
                             'Default is to use file start time.')
    parser.add_argument('--redis', type=str, default=None,
                        help='Redis host to connect to as Telescope State. '
                             'Default is to start a new local instance.')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Port to use when connecting to Redis instance '
                             '(or creating a new one).')
    parser.add_argument('--redis-only', dest='redis_only', action='store_true',
                        help='Only (re)build Redis DB - no object creation')
    parser.add_argument('--obj-size', type=int, default=20,
                        help='Target obj size as a power of 2. '
                             'Default: 2**20 (1 MB)')
    args = parser.parse_args()
    if not rados:
        logger.warning("No Ceph installation found - building Redis DB only")
        args.redis_only = True
    if not args.redis_only and args.pool is None:
        parser.error('argument --pool is required')
    if args.basename is None:
        args.basename = args.file[0].split(".")[0]
    return args


def redis_gen_proto(*args):
    proto = ['*%d\r\n' % (len(args),)]
    proto += ['$%d\r\n%s\r\n' % (len(arg), arg) for arg in args]
    return ''.join(proto)


def redis_bulk_str(r_str, host, port):
    bulk_cmd = "redis-cli --pipe -h {} -p {}".format(host, port)
    bulk_redis = subprocess.Popen(shlex.split(bulk_cmd), stdin=subprocess.PIPE,
                                  stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    (retout, reterr) = bulk_redis.communicate(input=r_str)
    if bulk_redis.returncode:
        logger.error("Failed on bulk key insert. Retcode: %d, Stderr: %s, "
                     "Stdout: %s", bulk_redis.returncode, retout, reterr)
        sys.exit()
    logger.debug("Bulk insert r_str of len %d completed: %s", len(r_str), retout)


def pack_chunk(x):
    return x.data


def unpack_chunk(s, dtype, shape):
    return np.frombuffer(s, dtype).reshape(shape)


def write_dump(ioctx, dump_index, dump_data, obj_base, freq_chunksize):
    total_write = 0
    chans = dump_data.shape[0]
    completions = []
    for obj_count, x in enumerate(np.arange(0, chans, freq_chunksize)):
        to_write = pack_chunk(dump_data[x:x + freq_chunksize])
        total_write += len(to_write)
        c = ioctx.aio_write_full('{}_{}_{}'.format(obj_base, dump_index, x), to_write,
                                 lambda completion: None, lambda completion: None)
        completions.append(c)
    for c in completions:
        c.wait_for_safe_and_cb()
    return total_write, obj_count + 1


def get_freq_chunk(data_shape, target_obj_size=20):
    """Get a frequency chunking that results in slices into this array
       being as close to 1MB as possible. Baselines are always grouped
       into a single object, so outliers in baseline number may produce
       objects much smaller or larger than 1MB"""
    bytes_per_baseline = data_shape[2] * data_shape[3] * 4
    logger.info("Bytes per baseline: %d", bytes_per_baseline)
    channels = data_shape[1]
    for chunk_power in range(int(np.log2(channels)) + 1):
        chunk_size = 2 ** chunk_power
        if (chunk_size * bytes_per_baseline) >= 2 ** target_obj_size:
            break
    real_obj_size = len(np.zeros((chunk_size, data_shape[2], data_shape[3]),
                                 dtype=np.float32).dumps())
     # figure out what the real written size of each dumps'ed object will be
    logger.info("Using chunk size %d giving obj size %d",
                chunk_size, real_obj_size)
    return chunk_size, real_obj_size


def main():
    args = parse_args()
    try:
        h5_file = h5py.File(args.file[0])
    except Exception as e:
        logger.error("Failed to open specified HDF5 file. %s", e)
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
        try:
            launch_cmd = "/usr/bin/redis-server --port {}".format(args.redis_port)
            local_redis = subprocess.Popen(shlex.split(launch_cmd),
                                           stderr=subprocess.PIPE,
                                           stdout=subprocess.PIPE)
        except OSError:
            launch_cmd = "/usr/local/bin/redis-server --port {}".format(args.redis_port)
            local_redis = subprocess.Popen(shlex.split(launch_cmd),
                                           stderr=subprocess.PIPE,
                                           stdout=subprocess.PIPE)
        time.sleep(3)
        if local_redis.poll():
            logger.error("Failed to launch local Redis instance, terminating. %s",
                         local_redis.communicate())
            sys.exit()
        logger.info("Local Redis instance launched successfully")
        redis_host = 'localhost'
    else:
        redis_host = args.redis
    ts = katsdptelstate.TelescopeState(endpoint='{}:{}'.format(redis_host,
                                                               args.redis_port))
    logger.info("Connected to Redis on %s:%d. DB has %d existing keys",
                redis_host, args.redis_port, len(ts.keys()))

    r_str = ""
    for attr in h5_file['TelescopeState'].attrs:
        r_str += redis_gen_proto("SET", attr, h5_file['TelescopeState'].attrs[attr])
    redis_bulk_str(r_str, redis_host, args.redis_port)

    for d_count, dset in enumerate(h5_file['TelescopeState'].keys()):
        st = time.time()
        r_str = ""
        d_val = h5_file['TelescopeState'][dset].value
         # much quicker to read it first and then iterate
        for (timestamp, pval) in d_val:
            packed_ts = struct.pack('>d', float(timestamp))
            r_str += redis_gen_proto("ZADD", str(dset), "0", packed_ts + pval)
        bss = time.time()
        redis_bulk_str(r_str, redis_host, args.redis_port)
        logger.info("Added %d items in %gs to key %s. Bulk insert time: %g",
                    len(d_val), time.time() - st, dset, time.time() - bss)
    logger.info("Added %d ranged keys to TelescopeState", d_count + 1)

    if args.redis_only:
        if args.redis is None:
            logger.warning("Terminating locally launched redis instance "
                           "(also saves telstate to local dump.rdb)")
            try:
                cli_cmd = "/usr/bin/redis-cli -p {} SHUTDOWN SAVE".format(args.redis_port)
                subprocess.call(shlex.split(cli_cmd))
            except OSError:
                cli_cmd = "/usr/local/bin/redis-cli -p {} SHUTDOWN SAVE".format(args.redis_port)
                subprocess.call(shlex.split(cli_cmd))
            local_redis.terminate()
        sys.exit(0)

    cluster = rados.Rados(conffile=args.ceph_conf)
    cluster.connect()
    available_pools = cluster.list_pools()
    if args.pool not in available_pools:
        logger.error("Specified pool %s not available in this cluster (%s)",
                     args.pool, available_pools)
        sys.exit()
    ioctx = cluster.open_ioctx(args.pool)
    pool_stats = ioctx.get_stats()
    logger.info("Connected to pool %s. Currently holds %d objects totalling %g GB",
                args.pool, pool_stats['num_objects'],
                pool_stats['num_bytes'] / 2 ** 30)

    freq_chunksize, obj_size = get_freq_chunk(data.shape, args.obj_size)
    max_dumps = args.max_dumps if args.max_dumps > 0 else data.shape[0]
    obj_count_per_dump = data.shape[1] // freq_chunksize
    obj_count = max_dumps * obj_count_per_dump

    ts.add("obj_basename", args.basename)
    ts.add("obj_chunk_size", freq_chunksize)
    ts.add("obj_size", obj_size)
    ts.add("obj_count", obj_count)
    ts.add("obj_pool", args.pool)
    f = open(args.ceph_conf, "r")
    ts.add("obj_ceph_conf", f.readlines())
    f.close()
    logger.info("Inserted obj schema metadata into telstate")

    logger.info("Processing %d dumps into %d objects of size %d with basename %s",
                max_dumps, obj_count, obj_size, args.basename)
    for dump_index, dump_data in enumerate(data):
        st = time.time()
        bytes_written, objs_written = write_dump(ioctx, dump_index, dump_data,
                                                 args.basename, freq_chunksize)
        et = time.time() - st
        if objs_written == obj_count_per_dump and \
           bytes_written == obj_size * obj_count_per_dump:
            logger.info("Stored dump index %d in %gs (%d objects totalling %gMBps)",
                        dump_index, et, obj_count_per_dump,
                        obj_count_per_dump * obj_size / (1024*1024) / et)
        else:
            logger.error("Failed to full write ts index %d. "
                         "Wrote %d/%d objects and %d/%d bytes",
                         dump_index, objs_written, obj_count_per_dump,
                         bytes_written, obj_size * obj_count_per_dump)
        if dump_index >= max_dumps - 1:
            logger.info("Reached specified ts limit (%d).", max_dumps)
            break
    logger.info("Staging complete...")
    if args.redis is None:
        raw_input("You have started a local Redis server. "
                  "Hit enter to kill this and cleanup.")
        local_redis.terminate()


if __name__ == '__main__':
    main()
