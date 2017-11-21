#!/usr/bin/env python

"""Simple script to take an HDF5 file and chunk it into objects.

This also populates a specified (or managed) Redis instance to hold
telescope state and other metadata.

The primary use is for testing the coming katdal access layer that will
wrap up such an object+Redis dataset and provide the usual standard access.

Objects are stored in chunks split over time and frequency but not baseline.
The chunking is chosen to produce objects with sizes on the order of 1 MB.
The schema used is as follows:

  <obj_base_name>/<dataset_name>[/<index1>_<index2>_<...>]

  - obj_base_name: for S3 this defaults to '<bucket>/<program_block>/<stream>'
  - dataset_name: 'correlator_data' / 'weights' / 'flags' / etc.
  - indexN: chunk start index along N'th dimension (suppressed if 1 chunk only)

The following useful object parameters are stored in telstate, prefixed by
'<program_block>.<stream>.':

  - ceph_pool: the name of the CEPH pool used
  - ceph_conf: copy of ceph.conf used to connect to target CEPH cluster
  - s3_endpoint: endpoint URL of S3 object store
  - <dataset_name>: dict containing chunk info (dtype, shape and chunks)
"""

import struct
import logging
import sys
import time
import shlex
import subprocess
import functools
from itertools import product

import numpy as np
import katdal
import katsdptelstate
import katsdpservices
import dask
import dask.array as da
from dask.diagnostics import ProgressBar

try:
    import rados
except ImportError:
    # librados (i.e. direct Ceph access) is only really available on Linux
    rados = None

try:
    import botocore
except ImportError:
    botocore = None
else:
    import botocore.config
    import botocore.session

try:
    from cStringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


logging.basicConfig()

logger = logging.getLogger('h5toobj')
logger.setLevel(logging.INFO)


def parse_args():
    parser = katsdpservices.ArgumentParser()
    parser.add_argument('file', type=str, nargs=1,
                        metavar='FILE', help='HDF5 file to process')
    parser.add_argument('--base-name', type=str, metavar='BASENAME',
                        help='Base name for objects (should include bucket '
                             'name for S3 object store)')
    parser.add_argument('--obj-size', type=float, default=2.0,
                        help='Target object size in MB')
    parser.add_argument('--max-dumps', type=int, default=0,
                        help='Number of dumps to process. Default is all.')
    parser.add_argument('--ceph-conf', type=str, default="/etc/ceph/ceph.conf",
                        metavar='CEPHCONF',
                        help='CEPH configuration file used for cluster connect')
    parser.add_argument('--ceph-pool', type=str, metavar='POOL',
                        help='CEPH pool to use for object storage')
    parser.add_argument('--s3-url', type=str,
                        help='S3 endpoint URL (includes leading "http")')
    parser.add_argument('--redis', type=str,
                        help='Redis host to connect to as Telescope State. '
                             'Default is to start a new local instance.')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Port to use when connecting to Redis instance '
                             '(or creating a new one)')
    parser.add_argument('--redis-only', action='store_true',
                        help='Only (re)build Redis DB - no object creation')
    parser.add_argument('--obj-only', action='store_true',
                        help='Only populate object store - no Redis update')
    args = parser.parse_args()
    if not rados and not botocore:
        logger.warning("No Ceph or S3 installation found - storing Redis DB only")
        args.redis_only = True
    if not args.redis_only:
        use_s3 = args.s3_url is not None
        use_rados = args.ceph_pool is not None
        if use_rados and use_s3:
            parser.error('Please specify either --ceph-pool or --s3-*')
        if use_rados and not rados:
            parser.error('Cannot use Ceph as rados Python library is not installed')
        if use_s3 and not botocore:
            parser.error('Cannot use S3 as botocore Python library is not installed')
    if args.base_name is None:
        args.base_name = args.file[0].split(".")[0]
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


def open_rados(ceph_conf, ceph_pool):
    cluster = rados.Rados(conffile=ceph_conf)
    cluster.connect()
    available_pools = cluster.list_pools()
    if ceph_pool not in available_pools:
        logger.error("Specified pool %s not available in this cluster (%s)",
                     ceph_pool, available_pools)
        sys.exit()
    ioctx = cluster.open_ioctx(ceph_pool)
    pool_stats = ioctx.get_stats()
    logger.info("Connected to pool %s. Currently holds %d objects totalling %g GB",
                ceph_pool, pool_stats['num_objects'],
                pool_stats['num_bytes'] / 2 ** 30)
    return ioctx


def write_chunk_rados(ioctx, name, chunk):
    data_fp = StringIO(chunk.data)
    data_str = data_fp.read()
    ioctx.write_full(name, data_str)


def open_s3(url):
    session = botocore.session.get_session()
    config = botocore.config.Config(max_pool_connections=200,
                                    s3={'addressing_style': 'path'})
    client = session.create_client(service_name='s3', endpoint_url=url,
                                   config=config)
    return client


def write_chunk_s3(client, name, chunk):
    bucket, key = name.split('/', 1)
    data_fp = StringIO(chunk.data)
    client.put_object(Bucket=bucket, Key=key, Body=data_fp)


def generate_chunks(shape, dtype, target_object_size, dims_to_split=(0, 1)):
    """"""
    dataset_size = np.prod(shape) * dtype.itemsize
    num_chunks = np.ceil(dataset_size / float(target_object_size))
    chunks = [(s,) for s in shape]
    for dim in dims_to_split:
        if dim >= len(shape):
            continue
        if num_chunks > 0.5 * shape[dim]:
            chunk_sizes = (1,) * shape[dim]
        else:
            items = np.arange(shape[dim])
            chunk_indices = np.array_split(items, num_chunks)
            chunk_sizes = tuple([len(chunk) for chunk in chunk_indices])
        chunks[dim] = chunk_sizes
        num_chunks = np.ceil(num_chunks / len(chunk_sizes))
    return tuple(chunks)


def dask_from_chunks(chunks, out_name, base_name):
    single_chunk = all(len(c) == 1 for c in chunks)
    last_index_per_dim = [np.r_[0, np.cumsum(c)[:-1]][-1] for c in chunks]
    widths = [len(str(i)) for i in last_index_per_dim]
    keys = list(product([out_name], *[range(len(bds)) for bds in chunks]))
    all_slices = da.core.slices_from_chunks(chunks)
    for key, slices in zip(keys, all_slices):
        index = [s.start for s in slices]
        if single_chunk:
            obj_name = base_name
        else:
            index_str = '_'.join(["{:0{width}d}".format(i, width=w)
                                  for (i, w) in zip(index, widths)])
            obj_name = base_name + '/' + index_str
        yield key, slices, obj_name


if __name__ == '__main__':
    args = parse_args()
    try:
        f = katdal.open(args.file[0])
        h5_file = f.file
    except Exception as e:
        logger.error("Failed to open specified HDF5 file. %s", e)
        sys.exit()
    try:
        vis = h5_file['Data/correlator_data']
    except KeyError:
        logger.error("This does not appear to be a valid MeerKAT HDF5 file")
        sys.exit()

    if args.obj_only:
        redis_endpoint = args.redis = redis_host = ''
    elif args.redis is None:
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
        redis_endpoint = '{}:{}'.format(redis_host, args.redis_port)
    else:
        redis_host = args.redis
        redis_endpoint = '{}:{}'.format(redis_host, args.redis_port)
    ts = katsdptelstate.TelescopeState(redis_endpoint)
    logger.info("Connected to Redis on %s. DB has %d existing keys",
                redis_endpoint, len(ts.keys()))

    r_str = ""
    for attr in h5_file['TelescopeState'].attrs:
        r_str += redis_gen_proto("SET", attr, h5_file['TelescopeState'].attrs[attr])
    if redis_endpoint:
        redis_bulk_str(r_str, redis_host, args.redis_port)

    if not args.obj_only:
        for d_count, dset in enumerate(h5_file['TelescopeState'].keys()):
            st = time.time()
            r_str = ""
            d_val = h5_file['TelescopeState'][dset].value
             # much quicker to read it first and then iterate
            for (timestamp, pval) in d_val:
                packed_ts = struct.pack('>d', float(timestamp))
                r_str += redis_gen_proto("ZADD", str(dset), "0", packed_ts + pval)
            bss = time.time()
            if redis_endpoint:
                redis_bulk_str(r_str, redis_host, args.redis_port)
                logger.info("Added %d items in %gs to key %s. Bulk insert time: %g",
                            len(d_val), time.time() - st, dset, time.time() - bss)
        logger.info("Added %d ranged keys to TelescopeState", d_count + 1)

    if args.redis_only and args.redis is None:
        logger.warning("Terminating locally launched redis instance "
                       "(also saves telstate to local dump.rdb)")
        try:
            cli_cmd = "/usr/bin/redis-cli -p {} SHUTDOWN SAVE".format(args.redis_port)
            subprocess.call(shlex.split(cli_cmd))
        except OSError:
            cli_cmd = "/usr/local/bin/redis-cli -p {} SHUTDOWN SAVE".format(args.redis_port)
            subprocess.call(shlex.split(cli_cmd))
        local_redis.terminate()

    if args.redis_only:
        logger.warning("Building Redis DB only - no data will be written...")
        sys.exit(0)

    program_block = f.experiment_id
    stream = 'sdp_l0'
    ts_pbs = ts.view(program_block + '.' + stream)
    max_dumps = args.max_dumps if args.max_dumps > 0 else vis.shape[0]

    use_rados = args.ceph_pool is not None
    if use_rados:
        store = open_rados(args.ceph_conf, args.ceph_pool)
        ts_pbs.add("ceph_pool", args.ceph_pool, immutable=True)
        with open(args.ceph_conf, "r") as ceph_conf:
            ts_pbs.add("ceph_conf", ceph_conf.readlines(), immutable=True)
        save = functools.partial(write_chunk_rados, store)
    else:
        store = open_s3(args.s3_url)
        ts_pbs.add("s3_endpoint", args.s3_url, immutable=True)
        save = functools.partial(write_chunk_s3, store)
    # Convert HDF5 visibilities to complex64 on the fly
    vis_save = lambda obj, x: save(obj, x.view(np.complex64)[..., 0])

    target_object_size = args.obj_size * 2 ** 20
    dask_graph = {}
    schedule = dask.threaded.get
    output_keys = []
    for dataset in ('timestamps', 'correlator_data', 'weights',
                    'weights_channel', 'flags'):
        arr = h5_file['Data'][dataset]
        dask_graph[dataset] = arr
        base_name = '/'.join((args.base_name, program_block, stream, dataset))
        shape = (min(arr.shape[0], max_dumps),) + arr.shape[1:]
        dtype = arr.dtype
        if dataset == 'correlator_data':
            shape = shape[:-1]
            dtype = np.dtype(np.complex64)
        chunks = generate_chunks(shape, dtype, target_object_size)
        num_chunks = np.prod([len(c) for c in chunks])
        chunk_size = np.prod([c[0] for c in chunks]) * dtype.itemsize
        logger.info("Splitting dataset %r with shape %s into %d chunk(s) of "
                    "~%d bytes each", base_name, shape, num_chunks, chunk_size)
        dask_info = {'dtype': dtype, 'shape': shape, 'chunks': chunks}
        ts_pbs.add(dataset, dask_info, immutable=True)
        obj_save = vis_save if dataset == 'correlator_data' else save
        dsk = {k: (obj_save, obj, (da.core.getter, dataset, s))
               for k, s, obj in dask_from_chunks(chunks, dataset, base_name)}
        dask_graph.update(dsk)
        output_keys.extend(dsk.keys())
    with ProgressBar():
        schedule(dask_graph, output_keys)
    logger.info("Staging complete...")

    if args.redis is None:
        raw_input("You have started a local Redis server. "
                  "Hit enter to kill this and cleanup.")
        local_redis.terminate()
