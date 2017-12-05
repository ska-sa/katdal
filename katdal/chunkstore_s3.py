################################################################################
# Copyright (c) 2017-2018, National Research Foundation (Square Kilometre Array)
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

"""A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API."""

import contextlib

import numpy as np
try:
    import botocore
except ImportError:
    botocore = None
else:
    import botocore.config
    import botocore.session
    from botocore.exceptions import EndpointConnectionError, ClientError

from .chunkstore import ChunkStore


@contextlib.contextmanager
def _convert_botocore_errors(chunk_name=None):
    try:
        yield
    except (EndpointConnectionError, ClientError) as e:
        prefix = 'Chunk {!r}: '.format(chunk_name) if chunk_name else ''
        raise OSError(prefix + str(e))


class S3ChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API.

    S3 authentication (i.e. the access + secret keys) is handled externally
    via the botocore config file or environment variables. The full identifier
    of each chunk (the "chunk name") is given by

      "<bucket>/<path>/<idx>"

    where "<bucket>" refers to the relevant S3 bucket, "<bucket>/<path>" is the
    name of the parent array of the chunk and "<idx>" is the index string
    of each chunk (e.g. "00001_00512"). The corresponding S3 key string of
    a chunk is therefore "<path>/<idx>".

    This object encapsulates the S3 client / session and its underlying
    connection pool, which allows subsequent get and put calls to share the
    connections.

    Parameters
    ----------
    url : string
        Endpoint of S3 service, e.g. 'http://127.0.0.1:9000'

    Raises
    ------
    ImportError
        If botocore is not installed (it's an optional dependency otherwise)
    OSError
        If S3 server interaction failed (it's down, authentication failed, etc)
    """

    def __init__(self, url):
        if not botocore:
            raise ImportError('Please install botocore for katdal S3 support')
        session = botocore.session.get_session()
        config = botocore.config.Config(max_pool_connections=200,
                                        s3={'addressing_style': 'path'})
        self.client = session.create_client(service_name='s3',
                                            endpoint_url=url, config=config)
        # Quick smoke test to see if the S3 server is available
        with _convert_botocore_errors():
            self.client.list_buckets()

    def get(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        bucket, key = self.split(chunk_name, 1)
        with _convert_botocore_errors(chunk_name):
            response = self.client.get_object(Bucket=bucket, Key=key)
        with contextlib.closing(response['Body']) as stream:
            data_str = stream.read()
        return np.ndarray(shape, dtype, data_str)

    def put(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, chunk=chunk)
        bucket, key = self.split(chunk_name, 1)
        data_str = chunk.tobytes()
        with _convert_botocore_errors(chunk_name):
            self.client.put_object(Bucket=bucket, Key=key, Body=data_str)

    get.__doc__ = ChunkStore.get.__doc__
    put.__doc__ = ChunkStore.put.__doc__
