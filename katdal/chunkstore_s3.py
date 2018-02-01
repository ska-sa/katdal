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
    _botocore_import_error = None
except ImportError as e:
    botocore = None
    _botocore_import_error = e
else:
    import botocore.config
    import botocore.session
    from botocore.exceptions import (EndpointConnectionError,
                                     NoCredentialsError, ClientError)

from .chunkstore import ChunkStore, StoreUnavailable, ChunkNotFound, BadChunk


class S3ChunkStore(ChunkStore):
    """A store of chunks (i.e. N-dimensional arrays) based on the Amazon S3 API.

    This object encapsulates the S3 client / session and its underlying
    connection pool, which allows subsequent get and put calls to share the
    connections.

    The full identifier of each chunk (the "chunk name") is given by

      "<bucket>/<path>/<idx>"

    where "<bucket>" refers to the relevant S3 bucket, "<bucket>/<path>" is
    the name of the parent array of the chunk and "<idx>" is the index string
    of each chunk (e.g. "00001_00512"). The corresponding S3 key string of
    a chunk is therefore "<path>/<idx>".

    Parameters
    ----------
    client : :class:`botocore.client.S3` object
        Pre-configured botocore S3 client

    Raises
    ------
    ImportError
        If botocore is not installed (it's an optional dependency otherwise)
    """

    def __init__(self, client):
        if not botocore:
            raise _botocore_import_error
        error_map = {EndpointConnectionError: StoreUnavailable,
                     client.exceptions.NoSuchKey: ChunkNotFound,
                     client.exceptions.NoSuchBucket: ChunkNotFound}
        super(S3ChunkStore, self).__init__(error_map)
        self.client = client

    @classmethod
    def from_url(cls, url, **kwargs):
        """Construct S3 chunk store from endpoint URL.

        S3 authentication (i.e. the access + secret keys) is handled externally
        via the botocore config file or environment variables. Extra keyword
        arguments are interpreted as botocore config settings (see
        :class:`botocore.config.Config`) or arguments to the client creation
        method (see :meth:`botocore.session.Session.create_client`), in that
        order, overriding the defaults.

        Parameters
        ----------
        url : string
            Endpoint of S3 service, e.g. 'http://127.0.0.1:9000'
        kwargs : dict
            Extra keyword arguments: config settings or create_client arguments

        Raises
        ------
        ImportError
            If botocore is not installed (it's an optional dependency otherwise)
        :exc:`chunkstore.StoreUnavailable`
            If S3 server interaction failed (it's down, no authentication, etc)
        """
        if not botocore:
            raise ImportError('Please install botocore for katdal S3 support')
        config_kwargs = dict(max_pool_connections=200,
                             s3={'addressing_style': 'path'})
        client_kwargs = {}
        # Split keyword arguments into config settings and create_client args
        for k, v in kwargs.items():
            if k in botocore.config.Config.OPTION_DEFAULTS:
                config_kwargs[k] = v
            else:
                client_kwargs[k] = v
        session = botocore.session.get_session()
        config = botocore.config.Config(**config_kwargs)
        try:
            client = session.create_client(service_name='s3',
                                           endpoint_url=url, config=config,
                                           **client_kwargs)
            # Quick smoke test to see if the S3 server is available
            client.list_buckets()
        except (EndpointConnectionError, NoCredentialsError, ValueError) as e:
            raise StoreUnavailable(str(e))
        return cls(client)

    def get_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.get_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        bucket, key = self.split(chunk_name, 1)
        with self._standard_errors(chunk_name):
            response = self.client.get_object(Bucket=bucket, Key=key)
        with contextlib.closing(response['Body']) as stream:
            data_str = stream.read()
        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if len(data_str) != expected_bytes:
            raise BadChunk('Chunk {!r}: dtype {} and shape {} implies an '
                           'object size of {} bytes, got {} bytes instead'
                           .format(chunk_name, dtype, shape, expected_bytes,
                                   len(data_str)))
        return np.ndarray(shape, dtype, data_str)

    def put_chunk(self, array_name, slices, chunk):
        """See the docstring of :meth:`ChunkStore.put_chunk`."""
        chunk_name, shape = self.chunk_metadata(array_name, slices, chunk=chunk)
        bucket, key = self.split(chunk_name, 1)
        data_str = chunk.tobytes()
        with self._standard_errors(chunk_name):
            self.client.put_object(Bucket=bucket, Key=key, Body=data_str)

    def has_chunk(self, array_name, slices, dtype):
        """See the docstring of :meth:`ChunkStore.has_chunk`."""
        dtype = np.dtype(dtype)
        chunk_name, shape = self.chunk_metadata(array_name, slices, dtype=dtype)
        bucket, key = self.split(chunk_name, 1)
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
        except ClientError as err:
            if err.response['Error']['Code'] != '404':
                raise
            return False
        else:
            actual_bytes = response['ContentLength']
            expected_bytes = int(np.prod(shape)) * dtype.itemsize
            return actual_bytes == expected_bytes

    get_chunk.__doc__ = ChunkStore.get_chunk.__doc__
    put_chunk.__doc__ = ChunkStore.put_chunk.__doc__
    has_chunk.__doc__ = ChunkStore.has_chunk.__doc__
