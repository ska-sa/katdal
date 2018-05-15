################################################################################
# Copyright (c) 2018, National Research Foundation (Square Kilometre Array)
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

"""Tests for :py:mod:`katdal.datasources`."""

import tempfile
import shutil

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal
import dask.array as da

from katdal.chunkstore import generate_chunks
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import ChunkStoreVisFlagsWeights


def ramp(shape, offset=0.0, slope=1.0, dtype=np.float_):
    """Generate a multidimensional ramp of values of the given dtype."""
    x = offset + slope * np.arange(np.prod(shape), dtype=np.float_)
    return x.astype(dtype).reshape(shape)


def to_dask_array(x):
    """Turn ndarray `x` into a dask array with the standard vis chunking."""
    n_corrprods = x.shape[2] if x.ndim >= 3 else x.shape[1] / 8
    chunk_size = 4 * n_corrprods * 8
    chunks = generate_chunks(x.shape, x.dtype, chunk_size,
                             dims_to_split=(0, 1), power_of_two=True)
    return da.from_array(x, chunks)


def put_fake_dataset(store, base_name, shape):
    """Write a fake dataset into the chunk store."""
    data = {'correlator_data': ramp(shape, dtype=np.float32) * (1 - 1j),
            'flags': np.zeros(shape, dtype=np.uint8),
            'weights': ramp(shape, slope=256. / np.prod(shape), dtype=np.uint8),
            'weights_channel': ramp(shape[:-1], dtype=np.float32)}
    ddata = {k: to_dask_array(array) for k, array in data.items()}
    chunk_info = {k: {'chunks': darray.chunks, 'dtype': darray.dtype,
                      'shape': darray.shape} for k, darray in ddata.items()}
    push = [store.put_dask_array(store.join(base_name, k), darray)
            for k, darray in ddata.items()]
    da.compute(push)
    return data, chunk_info


class TestChunkStoreVisFlagsWeights(object):
    """Test the :class:`ChunkStoreVisFlagsWeights` dataset store."""

    @classmethod
    def setup_class(cls):
        cls.tempdir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tempdir)

    def test_construction(self):
        store = NpyFileChunkStore(self.tempdir)
        base_name = 'cb1'
        shape = (10, 64, 30)
        data, chunk_info = put_fake_dataset(store, base_name, shape)
        vfw = ChunkStoreVisFlagsWeights(store, base_name, chunk_info)
        weights = data['weights'] * data['weights_channel'][..., np.newaxis]
        assert_equal(vfw.shape, data['correlator_data'].shape)
        assert_array_equal(vfw.vis.compute(), data['correlator_data'])
        assert_array_equal(vfw.flags.compute(), data['flags'])
        assert_array_equal(vfw.weights.compute(), weights)
