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
from __future__ import print_function, division, absolute_import

from builtins import object
import tempfile
import shutil
import os
import random
import itertools

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal
import dask.array as da

from katdal.chunkstore import generate_chunks
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import ChunkStoreVisFlagsWeights


def ramp(shape, offset=1.0, slope=1.0, dtype=np.float_):
    """Generate a multidimensional ramp of values of the given dtype."""
    x = offset + slope * np.arange(np.prod(shape), dtype=np.float_)
    return x.astype(dtype).reshape(shape)


def to_dask_array(x, chunks=None):
    """Turn ndarray `x` into dask array with the standard vis-like chunking."""
    if chunks is None:
        itemsize = np.dtype('complex64').itemsize
        # Special case for 2-D weights_channel array ensures one chunk per dump
        n_corrprods = x.shape[2] if x.ndim >= 3 else x.shape[1] // itemsize
        # This contrives to have a vis array with 1 dump and 4 channels per chunk
        chunk_size = 4 * n_corrprods * itemsize
        chunks = generate_chunks(x.shape, x.dtype, chunk_size,
                                 dims_to_split=(0, 1), power_of_two=True)
    return da.from_array(x, chunks)


def put_fake_dataset(store, prefix, shape, chunk_overrides=None, array_overrides={}):
    """Write a fake dataset into the chunk store."""
    data = {'correlator_data': ramp(shape, dtype=np.float32) * (1 - 1j),
            'flags': np.ones(shape, dtype=np.uint8),
            'weights': ramp(shape, slope=255. / np.prod(shape), dtype=np.uint8),
            'weights_channel': ramp(shape[:-1], dtype=np.float32)}
    for name in data:
        if name in array_overrides:
            data[name] = array_overrides[name]
    if chunk_overrides is None:
        chunk_overrides = {}
    ddata = {k: to_dask_array(array, chunk_overrides.get(k)) for k, array in data.items()}
    chunk_info = {k: {'prefix': prefix, 'chunks': darray.chunks,
                      'dtype': darray.dtype, 'shape': darray.shape}
                  for k, darray in ddata.items()}
    for k, darray in ddata.items():
        store.create_array(store.join(prefix, k))
    push = [store.put_dask_array(store.join(prefix, k), darray)
            for k, darray in ddata.items()]
    da.compute(*push)
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
        # Put fake dataset into chunk store
        store = NpyFileChunkStore(self.tempdir)
        prefix = 'cb1'
        shape = (10, 64, 30)
        data, chunk_info = put_fake_dataset(store, prefix, shape)
        vfw = ChunkStoreVisFlagsWeights(store, chunk_info, None)
        weights = data['weights'] * data['weights_channel'][..., np.newaxis]
        # Check that data is as expected when accessed via VisFlagsWeights
        assert_equal(vfw.shape, data['correlator_data'].shape)
        assert_array_equal(vfw.vis.compute(), data['correlator_data'])
        assert_array_equal(vfw.flags.compute(), data['flags'])
        assert_array_equal(vfw.weights.compute(), weights)

    def test_weight_power_scale(self):
        ants = 7
        index1, index2 = np.triu_indices(ants)
        inputs = ['m{:03}h'.format(i) for i in range(ants)]
        corrprods = np.array([(inputs[a], inputs[b]) for (a, b) in zip(index1, index2)])
        # Put fake dataset into chunk store
        store = NpyFileChunkStore(self.tempdir)
        prefix = 'cb1'
        shape = (10, 64, len(index1))

        # Make up some vis data where the expected scaling factors can be
        # computed by hand. Note: the autocorrs are all set to powers of
        # 2 so that we avoid any rounding errors.
        vis = np.full(shape, 2 + 3j, np.complex64)
        vis[:, :, index1 == index2] = 2     # Make all autocorrs real
        vis[3, :, index1 == index2] = 4     # Tests time indexing
        vis[:, 7, index1 == index2] = 4     # Tests frequency indexing
        vis[:, :, ants] *= 8                # The (1, 1) baseline
        vis[4, 5, 0] = 0                    # The (0, 0) baseline
        expected_scale = np.full(shape, 0.25, np.float32)
        expected_scale[3, :, :] = 1 / 16
        expected_scale[:, 7, :] = 1 / 16
        expected_scale[:, :, index1 == 1] /= 8
        expected_scale[:, :, index2 == 1] /= 8
        expected_scale[4, 5, index1 == 0] = 2.0**-32
        expected_scale[4, 5, index2 == 0] = 2.0**-32

        data, chunk_info = put_fake_dataset(
            store, prefix, shape, array_overrides={'correlator_data': vis})
        vfw = ChunkStoreVisFlagsWeights(store, chunk_info, corrprods)
        weights = data['weights'] * data['weights_channel'][..., np.newaxis] * expected_scale

        # Check that data is as expected when accessed via VisFlagsWeights
        assert_equal(vfw.shape, data['correlator_data'].shape)
        assert_array_equal(vfw.vis.compute(), data['correlator_data'])
        assert_array_equal(vfw.flags.compute(), data['flags'])
        assert_array_equal(vfw.weights.compute(), weights)

    def _test_missing_chunks(self, shape, chunk_overrides=None):
        # Put fake dataset into chunk store
        store = NpyFileChunkStore(self.tempdir)
        prefix = 'cb2'
        data, chunk_info = put_fake_dataset(store, prefix, shape, chunk_overrides)
        # Delete some random chunks in each array of the dataset
        missing_chunks = {}
        rs = random.Random(4)
        for array, info in chunk_info.items():
            array_name = store.join(prefix, array)
            slices = da.core.slices_from_chunks(info['chunks'])
            culled_slices = rs.sample(slices, len(slices) // 10 + 1)
            missing_chunks[array] = culled_slices
            for culled_slice in culled_slices:
                chunk_name, shape = store.chunk_metadata(array_name, culled_slice)
                os.remove(os.path.join(store.path, chunk_name) + '.npy')
        vfw = ChunkStoreVisFlagsWeights(store, chunk_info, None)
        # Check that (only) missing chunks have been replaced by zeros
        vis = data['correlator_data']
        for culled_slice in missing_chunks['correlator_data']:
            vis[culled_slice] = 0.
        assert_array_equal(vfw.vis, vis)
        weights = data['weights'] * data['weights_channel'][..., np.newaxis]
        for culled_slice in missing_chunks['weights'] + missing_chunks['weights_channel']:
            weights[culled_slice] = 0.
        assert_array_equal(vfw.weights, weights)
        # Check that (only) missing chunks have been flagged as 'data lost'
        flags = data['flags']
        for culled_slice in missing_chunks['flags']:
            flags[culled_slice] = 0
        for culled_slice in itertools.chain(*missing_chunks.values()):
            flags[culled_slice] |= 8
        assert_array_equal(vfw.flags, flags)

    def test_missing_chunks(self):
        self._test_missing_chunks((100, 256, 30))

    def test_missing_chunks_uneven_chunking(self):
        self._test_missing_chunks(
            (20, 210, 30),
            {
                'vis': (1, 6, 30),
                'weights': (5, 10, 15),
                'weights_channel': (1, 7),
                'flags': (4, 15, 30)
            })
