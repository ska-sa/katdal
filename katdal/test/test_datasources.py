################################################################################
# Copyright (c) 2018-2019, National Research Foundation (Square Kilometre Array)
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
from nose.tools import assert_equal, assert_raises
import dask.array as da
import katsdptelstate

from katdal.chunkstore import generate_chunks
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import ChunkStoreVisFlagsWeights, TelstateDataSource, view_l0_capture_stream
from katdal.flags import DATA_LOST


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


def put_fake_dataset(store, prefix, shape, chunk_overrides=None, array_overrides=None, flags_only=False):
    """Write a fake dataset into the chunk store."""
    if flags_only:
        data = {'flags': np.random.RandomState(1).randint(0, 7, shape, dtype=np.uint8)}
    else:
        data = {'correlator_data': ramp(shape, dtype=np.float32) * (1 - 1j),
                'flags': np.random.RandomState(2).randint(0, 7, shape, dtype=np.uint8),
                'weights': ramp(shape, slope=255. / np.prod(shape), dtype=np.uint8),
                'weights_channel': ramp(shape[:-1], dtype=np.float32)}
    if array_overrides is not None:
        for name in data:
            if name in array_overrides:
                data[name] = array_overrides[name]
    if chunk_overrides is None:
        chunk_overrides = {}
    ddata = {k: to_dask_array(array, chunk_overrides.get(k)) for k, array in data.items()}
    chunk_info = {k: {'prefix': prefix, 'chunks': darray.chunks,
                      'dtype': np.lib.format.dtype_to_descr(darray.dtype),
                      'shape': darray.shape}
                  for k, darray in ddata.items()}
    for k, darray in ddata.items():
        store.create_array(store.join(prefix, k))
    push = [store.put_dask_array(store.join(prefix, k), darray)
            for k, darray in ddata.items()]
    da.compute(*push)
    return data, chunk_info


def _make_fake_stream(telstate, store, cbid, stream, shape,
                      chunk_overrides=None, array_overrides=None, flags_only=False):
    telstate_prefix = telstate.join(cbid, stream)
    store_prefix = telstate_prefix.replace('_', '-')
    data, chunk_info = put_fake_dataset(store, store_prefix, shape,
                                        chunk_overrides=chunk_overrides, array_overrides=array_overrides,
                                        flags_only=flags_only)
    cs_view = telstate.view(telstate_prefix)
    s_view = telstate.view(stream)
    cs_view['chunk_info'] = chunk_info
    cs_view['first_timestamp'] = 123.0
    s_view['sync_time'] = 123456789.0
    s_view['int_time'] = 2.0
    s_view['bandwidth'] = 856e6
    s_view['center_freq'] = 1284e6
    s_view['n_chans'] = shape[1]
    s_view['n_bls'] = shape[2]
    # This isn't particularly representative - may need refinement depending on
    # what the test does
    n_ant = 1
    while n_ant * (n_ant + 1) * 2 < shape[2]:
        n_ant += 1
    if n_ant * (n_ant + 1) * 2 != shape[2]:
        raise ValueError('n_bls is not consistent with an integer number of antennas')
    bls_ordering = []
    for i in range(n_ant):
        for j in range(i, n_ant):
            for x in 'hv':
                for y in 'hv':
                    bls_ordering.append(('m{:03}{}'.format(i, x),
                                         'm{:03}{}'.format(j, y)))
    s_view['bls_ordering'] = np.array(bls_ordering)
    if not flags_only:
        s_view['need_weights_power_scale'] = True
        s_view['stream_type'] = 'sdp.vis'
    else:
        s_view['stream_type'] = 'sdp.flags'
    return data, cs_view, s_view


def make_fake_datasource(telstate, store, cbid, l0_shape, l1_flags_shape=None,
                         l0_chunk_overrides=None, l1_flags_chunk_overrides=None,
                         l0_array_overrides=None, l1_flags_array_overrides=None):
    """Create a complete fake data source.

    The resulting telstate and chunk store are suitable for constructing
    a :class:`~.TelstateDataSource`, including upgrading of flags. However,
    it adds about as little as possible to telstate for that, so may need
    to be extended from time to time.
    """
    if l1_flags_shape is None:
        l1_flags_shape = l0_shape
    l0_data, l0_cs_view, l0_s_view = \
        _make_fake_stream(telstate, store, cbid, 'sdp_l0', l0_shape,
                          chunk_overrides=l0_chunk_overrides,
                          array_overrides=l0_array_overrides)
    l1_flags_data, l1_flags_cs_view, l1_flags_s_view = \
        _make_fake_stream(telstate, store, cbid, 'sdp_l1_flags', l1_flags_shape,
                          chunk_overrides=l1_flags_chunk_overrides,
                          array_overrides=l1_flags_array_overrides,
                          flags_only=True)
    l1_flags_s_view['src_streams'] = ['sdp_l0']
    telstate['sdp_archived_streams'] = ['sdp_l0', 'sdp_l1_flags']
    return view_l0_capture_stream(telstate, cbid, 'sdp_l0') + (l0_data, l1_flags_data)


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
        assert_equal(vfw.store, store)
        assert_equal(vfw.vis_prefix, prefix)
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
            flags[culled_slice] |= DATA_LOST
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


class TestTelstateDataSource(object):
    def setup(self):
        self.tempdir = tempfile.mkdtemp()
        self.store = NpyFileChunkStore(self.tempdir)
        self.telstate = katsdptelstate.TelescopeState()
        self.cbid = 'cb'

    def teardown(self):
        shutil.rmtree(self.tempdir)

    def test_timestamps(self):
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_datasource(self.telstate, self.store, self.cbid, (20, 64, 40))
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        np.testing.assert_array_equal(
            data_source.timestamps,
            np.arange(20, dtype=np.float32) * 2 + 123456912)

    def test_upgrade_flags(self):
        shape = (20, 16, 40)
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_datasource(self.telstate, self.store, self.cbid, shape)
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        np.testing.assert_array_equal(data_source.data.vis.compute(), l0_data['correlator_data'])
        np.testing.assert_array_equal(data_source.data.flags.compute(), l1_flags_data['flags'])
        # Again, now explicitly disabling the upgrade
        data_source = TelstateDataSource(view, cbid, sn, self.store, upgrade_flags=False)
        np.testing.assert_array_equal(data_source.data.vis.compute(), l0_data['correlator_data'])
        np.testing.assert_array_equal(data_source.data.flags.compute(), l0_data['flags'])

    def test_upgrade_flags_extend_l1(self, l0_chunk_overrides=None, l1_flags_chunk_overrides=None):
        """L1 flags has fewer dumps than L0"""
        l0_shape = (20, 16, 40)
        l1_flags_shape = (18, 16, 40)
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_datasource(self.telstate, self.store, self.cbid, l0_shape, l1_flags_shape,
                                 l0_chunk_overrides=l0_chunk_overrides,
                                 l1_flags_chunk_overrides=l1_flags_chunk_overrides)
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        np.testing.assert_array_equal(
            data_source.timestamps,
            np.arange(l0_shape[0], dtype=np.float32) * 2 + 123456912)
        np.testing.assert_array_equal(data_source.data.vis.compute(), l0_data['correlator_data'])
        expected_flags = np.zeros(l0_shape, np.uint8)
        expected_flags[:l1_flags_shape[0]] = l1_flags_data['flags']
        expected_flags[l1_flags_shape[0]:] = DATA_LOST
        np.testing.assert_array_equal(data_source.data.flags.compute(), expected_flags)

    def test_upgrade_flags_extend_l1_multi_dump(self):
        """L1 flags has fewer dumps than L0, with multiple dumps per chunk"""
        self.test_upgrade_flags_extend_l1(
            l0_chunk_overrides={
                'correlator_data': (4, 4, 40),
                'weights': (4, 4, 40),
                'weights_channel': (4, 4),
                'flags': (4, 4, 40)
            },
            l1_flags_chunk_overrides={'flags': (9, 8, 40)}
        )

    def test_upgrade_flags_extend_l0(self, l0_chunk_overrides=None, l1_flags_chunk_overrides=None):
        """L1 flags has more dumps than L0"""
        l0_shape = (18, 16, 40)
        l1_flags_shape = (20, 16, 40)
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_datasource(self.telstate, self.store, self.cbid, l0_shape, l1_flags_shape,
                                 l0_chunk_overrides=l0_chunk_overrides,
                                 l1_flags_chunk_overrides=l1_flags_chunk_overrides)
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        np.testing.assert_array_equal(
            data_source.timestamps,
            np.arange(l1_flags_shape[0], dtype=np.float32) * 2 + 123456912)
        expected_vis = np.zeros(l1_flags_shape, l0_data['correlator_data'].dtype)
        expected_vis[:18] = l0_data['correlator_data']
        expected_flags = l1_flags_data['flags'].copy()
        # The visibilities for this extension are lost, so the flags will mark it as such
        expected_flags[18:20] |= DATA_LOST
        np.testing.assert_array_equal(data_source.data.vis.compute(), expected_vis)
        np.testing.assert_array_equal(data_source.data.flags.compute(), expected_flags)

    def test_upgrade_flags_extend_l0_multi_dump(self):
        """L1 flags has more dumps than L0, with multiple dumps per chunk"""
        self.test_upgrade_flags_extend_l0(
            l0_chunk_overrides={
                'correlator_data': (9, 4, 40),
                'weights': (9, 4, 40),
                'weights_channel': (9, 4),
                'flags': (9, 4, 40)
            },
            l1_flags_chunk_overrides={'flags': (5, 8, 40)}
        )

    def test_upgrade_flags_shape_mismatch(self):
        """L1 flags shape is incompatible with L0"""
        l0_shape = (18, 16, 40)
        l1_flags_shape = (20, 8, 40)
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_datasource(self.telstate, self.store, self.cbid, l0_shape, l1_flags_shape)
        with assert_raises(ValueError):
            TelstateDataSource(view, cbid, sn, self.store)
