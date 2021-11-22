################################################################################
# Copyright (c) 2018-2021, National Research Foundation (SARAO)
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

import os
import shutil
import tempfile
import urllib.parse

import katsdptelstate
import numpy as np
from katsdptelstate.rdb_writer import RDBWriter
from nose.tools import assert_raises

from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.datasources import (DataSourceNotFound, TelstateDataSource,
                                open_data_source, view_l0_capture_stream)
from katdal.flags import DATA_LOST
from katdal.test.test_vis_flags_weights import put_fake_dataset
from katdal.vis_flags_weights import correct_autocorr_quantisation


def _make_fake_stream(telstate, store, cbid, stream, shape,
                      chunk_overrides=None, array_overrides=None, flags_only=False,
                      bls_ordering_override=None):
    telstate_prefix = telstate.join(cbid, stream)
    store_prefix = telstate_prefix.replace('_', '-')
    data, chunk_info = put_fake_dataset(store, store_prefix, shape,
                                        chunk_overrides=chunk_overrides,
                                        array_overrides=array_overrides,
                                        flags_only=flags_only)
    cs_view = telstate.view(telstate_prefix)
    s_view = telstate.view(stream)
    cs_view['chunk_info'] = chunk_info
    cs_view['first_timestamp'] = 123.0
    s_view['sync_time'] = 1600000000.0
    s_view['int_time'] = 2.0
    s_view['bandwidth'] = 856e6
    s_view['center_freq'] = 1284e6
    s_view['n_chans'] = shape[1]
    s_view['n_bls'] = shape[2]
    # This isn't particularly representative - may need refinement depending on
    # what the test does
    if bls_ordering_override is None:
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
                        bls_ordering.append((f'm{i:03}{x}', f'm{j:03}{y}'))
        s_view['bls_ordering'] = np.array(bls_ordering)
    else:
        s_view['bls_ordering'] = np.asarray(bls_ordering_override)
    if not flags_only:
        s_view['need_weights_power_scale'] = True
        s_view['stream_type'] = 'sdp.vis'
    else:
        s_view['stream_type'] = 'sdp.flags'
    return data, cs_view, s_view


def make_fake_data_source(telstate, store, l0_shape, cbid='cb', l1_flags_shape=None,
                          l0_chunk_overrides=None, l1_flags_chunk_overrides=None,
                          l0_array_overrides=None, l1_flags_array_overrides=None,
                          bls_ordering_override=None):
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
                          array_overrides=l0_array_overrides,
                          bls_ordering_override=bls_ordering_override)
    l1_flags_data, l1_flags_cs_view, l1_flags_s_view = \
        _make_fake_stream(telstate, store, cbid, 'sdp_l1_flags', l1_flags_shape,
                          chunk_overrides=l1_flags_chunk_overrides,
                          array_overrides=l1_flags_array_overrides,
                          flags_only=True,
                          bls_ordering_override=bls_ordering_override)
    l1_flags_s_view['src_streams'] = ['sdp_l0']
    telstate['sdp_archived_streams'] = ['sdp_l0', 'sdp_l1_flags']
    return view_l0_capture_stream(telstate, cbid, 'sdp_l0') + (l0_data, l1_flags_data)


def assert_telstate_data_source_equal(source1, source2):
    """Assert that two :class:`~katdal.datasources.TelstateDataSource`s are equal."""
    # Check attributes (sensors left out for now)
    keys = source1.telstate.keys()
    assert set(source2.telstate.keys()) == set(keys)
    for key in keys:
        np.testing.assert_array_equal(source1.telstate[key], source2.telstate[key])
    # Check that we also have the same telstate view in both sources
    assert source1.telstate.prefixes == source2.telstate.prefixes
    # Check data arrays
    np.testing.assert_array_equal(source1.timestamps, source2.timestamps)
    np.testing.assert_array_equal(source1.data.vis.compute(), source2.data.vis.compute())
    np.testing.assert_array_equal(source1.data.flags.compute(), source2.data.flags.compute())
    np.testing.assert_array_equal(source1.data.weights.compute(), source2.data.weights.compute())


class TestTelstateDataSource:
    def setup(self):
        self.tempdir = tempfile.mkdtemp()
        self.store = NpyFileChunkStore(self.tempdir)
        self.telstate = katsdptelstate.TelescopeState()

    def teardown(self):
        shutil.rmtree(self.tempdir)

    def test_basic_timestamps(self):
        # Add a sensor to telstate to exercise the relevant code paths in TelstateDataSource
        self.telstate.add('obs_script_log', 'Digitisers synced', ts=1600000000.)
        view, cbid, sn, _, _ = make_fake_data_source(self.telstate, self.store, (20, 64, 40))
        data_source = TelstateDataSource(
            view, cbid, sn, chunk_store=None, url='http://hello')
        assert data_source.data is None
        expected_timestamps = np.arange(20, dtype=np.float32) * 2 + 1600000123
        np.testing.assert_array_equal(data_source.timestamps, expected_timestamps)

    def test_timestamps_preselect(self):
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_data_source(self.telstate, self.store, (20, 64, 40))
        data_source = TelstateDataSource(view, cbid, sn, self.store,
                                         preselect=dict(dumps=np.s_[2:10]))
        np.testing.assert_array_equal(
            data_source.timestamps,
            np.arange(2, 10, dtype=np.float32) * 2 + 1600000123)

    def test_bad_preselect(self):
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_data_source(self.telstate, self.store, (20, 64, 40))
        with assert_raises(IndexError):
            TelstateDataSource(view, cbid, sn, self.store, preselect=dict(dumps=np.s_[[1, 2]]))
        with assert_raises(IndexError):
            TelstateDataSource(view, cbid, sn, self.store, preselect=dict(dumps=np.s_[5:0:-1]))
        with assert_raises(IndexError):
            TelstateDataSource(view, cbid, sn, self.store, preselect=dict(frequencies=np.s_[:]))

    def test_preselect(self):
        view, cbid, sn, l0_data, l1_flags_data = \
            make_fake_data_source(self.telstate, self.store, (20, 64, 40))
        preselect = dict(dumps=np.s_[2:10], channels=np.s_[-20:])
        index = np.s_[2:10, -20:]
        data_source = TelstateDataSource(view, cbid, sn, self.store,
                                         upgrade_flags=False, preselect=preselect)
        np.testing.assert_array_equal(data_source.data.vis.compute(), l0_data['correlator_data'][index])
        np.testing.assert_array_equal(data_source.data.flags.compute(), l0_data['flags'][index])

    def test_upgrade_flags(self):
        shape = (20, 16, 40)
        view, cbid, sn, l0_data, l1_flags_data = make_fake_data_source(
            self.telstate, self.store, shape)
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
        view, cbid, sn, l0_data, l1_flags_data = make_fake_data_source(
            self.telstate, self.store, l0_shape, l1_flags_shape=l1_flags_shape,
            l0_chunk_overrides=l0_chunk_overrides,
            l1_flags_chunk_overrides=l1_flags_chunk_overrides)
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        expected_timestamps = np.arange(l0_shape[0], dtype=np.float32) * 2 + 1600000123
        np.testing.assert_array_equal(data_source.timestamps, expected_timestamps)
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
        view, cbid, sn, l0_data, l1_flags_data = make_fake_data_source(
            self.telstate, self.store, l0_shape, l1_flags_shape=l1_flags_shape,
            l0_chunk_overrides=l0_chunk_overrides,
            l1_flags_chunk_overrides=l1_flags_chunk_overrides)
        data_source = TelstateDataSource(view, cbid, sn, self.store)
        expected_timestamps = np.arange(l1_flags_shape[0], dtype=np.float32) * 2 + 1600000123
        np.testing.assert_array_equal(data_source.timestamps, expected_timestamps)
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
        view, cbid, sn, _, _ = make_fake_data_source(self.telstate, self.store, l0_shape,
                                                     l1_flags_shape=l1_flags_shape)
        with assert_raises(ValueError):
            TelstateDataSource(view, cbid, sn, self.store)

    def test_van_vleck(self):
        shape = (20, 16, 40)
        view, cbid, sn, l0_data, _ = make_fake_data_source(self.telstate, self.store, shape)
        # Uncorrected visibilities
        data_source = TelstateDataSource(view, cbid, sn, self.store, van_vleck='off')
        raw_vis = data_source.data.vis
        np.testing.assert_array_equal(raw_vis.compute(), l0_data['correlator_data'])
        # Corrected visibilities
        data_source2 = TelstateDataSource(view, cbid, sn, self.store, van_vleck='autocorr')
        corrected_vis = data_source2.data.vis
        expected_vis = correct_autocorr_quantisation(raw_vis, view['bls_ordering'])
        np.testing.assert_array_equal(corrected_vis.compute(), expected_vis.compute())
        # Check parameter validation
        with assert_raises(ValueError):
            TelstateDataSource(view, cbid, sn, self.store, van_vleck='blah')

    def test_construction_from_url(self):
        view, cbid, sn, _, _ = make_fake_data_source(self.telstate, self.store, (20, 16, 40))
        source_direct = TelstateDataSource(view, cbid, sn, self.store)
        # Save RDB file to e.g. 'tempdir/cb/cb_sdp_l0.rdb', as if 'tempdir' is a real S3 bucket
        rdb_dir = os.path.join(self.tempdir, cbid)
        os.mkdir(rdb_dir)
        rdb_filename = os.path.join(rdb_dir, f'{cbid}_{sn}.rdb')
        # Insert CBID and stream name at the top level, just like metawriter does
        self.telstate['capture_block_id'] = cbid
        self.telstate['stream_name'] = sn
        with RDBWriter(rdb_filename) as rdbw:
            rdbw.save(self.telstate)
        # Check that we can open RDB file and automatically infer the chunk store
        source_from_file = open_data_source(rdb_filename)
        assert_telstate_data_source_equal(source_from_file, source_direct)
        # Check that we can override the capture_block_id and stream name via query parameters
        query = urllib.parse.urlencode({'capture_block_id': cbid, 'stream_name': sn})
        url = urllib.parse.urlunparse(('file', '', rdb_filename, '', query, ''))
        source_from_url = TelstateDataSource.from_url(url, chunk_store=self.store)
        assert_telstate_data_source_equal(source_from_url, source_direct)
        # Check invalid URLs
        with assert_raises(DataSourceNotFound):
            open_data_source('ftp://unsupported')
        with assert_raises(DataSourceNotFound):
            open_data_source(rdb_filename[:-4])
        source_name = f'{cbid}_{sn}'
        assert source_from_file.name == source_name
        assert rdb_filename in source_from_file.url
