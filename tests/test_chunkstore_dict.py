################################################################################
# Copyright (c) 2017-2018,2021-2022, National Research Foundation (SARAO)
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

"""Tests for :py:mod:`katdal.chunkstore_dict`."""

import time

import numpy as np
import dask.array as da

from katdal.chunkstore_dict import DictChunkStore
from katdal.test.test_chunkstore import ChunkStoreTestBase, generate_arrays


class TestDictChunkStore(ChunkStoreTestBase):
    def setup_method(self):
        self.arrays = generate_arrays()
        self.store = DictChunkStore(**self.arrays)
        # This store is prepopulated so missing chunks can't be checked
        self.preloaded_chunks = True


def test_basic_overheads():
    """Check overheads of creating and transferring dask array between stores."""
    # The array is about 1 GB in size
    shape = (100, 1000, 1000)
    x = np.ones(shape)
    y = np.zeros(shape)
    store1 = DictChunkStore(x=x)
    store2 = DictChunkStore(y=y)
    # We have 1000 chunks of about 1 MB each
    chunk_size = (1, 100, 1000)
    chunks = da.core.normalize_chunks(chunk_size, shape)
    # Check that the time to set up dask arrays is not grossly inflated
    start_time = time.process_time()
    dx = store1.get_dask_array('x', chunks, float)
    py = store2.put_dask_array('y', dx)
    setup_duration = time.process_time() - start_time
    assert setup_duration < 1.0
    # Use basic array copy as a reference
    start_time = time.process_time()
    y[:] = x
    copy_duration = time.process_time() - start_time
    # Check ChunkStore / dask overhead on top of basic memory copy
    start_time = time.process_time()
    success = py.compute()
    dask_duration = time.process_time() - start_time
    assert dask_duration < 10 * copy_duration
    np.testing.assert_equal(success, None)
