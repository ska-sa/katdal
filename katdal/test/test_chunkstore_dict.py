################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
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

from katdal.chunkstore_dict import DictChunkStore
from katdal.test.test_chunkstore import ChunkStoreTestBase


class TestDictChunkStore(ChunkStoreTestBase):
    def setup(self):
        self.store = DictChunkStore(**vars(self))
        # This store is prepopulated so missing chunks can't be checked
        self.preloaded_chunks = True
