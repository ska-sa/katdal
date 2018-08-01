################################################################################
# Copyright (c) 2011-2018, National Research Foundation (Square Kilometre Array)
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

"""Common utilities for working with H5 files"""

from __future__ import print_function, division, absolute_import

import future.utils

import h5py


class AttributeManager(h5py.AttributeManager):
    """AttributeManager that fixes https://github.com/h5py/h5py/issues/379"""
    if future.utils.PY3:
        def __getitem__(self, name):
            value = super(AttributeManager, self).__getitem__(name)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return value


def h5attrs(obj):
    """Access the H5 attributes with a workaround for https://github.com/h5py/h5py/issues/379"""
    # The locking is based on the HLObject.attrs in h5py, but may not actually
    # be necessary.
    with h5py.h5a.phil:
        return AttributeManager(obj)
