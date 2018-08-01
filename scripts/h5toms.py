#!/usr/bin/env python

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

# Produce a CASA compatible Measurement Set from a KAT-7 HDF5 file (versions
# 1 and 2) or MeerKAT HDF5 file (version 3) using the casapy table tools
# in the ms_extra module (or pyrap/casacore if casapy is not available).

from __future__ import print_function, division, absolute_import

import mvftoms


if __name__ == '__main__':
    print("h5toms.py is deprecated and has been replaced by mvftoms.py")
    print("For now it is an alias to run mvftoms.py")
    print()
    mvftoms.main()
