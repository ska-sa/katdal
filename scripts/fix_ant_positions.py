#!/usr/bin/env python

################################################################################
# Copyright (c) 2011-2021, National Research Foundation (SARAO)
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

# Update the antenna positions in the specified HDF5 file.

import sys

import h5py

if len(sys.argv) < 2:
    print("Update antenna positions in the specified HDF5 file.\n\nUsage: fix_ant_positions.py <filename.h5>\n")
    sys.exit()

f = h5py.File(sys.argv[1])

f['MetaData/Configuration/Antennas/ant1'].attrs['description'] = 'ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.0950 -9.0950 0.0450, , 1.22'
f['MetaData/Configuration/Antennas/ant2'].attrs['description'] = 'ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.2844 26.3804 -0.22636, , 1.22'
f['MetaData/Configuration/Antennas/ant3'].attrs['description'] = 'ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.98474 26.8929 0.0004046, , 1.22'
f['MetaData/Configuration/Antennas/ant4'].attrs['description'] = 'ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.6053 25.4936 0.018615, , 1.22'
f['MetaData/Configuration/Antennas/ant5'].attrs['description'] = 'ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.2720 -2.5917 0.391362, , 1.22'
f['MetaData/Configuration/Antennas/ant6'].attrs['description'] = 'ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.5945 -79.6989 0.701598, , 1.22'
f['MetaData/Configuration/Antennas/ant7'].attrs['description'] = 'ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.9881 75.7543 0.138305, , 1.22'

f.close()

print("Updated antenna positions...")
