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

#
# Convert an FF holography file to an FF HDF5 v1 file as far as possible.
# Mostly this involves faking details of the reference antenna.
#
# Ludwig Schwardt
# 29 July 2013
#

from optparse import OptionParser

import h5py
import numpy as np

parser = OptionParser(usage="usage: %prog <file>",
                      description="Convert an FF holography HDF5 file to the FF HDF5 v1 "
                                  "format as far as possible. The file is modified in-place.")
opts, args = parser.parse_args()

filename = args[0]
f = h5py.File(filename)

ants_group, corr_group = f['Antennas'], f['Correlator']

# Vague attempt not to mess up the wrong HDF5 file
if 'version' in f.attrs or f.attrs['k7w_file_version'] != 3 or 'augment' in f.attrs or \
   len(ants_group) != 2 or len(ants_group[list(ants_group.keys())[-1]]) > 0:
    raise ValueError(f'{filename} does not seem to be an FF holography data file')

# First antenna => scan_ant (also reference), second antenna => ref_ant
scan_ant, ref_ant = ants_group.keys()
scan_ant, ref_ant = ants_group[scan_ant], ants_group[ref_ant]

# Fix description strings to have same location but unique antenna names and dish sizes
description = scan_ant.attrs['description'].split(',')[1:]
scan_ant.attrs['description'] = ','.join(['scan_ant'] + description)
ref_ant.attrs['description'] = ','.join(['ref_ant'] + description[:3] + [' 1.0'] + description[4:])

# Assume that the dish(es) first go onto the target for a 'cal' scan, implying
# that Scan0 is a 'slew' and Scan1 is a 'track' - pick a time within this scan
# to determine the (az, el) of the geostationary satellite
on_target_ts = f['Scans/CompoundScan0/Scan1/timestamps'][-1] / 1000.0
target_az, target_el = 0.0, 0.0
# Replicate the sensors if they don't exist
if 'Sensors' not in ref_ant:
    ref_ant.create_group('Sensors')
for k, v in scan_ant['Sensors'].items():
    # Create the position sensors with constant coordinates and copy the rest
    if k.startswith('pos_'):
        on_target = v['timestamp'].searchsorted(on_target_ts)
        long_ago = v['timestamp'][0]
        coord = v['value'][on_target]
        dataset = np.array([(long_ago, coord, 'nominal')], dtype=v.dtype)
        if k == 'pos_request_scan_azim':
            target_az = coord
        elif k == 'pos_request_scan_elev':
            target_el = coord
    else:
        dataset = scan_ant['Sensors'][k]
    if k not in ref_ant['Sensors']:
        ref_ant['Sensors'][k] = dataset

# Guess the satellite in use, based on rough (az, el) location
sat_name, lo_freq_hz = 'satellite', 1135.5e6
# We are using XDM
if description[0].startswith(' -25:53') and description[1].startswith(' 27:41'):
    print('It looks like the antennas were at HartRAO')
    # The default holography source at XDM
    if abs(target_az - (-26)) < 1.0 and abs(target_el - 57) < 1.0:
        sat_name, lo_freq_hz = 'EUTELSAT W2M', 1135.5e6
        print(f"It looks like the antennas pointed at '{sat_name}'")
# Create an azel target to replace the default dummy target
f['Scans']['CompoundScan0'].attrs['target'] = f'{sat_name}, azel, {target_az:g}, {target_el:g}'
f['Scans']['CompoundScan0'].attrs['label'] = 'holo'

# The correlator mapping is actually scan_ant => 0x and ref_ant => 0y, regardless of pol
pol = 'H'
if pol not in ref_ant:
    ref_ant.create_group(pol)
for k, v in scan_ant[pol].items():
    ref_ant[pol][k] = v
ref_ant[pol].attrs['dbe_input'] = '0y'
ref_ant[pol].attrs['delay_s'] = scan_ant[pol].attrs['delay_s']
# Only the first 4 corrprods are kept (0, 1, 2, 3) - update input map to reflect this
input_map = corr_group['input_map']
del corr_group['input_map']
corr_group['input_map'] = input_map[:4]
# The dump rate seems to be out by a factor of 128, based on timestamps vs shape
corr_group.attrs['accum_per_int'] *= 128
corr_group.attrs['dump_rate_hz'] /= 128
# The channel bandwidth is ideally ADC clock / 131072 = 524e6 / 2 / 8 / 8192
channel_bw = corr_group.attrs['channel_bandwidth_hz']
# The center frequency is set to the middle of channel 256 according
# to the frequency formula in Section 4.1 in Holography Manual SET/TM/001/E
# However, since katdal does not yet support USB mixing as in the holo downconverter,
# kludge this to channel 494 as the satellite RF signal is around channels 375-377
center_freq_hz = (375 - 256 + 375) * channel_bw + 10510e6 + lo_freq_hz + 56e6
corr_group.attrs['center_frequency_hz'] = center_freq_hz

# Update the augment attribute to flag the file as done
f.attrs['augment'] = 'Converted by ff_holo_to_h5v1'
f.close()
