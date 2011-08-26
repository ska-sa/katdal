#!/usr/bin/env python

# Update the antenna positions in the specified HDF5 file.

import h5py
import sys

if len(sys.argv) < 2:
    print "Update antenna positions in the specified HDF5 file.\n\nUsage: fix_ant_positions.py <filename.h5>\n"
    sys.exit()

f = h5py.File(sys.argv[1])

f['MetaData/Configuration/Antennas/ant1'].attrs['description'] = 'ant1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 25.095 -9.095 0.045, , 1.22'
f['MetaData/Configuration/Antennas/ant2'].attrs['description'] = 'ant2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 90.288 26.389 -0.238, , 1.22'
f['MetaData/Configuration/Antennas/ant3'].attrs['description'] = 'ant3, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 3.985 26.899 -0.012, , 1.22'
f['MetaData/Configuration/Antennas/ant4'].attrs['description'] = 'ant4, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -21.600 25.500 0.000, , 1.22'
f['MetaData/Configuration/Antennas/ant5'].attrs['description'] = 'ant5, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -38.264 -2.586 0.371, , 1.22'
f['MetaData/Configuration/Antennas/ant6'].attrs['description'] = 'ant6, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -61.580 -79.685 0.690, , 1.22'
f['MetaData/Configuration/Antennas/ant7'].attrs['description'] = 'ant7, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, -87.979 75.756 0.125, , 1.22'

f.close()

print "Updated antenna positions..."

