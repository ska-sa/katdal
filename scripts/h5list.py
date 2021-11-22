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
# Print compact one-line descriptions of multiple HDF5 files.
#
# Ludwig Schwardt
# 19 December 2011
#

import glob
import optparse
import os

import katdal

# See warnings while loading files (will appear *above* the relevant file)
# logging.basicConfig(format='%(levelname)s %(name)s %(message)s')

parser = optparse.OptionParser(usage="%prog [options] [filename or directory]*", description='List HDF5 files')
opts, args = parser.parse_args()
# Lists HDF5 files in current directory if no arguments where given
args = ['*.h5'] if not args else args

# Turn arguments (individual files, globs and directories) into a big list of HDF5 files
files = []
for arg in args:
    if '*' in arg:
        files.extend([name for name in glob.glob(arg) if name.endswith('.h5')])
    elif arg.endswith('.h5'):
        files.append(arg)
    else:
        for rootdir, subdirs, dirfiles in os.walk(arg):
            files.extend([os.path.join(rootdir, name) for name in dirfiles if name.endswith('.h5')])

# Open each file in turn and print a one-line summary
print("Name          Ver Observer   StartTimeSAST       Shape               SizeGB "
      "DumpHz SPW CFreqMHz Ants    Tgts Scans Description")
for f in files:
    try:
        d = katdal.open(f, quicklook=True)
    except Exception as e:
        print(f'{f} {e.__class__.__name__} - {e}')
        continue
    name = os.path.basename(f)
    name = (name[:10] + '...') if len(name) > 13 else name
    all_ants = ('ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7')
    file_ants = [ant.name for ant in d.ants]
    ants = ''.join([(ant[3:] if ant in file_ants else '-') for ant in all_ants])
    print('%13s %3s %10s %19s (%6d,%5d,%4d) %6.2f %6.3f %3d %8.3f %s %4d %5d %s' %
          (name, d.version, d.observer.strip()[:10].ljust(10), d.start_time.local()[:19],
           d.shape[0], d.shape[1], d.shape[2], d.size / 1024. / 1024. / 1024., 1.0 / d.dump_period,
           len(d.spectral_windows), d.spectral_windows[d.spw].centre_freq / 1e6, ants,
           len(d.catalogue), len(d.scan_indices), d.description))
    del d
