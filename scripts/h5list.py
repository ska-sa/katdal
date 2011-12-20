#! /usr/bin/env python
#
# Print compact one-line descriptions of multiple HDF5 files.
#
# Ludwig Schwardt
# 19 December 2011
#

import os
import optparse
import glob
import logging

import katfile

# See warnings while loading files (will appear *above* the relevant file, emphasised by an empty line above warning)
logging.basicConfig(format='\n%(levelname)s %(name)s %(message)s')

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
print "Name          Ver Observer   StartTimeSAST       Shape               SizeGB DumpHz SPW CFreqMHz Ants Tgts Scans Description"
for f in files:
    d = katfile.open(f)
    name = os.path.basename(f)
    name = (name[:10] + '...') if len(name) > 13 else name
    print '%13s %3s %10s %19s (%6d,%5d,%4d) %6.2f %6.3f %3d %8.3f %4d %4d %5d %s' % \
          (name, d.version, d.observer.strip()[:10].ljust(10), d.start_time.local()[:19],
           d.shape[0], d.shape[1], d.shape[2], d.size / 1024. / 1024. / 1024., 1.0 / d.dump_period,
           len(d.spectral_windows), d.spectral_windows[d.spw].centre_freq / 1e6, len(d.ants),
           len(d.catalogue), len(d.scan_indices), d.description)
    del d
