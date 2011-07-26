#!/usr/bin/python

# Produces a CASA compatible Measurement Set from a KAT-7 HDF5 file (versions 1 and 2),
# using the casacore table tools in the ms_extra module

import os.path
import sys
import shutil
import tarfile
import optparse

import numpy as np
import katpoint
from katfile import h5_data, ms_extra

 # NOTE: This should be checked before running (only for w stopping) to see how up to date the cable delays are !!!
#delays = [478.041e-9, 545.235e-9, 669.900e-9, 772.868e-9, 600.0e-9, 600.0e-9, 600.0e-9]
 # updated by simonr July 5th 2010
delays = [23243.947e-9, 23297.184e-9, 23406.078e-9, 23514.801e-9, 23676.916e-9, 23784.112e-9, 24047.285e-9]
 # updated by schwardt July 23rd 2011

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Convert HDF5 file to MeasurementSet')
parser.add_option("-b", "--blank-ms", default="/var/kat/static/blank.ms", help="Blank MS used as template (default=%default)")
parser.add_option("-r" , "--ref-ant", help="Reference antenna (default is first one used by script)")
parser.add_option("-t", "--tar", action="store_true", default=False, help="Tar-ball the MS")
parser.add_option("-w", "--stop-w", action="store_true", default=False, help="Use W term to stop fringes for each baseline")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide HDF5 filename as argument"
    sys.exit(1)
h5_filename = args[0]

if options.stop_w:
    print "W term in UVW coordinates will be used to stop the fringes."

if not ms_extra.casacore_binding:
    print "Failed to find casacore binding. You need to install both casacore and pyrap, or run the script from within a modified casapy containing h5py and katpoint."
    sys.exit(1)
else:
    print "Using '%s' casacore binding to produce MS" % (ms_extra.casacore_binding,)
ms_name = os.path.splitext(h5_filename)[0] + ".ms"

# The first step is to copy the blank template MS to our desired output (making sure it's not already there)
if os.path.exists(ms_name):
    print "MS '%s' already exists - please remove it before running this script" % (ms_name,)
    sys.exit(0)
try:
    shutil.copytree(options.blank_ms, ms_name)
except OSError:
    print "Failed to copy blank MS from %s to %s - please check presence of blank MS and/or permissions" % (options.blank_ms, ms_name)
    sys.exit(1)

print "Will create MS output in", ms_name

# Open HDF5 file
h5 = h5_data(h5_filename, ref_ant=options.ref_ant)

print "\nUsing %s as the reference antenna. All targets and activity detection will be based on this antenna.\n" % (h5.ref_ant,)
# MS expects timestamps in MJD seconds
start_time = katpoint.Timestamp(h5.start_time).to_mjd() * 24 * 60 * 60
end_time = katpoint.Timestamp(h5.end_time).to_mjd() * 24 * 60 * 60

ms_dict = {}
ms_dict['MAIN'] = []
ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in h5.ants], [ant.position_ecef for ant in h5.ants],
                                                    [ant.diameter for ant in h5.ants])
ms_dict['FEED'] = ms_extra.populate_feed_dict(len(h5.ants), num_receptors_per_feed=2)
ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(pol_type='HV')
ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(h5.start_time, h5.end_time, "KAT-7", h5.observer, h5.experiment_id)
ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(h5.channel_freqs, np.tile(h5.channel_bw, len(h5.channel_freqs)))

field_centers, field_times, field_names = [], [], []

#increment scans sequentially in the ms
scan_itr = 1
for scan_ind, compscan_ind, scan_state, target in h5.scans():
    tstamps = h5.timestamps()
    if scan_state != 'track':
        print "scan %3d (%4d samples) skipped '%s'" % (scan_ind, len(tstamps), scan_state)
        continue
    if len(tstamps) < 2:
        print "scan %3d (%4d samples) skipped - too short" % (scan_ind, len(tstamps))
        continue
    if target.body_type != 'radec':
        print "scan %3d (%4d samples) skipped - target '%s' not RADEC" % (scan_ind, len(tstamps), target.name)
        continue
    # MS expects timestamps in MJD seconds
    mjd_seconds = [katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60 for t in tstamps]
    # Update field lists if this is a new target
    if target.name not in field_names:
        # Since this will be an 'radec' target, we don't need an antenna or timestamp to get the (astrometric) ra, dec
        ra, dec = target.radec()
        field_names.append(target.name)
        field_centers.append((ra, dec))
        field_times.append(mjd_seconds[0])
        print "Added new field %d: '%s' %s %s" % (len(field_names) - 1, target.name, ra, dec)
    field_id = field_names.index(target.name)
    
    for ant1_index, ant1 in enumerate(h5.ants):
        for ant2_index, ant2 in enumerate(h5.ants):
            if ant2_index < ant1_index:
                continue
            # This is the order in which the polarisation products are expected, according to POLARIZATION table
            polprods = [(ant1.name + 'H', ant2.name + 'H'), (ant1.name + 'V', ant2.name + 'V'),
                        (ant1.name + 'H', ant2.name + 'V'), (ant1.name + 'V', ant2.name + 'H')]
            # Create 3-dim complex data array with shape (tstamps, channels, pols)
            vis_data = np.dstack([h5.vis(prod, zero_missing_data=True) for prod in polprods])
            uvw_coordinates = np.array(target.uvw(ant2, tstamps, ant1))
            if options.stop_w:
                # NB: this is not completely correct, as the cable delay is per pol (signal path) and not per antenna
                cable_delay_diff = delays[ant2_index] - delays[ant1_index]
                # Result of delay model in turns of phase. This is now frequency dependent so has shape (tstamps, channels)
                turns = np.outer((uvw_coordinates[2] / katpoint.lightspeed) + cable_delay_diff, h5.channel_freqs)
                vis_data *= np.exp(-2j * np.pi * turns[:, :, np.newaxis])
            ms_dict['MAIN'].append(ms_extra.populate_main_dict(uvw_coordinates, vis_data, mjd_seconds,
                                                               ant1_index, ant2_index, 1.0 / h5.dump_rate, field_id, scan_itr))
    scan_itr+=1

ms_dict['FIELD'] = ms_extra.populate_field_dict(field_centers, field_times, field_names)

# Finally we write the MS as per our created dicts
ms_extra.write_dict(ms_dict, ms_name)
if options.tar:
    tar = tarfile.open('%s.tar' % (ms_name,), 'w')
    tar.add(ms_name, arcname=os.path.basename(ms_name))
    tar.close()
