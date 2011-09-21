#!/usr/bin/python

# Produces a CASA compatible Measurement Set from a KAT-7 HDF5 file (versions 1 and 2),
# using the casacore table tools in the ms_extra module (or pyrap if casacore not available)

import os.path
import sys
import shutil
import tarfile
import optparse
import time

import numpy as np
import katpoint
import katfile
from katfile import ms_extra

 # NOTE: This should be checked before running (only for w stopping) to see how up to date the cable delays are !!!
delays = {}
delays['H'] = [23220.506e-9, 23283.799e-9, 23407.970e-9, 23514.801e-9, 23676.033e-9, 23782.854e-9, 24047.672e-9]
delays['V'] = [23228.551e-9, 23286.823e-9, 23400.221e-9, 23514.801e-9, 23668.223e-9, 23782.150e-9, 24039.237e-9]
 # updated by schwardt/simonr Aug 26th 2011

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Convert HDF5 file to MeasurementSet')
parser.add_option("-b", "--blank-ms", default="/var/kat/static/blank.ms", help="Blank MS used as template (default=%default)")
parser.add_option("-c", "--circular", action="store_true", default=False, help="Produce quad circular polarisation. (RR, RL, LR, LL) *** Currently just relabels the linear pols ****")
parser.add_option("-r" , "--ref-ant", help="Reference antenna (default is first one used by script)")
parser.add_option("-t", "--tar", action="store_true", default=False, help="Tar-ball the MS")
parser.add_option("-f", "--full_pol", action="store_true", default=False, help="Produce a full polarisation MS in CASA canonical order (HH, HV, VH, VV). Default is to produce HH,VV only")
parser.add_option("-v", "--verbose", action="store_true", default=False, help="More verbose progress information")
parser.add_option("-w", "--stop-w", action="store_true", default=False, help="Use W term to stop fringes for each baseline")
parser.add_option("-x", "--HH", action="store_true", default=False, help="Produce a Stokes I MeasurementSet using only HH")
parser.add_option("-y", "--VV", action="store_true", default=False, help="Produce a Stokes I MeasurementSet using only VV")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide one or more HDF5 filenames as arguments"
    sys.exit(1)

if options.HH and options.VV:
    print "You cannot include both HH and VV in the production of Stokes I (i.e. you specified --HH and --VV)."
    sys.exit(1)

if options.full_pol and (options.HH or options.VV):
    print "You have specified a full pol MS but also chosen to produce Stokes I (either HH or VV). Choose one or the other."
    sys.exit(1)

if len(args) > 1:
    print "Concatenating multiple h5 files into single MS."

if options.stop_w:
    print "W term in UVW coordinates will be used to stop the fringes."

if not ms_extra.casacore_binding:
    print "Failed to find casacore binding. You need to install both casacore and pyrap, or run the script from within a modified casapy containing h5py and katpoint."
    sys.exit(1)
else:
    print "Using '%s' casacore binding to produce MS" % (ms_extra.casacore_binding,)

pols_to_use = ['HH'] if options.HH else ['VV'] if options.VV else ['HH','HV','VH','VV'] if (options.full_pol or options.circular) else ['HH','VV']
 # which polarisation do we want to write into the MS and pull from the HDF5 file
pol_for_name = 'hh' if options.HH else 'vv' if options.VV else 'full_pol' if options.full_pol else 'circular_pol' if options.circular else 'hh_vv'
ms_name = os.path.splitext(args[0])[0] + ("." if len(args) == 1 else ".et_al.") + pol_for_name + ".ms"

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

if options.HH or options.VV: print "\n#### Producing Stokes I MS using " + ('HH' if options.HH else 'VV') + " only ####\n"
elif options.full_pol: print "\n#### Producing a full polarisation MS (HH,HV,VH,VV) ####\n"
else: print "\n#### Producing a two polarisation MS (HH, VV) ####\n"


# Open HDF5 file
h5 = katfile.open(args, ref_ant=options.ref_ant)
 # katfile can handle a list of files, which get virtually concatenated internally

print "\nUsing %s as the reference antenna. All targets and activity detection will be based on this antenna.\n" % (h5.ref_ant,)
# MS expects timestamps in MJD seconds
start_time = katpoint.Timestamp(h5.start_time).to_mjd() * 24 * 60 * 60
end_time = katpoint.Timestamp(h5.end_time).to_mjd() * 24 * 60 * 60

ms_dict = {}
ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in h5.ants], [ant.position_ecef for ant in h5.ants],
                                                    [ant.diameter for ant in h5.ants])
ms_dict['FEED'] = ms_extra.populate_feed_dict(len(h5.ants), num_receptors_per_feed=2)
ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(ms_pols=pols_to_use,stokes_i=(options.HH or options.VV),circular=options.circular)
ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(h5.start_time, h5.end_time, "KAT-7", h5.observer, h5.experiment_id)
ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(h5.channel_freqs, np.tile(h5.channel_bw, len(h5.channel_freqs)))

field_centers, field_times, field_names = [], [], []

print "Writing static meta data..."
ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)

ms_dict = {}
#increment scans sequentially in the ms
scan_itr = 1
print "\nIterating through scans in file(s)...\n"
main_table = ms_extra.open_main(ms_name, verbose=options.verbose)
 # prepare to write main dict
bls = None
file_scan_lengths = np.zeros((len(h5.files)))

for scan_ind, compscan_ind, scan_state, target in h5.scans():
    s = time.time()
    fid = h5.files.index(h5._current_file)
    file_scan_lengths[fid] = len(h5._current_file._scan_starts)
    scan_ind_relative = scan_ind if fid == 0 else scan_ind - (np.sum(file_scan_lengths[0:fid]))
    if bls is None:
        bls = h5._current_file.file['MetaData/Configuration/Correlator/'].attrs['bls_ordering'].tolist()
    tstamps = h5.timestamps()
    if scan_state != 'track':
        if options.verbose: print "scan %3d (%4d samples) skipped '%s'" % (scan_ind, len(tstamps), scan_state)
        continue
    if len(tstamps) < 2:
        if options.verbose: print "scan %3d (%4d samples) skipped - too short" % (scan_ind, len(tstamps))
        continue
    if target.body_type != 'radec':
        if options.verbose: print "scan %3d (%4d samples) skipped - target '%s' not RADEC" % (scan_ind, len(tstamps), target.name)
        continue
    print "scan %3d (%3d rel) (%4d samples) loaded. target: '%s'. Writing to disk..." % (scan_ind, scan_ind_relative, len(tstamps), target.name)

    # load all data for this scan...
    sstart = h5._current_file._scan_starts[scan_ind_relative]
    send = h5._current_file._scan_ends[scan_ind_relative]
    scan_data = h5._current_file.file['Data/correlator_data'][sstart:send+1,:,:].view(np.complex64).squeeze()
    #print "Scan start: %i, scan end: %i, length: %i\n" % (sstart, send, send - sstart)
    sz_mb = (scan_data.size * scan_data.dtype.itemsize) / (1024.0 * 1024.0)
    # MS expects timestamps in MJD seconds
    mjd_seconds = [katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60 for t in tstamps]
    # Update field lists if this is a new target
    if target.name not in field_names:
        # Since this will be an 'radec' target, we don't need an antenna or timestamp to get the (astrometric) ra, dec
        ra, dec = target.radec()
        field_names.append(target.name)
        field_centers.append((ra, dec))
        field_times.append(mjd_seconds[0])
        if options.verbose: print "Added new field %d: '%s' %s %s" % (len(field_names) - 1, target.name, ra, dec)
    field_id = field_names.index(target.name)
    for ant1_index, ant1 in enumerate(h5.ants):
        for ant2_index, ant2 in enumerate(h5.ants):
            if ant2_index < ant1_index:
                continue
            polprods = [("%s%s" % (ant1.name,p[0]), "%s%s" % (ant2.name,p[1])) for p in pols_to_use]
            pol_data = []
            uvw_coordinates = np.array(target.uvw(ant2, tstamps, ant1))
            for p in polprods:
                cable_delay = delays[p[0][-1]][ant2_index] - delays[p[1][-1]][ant1_index]
                 # cable delays specific to pol type
                vis_data = scan_data[:,:,bls.index(list(p))]
                if options.stop_w:
                 # Result of delay model in turns of phase. This is now frequency dependent so has shape (tstamps, channels)
                    turns = np.outer((uvw_coordinates[2] / katpoint.lightspeed) - cable_delay, h5.channel_freqs)
                    vis_data *= np.exp(-2j * np.pi * turns)
                pol_data.append(vis_data)
            vis_data = np.dstack(pol_data)
            ms_extra.write_rows(main_table, ms_extra.populate_main_dict(uvw_coordinates, vis_data, mjd_seconds, ant1_index, ant2_index, 1.0 / h5.dump_rate, field_id, scan_itr), verbose=options.verbose)
    s1 = time.time() - s
    print "Wrote scan data (%f MB) in %f s (%f MBps)\n" % (sz_mb, s1, sz_mb / s1)
    scan_itr+=1
main_table.close()
 # done writing main table

ms_dict = {}
ms_dict['FIELD'] = ms_extra.populate_field_dict(field_centers, field_times, field_names)
ms_dict['SOURCE'] = ms_extra.populate_source_dict(field_centers, field_times, h5.channel_freqs, field_names)

print "\nWriting dynamic fields to disk....\n"
# Finally we write the MS as per our created dicts
ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)
if options.tar:
    tar = tarfile.open('%s.tar' % (ms_name,), 'w')
    tar.add(ms_name, arcname=os.path.basename(ms_name))
    tar.close()
