#!/usr/bin/env python

################################################################################
# Copyright (c) 2011-2016, National Research Foundation (Square Kilometre Array)
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

import itertools
import os
import shutil
import tarfile
import optparse
import time
import pickle

import numpy as np

import katpoint
import katdal
from katdal import averager
from katdal import ms_extra


tag_to_intent = {'gaincal': 'CALIBRATE_PHASE,CALIBRATE_AMPLI',
                 'bpcal': 'CALIBRATE_BANDPASS,CALIBRATE_FLUX',
                 'target': 'TARGET'}

parser = optparse.OptionParser(usage="%prog [options] <filename.h5> [<filename2.h5>]*",
                               description='Convert HDF5 file(s) to MeasurementSet')
parser.add_option("-b", "--blank-ms", default="/var/kat/static/blank.ms",
                  help="Blank MS used as template (default=%default)")
parser.add_option("-o", "--output-ms", default=None,
                  help="Output Measurement Set")
parser.add_option("-c", "--circular", action="store_true", default=False,
                  help="Produce quad circular polarisation. (RR, RL, LR, LL) "
                       "*** Currently just relabels the linear pols ****")
parser.add_option("-r", "--ref-ant",
                  help="Reference antenna (default is first one used by script)")
parser.add_option("-t", "--tar", action="store_true", default=False,
                  help="Tar-ball the MS")
parser.add_option("-f", "--full_pol", action="store_true", default=False,
                  help="Produce a full polarisation MS in CASA canonical order "
                       "(HH, HV, VH, VV). Default is to produce HH,VV only")
parser.add_option("-v", "--verbose", action="store_true", default=False,
                  help="More verbose progress information")
parser.add_option("-w", "--stop-w", action="store_true", default=False,
                  help="Use W term to stop fringes for each baseline")
parser.add_option("-p", "--pols-to-use", default=None,
                  help="Select polarisation products to include in MS as comma separated list "
                       "(from: HH, HV, VH, VV). Default is all available from HH, VV")
parser.add_option("-u", "--uvfits", action="store_true", default=False,
                  help="Print command to convert MS to miriad uvfits in casapy")
parser.add_option("-a", "--no-auto", action="store_true", default=False,
                  help="MeasurementSet will exclude autocorrelation data")
parser.add_option("-s", "--keep-spaces", action="store_true", default=False,
                  help="Keep spaces in source names, default removes spaces")
parser.add_option("-C", "--channel-range",
                  help="Range of frequency channels to keep (zero-based inclusive "
                       "'first_chan,last_chan', default is all channels)")
parser.add_option("-e", "--elevation-range",
                  help="Flag elevations outside the range 'lowest_elevation,highest_elevation'")
parser.add_option("-m", "--model-data", action="store_true", default=False,
                  help="Add MODEL_DATA and CORRECTED_DATA columns to the MS. "
                       "MODEL_DATA initialised to unity amplitude zero phase, "
                       "CORRECTED_DATA initialised to DATA.")
parser.add_option("--flags", default="all",
                  help="List of online flags to apply "
                       "(from 'static,cam,data_lost,ingest_rfi,cal_rfi,predicted_rfi', "
                       "default is all flags, '' will apply no flags)")
parser.add_option("--dumptime", type=float, default=0.0,
                  help="Output time averaging interval in seconds, default is no averaging.")
parser.add_option("--chanbin", type=int, default=0,
                  help="Bin width for channel averaging in channels, default is no averaging.")
parser.add_option("--flagav", action="store_true", default=False,
                  help="If a single element in an averaging bin is flagged, flag the averaged bin.")
parser.add_option("--caltables", action="store_true", default=False,
                  help="Create calibration tables from gain solutions in the h5 file (if present).")

(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    parser.print_help()
    raise RuntimeError("Please provide one or more HDF5 filenames as arguments")

if options.elevation_range and len(options.elevation_range.split(',')) < 2:
    raise RuntimeError("You have selected elevation flagging. Please provide elevation "
                       "limits in the form 'lowest_elevation,highest_elevation'.")

if len(args) > 1:
    print "Concatenating multiple h5 files into single MS."

if not ms_extra.casacore_binding:
    raise RuntimeError("Failed to find casacore binding. You need to install both "
                       "casacore and pyrap, or run the script from within a modified "
                       "casapy containing h5py and katpoint.")
else:
    print "Using '%s' casacore binding to produce MS" % (ms_extra.casacore_binding,)

# Open HDF5 file
# if len(args) == 1: args = args[0]
# katdal can handle a list of files, which get virtually concatenated internally
h5 = katdal.open(args, ref_ant=options.ref_ant)

#Get list of unique polarisation products in the file
pols_in_file = np.unique([(cp[0][-1] + cp[1][-1]).upper() for cp in h5.corr_products])

#Which polarisation do we want to write into the MS
#select all possible pols if full-pol selected, otherwise the selected polarisations via pols_to_use
#otherwise finally select any of HH,VV present (the default).
pols_to_use = ['HH', 'HV', 'VH', 'VV'] if (options.full_pol or options.circular) else \
              list(np.unique(options.pols_to_use.split(','))) if options.pols_to_use else \
              [pol for pol in ['HH','VV'] if pol in pols_in_file]

#Check we have the chosen polarisations
if np.any([pol not in pols_in_file for pol in pols_to_use]):
    raise RuntimeError("Selected polarisation(s): %s not available. "
                       "Available polarisation(s): %s"%(','.join(pols_to_use),','.join(pols_in_file)))

#Set full_pol if this is selected via options.pols_to_use
if set(pols_to_use) == set(['HH', 'HV', 'VH', 'VV']) and not options.circular:
    options.full_pol=True

pol_for_name = 'full_pol' if options.full_pol else \
               'circular_pol' if options.circular else \
               '_'.join(pols_to_use).lower()

# ms_name = os.path.splitext(args[0])[0] + ("." if len(args) == 1 else ".et_al.") + pol_for_name + ".ms"

for win in range(len(h5.spectral_windows)):
    h5.select(reset='T')

    # Extract MS file per spectral window in H5 observation file
    print 'Extract MS for spw %d: central frequency %.2f MHz' % (win, (h5.spectral_windows[win]).centre_freq / 1e6)
    cen_freq = '%d' % int(h5.spectral_windows[win].centre_freq / 1e6)

    # If no output MS filename supplied, infer the output filename
    # from the first hdf5 file.
    if options.output_ms is None:
        basename = ('%s_%s' % (os.path.splitext(args[0])[0], cen_freq)) + \
                   ("." if len(args) == 1 else ".et_al.") + pol_for_name
        # create MS in current working directory
        ms_name = basename + ".ms"
    else:
        ms_name = options.output_ms
    # for the calibration table base name, use the ms name without the .ms extension, if there is a .ms extension
    # otherwise use the ms name
    caltable_basename = ms_name[:-3] if ms_name.lower().endswith('.ms') else ms_name

    h5.select(spw=win, scans='track', flags=options.flags)

    # The first step is to copy the blank template MS to our desired output (making sure it's not already there)
    if os.path.exists(ms_name):
        raise RuntimeError("MS '%s' already exists - please remove it before running this script" % (ms_name,))
    try:
        shutil.copytree(options.blank_ms, ms_name)
    except OSError:
        raise RuntimeError("Failed to copy blank MS from %s to %s - please check presence "
                           "of blank MS and/or permissions" % (options.blank_ms, ms_name))

    print "Will create MS output in", ms_name

    # Instructions to flag by elevation if requested
    if options.elevation_range is not None:
        emin, emax = options.elevation_range.split(',')
        print "\nThe MS can be flagged by elevation in casapy v3.4.0 or higher, with the command:"
        print "      tflagdata(vis='%s', mode='elevation', lowerlimit=%s, upperlimit=%s, action='apply')\n" % \
              (ms_name, emin, emax)

    # Instructions to create uvfits file if requested
    if options.uvfits:
        # uv_name = os.path.splitext(args[0])[0] + ("." if len(args) == 1 else ".et_al.") + pol_for_name + ".uvfits"
        uv_name = basename + ".uvfits"
        print "\nThe MS can be converted into a uvfits file in casapy, with the command:"
        print "      exportuvfits(vis='%s', fitsfile='%s', datacolumn='data')\n" % (ms_name, uv_name)

    if options.full_pol:
        print "\n#### Producing a full polarisation MS (HH,HV,VH,VV) ####\n"
    else:
        print "\n#### Producing MS with %s polarisation(s) ####\n"%(','.join(pols_to_use))

    # # Open HDF5 file
    # if len(args) == 1: args = args[0]
    # h5 = katdal.open(args, ref_ant=options.ref_ant)
    #  # katdal can handle a list of files, which get virtually concatenated internally

    # if fringe stopping is requested, check that it has not already been done in hardware
    if options.stop_w:
        print "W term in UVW coordinates will be used to stop the fringes."
        try:
            autodelay = [int(ad) for ad in h5.sensor['DBE/auto-delay']]
            if all(autodelay):
                print "Fringe-stopping already performed in hardware... do you really want to stop the fringes here?"
        except KeyError:
            pass

    # Select frequency channel range
    if options.channel_range is not None:
        channel_range = [int(chan_str) for chan_str in options.channel_range.split(',')]
        first_chan, last_chan = channel_range[0], channel_range[1]

        if (first_chan < 0) or (last_chan >= h5.shape[1]):
            raise RuntimeError("Requested channel range outside data set boundaries. "
                               "Set channels in the range [0,%s]" % (h5.shape[1] - 1,))

        chan_range = slice(first_chan, last_chan + 1)
        print "\nChannel range %s through %s." % (first_chan, last_chan)
        h5.select(channels=chan_range)

    # Are we averaging?
    average_data = False

    # Work out channel average and frequency increment
    if options.chanbin > 1:
        average_data = True
        # Check how many channels we are dropping
        numchans = len(h5.channels)
        chan_remainder = numchans % options.chanbin
        print "Averaging %s channels, output ms will have %s channels." % \
              (options.chanbin, int(numchans / min(numchans, options.chanbin)))
        if chan_remainder > 0:
            print "The last %s channels in the data will be dropped during averaging " \
                  "(%s does not divide %s)." % (chan_remainder, options.chanbin, numchans)
        chan_av = options.chanbin
    else:
        # No averaging in channel
        chan_av = 1

    # Get the frequency increment per averaged channel
    channel_freq_width = h5.channel_width * chan_av

    # Work out dump average and dump increment
    # Is the desired time bin greater than the dump period?
    if options.dumptime > h5.dump_period:
        average_data = True
        dump_av = int(np.round(options.dumptime / h5.dump_period))
        time_av = dump_av * h5.dump_period
        print "Averaging %s second dumps to %s seconds." % (h5.dump_period, time_av)
    else:
        # No averaging in time
        dump_av = 1
        time_av = h5.dump_period

    # Print a message if extending flags to averaging bins.
    if average_data and options.flagav and options.flags != '':
        print "Extending flags to averaging bins."

    # Optionally keep only cross-correlation products
    if options.no_auto:
        h5.select(corrprods='cross')
        print "\nCross-correlations only."

    print "\nUsing %s as the reference antenna. All targets and activity " \
          "detection will be based on this antenna.\n" % (h5.ref_ant,)
    # MS expects timestamps in MJD seconds
    start_time = h5.start_time.to_mjd() * 24 * 60 * 60
    end_time = h5.end_time.to_mjd() * 24 * 60 * 60
    # Version 1 and 2 files are KAT-7; the rest are MeerKAT
    telescope_name = 'KAT-7' if h5.version[0] in '12' else 'MeerKAT'

    ms_dict = {}
    ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in h5.ants],
                                                        [ant.position_ecef for ant in h5.ants],
                                                        [ant.diameter for ant in h5.ants])
    ms_dict['FEED'] = ms_extra.populate_feed_dict(len(h5.ants), num_receptors_per_feed=2)
    ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
    ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(ms_pols=pols_to_use,
                                                                  circular=options.circular)
    ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(start_time, end_time, telescope_name,
                                                                h5.observer, h5.experiment_id)

    print "Writing static meta data..."
    ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)

    # before resetting ms_dict, copy subset to caltable dictionary
    if options.caltables:
        caltable_dict = {}
        caltable_dict['ANTENNA'] = ms_dict['ANTENNA']
        caltable_dict['OBSERVATION'] = ms_dict['OBSERVATION']

    ms_dict = {}
    # increment scans sequentially in the ms
    scan_itr = 1
    print "\nIterating through scans in file(s)...\n"
    #  prepare to write main dict
    main_table = ms_extra.open_main(ms_name, verbose=options.verbose)
    corrprod_to_index = dict([(tuple(cp), ind) for cp, ind in zip(h5.corr_products, range(len(h5.corr_products)))])

    # ==========================================
    # Generate per-baseline antenna pairs and
    # correlator product indices
    # ==========================================

    # Generate baseline antenna pairs
    na = len(h5.ants)
    ant1_index, ant2_index = np.triu_indices(na, 1 if options.no_auto else 0)
    ant1 = [h5.ants[a1] for a1 in ant1_index]
    ant2 = [h5.ants[a2] for a2 in ant2_index]

    def _cp_index(a1, a2, pol):
        """
        Create individual correlator product index
        from antenna pair and polarisation
        """
        a1 = "%s%s" % (a1.name, pol[0].lower())
        a2 = "%s%s" % (a2.name, pol[1].lower())

        return corrprod_to_index.get((a1, a2))

    nbl = ant1_index.size
    npol = len(pols_to_use)

    # Create actual correlator product index
    cp_index = np.asarray([_cp_index(a1, a2, p)
                           for a1, a2 in itertools.izip(ant1, ant2)
                           for p in pols_to_use])

    # Identify missing correlator products
    # Reshape for broadcast on time and frequency dimensions
    missing_cp = np.logical_not([i is not None for i in cp_index])

    # Zero any None indices, but use the above masks to reason
    # about there existence in later code
    cp_index[missing_cp] = 0

    field_names, field_centers, field_times = [], [], []
    obs_modes = ['UNKNOWN']
    total_size_mb = 0.0

    for scan_ind, scan_state, target in h5.scans():
        s = time.time()
        scan_len = h5.shape[0]
        if scan_state != 'track':
            if options.verbose:
                print "scan %3d (%4d samples) skipped '%s' - not a track" % (scan_ind, scan_len, scan_state)
            continue
        if scan_len < 2:
            if options.verbose:
                print "scan %3d (%4d samples) skipped - too short" % (scan_ind, scan_len)
            continue
        if target.body_type != 'radec':
            if options.verbose:
                print "scan %3d (%4d samples) skipped - target '%s' not RADEC" % (scan_ind, scan_len, target.name)
            continue
        print "scan %3d (%4d samples) loaded. Target: '%s'. Writing to disk..." % (scan_ind, scan_len, target.name)

        # Get the average dump time for this scan (equal to scan length if the dump period is longer than a scan)
        dump_time_width = min(time_av, scan_len * h5.dump_period)

        scan_size_mb = 0.0
        # Get UTC timestamps
        utc_seconds = h5.timestamps[:]
        # Update field lists if this is a new target
        if target.name not in field_names:
            # Since this will be an 'radec' target, we don't need antenna or timestamp to get the (astrometric) ra, dec
            ra, dec = target.radec()

            field_names.append(target.name)
            field_centers.append((ra, dec))
            field_times.append(katpoint.Timestamp(utc_seconds[0]).to_mjd() * 60 * 60 * 24)
            if options.verbose:
                print "Added new field %d: '%s' %s %s" % (len(field_names) - 1, target.name, ra, dec)
        field_id = field_names.index(target.name)

        # Determine the observation tag for this scan
        obs_tag = []
        for tag in target.tags:
            if tag in tag_to_intent:
                obs_tag.append(tag_to_intent[tag])
        obs_tag = ','.join(obs_tag)
        # add tag to obs_modes list
        if obs_tag and obs_tag not in obs_modes:
            obs_modes.append(obs_tag)
        # get state_id from obs_modes list if it is in the list, else 0 'UNKNOWN'
        state_id = obs_modes.index(obs_tag) if obs_tag in obs_modes else 0

        # Iterate over time in some multiple of dump average
        ntime = utc_seconds.size
        tsize = dump_av
        ntime_av = 0

        for ltime in xrange(0, ntime - tsize + 1, tsize):

            utime = ltime + tsize
            tdiff = utime - ltime
            out_freqs = h5.channel_freqs
            nchan = out_freqs.size

            # load all visibility, weight and flag data
            # for this scan's timestamps.
            # Ordered (ntime, nchan, nbl*npol)
            scan_data = h5.vis[ltime:utime, :, :]
            scan_weight_data = h5.weights[ltime:utime, :, :]
            scan_flag_data = h5.flags[ltime:utime, :, :]

            # Select correlator products
            # cp_index could be used above when the LazyIndexer
            # supports advanced integer indices
            vis_data = scan_data[:, :, cp_index]

            weight_data = scan_weight_data[:, :, cp_index]
            flag_data = scan_flag_data[:, :, cp_index]

            # Zero and flag any missing correlator products
            vis_data[:, :, missing_cp] = 0 + 0j
            weight_data[:, :, missing_cp] = 0
            flag_data[:, :, missing_cp] = True

            out_utc = utc_seconds[ltime:utime]

            # Overwrite the input visibilities with averaged visibilities,flags,weights,timestamps,channel freqs
            if average_data:
                vis_data, weight_data, flag_data, out_utc, out_freqs = \
                    averager.average_visibilities(vis_data, weight_data, flag_data, out_utc, out_freqs,
                                                  timeav=dump_av, chanav=chan_av, flagav=options.flagav)

                # Infer new time and channel dimensions from averaged data
                tdiff, nchan = vis_data.shape[0], vis_data.shape[1]

            # Increment the number of averaged dumps
            ntime_av += tdiff

            def _separate_baselines_and_pols(array):
                """
                (1) Separate correlator product into baseline and polarisation,
                (2) rotate baseline between time and channel,
                (3) group time and baseline together
                """
                S = array.shape[:2] + (nbl, npol)
                return array.reshape(S).transpose(0, 2, 1, 3).reshape(-1, nchan, npol)

            def _create_uvw(a1, a2, times):
                """
                Return a (ntime, 3) array of UVW coordinates for baseline
                defined by a1 and a2. The sign convention matches `CASA`_,
                rather than the Measurement Set `definition`_.

                .. _CASA: https://casa.nrao.edu/Memos/CoordConvention.pdf
                .. _definition: https://casa.nrao.edu/Memos/229.html#SECTION00064000000000000000
                """
                uvw = target.uvw(a1, timestamp=times, antenna=a2)
                return np.asarray(uvw).T

            # Massage visibility, weight and flag data from
            # (ntime, nchan, nbl*npol) ordering to (ntime*nbl, nchan, npol)
            vis_data, weight_data, flag_data = (_separate_baselines_and_pols(a)
                                                for a in (vis_data, weight_data, flag_data))

            # Iterate through baselines, computing UVW coordinates
            # for a chunk of timesteps
            uvw_coordinates = np.concatenate([
                _create_uvw(a1, a2, out_utc)[:, np.newaxis, :]
                for a1, a2 in itertools.izip(ant1, ant2)], axis=1).reshape(-1, 3)

            # Convert averaged UTC timestamps to MJD seconds.
            # Blow time up to (ntime*nbl,)
            out_mjd = np.asarray([katpoint.Timestamp(time_utc).to_mjd() * 24 * 60 * 60
                                  for time_utc in out_utc])

            out_mjd = np.broadcast_to(out_mjd[:, np.newaxis], (tdiff, nbl)).ravel()

            # Repeat antenna indices to (ntime*nbl,)
            a1 = np.broadcast_to(ant1_index[np.newaxis, :], (tdiff, nbl)).ravel()
            a2 = np.broadcast_to(ant2_index[np.newaxis, :], (tdiff, nbl)).ravel()

            # Blow field ID up to (ntime*nbl,)
            big_field_id = np.full((tdiff * nbl,), field_id, dtype=np.int32)
            big_state_id = np.full((tdiff * nbl,), state_id, dtype=np.int32)
            big_scan_itr = np.full((tdiff * nbl,), scan_itr, dtype=np.int32)

            # Setup model_data and corrected_data if required
            model_data = None
            corrected_data = None

            if options.model_data:
                # unity intensity zero phase model data set, same shape as vis_data
                model_data = np.ones(vis_data.shape, dtype=np.complex64)
                # corrected data set copied from vis_data
                corrected_data = vis_data

            # write the data to the ms.
            main_dict = ms_extra.populate_main_dict(uvw_coordinates, vis_data, flag_data,
                                                    out_mjd, a1, a2,
                                                    dump_time_width, big_field_id, big_state_id,
                                                    big_scan_itr, model_data, corrected_data)
            ms_extra.write_rows(main_table, main_dict, verbose=options.verbose)

            # Increment the filesize.
            scan_size_mb += vis_data.dtype.itemsize * vis_data.size / (1024.0 * 1024.0)
            scan_size_mb += weight_data.dtype.itemsize * weight_data.size / (1024.0 * 1024.0)
            scan_size_mb += flag_data.dtype.itemsize * flag_data.size / (1024.0 * 1024.0)

            if options.model_data:
                scan_size_mb += model_data.dtype.itemsize * model_data.size / (1024.0 * 1024.0)
                scan_size_mb += corrected_data.dtype.itemsize * corrected_data.size / (1024.0 * 1024.0)

        s1 = time.time() - s
        if average_data and utc_seconds.shape != ntime_av:
            print "Averaged %s x %s second dumps to %s x %s second dumps" % \
                  (np.shape(utc_seconds)[0], h5.dump_period, ntime_av, dump_time_width)
        print "Wrote scan data (%f MB) in %f s (%f MBps)\n" % (scan_size_mb, s1, scan_size_mb / s1)
        scan_itr += 1
        total_size_mb += scan_size_mb

    if total_size_mb == 0.0:
        raise RuntimeError("No usable data found in HDF5 file (pick another reference antenna, maybe?)")

    # Remove spaces from source names, unless otherwise specified
    field_names = [f.replace(' ', '') for f in field_names] if not options.keep_spaces else field_names

    ms_dict = {}
    ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(out_freqs,
                                                                        channel_freq_width * np.ones(len(out_freqs)))
    ms_dict['FIELD'] = ms_extra.populate_field_dict(field_centers, field_times, field_names)
    ms_dict['STATE'] = ms_extra.populate_state_dict(obs_modes)
    ms_dict['SOURCE'] = ms_extra.populate_source_dict(field_centers, field_times, out_freqs, field_names)

    print "\nWriting dynamic fields to disk....\n"
    # Finally we write the MS as per our created dicts
    ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)
    if options.tar:
        tar = tarfile.open('%s.tar' % (ms_name,), 'w')
        tar.add(ms_name, arcname=os.path.basename(ms_name))
        tar.close()

    # --------------------------------------
    # Now write calibration product tables if required
    # Open first HDF5 file in the list to extract TelescopeState parameters from
    #   (can't extract telstate params from contatenated katdal file as it uses the hdf5 file directly)
    first_h5 = katdal.open(args[0], ref_ant=options.ref_ant)

    if options.caltables:
        # copy extra subtable dictionary values necessary for caltable
        caltable_dict['SPECTRAL_WINDOW'] = ms_dict['SPECTRAL_WINDOW']
        caltable_dict['FIELD'] = ms_dict['FIELD']

        solution_types = ['G', 'B', 'K']
        ms_soltype_lookup = {'G': 'G Jones', 'B': 'B Jones', 'K': 'K Jones'}

        print "\nWriting calibration solution tables to disk...."
        if 'TelescopeState' not in first_h5.file.keys():
            print " No TelescopeState in first H5 file. Can't create solution tables.\n"
        else:
            # first get solution antenna ordering
            #   newer h5 files have the cal antlist as a sensor
            if 'cal_antlist' in first_h5.file['TelescopeState'].keys():
                a0 = first_h5.file['TelescopeState']['cal_antlist'].value
                antlist = pickle.loads(a0[0][1])
            #   older h5 files have the cal antlist as an attribute
            elif 'cal_antlist' in first_h5.file['TelescopeState'].attrs.keys():
                antlist = np.safe_eval(first_h5.file['TelescopeState'].attrs['cal_antlist'])
            else:
                print " No calibration antenna ordering in first H5 file. Can't create solution tables.\n"
                continue
            antlist_indices = range(len(antlist))

            # for each solution type in the h5 file, create a table
            for sol in solution_types:
                caltable_name = '{0}.{1}'.format(caltable_basename, sol)
                sol_name = 'cal_product_{0}'.format(sol,)

                if sol_name in first_h5.file['TelescopeState'].keys():
                    print ' - creating {0} solution table: {1}\n'.format(sol, caltable_name)

                    # get solution values from the h5 file
                    solutions = first_h5.file['TelescopeState'][sol_name].value
                    soltimes, solvals = [], []
                    for t, s in solutions:
                        soltimes.append(t)
                        solvals.append(pickle.loads(s))
                    solvals = np.array(solvals)

                    # convert averaged UTC timestamps to MJD seconds.
                    sol_mjd = np.array([katpoint.Timestamp(time_utc).to_mjd() * 24 * 60 * 60 for time_utc in soltimes])

                    # determine solution characteristics
                    if len(solvals.shape) == 4:
                        ntimes, nchans, npols, nants = solvals.shape
                    else:
                        ntimes, npols, nants = solvals.shape
                        nchans = 1
                        solvals = solvals.reshape(ntimes, nchans, npols, nants)

                    # create calibration solution measurement set
                    caltable_desc = ms_extra.caltable_desc_float if sol == 'K' else ms_extra.caltable_desc_complex
                    caltable = ms_extra.open_table(caltable_name, tabledesc=caltable_desc)

                    # add other keywords for main table
                    if sol == 'K':
                        caltable.putkeyword('ParType', 'Float')
                    else:
                        caltable.putkeyword('ParType', 'Complex')
                    caltable.putkeyword('MSName', ms_name)
                    caltable.putkeyword('VisCal', ms_soltype_lookup[sol])
                    caltable.putkeyword('PolBasis', 'unknown')
                    # add necessary units
                    caltable.putcolkeywords('TIME', {'MEASINFO': {'Ref': 'UTC', 'type': 'epoch'},
                                                     'QuantumUnits': ['s']})
                    caltable.putcolkeywords('INTERVAL', {'QuantumUnits': ['s']})
                    # specify that this is a calibration table
                    caltable.putinfo({'readme': '', 'subType': ms_soltype_lookup[sol], 'type': 'Calibration'})

                    # get the solution data to write to the main table
                    solutions_to_write = solvals.transpose(0, 3, 1, 2).reshape(ntimes * nants, nchans, npols)

                    # MS's store delays in nanoseconds
                    if sol == 'K':
                        solutions_to_write = 1e9 * solutions_to_write

                    times_to_write = np.repeat(sol_mjd, nants)
                    antennas_to_write = np.tile(antlist_indices, ntimes)
                    # just mock up the scans -- this doesnt actually correspond to scans in the data
                    scans_to_write = np.repeat(range(len(sol_mjd)), nants)
                    # write the main table
                    main_cal_dict = ms_extra.populate_caltable_main_dict(times_to_write, solutions_to_write,
                                                                         antennas_to_write, scans_to_write)
                    ms_extra.write_rows(caltable, main_cal_dict, verbose=options.verbose)

                    # create and write subtables
                    subtables = ['OBSERVATION', 'ANTENNA', 'FIELD', 'SPECTRAL_WINDOW', 'HISTORY']
                    subtable_key = [(os.path.join(caltable.name(), st)) for st in subtables]

                    # Add subtable keywords and create subtables
                    # ------------------------------------------------------------------------------
                    # # this gives an error in casapy:
                    # # *** Error *** MSObservation(const Table &) - table is not a valid MSObservation
                    # for subtable, subtable_location in zip(subtables, subtable_key)
                    #     ms_extra.open_table(subtable_location, tabledesc=ms_extra.ms_desc[subtable])
                    #     caltable.putkeyword(subtable, 'Table: {0}'.format(subtable_location))
                    # # write the static info for the table
                    # ms_extra.write_dict(caltable_dict, caltable.name(), verbose=options.verbose)
                    # ------------------------------------------------------------------------------
                    # instead try just copying the main table subtables
                    #   this works to plot the data casapy, but the solutions still can't be applied in casapy...
                    for subtable, subtable_location in zip(subtables, subtable_key):
                        main_subtable = ms_extra.open_table(os.path.join(main_table.name(), subtable))
                        main_subtable.copy(subtable_location, deep=True)
                        caltable.putkeyword(subtable, 'Table: {0}'.format(subtable_location))
                        if subtable == 'ANTENNA':
                            caltable.putkeyword('NAME', antlist)
                            caltable.putkeyword('STATION', antlist)
                    if sol != 'B':
                        spw_table = ms_extra.open_table(os.path.join(caltable.name(), 'SPECTRAL_WINDOW'))
                        spw_table.removerows(spw_table.rownumbers())
                        cen_index = len(out_freqs) // 2
                        # the delay values in the cal pipeline are calculated relative to frequency 0
                        ref_freq = 0.0 if sol == 'K' else None
                        spw_dict = {'SPECTRAL_WINDOW':
                                    ms_extra.populate_spectral_window_dict(np.atleast_1d(out_freqs[cen_index]),
                                                                           np.atleast_1d(channel_freq_width),
                                                                           ref_freq=ref_freq)}
                        ms_extra.write_dict(spw_dict, caltable.name(), verbose=options.verbose)

                    # done with this caltable
                    caltable.flush()
                    caltable.close()

    main_table.close()
    # done writing main table
