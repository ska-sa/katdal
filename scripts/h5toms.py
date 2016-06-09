#! /usr/bin/env python

# Produce a CASA compatible Measurement Set from a KAT-7 HDF5 file (versions
# 1 and 2) or MeerKAT HDF5 file (version 3) using the casapy table tools
# in the ms_extra module (or pyrap/casacore if casapy is not available).

import os
import shutil
import tarfile
import optparse
import time

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
parser.add_option("-c", "--circular", action="store_true", default=False,
                  help="Produce quad circular polarisation. (RR, RL, LR, LL) "
                       "*** Currently just relabels the linear pols ****")
parser.add_option("-r" , "--ref-ant",
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
parser.add_option("-x", "--HH", action="store_true", default=False,
                  help="Produce a Stokes I MeasurementSet using only HH")
parser.add_option("-y", "--VV", action="store_true", default=False,
                  help="Produce a Stokes I MeasurementSet using only VV")
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
parser.add_option("--flags",
                  help="List of online flags to apply (from 'static,cam,detected_rfi,predicted_rfi', "
                       "default is all flags, '' will apply no flags)")
parser.add_option("--dumptime", type=float, default=0.0,
                  help="Output time averaging interval in seconds, default is no averaging.")
parser.add_option("--chanbin", type=int, default=0,
                  help="Bin width for channel averaging in channels, default is no averaging.")
parser.add_option("--flagav", action="store_true", default=False,
                  help="If a single element in an averaging bin is flagged, flag the averaged bin.")

(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    parser.print_help()
    raise RuntimeError("Please provide one or more HDF5 filenames as arguments")

if options.HH and options.VV:
    raise RuntimeError("You cannot include both HH and VV in the production of Stokes I (i.e. you specified --HH and --VV).")

if options.full_pol and (options.HH or options.VV):
    raise RuntimeError("You have specified a full pol MS but also chosen to produce Stokes I (either HH or VV). Choose one or the other.")

if options.elevation_range and len(options.elevation_range.split(',')) < 2:
    raise RuntimeError("You have selected elevation flagging. Please provide elevation limits in the form 'lowest_elevation,highest_elevation'.")

if len(args) > 1:
    print "Concatenating multiple h5 files into single MS."

if not ms_extra.casacore_binding:
    raise RuntimeError("Failed to find casacore binding. You need to install both casacore and pyrap, or run the script from within a modified casapy containing h5py and katpoint.")
else:
    print "Using '%s' casacore binding to produce MS" % (ms_extra.casacore_binding,)

pols_to_use = ['HH'] if options.HH else ['VV'] if options.VV else ['HH','HV','VH','VV'] if (options.full_pol or options.circular) else ['HH','VV']
 # which polarisation do we want to write into the MS and pull from the HDF5 file
pol_for_name = 'hh' if options.HH else 'vv' if options.VV else 'full_pol' if options.full_pol else 'circular_pol' if options.circular else 'hh_vv'

# Open HDF5 file
# if len(args) == 1: args = args[0]
# katdal can handle a list of files, which get virtually concatenated internally
h5 = katdal.open(args, ref_ant=options.ref_ant)

# ms_name = os.path.splitext(args[0])[0] + ("." if len(args) == 1 else ".et_al.") + pol_for_name + ".ms"

for win in range(len(h5.spectral_windows)):
    h5.select(reset='T')

    # Extract MS file per spectral window in H5 observation file
    print 'Extract MS for spw %d: central frequency %.2f MHz' % (win, (h5.spectral_windows[win]).centre_freq/1e6)
    cen_freq = '%d' % int(h5.spectral_windows[win].centre_freq/1e6)
    basename = ('%s_%s' % (os.path.splitext(args[0])[0], cen_freq)) + ("." if len(args) == 1 else ".et_al.") + pol_for_name

    h5.select(spw=win, scans='track')

    # create MS in current working directory
    ms_name = basename + ".ms"
    # The first step is to copy the blank template MS to our desired output (making sure it's not already there)
    if os.path.exists(ms_name):
        raise RuntimeError("MS '%s' already exists - please remove it before running this script" % (ms_name,))
    try:
        shutil.copytree(options.blank_ms, ms_name)
    except OSError:
        raise RuntimeError("Failed to copy blank MS from %s to %s - please check presence of blank MS and/or permissions" % (options.blank_ms, ms_name))

    print "Will create MS output in", ms_name

    # Instructions to flag by elevation if requested
    if options.elevation_range is not None:
        emin,emax = options.elevation_range.split(',')
        print "\nThe MS can be flagged by elevation in casapy v3.4.0 or higher, with the command:"
        print "      tflagdata(vis='%s', mode='elevation', lowerlimit=%s, upperlimit=%s, action='apply')\n" % (ms_name, emin, emax)

    # Instructions to create uvfits file if requested
    if options.uvfits:
        # uv_name = os.path.splitext(args[0])[0] + ("." if len(args) == 1 else ".et_al.") + pol_for_name + ".uvfits"
        uv_name = basename + ".uvfits"
        print "\nThe MS can be converted into a uvfits file in casapy, with the command:"
        print "      exportuvfits(vis='%s', fitsfile='%s', datacolumn='data')\n" % (ms_name, uv_name)

    if options.HH or options.VV: print "\n#### Producing Stokes I MS using " + ('HH' if options.HH else 'VV') + " only ####\n"
    elif options.full_pol: print "\n#### Producing a full polarisation MS (HH,HV,VH,VV) ####\n"
    else: print "\n#### Producing a two polarisation MS (HH, VV) ####\n"

    # # Open HDF5 file
    # if len(args) == 1: args = args[0]
    # h5 = katdal.open(args, ref_ant=options.ref_ant)
    #  # katdal can handle a list of files, which get virtually concatenated internally

    # if fringe stopping is requested, check that it has not already been done in hardware
    if options.stop_w:
        print "W term in UVW coordinates will be used to stop the fringes."
        try:
            autodelay=[int(ad) for ad in h5.sensor['DBE/auto-delay']]
            if all(autodelay): print "Fringe-stopping already performed in hardware... do you really want to stop the fringes here?"
        except KeyError:
            pass

    # Select frequency channel range
    if options.channel_range is not None:
        channel_range = [int(chan_str) for chan_str in options.channel_range.split(',')]
        first_chan, last_chan = channel_range[0], channel_range[1]

        if (first_chan < 0) or (last_chan >= h5.shape[1]):
            raise RuntimeError("Requested channel range outside data set boundaries. Set channels in the range [0,%s]" % (h5.shape[1]-1,))

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
        print "Averaging %s channels, output ms will have %s channels." % (options.chanbin,int(numchans/min(numchans,options.chanbin)))
        if chan_remainder > 0:
            print "The last %s channels in the data will be dropped during averaging (%s does not divide %s)." % (chan_remainder,options.chanbin,numchans)
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
    if average_data and options.flagav and options.flags!='': print "Extending flags to averaging bins."


    # Optionally keep only cross-correlation products
    if options.no_auto:
        h5.select(corrprods='cross')
        print "\nCross-correlations only."

    print "\nUsing %s as the reference antenna. All targets and activity detection will be based on this antenna.\n" % (h5.ref_ant,)
    # MS expects timestamps in MJD seconds
    start_time = h5.start_time.to_mjd() * 24 * 60 * 60
    end_time = h5.end_time.to_mjd() * 24 * 60 * 60
    # Version 1 and 2 files are KAT-7; the rest are MeerKAT
    telescope_name = 'KAT-7' if h5.version[0] in '12' else 'MeerKAT'

    ms_dict = {}
    ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in h5.ants], [ant.position_ecef for ant in h5.ants],
                                                        [ant.diameter for ant in h5.ants])
    ms_dict['FEED'] = ms_extra.populate_feed_dict(len(h5.ants), num_receptors_per_feed=2)
    ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
    ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(ms_pols=pols_to_use,stokes_i=(options.HH or options.VV),circular=options.circular)
    ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(start_time, end_time, telescope_name, h5.observer, h5.experiment_id)

    print "Writing static meta data..."
    ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)

    ms_dict = {}
    #increment scans sequentially in the ms
    scan_itr = 1
    print "\nIterating through scans in file(s)...\n"
    main_table = ms_extra.open_main(ms_name, verbose=options.verbose)
     # prepare to write main dict
    corrprod_to_index = dict([(tuple(cp), ind) for cp, ind in zip(h5.corr_products, range(len(h5.corr_products)))])
    field_names, field_centers, field_times = [], [], []
    obs_modes = ['UNKNOWN']

    for scan_ind, scan_state, target in h5.scans():
        s = time.time()
        scan_len = h5.shape[0]
        if scan_state != 'track':
            if options.verbose: print "scan %3d (%4d samples) skipped '%s' - not a track" % (scan_ind, scan_len, scan_state)
            continue
        if scan_len < 2:
            if options.verbose: print "scan %3d (%4d samples) skipped - too short" % (scan_ind, scan_len)
            continue
        if target.body_type != 'radec':
            if options.verbose: print "scan %3d (%4d samples) skipped - target '%s' not RADEC" % (scan_ind, scan_len, target.name)
            continue
        print "scan %3d (%4d samples) loaded. Target: '%s'. Writing to disk..." % (scan_ind, scan_len, target.name)

        # load all data for this scan up front, as this improves disk throughput
        scan_data = h5.vis[:]
        # load the weights for this scan.
        scan_weight_data = h5.weights()[:]
        # load flags selected from 'options.flags' for this scan
        scan_flag_data = h5.flags(options.flags)[:]

        # Get the average dump time for this scan (equal to scan length if the dump period is longer than a scan)
        dump_time_width = min(time_av,scan_len*h5.dump_period)

        sz_mb = 0.0
        # Get UTC timestamps
        utc_seconds = h5.timestamps[:]
        # Update field lists if this is a new target
        if target.name not in field_names:
            # Since this will be an 'radec' target, we don't need an antenna or timestamp to get the (astrometric) ra, dec
            ra, dec = target.radec()

            field_names.append(target.name)
            field_centers.append((ra, dec))
            field_times.append(katpoint.Timestamp(utc_seconds[0]).to_mjd() * 60 * 60 * 24)
            if options.verbose: print "Added new field %d: '%s' %s %s" % (len(field_names) - 1, target.name, ra, dec)
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

        for ant1_index, ant1 in enumerate(h5.ants):
            for ant2_index, ant2 in enumerate(h5.ants):
                if ant2_index < ant1_index:
                    continue
                if options.no_auto and (ant2_index == ant1_index):
                    continue
                polprods = [("%s%s" % (ant1.name,p[0].lower()), "%s%s" % (ant2.name,p[1].lower())) for p in pols_to_use]

                pol_data,flag_pol_data,weight_pol_data = [],[],[]

                for p in polprods:
                    #cable_delay = delays[p[1][-1]][ant2.name] - delays[p[0][-1]][ant1.name]
                    # cable delays specific to pol type
                    cp_index = corrprod_to_index.get(p)
                    vis_data = scan_data[:,:,cp_index] if cp_index is not None else np.zeros(h5.shape[:2], dtype=np.complex64)
                    weight_data = scan_weight_data[:,:,cp_index] if cp_index is not None else np.ones(h5.shape[:2], dtype=np.float32)
                    flag_data = scan_flag_data[:,:,cp_index] if cp_index is not None else np.zeros(h5.shape[:2], dtype=np.bool)
                    #if options.stop_w:
                        # Result of delay model in turns of phase. This is now frequency dependent so has shape (tstamps, channels)
                    #    turns = np.outer((h5.w[:, cp_index] / katpoint.lightspeed) - cable_delay, h5.channel_freqs)
                    #    vis_data *= np.exp(-2j * np.pi * turns)

                    out_utc = utc_seconds
                    out_freqs = h5.channel_freqs

                    # Overwrite the input visibilities with averaged visibilities,flags,weights,timestamps,channel freqs
                    if average_data: vis_data,weight_data,flag_data,out_utc,out_freqs=averager.average_visibilities(vis_data, weight_data, flag_data, out_utc, out_freqs,timeav=dump_av,chanav=chan_av,flagav=options.flagav)

                    pol_data.append(vis_data)
                    weight_pol_data.append(weight_data)
                    flag_pol_data.append(flag_data)

                vis_data = np.dstack(pol_data)
                weight_data = np.dstack(weight_pol_data)
                flag_data = np.dstack(flag_pol_data)

                model_data = None
                corrected_data = None
                if options.model_data:
                    # unity intensity zero phase model data set, same shape as vis_data
                    model_data = np.ones(vis_data.shape, dtype=np.complex64)
                    # corrected data set copied from vis_data
                    corrected_data = vis_data

                uvw_coordinates = np.array(target.uvw(ant2, timestamp=out_utc, antenna=ant1))

                #Convert averaged UTC timestamps to MJD seconds.
                out_mjd  =  [katpoint.Timestamp(time_utc).to_mjd() * 24 * 60 * 60 for time_utc in out_utc]

                #Increment the filesize.
                sz_mb += vis_data.dtype.itemsize * vis_data.size / (1024.0 * 1024.0)
                sz_mb += weight_data.dtype.itemsize * weight_data.size / (1024.0 * 1024.0)
                sz_mb += flag_data.dtype.itemsize * flag_data.size / (1024.0 * 1024.0)

                if options.model_data:
                    sz_mb += model_data.dtype.itemsize * model_data.size / (1024.0 * 1024.0)
                    sz_mb += corrected_data.dtype.itemsize * corrected_data.size / (1024.0 * 1024.0)                

                #write the data to the ms.
                ms_extra.write_rows(main_table, ms_extra.populate_main_dict(uvw_coordinates, vis_data, flag_data, out_mjd, ant1_index, ant2_index, dump_time_width, field_id, state_id, scan_itr, model_data, corrected_data), verbose=options.verbose)

        s1 = time.time() - s
        if average_data and np.shape(utc_seconds)[0]!=np.shape(out_utc)[0]:
            print "Averaged %s x %s second dumps to %s x %s second dumps" % (np.shape(utc_seconds)[0],h5.dump_period,np.shape(out_utc)[0],dump_time_width)
        print "Wrote scan data (%f MB) in %f s (%f MBps)\n" % (sz_mb, s1, sz_mb / s1)
        scan_itr+=1
    main_table.close()
    # done writing main table

    # Remove spaces from source names, unless otherwise specified
    field_names = [f.replace(' ','') for f in field_names] if not options.keep_spaces else field_names

    ms_dict = {}
    ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(out_freqs, channel_freq_width * np.ones(len(out_freqs)))
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

