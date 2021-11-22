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

# Produce a CASA-compatible MeasurementSet from a MeerKAT Visibility Format
# (MVF) dataset using casacore.

import argparse
import multiprocessing
import multiprocessing.sharedctypes
import os
import queue
import re
import tarfile
import time
import urllib.parse
from collections import namedtuple

import dask
import katpoint
import numba
import numpy as np

import katdal
from katdal import averager, ms_async, ms_extra
from katdal.flags import NAMES as FLAG_NAMES
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.sensordata import telstate_decode

SLOTS = 4    # Controls overlap between loading and writing

CPInfo = namedtuple('CPInfo', 'ant1_index ant2_index ant1 ant2 cp_index')


def casa_style_int_list(range_string, use_argparse=False, opt_unit="m"):
    """Turn CASA style range string "[0-9]*~[0-9]*, ..." into a list of ints.

    The range may contain unit strings such as 10~50m if the accepted unit is
    specified in `opt_unit`. Returns None if no range is specified, otherwise
    a list of range values.
    """
    range_string = range_string.strip()
    if range_string in ("", "*"):
        return None
    RangeException = argparse.ArgumentTypeError if use_argparse else ValueError
    range_type = r"^((\d+)[{0}]?)?(~(\d+)[{0}]?)?$".format(opt_unit)
    vals = []
    for val in range_string.split(","):
        match = re.match(range_type, val.strip())
        if not match:
            raise RangeException("Value must be CASA range, comma list or blank")
        elif match.group(4):
            # range
            rmin = int(match.group(2))
            rmax = int(match.group(4))
            vals.extend(range(rmin, rmax + 1))
        elif match.group(2):
            # int
            vals.append(int(match.group(2)))
        else:
            raise RangeException("Value must be CASA range, comma list or blank")
    return vals


def default_ms_name(args, centre_freq=None):
    """Infer default MS name from argument list and optional frequency label."""
    # Use the first dataset in the list to generate the base part of the MS name
    url_parts = urllib.parse.urlparse(args[0], scheme='file')
    # Create MS in current working directory (strip off directories)
    dataset_filename = os.path.basename(url_parts.path)
    # Get rid of the ".full" bit on RDB files (it's the same dataset)
    full_rdb_ext = '.full.rdb'
    if dataset_filename.endswith(full_rdb_ext):
        dataset_basename = dataset_filename[:-len(full_rdb_ext)]
    else:
        dataset_basename = os.path.splitext(dataset_filename)[0]
    # Add frequency to name to disambiguate multiple spectral windows
    if centre_freq:
        dataset_basename += f'_{int(centre_freq)}Hz'
    # Add ".et_al" as reminder that we concatenated multiple datasets
    return '{}{}.ms'.format(dataset_basename, "" if len(args) == 1 else ".et_al")


def load(dataset, indices, vis, weights, flags):
    """Load data from lazy indexers into existing storage.

    This is optimised for the MVF v4 case where we can use dask directly
    to eliminate one copy, and also load vis, flags and weights in parallel.
    In older formats it causes an extra copy.

    Parameters
    ----------
    dataset : :class:`katdal.DataSet`
        Input dataset, possibly with an existing selection
    indices : tuple
        Index expression for subsetting the dataset
    vis, weights, flags : array-like
        Outputs, which must have the correct shape and type
    """
    if isinstance(dataset.vis, DaskLazyIndexer):
        DaskLazyIndexer.get([dataset.vis, dataset.weights, dataset.flags], indices,
                            out=[vis, weights, flags])
    else:
        vis[:] = dataset.vis[indices]
        weights[:] = dataset.weights[indices]
        flags[:] = dataset.flags[indices]


@numba.jit(nopython=True, parallel=True)
def permute_baselines(in_vis, in_weights, in_flags, cp_index, out_vis, out_weights, out_flags):
    """Reorganise baselines and axis order.

    The inputs have dimensions (time, channel, pol-baseline), and the output has shape
    (time, baseline, channel, pol). cp_index is a 2D array which is indexed by baseline and
    pol to get the input pol-baseline.

    cp_index may contain negative indices if the data is not present, in which
    case it is filled with 0s and flagged.

    This could probably be optimised further: the current implementation isn't
    particularly cache-friendly, and it could benefit from unrolling the loop
    over polarisations in some way
    """
    # Workaround for https://github.com/numba/numba/issues/2921
    in_flags_u8 = in_flags.view(np.uint8)
    n_time, n_bls, n_chans, n_pols = out_vis.shape  # noqa: W0612
    bstep = 128
    bblocks = (n_bls + bstep - 1) // bstep
    for t in range(n_time):
        for bblock in numba.prange(bblocks):  # noqa: E1133
            bstart = bblock * bstep
            bstop = min(n_bls, bstart + bstep)
            for c in range(n_chans):
                for b in range(bstart, bstop):
                    for p in range(out_vis.shape[3]):
                        idx = cp_index[b, p]
                        if idx >= 0:
                            vis = in_vis[t, c, idx]
                            weight = in_weights[t, c, idx]
                            flag = in_flags_u8[t, c, idx] != 0
                        else:
                            vis = np.complex64(0 + 0j)
                            weight = np.float32(0)
                            flag = np.bool_(True)
                        out_vis[t, b, c, p] = vis
                        out_weights[t, b, c, p] = weight
                        out_flags[t, b, c, p] = flag
    return out_vis, out_weights, out_flags


def main():
    tag_to_intent = {'gaincal': 'CALIBRATE_PHASE,CALIBRATE_AMPLI',
                     'bpcal': 'CALIBRATE_BANDPASS,CALIBRATE_FLUX',
                     'target': 'TARGET'}

    usage = "%(prog)s [options] <dataset> [<dataset2>]*"
    description = "Convert MVF dataset(s) to CASA MeasurementSet. The datasets may " \
                  "be local filenames or archive URLs (including access tokens). " \
                  "If there are multiple datasets they will be concatenated via " \
                  "katdal before conversion."
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument("-o", "--output-ms", default=None,
                        help="Name of output MeasurementSet")
    parser.add_argument("-c", "--circular", action="store_true", default=False,
                        help="Produce quad circular polarisation. (RR, RL, LR, LL) "
                             "*** Currently just relabels the linear pols ****")
    parser.add_argument("-r", "--ref-ant",
                        help="Override the reference antenna used to pick targets "
                             "and scans (default is the 'array' antenna in MVFv4 "
                             "and the first antenna in older formats)")
    parser.add_argument("-t", "--tar", action="store_true", default=False,
                        help="Tar-ball the MS")
    parser.add_argument("-f", "--full_pol", action="store_true", default=False,
                        help="Produce a full polarisation MS in CASA canonical order "
                             "(HH, HV, VH, VV). Default is to produce HH,VV only.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="More verbose progress information")
    parser.add_argument("-w", "--stop-w", action="store_true", default=False,
                        help="Use W term to stop fringes for each baseline")
    parser.add_argument("-p", "--pols-to-use", default=None,
                        help="Select polarisation products to include in MS as "
                             "comma-separated list (from: HH, HV, VH, VV). "
                             "Default is all available from (HH, VV).")
    parser.add_argument("-u", "--uvfits", action="store_true", default=False,
                        help="Print command to convert MS to miriad uvfits in casapy")
    parser.add_argument("-a", "--no-auto", action="store_true", default=False,
                        help="MeasurementSet will exclude autocorrelation data")
    parser.add_argument("-s", "--keep-spaces", action="store_true", default=False,
                        help="Keep spaces in source names, default removes spaces")
    parser.add_argument("-C", "--channel-range",
                        help="Range of frequency channels to keep (zero-based inclusive "
                             "'first_chan,last_chan', default is all channels)")
    parser.add_argument("-e", "--elevation-range",
                        help="Flag elevations outside the range "
                             "'lowest_elevation,highest_elevation'")
    parser.add_argument("-m", "--model-data", action="store_true", default=False,
                        help="Add MODEL_DATA and CORRECTED_DATA columns to the MS. "
                             "MODEL_DATA initialised to unity amplitude zero phase, "
                             "CORRECTED_DATA initialised to DATA.")
    flag_names = ', '.join(name for name in FLAG_NAMES if not name.startswith('reserved'))
    parser.add_argument("--flags", default="all",
                        help="List of online flags to apply "
                             "(from " + flag_names + ") "
                             "default is all flags, '' will apply no flags)")
    parser.add_argument("--dumptime", type=float, default=0.0,
                        help="Output time averaging interval in seconds, "
                             "default is no averaging")
    parser.add_argument("--chanbin", type=int, default=0,
                        help="Bin width for channel averaging in channels, "
                             "default is no averaging")
    parser.add_argument("--flagav", action="store_true", default=False,
                        help="If a single element in an averaging bin is flagged, "
                             "flag the averaged bin")
    parser.add_argument("--caltables", action="store_true", default=False,
                        help="Create calibration tables from gain solutions in "
                             "the dataset (if present)")
    parser.add_argument("--quack", type=int, default=1, metavar='N',
                        help="Discard the first N dumps "
                             "(which are frequently incomplete)")
    parser.add_argument("--applycal", default="",
                        help="List of calibration solutions to apply to data as "
                             "a string of comma-separated names, e.g. 'l1' or "
                             "'K,B,G'. Use 'default' for L1 + L2 and 'all' for "
                             "all available products.")
    parser.add_argument("--target", default=[], action="append",
                        help="Select only specified target field (name). This switch can "
                             "be specified multiple times if need be. Default is select "
                             "all fields.")
    parser.add_argument("--ant", default=[], action="append",
                        help="Select only specified antenna (specified as name). This switch can "
                             "be specified multiple times if need be. Default is select "
                             "all antennas.")
    parser.add_argument("--scans", default="",
                        type=lambda x: casa_style_int_list(x, use_argparse=True, opt_unit="m"),
                        help="Only select a range of tracking scans. Default is select all "
                             "tracking scans. Accepts a comma list or a casa style range "
                             "such as 5~10.")
    parser.add_argument("datasets", help="Dataset path", nargs='+')

    parseargs = parser.parse_args()
    # positional arguments are now part of arguments, but lets keep the logic the same
    options = parseargs
    args = parseargs.datasets

    # Loading is I/O-bound, so give more threads than CPUs
    dask.config.set(pool=multiprocessing.pool.ThreadPool(4 * multiprocessing.cpu_count()))

    if len(args) < 1:
        parser.print_help()
        raise RuntimeError("Please provide one or more MVF dataset names as arguments")

    if options.elevation_range and len(options.elevation_range.split(',')) < 2:
        raise RuntimeError("You have selected elevation flagging. Please provide elevation "
                           "limits in the form 'lowest_elevation,highest_elevation'.")

    if len(args) > 1:
        print("Concatenating multiple datasets into single MS.")

    def antenna_indices(na, no_auto_corr):
        """Get default antenna1 and antenna2 arrays."""
        return np.triu_indices(na, 1 if no_auto_corr else 0)

    def corrprod_index(dataset):
        """The correlator product index (with -1 representing missing indices)."""
        corrprod_to_index = {tuple(cp): n for n, cp in enumerate(dataset.corr_products)}

        # ==========================================
        # Generate per-baseline antenna pairs and
        # correlator product indices
        # ==========================================

        def _cp_index(a1, a2, pol):
            """Create correlator product index from antenna pair and pol."""
            a1 = a1.name + pol[0].lower()
            a2 = a2.name + pol[1].lower()
            return corrprod_to_index.get((a1, a2), -1)

        # Generate baseline antenna pairs
        ant1_index, ant2_index = antenna_indices(len(dataset.ants), options.no_auto)
        # Order as similarly to the input as possible, which gives better performance
        # in permute_baselines.
        bl_indices = list(zip(ant1_index, ant2_index))
        bl_indices.sort(key=lambda ants: _cp_index(dataset.ants[ants[0]],
                                                   dataset.ants[ants[1]],
                                                   pols_to_use[0]))
        # Undo the zip
        ant1_index[:] = [bl[0] for bl in bl_indices]
        ant2_index[:] = [bl[1] for bl in bl_indices]
        ant1 = [dataset.ants[a1] for a1 in ant1_index]
        ant2 = [dataset.ants[a2] for a2 in ant2_index]

        # Create actual correlator product index
        cp_index = [_cp_index(a1, a2, p)
                    for a1, a2 in zip(ant1, ant2)
                    for p in pols_to_use]
        cp_index = np.array(cp_index, dtype=np.int32)

        return CPInfo(ant1_index, ant2_index, ant1, ant2, cp_index)

    # Open dataset
    open_args = args[0] if len(args) == 1 else args
    # katdal can handle a list of datasets, which get virtually concatenated internally
    dataset = katdal.open(open_args, ref_ant=options.ref_ant, applycal=options.applycal)
    if dataset.applycal_products:
        print('The following calibration products will be applied:',
              ', '.join(dataset.applycal_products))
    else:
        print('No calibration products will be applied')

    # select a subset of antennas
    avail_ants = [a.name for a in dataset.ants]
    dump_ants = options.ant if len(options.ant) != 0 else avail_ants
    # some users may specify comma-separated lists although we said the switch should
    # be specified multiple times. Put in a guard to split comma separated lists as well
    split_ants = []
    for a in dump_ants:
        split_ants += [x.strip() for x in a.split(",")]
    dump_ants = split_ants
    dump_ant_str = ", ".join(map(repr, dump_ants))
    avail_ant_str = ", ".join(map(repr, avail_ants))

    if not set(dump_ants) <= set(avail_ants):
        raise RuntimeError("One or more antennas cannot be found in the dataset. "
                           f"You requested {dump_ant_str} but only {avail_ant_str} are available.")

    if len(dump_ants) == 0:
        print('User antenna criterion resulted in empty database, nothing to be done. '
              f'Perhaps you wanted to select from the following: {avail_ant_str}')

    print(f'Per user request the following antennas will be selected: {dump_ant_str}')

    # select a subset of targets
    avail_fields = [f.name for f in dataset.catalogue.targets]
    dump_fields = options.target if len(options.target) != 0 else avail_fields

    # some users may specify comma-separated lists although we said the switch should
    # be specified multiple times. Put in a guard to split comma separated lists as well
    split_fields = []
    for f in dump_fields:
        split_fields += [x.strip() for x in f.split(",")]
    dump_fields = split_fields
    dump_field_str = ", ".join(map(repr, dump_fields))
    avail_field_str = ", ".join(map(repr, avail_fields))

    if not set(dump_fields) <= set(avail_fields):
        raise RuntimeError("One or more fields cannot be found in the dataset. "
                           f"You requested {dump_field_str} but only {avail_field_str} are available.")

    if len(dump_fields) == 0:
        print('User target field criterion resulted in empty database, nothing to be done. '
              f'Perhaps you wanted to select from the following: {avail_field_str}')

    print(f'Per user request the following target fields will be selected: {dump_field_str}')

    dataset.select(targets=dump_fields)

    # get a set of user selected available tracking scans, ignore slew scans
    avail_tracks = list(map(lambda x: x[0],
                            filter(lambda x: x[1] == 'track',
                                   dataset.scans())))
    dump_scans = options.scans if options.scans else avail_tracks
    dump_scans = list(set(dump_scans).intersection(set(avail_tracks)))
    if len(dump_scans) == 0:
        avail_track_str = ", ".join(map(str, avail_tracks))
        raise RuntimeError('User scan criterion resulted in empty database, nothing to be done. '
                           f'Perhaps you wanted to select from the following: {avail_track_str}')

    print(f'Per user request the following scans will be dumped: {", ".join(map(str, dump_scans))}')

    # Get list of unique polarisation products in the dataset
    pols_in_dataset = np.unique([(cp[0][-1] + cp[1][-1]).upper() for cp in dataset.corr_products])

    # Which polarisation do we want to write into the MS
    # select all possible pols if full-pol selected, otherwise the selected polarisations via pols_to_use
    # otherwise finally select any of HH,VV present (the default).
    pols_to_use = ['HH', 'HV', 'VH', 'VV'] if (options.full_pol or options.circular) else \
        list(np.unique(options.pols_to_use.split(','))) if options.pols_to_use else \
        [pol for pol in ['HH', 'VV'] if pol in pols_in_dataset]

    # Check we have the chosen polarisations
    if np.any([pol not in pols_in_dataset for pol in pols_to_use]):
        raise RuntimeError(f"Selected polarisation(s): {', '.join(pols_to_use)} not available. "
                           f"Available polarisation(s): {','.join(pols_in_dataset)}")

    # Set full_pol if this is selected via options.pols_to_use
    if set(pols_to_use) == {'HH', 'HV', 'VH', 'VV'} and not options.circular:
        options.full_pol = True

    # Extract one MS per spectral window in the dataset(s)
    for win in range(len(dataset.spectral_windows)):
        dataset.select(reset='T')

        centre_freq = dataset.spectral_windows[win].centre_freq
        print(f'Extract MS for spw {win}: centre frequency {int(centre_freq)} Hz')

        # If no output MS directory name supplied, infer it from dataset(s)
        if options.output_ms is None:
            if len(dataset.spectral_windows) > 1:
                # Use frequency label to disambiguate multiple spectral windows
                ms_name = default_ms_name(args, centre_freq)
            else:
                ms_name = default_ms_name(args)
        else:
            ms_name = options.output_ms
        basename = os.path.splitext(ms_name)[0]

        # Discard first N dumps which are frequently incomplete
        dataset.select(spw=win,
                       scans=dump_scans,  # should already be filtered to target type only
                       targets=dump_fields,
                       flags=options.flags,
                       ants=dump_ants,
                       dumps=slice(options.quack, None))

        # The first step is to copy the blank template MS to our desired output
        # (making sure it's not already there)
        if os.path.exists(ms_name):
            raise RuntimeError(f"MS '{ms_name}' already exists - please remove it "
                               "before running this script")

        print("Will create MS output in " + ms_name)

        # Instructions to flag by elevation if requested
        if options.elevation_range is not None:
            emin, emax = options.elevation_range.split(',')
            print("\nThe MS can be flagged by elevation in casapy v3.4.0 or higher, with the command:")
            print(f"      tflagdata(vis='{ms_name}', mode='elevation', lowerlimit={emin}, "
                  f"upperlimit={emax}, action='apply')\n")

        # Instructions to create uvfits file if requested
        if options.uvfits:
            uv_name = basename + ".uvfits"
            print("\nThe MS can be converted into a uvfits file in casapy, with the command:")
            print(f"      exportuvfits(vis='{ms_name}', fitsfile='{uv_name}', datacolumn='data')\n")

        if options.full_pol:
            print("\n#### Producing a full polarisation MS (HH,HV,VH,VV) ####\n")
        else:
            print(f"\n#### Producing MS with {','.join(pols_to_use)} polarisation(s) ####\n")

        # if fringe stopping is requested, check that it has not already been done in hardware
        if options.stop_w:
            print("W term in UVW coordinates will be used to stop the fringes.")
            try:
                autodelay = [int(ad) for ad in dataset.sensor['DBE/auto-delay']]
                if all(autodelay):
                    print("Fringe-stopping already performed in hardware... "
                          "do you really want to stop the fringes here?")
            except KeyError:
                pass

        # Select frequency channel range
        if options.channel_range is not None:
            channel_range = [int(chan_str) for chan_str in options.channel_range.split(',')]
            first_chan, last_chan = channel_range[0], channel_range[1]

            if (first_chan < 0) or (last_chan >= dataset.shape[1]):
                raise RuntimeError("Requested channel range outside data set boundaries. "
                                   f"Set channels in the range [0,{dataset.shape[1] - 1}]")
            if first_chan > last_chan:
                raise RuntimeError(f"First channel ({first_chan}) bigger than last channel "
                                   f"({last_chan}) - did you mean it the other way around?")

            chan_range = slice(first_chan, last_chan + 1)
            print(f"\nChannel range {first_chan} through {last_chan}.")
            dataset.select(channels=chan_range)

        # Are we averaging?
        average_data = False

        # Determine the number of channels
        nchan = len(dataset.channels)

        # Work out channel average and frequency increment
        if options.chanbin > 1:
            average_data = True
            # Check how many channels we are dropping
            chan_remainder = nchan % options.chanbin
            avg_nchan = int(nchan / min(nchan, options.chanbin))
            print(f"Averaging {options.chanbin} channels, output ms will have {avg_nchan} channels.")
            if chan_remainder > 0:
                print(f"The last {chan_remainder} channels in the data will be dropped "
                      f"during averaging ({options.chanbin} does not divide {nchan}).")
            chan_av = options.chanbin
            nchan = avg_nchan
        else:
            # No averaging in channel
            chan_av = 1

        # Get the frequency increment per averaged channel
        channel_freq_width = dataset.channel_width * chan_av

        # Work out dump average and dump increment
        # Is the desired time bin greater than the dump period?
        if options.dumptime > dataset.dump_period:
            average_data = True
            dump_av = int(np.round(options.dumptime / dataset.dump_period))
            time_av = dump_av * dataset.dump_period
            print(f"Averaging {dataset.dump_period} second dumps to {time_av} seconds.")
        else:
            # No averaging in time
            dump_av = 1
            time_av = dataset.dump_period

        # Print a message if extending flags to averaging bins.
        if average_data and options.flagav and options.flags != '':
            print("Extending flags to averaging bins.")

        # Optionally keep only cross-correlation products
        if options.no_auto:
            dataset.select(corrprods='cross')
            print("\nCross-correlations only.")

        print(f"\nUsing {dataset.ref_ant} as the reference antenna. All targets and scans "
              "will be based on this antenna.\n")
        # MS expects timestamps in MJD seconds
        start_time = dataset.start_time.to_mjd() * 24 * 60 * 60
        end_time = dataset.end_time.to_mjd() * 24 * 60 * 60
        # MVF version 1 and 2 datasets are KAT-7; the rest are MeerKAT
        telescope_name = 'KAT-7' if dataset.version[0] in '12' else 'MeerKAT'

        # increment scans sequentially in the ms
        scan_itr = 1
        print("\nIterating through scans in dataset(s)...\n")

        cp_info = corrprod_index(dataset)
        nbl = cp_info.ant1_index.size
        npol = len(pols_to_use)

        field_names, field_centers, field_times = [], [], []
        obs_modes = ['UNKNOWN']
        total_size = 0

        # Create the MeasurementSet
        table_desc, dminfo = ms_extra.kat_ms_desc_and_dminfo(
            nbl=nbl, nchan=nchan, ncorr=npol, model_data=options.model_data)
        ms_extra.create_ms(ms_name, table_desc, dminfo)

        ms_dict = {}
        ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict([ant.name for ant in dataset.ants],
                                                            [ant.position_ecef for ant in dataset.ants],
                                                            [ant.diameter for ant in dataset.ants])
        ms_dict['FEED'] = ms_extra.populate_feed_dict(len(dataset.ants), num_receptors_per_feed=2)
        ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
        ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(ms_pols=pols_to_use,
                                                                      circular=options.circular)
        ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(
            start_time, end_time, telescope_name, dataset.observer, dataset.experiment_id)

        # before resetting ms_dict, copy subset to caltable dictionary
        if options.caltables:
            caltable_dict = {}
            caltable_dict['ANTENNA'] = ms_dict['ANTENNA']
            caltable_dict['OBSERVATION'] = ms_dict['OBSERVATION']

        print("Writing static meta data...")
        ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)

        # Pre-allocate memory buffers
        tsize = dump_av
        in_chunk_shape = (tsize,) + dataset.shape[1:]
        scan_vis_data = np.empty(in_chunk_shape, dataset.vis.dtype)
        scan_weight_data = np.empty(in_chunk_shape, dataset.weights.dtype)
        scan_flag_data = np.empty(in_chunk_shape, dataset.flags.dtype)

        ms_chunk_shape = (SLOTS, tsize // dump_av, nbl, nchan, npol)
        raw_vis_data = ms_async.RawArray(ms_chunk_shape, scan_vis_data.dtype)
        raw_weight_data = ms_async.RawArray(ms_chunk_shape, scan_weight_data.dtype)
        raw_flag_data = ms_async.RawArray(ms_chunk_shape, scan_flag_data.dtype)
        ms_vis_data = raw_vis_data.asarray()
        ms_weight_data = raw_weight_data.asarray()
        ms_flag_data = raw_flag_data.asarray()

        # Need to limit the queue to prevent overwriting slots before they've
        # been processed. The -2 allows for the one we're writing and the one
        # the writer process is reading.
        work_queue = multiprocessing.Queue(maxsize=SLOTS - 2)
        result_queue = multiprocessing.Queue()
        writer_process = multiprocessing.Process(
            target=ms_async.ms_writer_process,
            args=(work_queue, result_queue, options, dataset.ants, cp_info, ms_name,
                  raw_vis_data, raw_weight_data, raw_flag_data))
        writer_process.start()

        try:
            slot = 0
            for scan_ind, scan_state, target in dataset.scans():
                s = time.time()
                scan_len = dataset.shape[0]
                prefix = f'scan {scan_ind:3d} ({scan_len:4d} samples)'
                if scan_state not in {'track', 'scan'}:
                    if options.verbose:
                        print(f"{prefix} skipped '{scan_state}' - not a track or scan")
                    continue
                if scan_len < 2:
                    if options.verbose:
                        print(f'{prefix} skipped - too short')
                    continue
                print(f"{prefix} loaded. Target: '{target.name}'. Writing to disk...")

                # Get the average dump time for this scan (equal to scan length
                # if the dump period is longer than a scan)
                dump_time_width = min(time_av, scan_len * dataset.dump_period)

                # Get UTC timestamps
                utc_seconds = dataset.timestamps[:]
                # Update field lists if this is a new target
                if target.name not in field_names:
                    field_names.append(target.name)
                    # Set direction of the field center to the (ra, dec) at the start
                    # of the first valid scan, based on the reference (catalogue) antenna.
                    field_time = katpoint.Timestamp(utc_seconds[0])
                    ra, dec = target.radec(field_time)
                    field_centers.append((ra, dec))
                    field_times.append(field_time.to_mjd() * 60 * 60 * 24)
                    if options.verbose:
                        print(f"Added new field {len(field_names) - 1}: '{target.name}' {ra} {dec}")
                field_id = field_names.index(target.name)

                # Determine the observation tag for this scan
                obs_tag = ','.join(tag_to_intent[tag]
                                   for tag in target.tags if tag in tag_to_intent)

                # add tag to obs_modes list
                if obs_tag and obs_tag not in obs_modes:
                    obs_modes.append(obs_tag)
                # get state_id from obs_modes list if it is in the list, else 0 'UNKNOWN'
                state_id = obs_modes.index(obs_tag) if obs_tag in obs_modes else 0

                # Iterate over time in some multiple of dump average
                ntime = utc_seconds.size
                ntime_av = 0

                for ltime in range(0, ntime - tsize + 1, tsize):
                    utime = ltime + tsize
                    tdiff = utime - ltime
                    out_freqs = dataset.channel_freqs

                    # load all visibility, weight and flag data
                    # for this scan's timestamps.
                    # Ordered (ntime, nchan, nbl*npol)
                    load(dataset, np.s_[ltime:utime, :, :],
                         scan_vis_data, scan_weight_data, scan_flag_data)

                    # This are updated as we go to point to the current storage
                    vis_data = scan_vis_data
                    weight_data = scan_weight_data
                    flag_data = scan_flag_data

                    out_utc = utc_seconds[ltime:utime]

                    # Overwrite the input visibilities with averaged visibilities,
                    # flags, weights, timestamps, channel freqs
                    if average_data:
                        vis_data, weight_data, flag_data, out_utc, out_freqs = \
                            averager.average_visibilities(vis_data, weight_data, flag_data,
                                                          out_utc, out_freqs, timeav=dump_av,
                                                          chanav=chan_av, flagav=options.flagav)

                        # Infer new time dimension from averaged data
                        tdiff = vis_data.shape[0]

                    # Select correlator products and permute axes
                    cp_index = cp_info.cp_index.reshape((nbl, npol))
                    vis_data, weight_data, flag_data = permute_baselines(
                        vis_data, weight_data, flag_data, cp_index,
                        ms_vis_data[slot], ms_weight_data[slot], ms_flag_data[slot])

                    # Increment the number of averaged dumps
                    ntime_av += tdiff

                    # Check if writer process has crashed and abort if so
                    try:
                        result = result_queue.get_nowait()
                        raise result
                    except queue.Empty:
                        pass

                    work_queue.put(ms_async.QueueItem(
                        slot=slot, target=target, time_utc=out_utc, dump_time_width=dump_time_width,
                        field_id=field_id, state_id=state_id, scan_itr=scan_itr))
                    slot += 1
                    if slot == SLOTS:
                        slot = 0

                work_queue.put(ms_async.EndOfScan())
                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                scan_size = result.scan_size
                s1 = time.time() - s

                if average_data and utc_seconds.shape != ntime_av:
                    print(f'Averaged {np.shape(utc_seconds)[0]} x {dataset.dump_period} second dumps '
                          f'to {ntime_av} x {dump_time_width} second dumps')

                scan_size_mb = float(scan_size) / (1024**2)

                print(f'Wrote scan data ({scan_size_mb:.3f} MiB) '
                      f'in {s1:.3f} s ({scan_size_mb / s1:.3f} MiBps)\n')

                scan_itr += 1
                total_size += scan_size

        finally:
            work_queue.put(None)
            writer_exc = None
            # Drain the result_queue so that we unblock the writer process
            while True:
                result = result_queue.get()
                if isinstance(result, Exception):
                    writer_exc = result
                elif result is None:
                    break
            writer_process.join()
        # This raise is deferred to outside the finally block, so that we don't
        # raise an exception while unwinding another one.
        if isinstance(writer_exc, Exception):
            raise writer_exc

        if total_size == 0:
            raise RuntimeError("No usable data found in MVF dataset "
                               "(pick another reference antenna, maybe?)")

        # Remove spaces from source names, unless otherwise specified
        field_names = [f.replace(' ', '') for f in field_names] \
            if not options.keep_spaces else field_names

        ms_dict = {}
        ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(
            out_freqs, channel_freq_width * np.ones(len(out_freqs)))
        ms_dict['FIELD'] = ms_extra.populate_field_dict(
            field_centers, field_times, field_names)
        ms_dict['STATE'] = ms_extra.populate_state_dict(obs_modes)
        ms_dict['SOURCE'] = ms_extra.populate_source_dict(
            field_centers, field_times, field_names)

        print("\nWriting dynamic fields to disk....\n")
        # Finally we write the MS as per our created dicts
        ms_extra.write_dict(ms_dict, ms_name, verbose=options.verbose)
        if options.tar:
            tar = tarfile.open(f'{ms_name}.tar', 'w')
            tar.add(ms_name, arcname=os.path.basename(ms_name))
            tar.close()

        # --------------------------------------
        # Now write calibration product tables if required
        # Open first HDF5 file in the list to extract TelescopeState parameters from
        #   (can't extract telstate params from contatenated katdal file as it
        #    uses the hdf5 file directly)
        first_dataset = katdal.open(args[0], ref_ant=options.ref_ant)
        main_table = ms_extra.open_table(ms_name, verbose=options.verbose)

        if options.caltables:
            # copy extra subtable dictionary values necessary for caltable
            caltable_dict['SPECTRAL_WINDOW'] = ms_dict['SPECTRAL_WINDOW']
            caltable_dict['FIELD'] = ms_dict['FIELD']

            solution_types = ['G', 'B', 'K']
            ms_soltype_lookup = {'G': 'G Jones', 'B': 'B Jones', 'K': 'K Jones'}

            print("\nWriting calibration solution tables to disk....")
            if 'TelescopeState' not in first_dataset.file.keys():
                print(" No TelescopeState in first dataset. Can't create solution tables.\n")
            else:
                # first get solution antenna ordering
                #   newer files have the cal antlist as a sensor
                if 'cal_antlist' in first_dataset.file['TelescopeState'].keys():
                    a0 = first_dataset.file['TelescopeState/cal_antlist'][()]
                    antlist = telstate_decode(a0[0][1])
                #   older files have the cal antlist as an attribute
                elif 'cal_antlist' in first_dataset.file['TelescopeState'].attrs.keys():
                    antlist = np.safe_eval(first_dataset.file['TelescopeState'].attrs['cal_antlist'])
                else:
                    print(" No calibration antenna ordering in first dataset. "
                          "Can't create solution tables.\n")
                    continue
                antlist_indices = list(range(len(antlist)))

                # for each solution type in the file, create a table
                for sol in solution_types:
                    caltable_name = f'{basename}.{sol}'
                    sol_name = f'cal_product_{sol}'

                    if sol_name in first_dataset.file['TelescopeState'].keys():
                        print(f' - creating {sol} solution table: {caltable_name}\n')

                        # get solution values from the file
                        solutions = first_dataset.file['TelescopeState'][sol_name][()]
                        soltimes, solvals = [], []
                        for t, s in solutions:
                            soltimes.append(t)
                            solvals.append(telstate_decode(s))
                        solvals = np.array(solvals)

                        # convert averaged UTC timestamps to MJD seconds.
                        sol_mjd = np.array([katpoint.Timestamp(time_utc).to_mjd() * 24 * 60 * 60
                                            for time_utc in soltimes])

                        # determine solution characteristics
                        if len(solvals.shape) == 4:
                            ntimes, nchans, npols, nants = solvals.shape
                        else:
                            ntimes, npols, nants = solvals.shape
                            nchans = 1
                            solvals = solvals.reshape((ntimes, nchans, npols, nants))

                        # create calibration solution measurement set
                        caltable_desc = ms_extra.caltable_desc_float \
                            if sol == 'K' else ms_extra.caltable_desc_complex
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
                        caltable.putinfo({'readme': '', 'subType': ms_soltype_lookup[sol],
                                          'type': 'Calibration'})

                        # get the solution data to write to the main table
                        solutions_to_write = solvals.transpose(0, 3, 1, 2).reshape((
                            ntimes * nants, nchans, npols))

                        # MS's store delays in nanoseconds
                        if sol == 'K':
                            solutions_to_write = 1e9 * solutions_to_write

                        times_to_write = np.repeat(sol_mjd, nants)
                        antennas_to_write = np.tile(antlist_indices, ntimes)
                        # just mock up the scans -- this doesnt actually correspond to scans in the data
                        scans_to_write = np.repeat(list(range(len(sol_mjd))), nants)
                        # write the main table
                        main_cal_dict = ms_extra.populate_caltable_main_dict(
                            times_to_write, solutions_to_write, antennas_to_write, scans_to_write)
                        ms_extra.write_rows(caltable, main_cal_dict, verbose=options.verbose)

                        # create and write subtables
                        subtables = ['OBSERVATION', 'ANTENNA', 'FIELD', 'SPECTRAL_WINDOW', 'HISTORY']
                        subtable_key = [(os.path.join(caltable.name(), st)) for st in subtables]

                        # Add subtable keywords and create subtables
                        # ------------------------------------------------------------------------------
                        # # this gives an error in casapy:
                        # *** Error *** MSObservation(const Table &) - table is not a valid MSObservation
                        # for subtable, subtable_location in zip(subtables, subtable_key)
                        #    ms_extra.open_table(subtable_location, tabledesc=ms_extra.ms_desc[subtable])
                        #    caltable.putkeyword(subtable, 'Table: {0}'.format(subtable_location))
                        # # write the static info for the table
                        # ms_extra.write_dict(caltable_dict, caltable.name(), verbose=options.verbose)
                        # ------------------------------------------------------------------------------
                        # instead try just copying the main table subtables
                        #   this works to plot the data casapy, but the solutions still can't be
                        #   applied in casapy...
                        for subtable, subtable_location in zip(subtables, subtable_key):
                            main_subtable = ms_extra.open_table(os.path.join(main_table.name(),
                                                                             subtable))
                            main_subtable.copy(subtable_location, deep=True)
                            caltable.putkeyword(subtable, f'Table: {subtable_location}')
                            if subtable == 'ANTENNA':
                                caltable.putkeyword('NAME', antlist)
                                caltable.putkeyword('STATION', antlist)
                        if sol != 'B':
                            spw_table = ms_extra.open_table(os.path.join(caltable.name(),
                                                                         'SPECTRAL_WINDOW'))
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


if __name__ == '__main__':
    main()
