#!/usr/bin/python

# Produces a CASA compatible Measurement Set from a KAT-7 HDF5 file (versions 1 and 2).
#
# Uses the pyrap python CASA bindings from the ATNF (imported via ms_extra package).

import os.path
import sys
import shutil
import tarfile
import optparse

import h5py
import numpy as np
import katpoint
from k7augment import ms_extra

 # NOTE: This should be checked before running (only for w stopping) to see how up to date the cable delays are !!!
delays = [478.041e-9, 545.235e-9, 669.900e-9, 772.868e-9, 600.0e-9, 600.0e-9, 600.0e-9]
 # updated by simonr July 5th 2010

def get_single_value(group, name):
    """Return single value from attribute or dataset with given name in group.

    If data is retrieved from a dataset, this functions raises an error if the
    values in the dataset are not all the same. Otherwise it returns the first
    value.

    """
    value = group.attrs.get(name, None)
    if value is not None:
        return value
    dataset = group.get(name, None)
    if dataset is None:
        raise ValueError("Could not find attribute or dataset named %r/%r" % (group.name, name))
    if not dataset.len():
        raise ValueError("Found dataset named %r/%r but it was empty" % (group.name, name))
    if not all(dataset.value == dataset.value[0]):
        raise ValueError("Not all values in %r/%r are equal. Values found: %r" % (group.name, name, dataset.value))
    return dataset.value[0]

parser = optparse.OptionParser()
parser.add_option("-b", "--blank-ms", default="/var/kat/static/blank.ms", help="Blank MS used as a template (default=%default)")
parser.add_option("-r" , "--ref-ant", help="Reference antenna (default is first one used by script)")
parser.add_option("-t", "--tar", action="store_true", default=False, help="Tar-ball the MS")
parser.add_option("-w", "--stop-w", action="store_true", default=False, help="Use the W term to stop the fringes for each baseline")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "No HDF5 filename supplied.\n"
    print "Usage: h5toms.py [options] <filename.h5>"
    sys.exit(1)
h5_filename = args[0]

if options.stop_w:
    print "W term in UVW coordinates will be used to stop the fringes."

if ms_extra.pyrap_fail:
    print "Failed to import pyrap. You need to have both casacore and pyrap installed in order to produce measurement sets."
    sys.exit(1)

ms_name = os.path.splitext(h5_filename)[0] + ".ms"

 # first step is to copy the blank template MS to our desired output...
#if os.path.isdir(ms_name) or os.path.isdir(
try:
    shutil.copytree(options.blank_ms, ms_name)
except OSError:
    print "Failed to copy blank MS from %s to %s - please check presence of blank MS and/or permissions" % (options.blank_ms, ms_name)
    sys.exit(1)

print "Will create MS output in", ms_name

f = h5py.File(h5_filename, 'r+')

# Only continue if file is correct version and has been properly augmented
version = f.attrs.get('version', '1.x')
if not version.startswith('2.'):
    print "Attempting to load version '%s' file with version 2 loader" % (version,))
    sys.exit(1)
if not 'augment_ts' in f.attrs:
    print "HDF5 file not augmented - please run k7_augment.py (provided by katsdisp package)"
    sys.exit(1)

 # build the antenna table
antenna_objs = {}
antenna_positions = []
antenna_diameter = 0
dbe_map = {}
id_map = {}
ant_map = {}
ant_config = f['/MetaData/Configuration/Antennas']
for ant_name in ant_config:
    print ant_name
    ant = katpoint.Antenna(ant_config[ant_name].attrs['description'])
    antenna_objs[ant_name] = ant
    antenna_diameter = ant.diameter
    antenna_positions.append(ant.position_ecef)
    num_receptors_per_feed = 2

refant = f['MetaData/Configuration/Observation'].attrs['script_ants'].split(",")[0] if options.ref_ant is None else options.ref_ant
refant_obj = antenna_objs[refant]
ant_sensors = f['/MetaData/Sensors/Antennas'][refant]
print "\nUsing %s as the reference antenna. All targets and activity detection will be based on this antenna.\n" % (refant,)

telescope_name = "KAT-7"
observer_name = f['/MetaData/Configuration/Observation'].attrs['script_observer']
project_name = f['/MetaData/Configuration/Observation'].attrs['script_experiment_id']

# Get center frequency in Hz, assuming it hasn't changed during the experiment
try:
    band_center = f['/MetaData/Sensors/RFE/center-frequency-hz']['value'][0]
except (KeyError, h5py.H5Error):
    raise ValueError("Center frequency sensor '/MetaData/Sensors/RFE/center-frequency-hz' not found")

# Load correlator configuration group
corr_config = f['MetaData/Configuration/Correlator']
# Construct channel center frequencies from DBE attributes
num_channels = get_single_value(corr_config, 'n_chans')
channel_bw = get_single_value(corr_config, 'bandwidth') / num_channels
# Assume that lower-sideband downconversion has been used, which flips frequency axis
# Also subtract half a channel width to get frequencies at center of each channel
center_freqs = band_center - channel_bw * (np.arange(num_channels) - num_channels / 2 + 0.5)
sample_period = get_single_value(corr_config, 'int_time')
dump_rate = 1.0 / sample_period

n_bls = corr_config.attrs['n_bls']
n_pol = corr_config.attrs['n_stokes']
bls_ordering = corr_config.attrs['bls_ordering']

field_id = 0
field_counter = -1
fields = {}
obs_start = 0
obs_end = 0

data = f['/Data/correlator_data']
data_timestamps = f['/Data/timestamps'].value
dump_endtimes = data_timestamps + 0.5 * sample_period

# Use the activity sensor of ref antenna to partition the data set into scans (and to label the scans)
activity_sensor = ant_sensors['activity']
# Simplify the activities to derive the basic state of the antenna (slewing, scanning, tracking, stopped)
simplify = {'scan': 'scan', 'track': 'track', 'slew': 'slew', 'scan_ready': 'slew', 'scan_complete': 'slew'}
state = np.array([simplify.get(act, 'stop') for act in activity_sensor['value']])
state_changes = [n for n in xrange(len(state)) if (n == 0) or (state[n] != state[n - 1])]
scan_labels, state_timestamps = state[state_changes], activity_sensor['timestamp'][state_changes]
scan_starts = dump_endtimes.searchsorted(state_timestamps)
scan_ends = np.r_[scan_starts[1:] - 1, len(dump_endtimes) - 1]

target_sensor = ant_sensors['target']
target, target_timestamps = target_sensor['value'], target_sensor['timestamp']
target_changes = [n for n in xrange(len(target)) if target[n] and ((n== 0) or (target[n] != target[n - 1]))]
target, target_timestamps = target[target_changes],target_timestamps[target_changes]
compscan_starts = dump_endtimes.searchsorted(target_timestamps)
compscan_ends = np.r_[compscan_starts[1:] - 1, len(dump_endtimes) - 1]

ms_dict = {}
ms_dict['ANTENNA'] = ms_extra.populate_antenna_dict(antenna_positions, antenna_diameter)
ms_dict['FEED'] = ms_extra.populate_feed_dict(len(antenna_positions), num_receptors_per_feed)
ms_dict['DATA_DESCRIPTION'] = ms_extra.populate_data_description_dict()
ms_dict['POLARIZATION'] = ms_extra.populate_polarization_dict(pol_type='HV')
ms_dict['MAIN'] = []
ms_dict['FIELD'] = []

for i,c_start_id in enumerate(compscan_starts):
    c_end_id = compscan_ends[i] + 1
    print "Cscan runs from id %i to id %i\n" % (c_start_id, c_end_id)
    tstamps = data_timestamps[c_start_id:c_end_id]
    c_start = tstamps[0]
    c_end = tstamps[-1]
    tgt = katpoint.Target(target[i][1:-1]) # strip out " from sensor values
    tgt.antenna = refant_obj
    radec = tgt.radec()
    if fields.has_key(tgt.description):
        field_id = fields[tgt.description]
    else:
        field_counter += 1
        field_id = field_counter
        fields[tgt.description] = field_counter
        ms_dict['FIELD'].append(ms_extra.populate_field_dict(tgt.radec(), katpoint.Timestamp(c_start).to_mjd() * 24 * 60 * 60, field_name=tgt.name))
          # append this new field
        print "Adding new field id",field_id,"with radec",tgt.radec()

    tstamps = tstamps + (0.5/dump_rate)
             # move timestamps to middle of integration
    mjd_tstamps = [katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60 for t in tstamps]
    data = data_ref[c_start_id:c_end_id].astype(np.float32).view(np.complex64)[:,:,:,:,0].swapaxes(1,2).swapaxes(0,1)
     # pick up the data segement for this compound scan, reorder into bls, timestamp, channels, pol, complex
    for bl in range(n_bls):
        (a1, a2) = bls_ordering[bl]
        if a1 > 6 or a2 > 6: continue
        a1_name = 'ant' + str(a1 + 1)
        a2_name = 'ant' + str(a2 + 1)
        uvw_coordinates = np.array(tgt.uvw(antenna_objs[a2_name], tstamps / 1000, antenna_objs[a1_name]))
	    vis_data = data[bl]
        if options.stop_w:
            cable_delay_diff = (delays[int(a2)-1] - delays[int(a1)-1])
            w = np.outer(((uvw_coordinates[2] / katpoint.lightspeed) + cable_delay_diff), center_freqs)
             # get w in terms of phase (in radians). This is now frequency dependent so has shape(tstamps, channels)
            vis_data *= np.exp(-2j * np.pi * w)
            # recast the data into (ts, channels, polarisations, complex64)
        ms_dict['MAIN'].append(ms_extra.populate_main_dict(uvw_coordinates, vis_data, mjd_tstamps, int(a1), int(a2), 1.0/dump_rate, field_id))

     # handle the per compound scan specific MS rows. (And those that are common, but only known after at least on CS)
     # the field will in theory change per compound scan. The spectral window should be constant, but is only known
     # after parsing at least one scan.
    if not ms_dict.has_key('SPECTRAL_WINDOW'):
        ms_dict['SPECTRAL_WINDOW'] = ms_extra.populate_spectral_window_dict(center_frequency, channel_bw, num_channels)
 # end of compound scans
ms_dict['OBSERVATION'] = ms_extra.populate_observation_dict(obs_start, obs_end, telescope_name, observer_name, project_name)

 # finally we write the ms as per our created dicts
ms_extra.write_dict(ms_dict, ms_name)
if options.tar:
    tar = tarfile.open('%s.tar' % ms_name, 'w')
    tar.add(ms_name, arcname=os.path.basename(ms_name))
    tar.close()
