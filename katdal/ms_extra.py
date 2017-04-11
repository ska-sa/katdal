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

"""Create MS compatible data and write this data into a template MeasurementSet.

This can use either casapy (the default) or pyrap to create the MS.

"""
#
# Ludwig Schwardt
# 25 March 2008
#

import sys
import os.path
from copy import deepcopy

import numpy as np

# Look for a casacore library binding that will provide Table tools
try:
    # Try to use the casapy table tool first
    import casac
    tb = casac.homefinder.find_home_by_name('tableHome').create()
    casacore_binding = 'casapy'
except:
    try:
        # Otherwise fall back to pyrap
        from pyrap import tables
        casacore_binding = 'pyrap'
    except ImportError:
        casacore_binding = ''

# Table descriptions for the standard MS tables
# ---------------------------------------------
# These dicts are used to create a new MS via pyrap. They are obtained by first
# creating a blank MS in CASA, using::
#
#   sm.open('blank.ms')
#   sm.close()
#
# The structure of this MS is then extracted in pyrap by getting the description
# of the MAIN and sub-tables::
#
#   import pyrap.tables as tables
#   import os
#   ms_name = 'blank.ms'
#   ms_desc = {'MAIN' : tables.table(ms_name).getdesc(actual=False)}
#   for sub_table in os.walk(ms_name).next()[1]:
#       ms_desc[sub_table] = tables.table(os.path.join(ms_name, sub_table)).getdesc(actual=False)
#   ...
#


def std_scalar(comment, valueType='integer', option=0, **kwargs):
    """Description for standard scalar column."""
    return dict(comment=comment, valueType=valueType, dataManagerType='StandardStMan',
                dataManagerGroup='StandardStMan', option=option, maxlen=0, **kwargs)


def std_array(comment, valueType, ndim, **kwargs):
    """Description for standard array column with variable shape (used for smaller arrays)."""
    return dict(comment=comment, valueType=valueType, ndim=ndim, dataManagerType='StandardStMan',
                dataManagerGroup='StandardStMan', _c_order=True, option=0, maxlen=0, **kwargs)


def fixed_array(comment, valueType, shape, **kwargs):
    """Description for direct array column with fixed shape (used for smaller arrays)."""
    return dict(comment=comment, valueType=valueType, shape=np.asarray(shape, dtype=np.int32), ndim=len(shape),
                dataManagerType='StandardStMan', dataManagerGroup='StandardStMan',
                _c_order=True, option=5, maxlen=0, **kwargs)


def tiled_array(comment, valueType, ndim, dataManagerGroup, **kwargs):
    """Description for array column with tiled storage manager (used for bigger arrays)."""
    return dict(comment=comment, valueType=valueType, ndim=ndim, dataManagerType='TiledShapeStMan',
                dataManagerGroup=dataManagerGroup, _c_order=True, option=0, maxlen=0, **kwargs)


def define_hypercolumn(desc):
    """Add hypercolumn definitions to table description."""
    desc['_define_hypercolumn_'] = dict([(v['dataManagerGroup'],
                                          dict(HCdatanames=[k], HCndim=v['ndim'] + 1, HCcoordnames=[], HCidnames=[]))
                                         for k, v in desc.iteritems() if v['dataManagerType'] == 'TiledShapeStMan'])

ms_desc = {}

ms_desc['MAIN'] = {
    'ANTENNA1': std_scalar('ID of first antenna in interferometer'),
    'ANTENNA2': std_scalar('ID of second antenna in interferometer'),
    'ARRAY_ID': std_scalar('ID of array or subarray'),
    'CORRECTED_DATA': tiled_array('The corrected data column', 'complex', 2, 'CorrectedDataColumn'),
    'DATA': tiled_array('The data column', 'complex', 2, 'dataHyperColumn'),
    'DATA_DESC_ID': std_scalar('The data description table index'),
    'EXPOSURE': std_scalar('The effective integration time', 'double'),
    'FEED1': std_scalar('The feed index for ANTENNA1'),
    'FEED2': std_scalar('The feed index for ANTENNA2'),
    'FIELD_ID': std_scalar('Unique id for this pointing'),
    'FLAG': tiled_array('The data flags, array of bools with same shape as data', 'boolean', 2, 'FlagColumn'),
    'FLAG_CATEGORY': tiled_array('The flag category, NUM_CAT flags for each datum', 'boolean', 3, 'flagHyperColumn'),
    'FLAG_ROW': std_scalar('Row flag - flag all data in this row if True', valueType='boolean'),
    'IMAGING_WEIGHT': tiled_array('Weight set by imaging task (e.g. uniform weighting)',
                                  'float', 1, 'imWeightHyperColumn'),
    'INTERVAL': std_scalar('The sampling interval', 'double'),
    'MODEL_DATA': tiled_array('The model data column', 'complex', 2, 'ModelDataColumn'),
    'OBSERVATION_ID': std_scalar('ID for this observation, index in OBSERVATION table'),
    'PROCESSOR_ID': std_scalar('Id for backend processor, index in PROCESSOR table'),
    'SCAN_NUMBER': std_scalar('Sequential scan number from on-line system'),
    'SIGMA': tiled_array('Estimated rms noise for channel with unity bandpass response', 'float', 1, 'SigmaColumn'),
    'STATE_ID': std_scalar('ID for this observing state'),
    'TIME': std_scalar('Modified Julian Day', 'double'),
    'TIME_CENTROID': std_scalar('Modified Julian Day', 'double'),
    'UVW': fixed_array('Vector with uvw coordinates (in meters)', 'double', [3]),
    'WEIGHT': tiled_array('Weight for each polarization spectrum', 'float', 1, 'WeightColumn'),
}

ms_desc['ANTENNA'] = {
    'DISH_DIAMETER': std_scalar('Physical diameter of dish', 'double'),
    'FLAG_ROW': std_scalar('Flag for this row', 'boolean'),
    'MOUNT': std_scalar('Mount type e.g. alt-az, equatorial, etc.', 'string'),
    'NAME': std_scalar('Antenna name, e.g. VLA22, CA03', 'string'),
    'OFFSET': fixed_array('Axes offset of mount to FEED REFERENCE point', 'double', [3]),
    'POSITION': fixed_array('Antenna X,Y,Z phase reference position', 'double', [3]),
    'STATION': std_scalar('Station (antenna pad) name', 'string'),
    'TYPE': std_scalar('Antenna type (e.g. SPACE-BASED)', 'string'),
}

ms_desc['FEED'] = {
    'ANTENNA_ID': std_scalar('ID of antenna in this array'),
    'BEAM_ID': std_scalar('Id for BEAM model'),
    'BEAM_OFFSET': std_array('Beam position offset (on sky but in antennareference frame)', 'double', 2),
    'FEED_ID': std_scalar('Feed id'),
    'INTERVAL': std_scalar('Interval for which this set of parameters is accurate', 'double'),
    'NUM_RECEPTORS': std_scalar('Number of receptors on this feed (probably 1 or 2'),
    'POLARIZATION_TYPE': std_array('Type of polarization to which a given RECEPTOR responds', 'string', 1),
    'POL_RESPONSE': std_array('D-matrix i.e. leakage between two receptors', 'complex', 2),
    'POSITION': fixed_array('Position of feed relative to feed reference position', 'double', [3]),
    'RECEPTOR_ANGLE': std_array('The reference angle for polarization', 'double', 1),
    'SPECTRAL_WINDOW_ID': std_scalar('ID for this spectral window setup'),
    'TIME': std_scalar('Midpoint of time for which this set of parameters is accurate', 'double'),
}

ms_desc['DATA_DESCRIPTION'] = {
    'FLAG_ROW': std_scalar('Flag this row', 'boolean'),
    'POLARIZATION_ID': std_scalar('Pointer to polarization table'),
    'SPECTRAL_WINDOW_ID': std_scalar('Pointer to spectralwindow table'),
}

ms_desc['POLARIZATION'] = {
    'CORR_PRODUCT': std_array('Indices describing receptors of feed going into correlation', 'integer', 2),
    'CORR_TYPE': std_array('The polarization type for each correlation product, as a Stokes enum.', 'integer', 1),
    'FLAG_ROW': std_scalar('Row flag', 'boolean'),
    'NUM_CORR': std_scalar('Number of correlation products'),
}

ms_desc['OBSERVATION'] = {
    'FLAG_ROW': std_scalar('Row flag', 'boolean'),
    'LOG': std_array('Observing log', 'string', 1),
    'PROJECT': std_scalar('Project identification string', 'string'),
    'OBSERVER': std_scalar('Name of observer(s)', 'string'),
    'RELEASE_DATE': std_scalar('Release date when data becomes public', 'double'),
    'SCHEDULE': std_array('Observing schedule', 'string', 1),
    'SCHEDULE_TYPE': std_scalar('Observing schedule type', 'string'),
    'TELESCOPE_NAME': std_scalar('Telescope Name (e.g. WSRT, VLBA)', 'string'),
    'TIME_RANGE': fixed_array('Start and end of observation', 'double', [2]),
}

ms_desc['SPECTRAL_WINDOW'] = {
    'CHAN_FREQ': std_array('Center frequencies for each channel in the data matrix', 'double', 1),
    'CHAN_WIDTH': std_array('Channel width for each channel', 'double', 1),
    'EFFECTIVE_BW': std_array('Effective noise bandwidth of each channel', 'double', 1),
    'FLAG_ROW': std_scalar('Row flag', 'boolean'),
    'FREQ_GROUP': std_scalar('Frequency group'),
    'FREQ_GROUP_NAME': std_scalar('Frequency group name', 'string'),
    'IF_CONV_CHAIN': std_scalar('The IF conversion chain number'),
    'MEAS_FREQ_REF': std_scalar('Frequency Measure reference'),
    'NAME': std_scalar('Spectral window name', 'string'),
    'NET_SIDEBAND': std_scalar('Net sideband'),
    'NUM_CHAN': std_scalar('Number of spectral channels'),
    'REF_FREQUENCY': std_scalar('The reference frequency', 'double'),
    'RESOLUTION': std_array('The effective noise bandwidth for each channel', 'double', 1),
    'TOTAL_BANDWIDTH': std_scalar('The total bandwidth for this window', 'double'),
}

ms_desc['FIELD'] = {
    'CODE': std_scalar('Special characteristics of field, e.g. Bandpass calibrator', 'string'),
    'DELAY_DIR': std_array('Direction of delay center (e.g. RA, DEC)as polynomial in time.', 'double', 2),
    'FLAG_ROW': std_scalar('Row Flag', 'boolean'),
    'NAME': std_scalar('Name of this field', 'string'),
    'NUM_POLY': std_scalar('Polynomial order of _DIR columns'),
    'PHASE_DIR': std_array('Direction of phase center (e.g. RA, DEC).', 'double', 2),
    'REFERENCE_DIR': std_array('Direction of REFERENCE center (e.g. RA, DEC).as polynomial in time.', 'double', 2),
    'SOURCE_ID': std_scalar('Source id'),
    'TIME': std_scalar('Time origin for direction and rate', 'double'),
}

ms_desc['STATE'] = {
    'SIG': std_scalar('True if the source signal is being observed.'),
    'REF': std_scalar('True for a reference phase.'),
    'CAL': std_scalar('Noise calibration temperature (zero if not added).'),
    'LOAD': std_scalar('Load temperature (zero if no load).'),
    'SUB_SCAN': std_scalar('Sub-scan number, relative to the SCAN_NUMBER in MAIN.'),
    'OBS_MODE': std_scalar('Observing mode, defined by a set of reserved keywords.'),
    'FLAG_ROW': std_scalar('Row Flag', 'boolean'),
}

ms_desc['SOURCE'] = {
    'DIRECTION': std_array('Source direction at specified time', 'double', 2),
    'PROPER_MOTION': std_array('Source proper motion at specified time', 'double', 2),
    'SOURCE_ID': std_scalar('Source identifier as specified in FIELD table', 'integer'),
    'TIME': std_scalar('Mid point of time interval for which this row is valid', 'double'),
    'CALIBRATION_GROUP': std_scalar('Cal group to which this source belongs', 'integer'),
    'NAME': std_scalar('Name of this source', 'string'),
    'NUM_LINES': std_scalar('Number of line transitions associated with this source', 'integer'),
    'REST_FREQUENCY': std_scalar('Rest frequency of line', 'double'),
    'SYSVEL': std_scalar('Systemic velocity of line', 'double'),
}

ms_desc['POINTING'] = {
    'ANTENNA_ID': std_scalar('Antenna Id'),
    'DIRECTION': std_array('Antenna pointing direction as polynomial in time', 'double', 2),
    'INTERVAL': std_scalar('Time interval', 'double'),
    'NAME': std_scalar('Pointing position name', 'string'),
    'NUM_POLY': std_scalar('Series order'),
    'TARGET': std_array('target direction as polynomial in time', 'double', -1),
    'TIME': std_scalar('Time interval midpoint', 'double'),
    'TIME_ORIGIN': std_scalar('Time origin for direction', 'double'),
    'TRACKING': std_scalar('Tracking flag - True if on position', 'boolean'),
}

for sub_table in ms_desc:
    define_hypercolumn(ms_desc[sub_table])

caltable_desc = {}
caltable_desc['TIME'] = std_scalar('Timestamp of solution', 'double', option=5)
caltable_desc['FIELD_ID'] = std_scalar('Unique id for this pointing', 'integer', option=5)
caltable_desc['SPECTRAL_WINDOW_ID'] = std_scalar('Spectral window', 'integer', option=5)
caltable_desc['ANTENNA1'] = std_scalar('ID of first antenna in interferometer', 'integer', option=5)
caltable_desc['ANTENNA2'] = std_scalar('ID of second antenna in interferometer', 'integer', option=5)
caltable_desc['INTERVAL'] = std_scalar('The effective integration time', 'double', option=5)
caltable_desc['SCAN_NUMBER'] = std_scalar('Scan number', 'integer', option=5)
caltable_desc['OBSERVATION_ID'] = std_scalar('Observation id (index in OBSERVATION table)', 'integer', option=5)
caltable_desc['PARAMERR'] = std_array('Parameter error', 'float', -1)
caltable_desc['FLAG'] = std_array('Solution values', 'boolean', -1)
caltable_desc['SNR'] = std_array('Signal to noise ratio', 'float', -1)
caltable_desc['WEIGHT'] = std_array('Weight', 'float', -1)
# float version of caltable
caltable_desc_float = deepcopy(caltable_desc)
caltable_desc_float['FPARAM'] = std_array('Solution values', 'float', -1)
define_hypercolumn(caltable_desc_float)
# complex version of caltable
caltable_desc_complex = deepcopy(caltable_desc)
caltable_desc_complex['CPARAM'] = std_array('Solution values', 'complex', -1)
define_hypercolumn(caltable_desc_complex)

# Define the appropriate way to open a table using the selected binding
if casacore_binding == 'casapy':
    def open_table(filename, readonly=False, ack=False, **kwargs):
        success = tb.open(filename, nomodify=readonly, **kwargs)
        return tb if success else None

elif casacore_binding == 'pyrap':
    def open_table(filename, readonly=False, ack=False, **kwargs):
        t = tables.table(filename, readonly=readonly, ack=ack, **kwargs)

        return t if type(t) == tables.table else None

else:
    def open_table(filename, readonly=False):
        raise ImportError("Cannot open MS '%s', as neither casapy nor pyrap were found" % (filename,))


# -------- Routines that create MS data structures in dictionaries -----------

def populate_main_dict(uvw_coordinates, vis_data, flag_data, timestamps, antenna1_index,
                       antenna2_index, integrate_length, field_id=0, state_id=1,
                       scan_number=0, model_data=None, corrected_data=None):
    """Construct a dictionary containing the columns of the MAIN table.

    The MAIN table contains the visibility data itself. The vis data has shape
    (num_vis_samples, num_pols, num_channels). The table has one row per
    visibility sample, which is one row per baseline per snapshot (time sample).

    Parameters
    ----------
    uvw_coordinates : array of float, shape (num_vis_samples, 3)
        Array containing (u,v,w) coordinates in metres
    vis_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
        Array containing complex visibility data in Janskys
    flag_data : array of boolean, shape same as vis_data
    timestamps : array of float, shape (num_vis_samples,)
        Array of timestamps as Modified Julian Dates in seconds
        (may contain duplicate times for multiple baselines)
    antenna1_index : int or array of int, shape (num_vis_samples,)
        Array containing the index of the first antenna of each vis sample
    antenna2_index : int or array of int, shape (num_vis_samples,)
        Array containing the index of the second antenna of each vis sample
    integrate_length : float
        The integration time (one over dump rate), in seconds
    field_id : int or array of int, shape (num_vis_samples,), optional
        The field ID (pointing) associated with this data
    state_id : int or array of int, shape (num_vis_samples,), optional
        The state ID (observation intent) associated with this data
    scan_number : int or array of int, shape (num_vis_samples,), optional
        The scan index (compound scan index in the case of KAT-7)
    model_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
        Array containing complex visibility data in Janskys
    corrected_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
        Array containing complex visibility data in Janskys

    Returns
    -------
    main_dict : dict
        Dictionary containing columns of MAIN table

    Raises
    ------
    ValueError
        If there is a shape mismatch between some input arrays

    """
    num_vis_samples, num_channels, num_pols = vis_data.shape
    timestamps = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))

    main_dict = {}
    # ID of first antenna in interferometer (integer)
    main_dict['ANTENNA1'] = antenna1_index
    # ID of second antenna in interferometer (integer)
    main_dict['ANTENNA2'] = antenna2_index
    # ID of array or subarray (integer)
    main_dict['ARRAY_ID'] = np.zeros(num_vis_samples, dtype=np.int32)
    # The corrected data column (complex, 3-dim)
    if corrected_data is not None:
        main_dict['CORRECTED_DATA'] = corrected_data
    # The data column (complex, 3-dim)
    main_dict['DATA'] = vis_data
    # The data description table index (integer)
    main_dict['DATA_DESC_ID'] = np.zeros(num_vis_samples, dtype=np.int32)
    # The effective integration time (double)
    main_dict['EXPOSURE'] = integrate_length * np.ones(num_vis_samples)
    # The feed index for ANTENNA1 (integer)
    main_dict['FEED1'] = np.zeros(num_vis_samples, dtype=np.int32)
    # The feed index for ANTENNA1 (integer)
    main_dict['FEED2'] = np.zeros(num_vis_samples, dtype=np.int32)
    # Unique id for this pointing (integer)
    main_dict['FIELD_ID'] = field_id
    # The data flags, array of bools with same shape as data
    main_dict['FLAG'] = flag_data
    # The flag category, NUM_CAT flags for each datum [snd 1 is num channels] (boolean, 4-dim)
    main_dict['FLAG_CATEGORY'] = np.zeros((num_vis_samples, 1, num_channels, num_pols), dtype='uint8')
    # Row flag - flag all data in this row if True (boolean)
    main_dict['FLAG_ROW'] = np.zeros(num_vis_samples, dtype=np.uint8)
    # Weight set by imaging task (e.g. uniform weighting) (float, 1-dim)
    # main_dict['IMAGING_WEIGHT'] = np.ones((num_vis_samples, 1), dtype=np.float32)
    # The sampling interval (double)
    main_dict['INTERVAL'] = integrate_length * np.ones(num_vis_samples)
    # The model data column (complex, 3-dim)
    if model_data is not None:
        main_dict['MODEL_DATA'] = model_data
    # ID for this observation, index in OBSERVATION table (integer)
    main_dict['OBSERVATION_ID'] = np.zeros(num_vis_samples, dtype=np.int32)
    # Id for backend processor, index in PROCESSOR table (integer)
    main_dict['PROCESSOR_ID'] = - np.ones(num_vis_samples, dtype=np.int32)
    # Sequential scan number from on-line system (integer)
    main_dict['SCAN_NUMBER'] = scan_number
    # Estimated rms noise for channel with unity bandpass response (float, 1-dim)
    main_dict['SIGMA'] = np.ones((num_vis_samples, num_pols), dtype=np.float32)
    # ID for this observing state (integer)
    main_dict['STATE_ID'] = state_id
    # Modified Julian Dates in seconds (double)
    main_dict['TIME'] = timestamps
    # Modified Julian Dates in seconds (double)
    main_dict['TIME_CENTROID'] = timestamps
    # Vector with uvw coordinates (in metres) (double, 1-dim, shape=(3,))
    main_dict['UVW'] = np.asarray(uvw_coordinates)
    # Weight for each polarisation spectrum (float, 1-dim)
    main_dict['WEIGHT'] = np.ones((num_vis_samples, num_pols), dtype=np.float32)
    return main_dict


def populate_caltable_main_dict(solution_times, solution_values, antennas, scans):
    """Construct a dictionary containing the columns of the MAIN table.

    The MAIN table contains the gain solution data itself. The shape of the data
    sepends on the nature of the solution: (npol,1) for gains and delays and
    (npol, nchan) for bandpasses.
    The table has one row per antenna per time.

    Parameters
    ----------
    solution_times : array of float, shape (num_solutions,)
        Calibration solution times
    solution_values : array of float, shape (num_solutions,)
        Calibration solution values
    antennas: array of float, shape (num_solutions,)
        Antenna corresponding to each solution value
    scans: array of float, shape (num_solutions,)
        Scan number corresponding to each solution value

    Returns
    -------
    calibration_main_dict : dict
        Dictionary containing columns of the caltable MAIN table

    """
    num_rows = len(solution_times)
    calibration_main_dict = {}
    calibration_main_dict['TIME'] = solution_times
    calibration_main_dict['FIELD_ID'] = np.zeros(num_rows, dtype=np.int32)
    calibration_main_dict['SPECTRAL_WINDOW_ID'] = np.zeros(num_rows, dtype=np.int32)
    calibration_main_dict['ANTENNA1'] = antennas
    calibration_main_dict['ANTENNA2'] = np.zeros(num_rows, dtype=np.int32)
    calibration_main_dict['INTERVAL'] = np.zeros(num_rows, dtype=np.int32)
    calibration_main_dict['SCAN_NUMBER'] = scans
    calibration_main_dict['OBSERVATION_ID'] = np.zeros(num_rows, dtype=np.int32)
    if np.iscomplexobj(solution_values):
        calibration_main_dict['CPARAM'] = solution_values
    else:
        calibration_main_dict['FPARAM'] = solution_values
    calibration_main_dict['PARAMERR'] = np.zeros_like(solution_values, dtype=np.float32)
    calibration_main_dict['FLAG'] = np.zeros_like(solution_values, dtype=np.int32)
    calibration_main_dict['SNR'] = np.ones_like(solution_values, dtype=np.float32)
    return calibration_main_dict


def populate_antenna_dict(antenna_names, antenna_positions, antenna_diameters):
    """Construct a dictionary containing the columns of the ANTENNA subtable.

    The ANTENNA subtable contains info about each antenna, such as its name,
    position, mount type and diameter. It has one row per antenna.

    Parameters
    ----------
    antenna_names : array of string, shape (num_antennas,)
        Array of antenna names, one per antenna
    antenna_positions : array of float, shape (num_antennas, 3)
        Array of antenna positions in ECEF (aka XYZ) coordinates, in metres
    antenna_diameters : array of float, shape (num_antennas,)
        Array of antenna diameters, in metres

    Returns
    -------
    antenna_dict : dict
        Dictionary containing columns of ANTENNA subtable

    """
    num_antennas = len(antenna_names)
    antenna_dict = {}
    # Physical diameter of dish (double)
    antenna_dict['DISH_DIAMETER'] = np.asarray(antenna_diameters, np.float64)
    # Flag for this row (boolean)
    antenna_dict['FLAG_ROW'] = np.zeros(num_antennas, np.uint8)
    # Mount type e.g. alt-az, equatorial, etc. (string)
    antenna_dict['MOUNT'] = np.tile('ALT-AZ', num_antennas)
    # Antenna name, e.g. VLA22, CA03 (string)
    antenna_dict['NAME'] = np.asarray(antenna_names)
    # Axes offset of mount to FEED REFERENCE point (double, 1-dim, shape=(3,))
    antenna_dict['OFFSET'] = np.zeros((num_antennas, 3), np.float64)
    # Antenna X,Y,Z phase reference position (double, 1-dim, shape=(3,))
    antenna_dict['POSITION'] = np.asarray(antenna_positions, dtype=np.float64)
    # Station (antenna pad) name (string)
    antenna_dict['STATION'] = np.asarray(antenna_names)
    # Antenna type (e.g. SPACE-BASED) (string)
    antenna_dict['TYPE'] = np.tile('GROUND-BASED', num_antennas)
    return antenna_dict


def populate_feed_dict(num_feeds, num_receptors_per_feed=2):
    """Construct a dictionary containing the columns of the FEED subtable.

    The FEED subtable specifies feed characteristics such as polarisation and
    beam offsets. It has one row per feed (typically one feed per antenna).
    Each feed has a number of receptors (typically one per polarisation type).

    Parameters
    ----------
    num_feeds : integer
        Number of feeds in telescope (typically equal to number of antennas)
    num_receptors_per_feed : integer, optional
        Number of receptors per feed (usually one per polarisation type)

    Returns
    -------
    feed_dict : dict
        Dictionary containing columns of FEED subtable

    """
    feed_dict = {}
    # ID of antenna in this array (integer)
    feed_dict['ANTENNA_ID'] = np.arange(num_feeds, dtype=np.int32)
    # Id for BEAM model (integer)
    feed_dict['BEAM_ID'] = np.ones(num_feeds, dtype=np.int32)
    # Beam position offset (on sky but in antenna reference frame): (double, 2-dim)
    feed_dict['BEAM_OFFSET'] = np.zeros((num_feeds, 2, 2), dtype=np.float64)
    # Feed id (integer)
    feed_dict['FEED_ID'] = np.zeros(num_feeds, dtype=np.int32)
    # Interval for which this set of parameters is accurate (double)
    feed_dict['INTERVAL'] = np.zeros(num_feeds, dtype=np.float64)
    # Number of receptors on this feed (probably 1 or 2) (integer)
    feed_dict['NUM_RECEPTORS'] = np.tile(np.int32(num_receptors_per_feed), num_feeds)
    # Type of polarisation to which a given RECEPTOR responds (string, 1-dim)
    feed_dict['POLARIZATION_TYPE'] = np.tile(['X', 'Y'], (num_feeds, 1))
    # D-matrix i.e. leakage between two receptors (complex, 2-dim)
    feed_dict['POL_RESPONSE'] = np.dstack([np.eye(2, dtype=np.complex64) for n in range(num_feeds)]).transpose()
    # Position of feed relative to feed reference position (double, 1-dim, shape=(3,))
    feed_dict['POSITION'] = np.zeros((num_feeds, 3), np.float64)
    # The reference angle for polarisation (double, 1-dim)
    feed_dict['RECEPTOR_ANGLE'] = np.zeros((num_feeds, num_receptors_per_feed), dtype=np.float64)
    # ID for this spectral window setup (integer)
    feed_dict['SPECTRAL_WINDOW_ID'] = - np.ones(num_feeds, dtype=np.int32)
    # Midpoint of time for which this set of parameters is accurate (double)
    feed_dict['TIME'] = np.zeros(num_feeds, dtype=np.float64)
    return feed_dict


def populate_data_description_dict():
    """Construct a dictionary containing the columns of the DATA_DESCRIPTION subtable.

    The DATA_DESCRIPTION subtable groups together a set of polarisation and
    frequency parameters, which may differ for various experiments done on the
    same data set. It has one row per data setting.

    Returns
    -------
    data_description_dict : dict
        Dictionary containing columns of DATA_DESCRIPTION subtable

    """
    data_description_dict = {}
    # Flag this row (boolean)
    data_description_dict['FLAG_ROW'] = np.zeros(1, dtype=np.uint8)
    # Pointer to polarisation table (integer)
    data_description_dict['POLARIZATION_ID'] = np.zeros(1, dtype=np.int32)
    # Pointer to spectralwindow table (integer)
    data_description_dict['SPECTRAL_WINDOW_ID'] = np.zeros(1, dtype=np.int32)
    return data_description_dict


def populate_polarization_dict(ms_pols=['HH', 'VV'], stokes_i=False, circular=False):
    """Construct a dictionary containing the columns of the POLARIZATION subtable.

    The POLARIZATION subtable describes how the various receptors are correlated
    to create the Stokes terms. It has one row per polarisation setting.

    Parameters
    ----------
    ms_pols : ['HH'] | ['VV'] | ['HH','VV'] | ['HH','VV','HV','VH']
        The polarisations used in this dataset
    stokes_i : False
        Mark single pol as Stokes I
    circular : False
        Label the linear pols with circular (for fun and/or profit)

    Returns
    -------
    polarization_dict : dict
        Dictionary containing columns of POLARIZATION subtable

    """
    pol_num = {'H': 0, 'V': 1}
    #  lookups for converting to CASA speak...
    pol_types = {'I': 1, 'Q': 2, 'U': 3, 'V': 4, 'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8,
                 'HH': 9, 'VV': 12, 'HV': 10, 'VH': 11}
    if len(ms_pols) > 1 and stokes_i:
        print "Warning: Polarisation to be marked as stokes, but more than 1 polarisation " \
              "product specified. Using first specified pol (%s)" % ms_pols[0]
        ms_pols = [ms_pols[0]]
    #  Indices describing receptors of feed going into correlation (integer, 2-dim)
    polarization_dict = {}
    #  The polarisation type for each correlation product, as a Stokes enum (4 integer, 1-dim)
    #  Stokes enum (starting at 1) = {I, Q, U, V, RR, RL, LR, LL, XX, XY, YX, YY, ...}
    #  The native correlator data are in XX, YY, XY, YX for HV pol, XX for H pol and YY for V pol
    polarization_dict['CORR_PRODUCT'] = np.array([[pol_num[p[0]], pol_num[p[1]]]
                                                  for p in ms_pols], dtype=np.int32)[np.newaxis, :, :]
    polarization_dict['CORR_TYPE'] = np.array([pol_types[p] - (4 if circular else 0)
                                               for p in (['I'] if stokes_i else ms_pols)])[np.newaxis, :]
    #  Number of correlation products (integer)
    polarization_dict['FLAG_ROW'] = np.zeros(1, dtype=np.uint8)
    polarization_dict['NUM_CORR'] = np.array([len(ms_pols)], dtype=np.int32)
    return polarization_dict


def populate_observation_dict(start_time, end_time, telescope_name='unknown',
                              observer_name='unknown', project_name='unknown'):
    """Construct a dictionary containing the columns of the OBSERVATION subtable.

    The OBSERVATION subtable describes the overall project and the people doing
    the observing. It has one row per observation project/schedule?

    Parameters
    ----------
    start_time : float
        Start time of project, as a Modified Julian Date in seconds
    end_time : float
        End time of project, as a Modified Julian Date in seconds
    telescope_name : string, optional
        Telescope name
    observer_name : string, optional
        Name of observer
    project_name : string, optional
        Description of project

    Returns
    -------
    observation_dict : dict
        Dictionary containing columns of OBSERVATION subtable

    """
    observation_dict = {}
    # Row flag (boolean)
    observation_dict['FLAG_ROW'] = np.zeros(1, dtype=np.uint8)
    # Observing log (string, 1-dim)
    observation_dict['LOG'] = np.array(['unavailable']).reshape((1, 1))
    # Name of observer(s) (string)
    observation_dict['OBSERVER'] = np.array([observer_name])
    # Project identification string
    observation_dict['PROJECT'] = np.array([project_name])
    # Release date when data becomes public (double)
    observation_dict['RELEASE_DATE'] = np.array([end_time])
    # Observing schedule (string, 1-dim)
    observation_dict['SCHEDULE'] = np.array(['unavailable']).reshape((1, 1))
    # Observing schedule type (string)
    observation_dict['SCHEDULE_TYPE'] = np.array(['unknown'])
    # Telescope Name (e.g. WSRT, VLBA) (string)
    observation_dict['TELESCOPE_NAME'] = np.array([telescope_name])
    # Start and end of observation (double, 1-dim, shape=(2,))
    observation_dict['TIME_RANGE'] = np.array([[start_time, end_time]])
    return observation_dict


def populate_spectral_window_dict(center_frequencies, channel_bandwidths, ref_freq=None):
    """Construct a dictionary containing the columns of the SPECTRAL_WINDOW subtable.

    The SPECTRAL_WINDOW subtable describes groupings of frequency channels into
    spectral windows. It has one row per spectral window. At the moment, only a
    single spectral window is considered. The reference frequency is chosen to
    be the center frequency of the middle channel.

    Parameters
    ----------
    center_frequencies : array of float, shape (num_channels,)
        Observation center frequencies for each channel, in Hz
    channel_bandwidths : array of float, shape (num_channels,)
        Bandwidth for each channel, in Hz

    Returns
    -------
    spectral_window_dict : dict
        Dictionary containing columns of SPECTRAL_WINDOW subtable

    """
    num_channels = len(center_frequencies)
    if len(channel_bandwidths) != num_channels:
        raise ValueError('Lengths of center_frequencies and channel_bandwidths differ (%d vs %d)' %
                         (len(center_frequencies), len(channel_bandwidths)))
    spectral_window_dict = {}
    # Center frequencies for each channel in the data matrix (double, 1-dim)
    spectral_window_dict['CHAN_FREQ'] = np.array([center_frequencies], dtype=np.float64)
    # Channel width for each channel (double, 1-dim)
    spectral_window_dict['CHAN_WIDTH'] = np.array([channel_bandwidths], dtype=np.float64)
    # Effective noise bandwidth of each channel (double, 1-dim)
    spectral_window_dict['EFFECTIVE_BW'] = np.array([channel_bandwidths], dtype=np.float64)
    # Row flag (boolean)
    spectral_window_dict['FLAG_ROW'] = np.zeros(1, dtype=np.uint8)
    # Frequency group (integer)
    spectral_window_dict['FREQ_GROUP'] = np.zeros(1, dtype=np.int32)
    # Frequency group name (string)
    spectral_window_dict['FREQ_GROUP_NAME'] = np.array(['none'])
    # The IF conversion chain number (integer)
    spectral_window_dict['IF_CONV_CHAIN'] = np.zeros(1, dtype=np.int32)
    # Frequency Measure reference (integer) (5=Topocentric)
    spectral_window_dict['MEAS_FREQ_REF'] = np.array([5], dtype=np.int32)
    # Spectral window name (string)
    spectral_window_dict['NAME'] = np.array(['none'])
    # Net sideband (integer)
    spectral_window_dict['NET_SIDEBAND'] = np.ones(1, dtype=np.int32)
    # Number of spectral channels (integer)
    spectral_window_dict['NUM_CHAN'] = np.array([num_channels], dtype=np.int32)
    # The reference frequency (double) - pick the frequency of the middle channel
    if ref_freq is None:
        spectral_window_dict['REF_FREQUENCY'] = np.array([center_frequencies[num_channels // 2]], dtype=np.float64)
    else:
        spectral_window_dict['REF_FREQUENCY'] = np.array([ref_freq], dtype=np.float64)
    # The effective noise bandwidth for each channel (double, 1-dim)
    spectral_window_dict['RESOLUTION'] = np.array([channel_bandwidths], dtype=np.float64)
    # The total bandwidth for this window (double)
    spectral_window_dict['TOTAL_BANDWIDTH'] = np.array([channel_bandwidths.sum()], dtype=np.float64)
    return spectral_window_dict


def populate_source_dict(phase_centers, time_origins, center_frequencies, field_names=None):
    """Construct a dictionary containing the columns of the SOURCE subtable.

    The SOURCE subtable describes time-variable source information, that may
    be associated with a given FIELD_ID. It appears to be optional, but for
    completeness it is included here (with no time varying terms).
    """
    num_channels = len(center_frequencies)
    phase_centers = np.atleast_2d(np.asarray(phase_centers, np.float64))
    num_fields = len(phase_centers)
    if field_names is None:
        field_names = ['Source%d' % (field,) for field in range(num_fields)]
    source_dict = {}
    source_dict['SOURCE_ID'] = np.arange(num_fields,dtype=np.int32)
    source_dict['PROPER_MOTION'] = np.zeros((num_fields, 2), dtype=np.float32)
    source_dict['DIRECTION'] = phase_centers
    source_dict['CALIBRATION_GROUP'] = np.ones(num_fields,dtype=np.int32) * -1 # Grouping for calibration purpose
    source_dict['NAME'] = np.atleast_1d(field_names)
    source_dict['NUM_LINES'] = np.ones(num_fields,dtype=np.int32) # Number of spectral lines
    source_dict['TIME'] = np.atleast_1d(np.asarray(time_origins, dtype=np.float64))
    source_dict['REST_FREQUENCY'] = np.tile(np.array([center_frequencies[num_channels // 2]],
                                                     dtype=np.float64), (num_fields, 1))
    source_dict['SYSVEL'] = np.ones((num_fields, 1), dtype=np.float64)
    return source_dict


def populate_field_dict(phase_centers, time_origins, field_names=None):
    """Construct a dictionary containing the columns of the FIELD subtable.

    The FIELD subtable describes each field (or pointing) by its sky coordinates.
    It has one row per field/pointing.

    Parameters
    ----------
    phase_centers : array of float, shape (M, 2)
        Direction of *M* phase centers as (ra, dec) coordinates in radians
    time_origins : array of float, shape (M,)
        Time origins where the *M* phase centers are correct, as Modified Julian
        Dates in seconds
    field_names : array of string, shape (M,), optional
        Names of fields/pointings (typically some source names)

    Returns
    -------
    field_dict : dict
        Dictionary containing columns of FIELD subtable

    """
    phase_centers = np.atleast_2d(np.asarray(phase_centers, np.float64))[:, np.newaxis, :]
    num_fields = len(phase_centers)
    if field_names is None:
        field_names = ['Field%d' % (field,) for field in range(num_fields)]
    field_dict = {}
    # Special characteristics of field, e.g. position code (string)
    field_dict['CODE'] = np.tile('T', num_fields)
    # Direction of delay center (e.g. RA, DEC) as polynomial in time (double, 2-dim)
    field_dict['DELAY_DIR'] = phase_centers
    # Row flag (boolean)
    field_dict['FLAG_ROW'] = np.zeros(num_fields, dtype=np.uint8)
    # Name of this field (string)
    field_dict['NAME'] = np.atleast_1d(field_names)
    # Polynomial order of *_DIR columns (integer)
    field_dict['NUM_POLY'] = np.zeros(num_fields, dtype=np.int32)
    # Direction of phase center (e.g. RA, DEC) (double, 2-dim)
    field_dict['PHASE_DIR'] = phase_centers
    # Direction of REFERENCE center (e.g. RA, DEC) as polynomial in time (double, 2-dim)
    field_dict['REFERENCE_DIR'] = phase_centers
    # Source id (integer), or a value of -1 indicates there is no corresponding source defined
    field_dict['SOURCE_ID'] = np.arange(num_fields,dtype=np.int32) # the same as source id
    # Time origin for direction and rate (double)
    field_dict['TIME'] = np.atleast_1d(np.asarray(time_origins, dtype=np.float64))
    return field_dict


def populate_state_dict(obs_modes=['UNKNOWN']):
    """Construct a dictionary containing the columns of the STATE subtable.

    The STATE subtable describes observing modes.
    It has one row per observing modes.

    Parameters
    ----------
    obs_modes : array of string
        Observing modes, used to define the schedule strategy.

    Returns
    -------
    state_dict : dict
        Dictionary containing columns of STATE subtable

    """
    num_states = len(obs_modes)
    state_dict = {}
    # Signal (boolean)
    state_dict['SIG'] = np.ones(num_states, dtype=np.uint8)
    # Reference (boolean)
    state_dict['REF'] = np.zeros(num_states, dtype=np.uint8)
    # Noise calibration temperature (double)
    state_dict['CAL'] = np.zeros(num_states, dtype=np.float64)
    # Load temperature (double)
    state_dict['LOAD'] = np.zeros(num_states, dtype=np.float64)
    # Sub-scan number (int)
    state_dict['SUB_SCAN'] = np.zeros(num_states, dtype=np.int32)
    # Observing mode (string)
    state_dict['OBS_MODE'] = np.atleast_1d(obs_modes)
    # Row flag (boolean)
    state_dict['FLAG_ROW'] = np.zeros(num_states, dtype=np.uint8)
    return state_dict


def populate_pointing_dict(num_antennas, observation_duration, start_time, phase_center, pointing_name='default'):
    """Construct a dictionary containing the columns of the POINTING subtable.

    The POINTING subtable contains data on individual antennas tracking a target.
    It has one row per pointing/antenna?

    Parameters
    ----------
    num_antennas : integer
        Number of antennas
    observation_duration : float
        Length of observation, in seconds
    start_time : float
        Start time of observation, as a Modified Julian Date in seconds
    phase_center : array of float, shape (2,)
        Direction of phase center, in ra-dec coordinates as 2-element array
    pointing_name : string, optional
        Name for pointing

    Returns
    -------
    pointing_dict : dict
        Dictionary containing columns of POINTING subtable

    """
    phase_center = phase_center.reshape((2, 1, 1))
    pointing_dict = {}
    # Antenna Id (integer)
    pointing_dict['ANTENNA_ID'] = np.arange(num_antennas, dtype=np.int32)
    # Antenna pointing direction as polynomial in time (double, 2-dim)
    pointing_dict['DIRECTION'] = np.repeat(phase_center, num_antennas)
    # Time interval (double)
    pointing_dict['INTERVAL'] = np.tile(np.float64(observation_duration), num_antennas)
    # Pointing position name (string)
    pointing_dict['NAME'] = np.array([pointing_name] * num_antennas)
    # Series order (integer)
    pointing_dict['NUM_POLY'] = np.zeros(num_antennas, dtype=np.int32)
    # Target direction as polynomial in time (double, -1-dim)
    pointing_dict['TARGET'] = np.repeat(phase_center, num_antennas)
    # Time interval midpoint (double)
    pointing_dict['TIME'] = np.tile(np.float64(start_time), num_antennas)
    # Time origin for direction (double)
    pointing_dict['TIME_ORIGIN'] = np.tile(np.float64(start_time), num_antennas)
    # Tracking flag - True if on position (boolean)
    pointing_dict['TRACKING'] = np.ones(num_antennas, dtype=np.uint8)
    return pointing_dict


def populate_ms_dict(uvw_coordinates, vis_data, timestamps, antenna1_index, antenna2_index,
                     integrate_length, center_frequencies, channel_bandwidths,
                     antenna_names, antenna_positions, antenna_diameter,
                     num_receptors_per_feed, start_time, end_time,
                     telescope_name, observer_name, project_name, phase_center, obs_modes):
    """Construct a dictionary containing all the tables in a MeasurementSet.

    Parameters
    ----------
    uvw_coordinates : array of float, shape (num_vis_samples, 3)
        Array containing (u,v,w) coordinates in multiples of the wavelength
    vis_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
        Array containing complex visibility data in Janskys
    timestamps : array of float, shape (num_vis_samples,)
        Array of timestamps as Modified Julian Dates in seconds
    antenna1_index : int or array of int, shape (num_vis_samples,)
        Array containing the index of the first antenna of each uv sample
    antenna2_index : int or array of int, shape (num_vis_samples,)
        Array containing the index of the second antenna of each uv sample
    integrate_length : float
        The integration time (one over dump rate), in seconds
    center_frequencies : array of float, shape (num_channels,)
        Observation center frequencies for each channel, in Hz
    channel_bandwidths : array of float, shape (num_channels,)
        Bandwidth for each channel, in Hz
    antenna_names : array of string, shape (num_antennas,)
        Array of antenna names, one per antenna
    antenna_positions : array of float, shape (num_antennas, 3)
        Array of antenna positions in ECEF (aka XYZ) coordinates, in metres
    antenna_diameter : array of float, shape (num_antennas,)
        Array of antenna diameters, in metres
    num_receptors_per_feed : integer
        Number of receptors per feed (usually one per polarisation type)
    start_time : float
        Start time of project, as a Modified Julian Date in seconds
    end_time : float
        End time of project, as a Modified Julian Date in seconds
    telescope_name : string
        Telescope name
    observer_name : string
        Observer name
    project_name : string
        Description of project
    phase_center : array of float, shape (2,)
        Direction of phase center, in ra-dec coordinates as 2-element array
    obs_modes: array of strings
        Observing modes

    Returns
    -------
    ms_dict : dict
        Dictionary containing all tables and subtables of a measurement set

    """
    ms_dict = {}
    ms_dict['MAIN'] = populate_main_dict(uvw_coordinates, vis_data, timestamps,
                                         antenna1_index, antenna2_index, integrate_length)
    ms_dict['ANTENNA'] = populate_antenna_dict(antenna_names, antenna_positions, antenna_diameter)
    ms_dict['FEED'] = populate_feed_dict(len(antenna_positions), num_receptors_per_feed)
    ms_dict['DATA_DESCRIPTION'] = populate_data_description_dict()
    ms_dict['POLARIZATION'] = populate_polarization_dict()
    ms_dict['OBSERVATION'] = populate_observation_dict(start_time, end_time,
                                                       telescope_name, observer_name, project_name)
    ms_dict['SPECTRAL_WINDOW'] = populate_spectral_window_dict(center_frequencies, channel_bandwidths)
    ms_dict['FIELD'] = populate_field_dict(phase_center, start_time)
    ms_dict['STATE'] = populate_state_dict(obs_modes)
    ms_dict['SOURCE'] = populate_source_dict(phase_center)
    return ms_dict

# ----------------- Write completed dictionary to MS file --------------------


def open_main(ms_name='./blank.ms', verbose=True):
    t = open_table(ms_name, ack=verbose)
    if t is None:
        print "Failed to open main table for writing."
        sys.exit(1)
    return t


def write_rows(t, row_dict, verbose=True):
    num_rows = row_dict.values()[0].shape[0]
    # Append rows to the table by starting after the last row in table
    startrow = t.nrows()
    # Add the space required for this group of rows
    t.addrows(num_rows)
    if verbose:
        print "  added %d rows" % (num_rows,)
    for col_name, col_data in row_dict.iteritems():
        if col_name in t.colnames():
            try:
                t.putcol(col_name, col_data.T if casacore_binding == 'casapy' else col_data, startrow)
                if verbose:
                    print "  wrote column '%s' with shape %s" % (col_name, col_data.shape)
            except RuntimeError, err:
                print "  error writing column '%s' with shape %s (%s)" % (col_name, col_data.shape, err)
        else:
            if verbose:
                print "  column '%s' not in table" % (col_name,)
    # Flush table to disk
    t.flush()


def write_dict(ms_dict, ms_name='./blank.ms', verbose=True):
    # Iterate through subtables
    for sub_table_name, sub_dict in ms_dict.iteritems():
        # Allow parsing of single dict and array of dicts in the same fashion
        if isinstance(sub_dict, dict):
            sub_dict = [sub_dict]
        # Iterate through row groups that are separate dicts within the sub_dict array
        for row_dict in sub_dict:
            if verbose:
                print "Table %s:" % (sub_table_name,)
            # Open table using whichever casacore library was found
            t = open_table(ms_name, ack=verbose) if sub_table_name == 'MAIN' else \
                open_table(os.path.join(ms_name, sub_table_name))
            if verbose and t is not None:
                print "  opened successfully"
            if t is None:
                print "  could not open table!"
                break
            write_rows(t, row_dict, verbose)
            t.close()
            if verbose:
                print "  closed successfully"
