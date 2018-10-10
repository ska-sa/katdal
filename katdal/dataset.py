################################################################################
# Copyright (c) 2011-2018, National Research Foundation (Square Kilometre Array)
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

"""Base class for accessing a visibility data set."""
from __future__ import print_function, division, absolute_import

from builtins import zip
from past.builtins import basestring
from builtins import object
import time
import logging
import numbers
import threading

import numpy as np

import katpoint
from katpoint import is_iterable, rad2deg

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# -- CLASS :  Helper classes
# -------------------------------------------------------------------------------------------------


class WrongVersion(Exception):
    """Trying to access data using accessor class with the wrong version."""


class BrokenFile(Exception):
    """Data set could not be loaded because file is inconsistent or misses critical bits."""


class Subarray(object):
    """Subarray specification.

    A subarray is determined by the specific correlation products produced by the
    correlator and the antenna objects associated with the inputs found in the
    correlation products.

    Parameters
    ----------
    ants : sequence of :class:`katpoint.Antenna` objects
        List of antenna objects, culled to contain only antennas found in
        `corr_products`
    corr_products : sequence of (string, string) pairs, length *B*
        Correlation products as pairs of input labels, e.g. ('ant1h', 'ant2v'),
        exposed as an array of strings with shape (*B*, 2)

    Attributes
    ----------
    inputs : list of strings
        List of correlator input labels found in `corr_products`, e.g. 'ant1h'

    """

    def __init__(self, ants, corr_products):
        self.corr_products = np.array([(inpA.lower(), inpB.lower())
                                       for inpA, inpB in corr_products])
        # Extract all inputs (and associated antennas) from corr product list
        self.inputs = sorted(set(np.ravel(self.corr_products)))
        input_ants = set([inp[:-1] for inp in self.inputs])
        # Only keep antennas that are involved in correlation products
        self.ants = [ant for ant in ants if ant.name in input_ants]

    def __repr__(self):
        """Short human-friendly string representation of subarray object."""
        return "<katdal.Subarray antennas=%d inputs=%d corrprods=%d at 0x%x>" % \
               (len(self.ants), len(self.inputs), len(self.corr_products),
                id(self))

    @property
    def _description(self):
        """Complete string representation, used internally for comparisons."""
        ants = '\n'.join(ant.description for ant in self.ants)
        corrprods = ' '.join('%s,%s' % (inpA, inpB)
                             for inpA, inpB in self.corr_products)
        return '\n'.join((ants, corrprods))

    def __eq__(self, other):
        """Equality comparison operator."""
        return self._description == (other._description
                                     if isinstance(other, Subarray) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self._description < (other._description
                                    if isinstance(other, Subarray) else other)

    def __hash__(self):
        """Base hash on description string, just like equality operator."""
        return hash(self._description)


class SpectralWindow(object):
    """Spectral window specification.

    A spectral window is determined by the number of frequency channels produced
    by the correlator and their corresponding centre frequencies, as well as the
    channel width. The channels are assumed to be regularly spaced and to be the
    result of either lower-sideband downconversion (channel frequencies
    decreasing with channel index) or upper-sideband downconversion (frequencies
    increasing with index). For further information the receiver band and
    correlator product names are also available.

    .. warning::

        Instances should be treated as immutable. Changing the attributes will
        lead to inconsistencies between them.

    Parameters
    ----------
    centre_freq : float
        Centre frequency of spectral window, in Hz
    channel_width : float
        Bandwidth of each frequency channel, in Hz
    num_chans : int
        Number of frequency channels
    product : string, optional
        Name of data product / correlator mode
    sideband : {-1, +1}, optional
        Type of downconversion (-1 => lower sideband, +1 => upper sideband)
    band : {'L', 'UHF', 'S', 'X', 'Ku'}, optional
        Name of receiver / band
    bandwidth : float, optional
        The bandwidth of the whole spectral window, in Hz. If specified,
        `channel_width` is ignored and computed from the bandwidth. If not
        specified, bandwidth is computed from the channel width. Specifying
        this is a good idea if the channel width cannot be exactly represented
        in floating point.

    Attributes
    ----------
    channel_freqs : array of float, shape (*F*,)
        Centre frequency of each frequency channel (assuming LSB mixing), in Hz
    """

    def __init__(self, centre_freq, channel_width, num_chans, product=None,
                 sideband=-1, band='L', bandwidth=None):
        if bandwidth is None:
            bandwidth = channel_width * num_chans
        else:
            channel_width = bandwidth / num_chans
        self.centre_freq = centre_freq
        self.channel_width = channel_width
        self.bandwidth = bandwidth
        self.num_chans = num_chans
        self.product = product if product is not None else ''
        self.sideband = sideband
        self.band = band
        # channel_freqs is computed on demand
        self._channel_freqs_lock = threading.Lock()
        self._channel_freqs = None

    @property
    def channel_freqs(self):
        with self._channel_freqs_lock:
            if self._channel_freqs is None:
                # Don't subtract half a channel width as channel 0 is centred on 0 Hz in baseband
                # We use self.bandwidth and self.num_chans to avoid rounding
                # errors that might accumulate if channel_width is inexact.
                self._channel_freqs = self.centre_freq + self.sideband * self.bandwidth * (
                    np.arange(self.num_chans) - self.num_chans // 2) / self.num_chans
            return self._channel_freqs

    def __repr__(self):
        """Short human-friendly string representation of spectral window object."""
        return "<katdal.SpectralWindow %s-band product=%s centre=%.3f MHz " \
               "bandwidth=%.3f MHz channels=%d at 0x%x>" % \
               (self.band if self.band else 'unknown',
                repr(self.product) if self.product else 'unknown',
                self.centre_freq / 1e6,
                self.bandwidth / 1e6,
                self.num_chans, id(self))

    @property
    def _description(self):
        """Complete hashable representation, used internally for comparisons."""
        # Pick values that enable a sensible ordering of spectral windows
        # Using self.bandwidth is generally redundant but may play a role in
        # obscure rounding cases.
        return (self.centre_freq,
                -self.channel_width, self.num_chans, self.sideband,
                self.band, self.product, -self.bandwidth)

    def __eq__(self, other):
        """Equality comparison operator."""
        return self._description == (
            other._description if isinstance(other, SpectralWindow) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self._description < (
            other._description if isinstance(other, SpectralWindow) else other)

    def __hash__(self):
        """Base hash on description tuple, just like equality operator."""
        return hash(self._description)

    def subrange(self, first, last):
        """Get a new :class:`SpectralWindow` representing a subset of the channels.

        The returned :class:`SpectralWindow` covers the same frequencies as
        channels [first, last) of the original.

        Raises
        ------
        IndexError
            If [first, last) is not a (non-empty) subinterval of the channels
        """
        if not (0 <= first < last <= self.num_chans):
            raise IndexError('channel indices out of range')
        channel_shift = (first + last) // 2 - self.num_chans // 2
        num_chans = last - first
        # We use self.bandwidth and self.num_chans to avoid rounding errors
        # that might accumulate if channel_width is inexact.
        centre_freq = self.centre_freq \
            + channel_shift * self.bandwidth * self.sideband / self.num_chans
        return SpectralWindow(
            centre_freq, self.channel_width, num_chans,
            self.product, self.sideband, self.band,
            bandwidth=self.bandwidth * num_chans / self.num_chans)

    def rechannelise(self, num_chans):
        """Get a new :class:`SpectralWindow` with a different number of channels.

        The returned :class:`SpectralWindow` covers the same frequencies as the
        original, but dividing the bandwidth into a different number of
        channels.
        """
        if num_chans == self.num_chans:
            return self
        # Find the centre of the bandwidth (whereas centre_freq is the centre
        # of the middle channel)
        centre_freq = self.centre_freq
        if self.num_chans % 2 == 0:
            centre_freq -= self.sideband * 0.5 * self.channel_width
        channel_width = self.bandwidth / num_chans
        # Now convert to the centre of the new middle channel
        if num_chans % 2 == 0:
            centre_freq += self.sideband * 0.5 * channel_width
        return SpectralWindow(
            centre_freq, channel_width, num_chans,
            self.product, self.sideband, self.band,
            bandwidth=self.bandwidth)


def _robust_target(description):
    """Robust build of :class:`katpoint.Target` object from description string."""
    if not description:
        return katpoint.Target('Nothing, special')
    try:
        return katpoint.Target(description)
    except ValueError:
        logger.warning("Invalid target description '%s' - replaced with dummy target" % (description,))
        return katpoint.Target('Nothing, special')


DEFAULT_SENSOR_PROPS = {
    '*nd_coupler': {'categorical': True, 'greedy_values': (True,), 'initial_value': '0',
                    'transform': lambda x: x not in ('0', 'False', 0)},
    '*nd_pin': {'categorical': True, 'greedy_values': (True,), 'initial_value': '0',
                'transform': lambda x: x not in ('0', 'False', 0)},
    'Observation/label': {'initial_value': '', 'transform': str, 'allow_repeats': True},
    'Observation/scan_state': {'allow_repeats': True},
}

# -------------------------------------------------------------------------------------------------
# -- FUNCTION :  Virtual sensor calculations
# -------------------------------------------------------------------------------------------------


def _calc_mjd(cache, name):
    """Calculate Modified Julian Day (MJD) timestamps using sensor cache contents."""
    cache[name] = mjd = np.array([katpoint.Timestamp(t).to_mjd() for t in cache.timestamps[:]])
    return mjd


def _calc_lst(cache, name, ant):
    """Calculate local sidereal time (LST) timestamps using sensor cache contents."""
    antenna = cache.get('Antennas/%s/antenna' % (ant,))[0]
    cache[name] = lst = antenna.local_sidereal_time(cache.timestamps[:])
    return lst


def _calc_radec(cache, name, ant):
    """Calculate (ra, dec) pointing coordinates using sensor cache contents."""
    ant_group = 'Antennas/%s/' % (ant,)
    antenna = cache.get(ant_group + 'antenna')[0]
    az, el = cache.get(ant_group + 'az'), cache.get(ant_group + 'el')
    radec = np.array([katpoint.construct_azel_target(a, e).radec(t, antenna)
                      for t, a, e in zip(cache.timestamps[:], az, el)])
    cache[ant_group + 'ra'] = radec[:, 0]
    cache[ant_group + 'dec'] = radec[:, 1]
    return radec[:, 0] if name == ant_group + 'ra' else radec[:, 1]


def _calc_parangle(cache, name, ant):
    """Calculate parallactic angle using sensor cache contents."""
    ant_group = 'Antennas/%s/' % (ant,)
    antenna = cache.get(ant_group + 'antenna')[0]
    az, el = cache.get(ant_group + 'az'), cache.get(ant_group + 'el')
    parangle = np.array([katpoint.construct_azel_target(a, e).parallactic_angle(t, antenna)
                         for t, a, e in zip(cache.timestamps[:], az, el)])
    cache[name] = parangle
    return parangle


def _calc_target_coords(cache, name, ant, projection, coordsys):
    """Calculate target coordinates using sensor cache contents."""
    ant_group = 'Antennas/%s/' % (ant,)
    antenna = cache.get(ant_group + 'antenna')[0]
    lon = cache.get(ant_group + 'ra') if coordsys == 'radec' else cache.get(ant_group + 'az')
    lat = cache.get(ant_group + 'dec') if coordsys == 'radec' else cache.get(ant_group + 'el')
    # Fix over-the-top elevations (projections can only handle elevations in range +- 90 degrees)
    over_the_top = (lat > np.pi / 2.0) & (lat < np.pi)
    lon[over_the_top] += np.pi
    lat[over_the_top] = np.pi - lat[over_the_top]
    x, y = np.empty(len(cache.timestamps)), np.empty(len(cache.timestamps))
    targets = cache.get('Observation/target')
    for segm, target in targets.segments():
        x[segm], y[segm] = target.sphere_to_plane(lon[segm], lat[segm], cache.timestamps[segm],
                                                  antenna, projection, coordsys)
    cache[ant_group + 'target_x_%s_%s' % (projection, coordsys)] = x
    cache[ant_group + 'target_y_%s_%s' % (projection, coordsys)] = y
    return x if name.startswith(ant_group + 'target_x') else y


def _calc_uvw(cache, name, antA, antB):
    """Calculate (u,v,w) coordinates using sensor cache contents."""
    antA_group, antB_group = 'Antennas/%s/' % (antA,), 'Antennas/%s/' % (antB,)
    antennaA, antennaB = cache.get(antA_group + 'antenna')[0], cache.get(antB_group + 'antenna')[0]
    u, v, w = np.empty(len(cache.timestamps)), np.empty(len(cache.timestamps)), np.empty(len(cache.timestamps))
    targets = cache.get('Observation/target')
    for segm, target in targets.segments():
        u[segm], v[segm], w[segm] = target.uvw(antennaA, cache.timestamps[segm], antennaB)
    cache[antA_group + 'u_%s' % (antB,)] = u
    cache[antA_group + 'v_%s' % (antB,)] = v
    cache[antA_group + 'w_%s' % (antB,)] = w
    return u if name.startswith(antA_group + 'u') else v if name.startswith(antA_group + 'v') else w


DEFAULT_VIRTUAL_SENSORS = {
    'Timestamps/mjd': _calc_mjd, 'Antennas/{ant}/lst': _calc_lst,
    'Antennas/{ant}/ra': _calc_radec, 'Antennas/{ant}/dec': _calc_radec,
    'Antennas/{ant}/parangle': _calc_parangle,
    'Antennas/{ant}/target_[xy]_{projection}_{coordsys}': _calc_target_coords,
    'Antennas/{antA}/[uvw]_{antB}': _calc_uvw,
}

# -------------------------------------------------------------------------------------------------
# -- CLASS :  DataSet
# -------------------------------------------------------------------------------------------------


class DataSet(object):
    """Base class for accessing a visibility data set.

    This provides a simple interface to a generic file (or files) containing
    visibility data (both single-dish and interferometer data supported).
    The data are not loaded into memory on opening the file, but are accessible
    via properties after typically selecting a subset of the data. This allows
    the reading of huge files.

    Parameters
    ----------
    name : string
        Name / identifier of data set
    ref_ant : string, optional
        Name of reference antenna, used to partition data set into scans
        (default is first antenna in use by script)
    time_offset : float, optional
        Offset to add to all correlator timestamps, in seconds

    Attributes
    ----------
    version : string
        Format version string
    observer : string
        Name of person that recorded the data set
    description : string
        Short description of the purpose of the data set
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    obs_params : dict mapping string to string or list of strings
        Observation parameters, typically set in observation script
    obs_script_log : list of strings
        Observation script output log (useful for debugging)

    subarrays : list of :class:`SubArray` objects
        List of all subarrays in data set
    subarray : int
        Index of currently selected subarray
    ants : list of :class:`katpoint.Antenna` objects
        List of selected antennas
    inputs : array of strings
        List of selected correlator input labels ('ant1h')
    corr_products : array of strings, shape (*B*, 2)
        Array of selected correlation products as pairs of input labels
        (e.g. [('ant1h', 'ant1h'), ('ant1h', 'ant2h')])
    receivers : dict mapping string to string or list of strings
        Identifier of the active receiver on each antenna

    spectral_windows : list of :class:`SpectralWindow` objects
        List of all spectral windows in data set
    spw : int
        Index of currently selected spectral window
    channel_width : float
        Channel bandwidth of selected spectral window, in Hz
    freqs / channel_freqs : array of float, shape (*F*,)
        Centre frequency of each selected channel, in Hz
    channels : array of int, shape (*F*,)
        Original channel indices of selected channels

    dump_period : float
        Dump period, in seconds
    sensor : :class:`SensorCache` object
        Sensor cache
    catalogue : :class:`katpoint.Catalogue` object
        Catalogue of all targets / sources / fields in data set
    start_time : :class:`katpoint.Timestamp` object
        Timestamp of start of first sample in file, in UT seconds since Unix epoch
    end_time : :class:`katpoint.Timestamp` object
        Timestamp of end of last sample in file, in UT seconds since Unix epoch
    dumps : array of int, shape (*T*,)
        Original dump indices of selected dumps
    scan_indices : list of int
        List of currently selected scans as indices
    compscan_indices : list of int
        List of currently selected compound scans as indices
    target_indices : list of int
        List of currently selected targets as indices into catalogue
    target_projection : {'ARC', 'SIN', 'TAN', 'STG', 'CAR'}, optional
        Type of spherical projection for target coordinates
    target_coordsys : {'azel', 'radec'}, optional
        Spherical pointing coordinate system for target coordinates
    shape : tuple of 3 ints
        Shape of selected visibility data array, as (*T*, *F*, *B*)
    size : int
        Size of selected visibility data array, in bytes

    """

    def __init__(self, name, ref_ant='', time_offset=0.0):
        self.name = name
        self.ref_ant = ref_ant
        self.time_offset = time_offset
        self.version = ''
        self.observer = ''
        self.description = ''
        self.experiment_id = ''
        self.obs_params = {}
        self.obs_script_log = []

        self.subarrays = []
        self.subarray = -1
        self.ants = []
        self.inputs = []
        self.corr_products = np.empty(shape=(0, 2), dtype='|S5')
        self.receivers = {}

        self.spectral_windows = []
        self.spw = -1
        self.channel_width = 0.0
        self.freqs = self.channel_freqs = np.empty(0)
        self.channels = np.empty(0, dtype=np.int)

        self.dump_period = 0.0
        self.sensor = {}
        self.catalogue = katpoint.Catalogue()
        self.start_time = katpoint.Timestamp(0.0)
        self.end_time = katpoint.Timestamp(0.0)
        self.dumps = np.empty(0, dtype=np.int)
        self.scan_indices = []
        self.compscan_indices = []
        self.target_indices = []
        self.target_projection = 'ARC'
        self.target_coordsys = 'azel'
        self.shape = (0, 0, 0)
        self.size = 0

        self._selection = {}
        self._time_keep = []
        self._freq_keep = []
        self._corrprod_keep = []
        self._weights_keep = 'all'
        self._flags_keep = 'all'

    def __repr__(self):
        """Short human-friendly string representation of data set object."""
        return "<katdal.%s '%s' shape %s at 0x%x>" % (self.__class__.__name__, self.name, self.shape, id(self))

    def __str__(self):
        """Verbose human-friendly string representation of data set."""
        # Start with static file information
        descr = ['===============================================================================',
                 'Name: %s (version %s)' % (self.name, self.version),
                 '===============================================================================',
                 'Observer: %s  Experiment ID: %s' % (self.observer if self.observer else '-',
                                                      self.experiment_id if self.experiment_id else '-'),
                 "Description: '%s'" % (self.description if self.description else 'No description',),
                 'Observed from %s to %s' % (self.start_time.local(), self.end_time.local()),
                 'Dump rate / period: %.5f Hz / %.3f s' % (1 / self.dump_period, self.dump_period),
                 'Subarrays: %d' % (len(self.subarrays),),
                 '  ID  Antennas                            Inputs  Corrprods']
        for n, sub in enumerate(self.subarrays):
            ant_names = ','.join([ant.name for ant in sub.ants])
            descr.append('  %2d  %28s  %2d      %3d' %
                         (n, ant_names.ljust(7 * 4 + 6), len(sub.inputs), len(sub.corr_products)))
        descr += ['Spectral Windows: %d' % (len(self.spectral_windows),),
                  '  ID Band Product  CentreFreq(MHz)  Bandwidth(MHz)  Channels  ChannelWidth(kHz)']
        for n, spw in enumerate(self.spectral_windows):
            descr.append('  %2d %-4s %-9s %9.3f        %8.3f          %5d     %9.3f' %
                         (n, spw.band if spw.band else '-', spw.product if spw.product else '-',
                          spw.centre_freq / 1e6,
                          spw.channel_width / 1e6 * spw.num_chans,
                          spw.num_chans, spw.channel_width / 1e3))
        # Now add dynamic information, which depends on the current selection criteria
        descr += ['-------------------------------------------------------------------------------',
                  'Data selected according to the following criteria:']
        for k, v in sorted(self._selection.items()):
            descr.append('  %s=%s' % (k, ("'%s'" % (v,)) if isinstance(v, basestring) else v))
        descr.append('-------------------------------------------------------------------------------')
        descr.append('Shape: (%d dumps, %d channels, %d correlation products) => Size: %s' %
                     tuple(list(self.shape) + ['%.3f %s' % ((self.size / 1e9, 'GB') if self.size > 1e9 else
                                                            (self.size / 1e6, 'MB') if self.size > 1e6 else
                                                            (self.size / 1e3, 'KB'))]))
        autocorrs = np.array([(inpA[:-1] == inpB[:-1]) for inpA, inpB in self.corr_products])
        crosscorrs = np.array([(inpA[:-1] != inpB[:-1]) for inpA, inpB in self.corr_products])
        descr.append('Antennas: %s  Inputs: %d  Autocorr: %s  Crosscorr: %s' %
                     (','.join([('*' + ant.name if ant.name == self.ref_ant else ant.name) for ant in self.ants]),
                      len(self.inputs), 'yes' if np.any(autocorrs) else 'no', 'yes' if np.any(crosscorrs) else 'no'))
        chan_min, chan_max = self.channels.argmin(), self.channels.argmax()
        descr.append('Channels: %d (index %d - %d, %8.3f MHz - %8.3f MHz), each %7.3f kHz wide' %
                     (len(self.channels), self.channels[chan_min], self.channels[chan_max],
                      self.freqs[chan_min] / 1e6, self.freqs[chan_max] / 1e6, self.channel_width / 1e3))
        # Discover maximum name and tag string lengths for targets beforehand
        name_len, tag_len = 4, 4
        for n in self.target_indices:
            target = self.catalogue.targets[n]
            name_len = max(len(target.name), name_len)
            tag_len = max(len(' '.join(target.tags[1:] if target.body_type != 'xephem' else target.tags[2:])), tag_len)
        descr += ['Targets: %d selected out of %d in catalogue' % (len(self.target_indices), len(self.catalogue)),
                  '  ID  %s  Type      RA(J2000)     DEC(J2000)  %s  Dumps  ModelFlux(Jy)' %
                  ('Name'.ljust(name_len), 'Tags'.ljust(tag_len))]
        for n in self.target_indices:
            target = self.catalogue.targets[n]
            target_type = target.body_type if target.body_type != 'xephem' else target.tags[1]
            tags = ' '.join(target.tags[1:] if target.body_type != 'xephem' else target.tags[2:])
            ra, dec = target.radec() if target_type == 'radec' else ('-', '-')
            # Calculate average target flux over selected frequency band
            flux_spectrum = target.flux_density(self.freqs / 1e6)
            flux_valid = ~np.isnan(flux_spectrum)
            flux = ('%9.2f' % (flux_spectrum[flux_valid].mean(),)) if np.any(flux_valid) else ''
            target_dumps = ((self.sensor.get('Observation/target_index') == n) & self._time_keep).sum()
            descr.append('  %2d  %s  %s  %11s  %11s  %s  %5d  %s' %
                         (n, target.name.ljust(name_len), target_type.ljust(8),
                          ra, dec, tags.ljust(tag_len), target_dumps, flux))
        scans = self.sensor.get('Observation/scan_index')
        compscans = self.sensor.get('Observation/compscan_index')
        total_scans, total_compscans = len(scans.unique_values), len(compscans.unique_values)
        descr += ['Scans: %d selected out of %d total       Compscans: %d selected out of %d total' %
                  (len(self.scan_indices), total_scans, len(self.compscan_indices), total_compscans),
                  '  Date        Timerange(UTC)       ScanState  CompScanLabel  Dumps  Target']
        current_date = ''
        for scan, state, target in self.scans():
            start, end = time.gmtime(self.timestamps[0]), time.gmtime(self.timestamps[-1])
            start_date, start_time = time.strftime('%d-%b-%Y/', start), time.strftime('%H:%M:%S', start)
            end_date, end_time = time.strftime('%d-%b-%Y/', end), time.strftime('%H:%M:%S', end)
            timerange = (start_date if start_date != current_date else '') + start_time
            current_date = start_date
            timerange += ' - ' + (end_date if end_date != current_date else '') + end_time
            compscan, label = self.sensor['Observation/compscan_index'][0], self.sensor['Observation/label'][0]
            descr.append('  %31s  %3d:%5s  %3d:%9s  %5d  %3d:%s' %
                         (timerange, scan, state.ljust(5), compscan, label.ljust(9),
                          self.shape[0], self.target_indices[0], target.name))
        return '\n'.join(descr)

    def _fix_flux_freq_range(self):
        """Ensure that target flux density models are valid on data frequencies."""
        # Pad the minimum and maximum channel freqs with a few extra channels
        min_freq = min((spw.channel_freqs.min() - 2 * spw.channel_width) / 1e6
                       for spw in self.spectral_windows)
        max_freq = max((spw.channel_freqs.max() + 2 * spw.channel_width) / 1e6
                       for spw in self.spectral_windows)
        for target in self.catalogue:
            model = target.flux_model
            if not model:
                continue
            if min_freq < model.min_freq_MHz and \
               min_freq > 0.6 * model.min_freq_MHz or \
               max_freq > model.max_freq_MHz and \
               max_freq < 1.6 * model.max_freq_MHz:
                new_min_freq = min(min_freq, model.min_freq_MHz)
                new_max_freq = max(max_freq, model.max_freq_MHz)
                logger.warning('Extending flux density model frequency range '
                               'of %r from %d-%d MHz to %d-%d MHz', target.name,
                               model.min_freq_MHz, model.max_freq_MHz,
                               new_min_freq, new_max_freq)
                model.min_freq_MHz = new_min_freq
                model.max_freq_MHz = new_max_freq

    def _set_keep(self, time_keep=None, freq_keep=None, corrprod_keep=None,
                  weights_keep=None, flags_keep=None):
        """Set time, frequency and/or correlation product selection masks.

        Set the selection masks for those parameters that are present. Also
        include weights and flags selections as options.

        Parameters
        ----------
        time_keep : array of bool, shape (*T*,), optional
            Boolean selection mask with one entry per timestamp
        freq_keep : array of bool, shape (*F*,), optional
            Boolean selection mask with one entry per frequency channel
        corrprod_keep : array of bool, shape (*B*,), optional
            Boolean selection mask with one entry per correlation product
        weights_keep : 'all' or string or sequence of strings, optional
            Names of selected weight types (or 'all' for the lot)
        flags_keep : 'all' or string or sequence of strings, optional
            Names of selected flag types (or 'all' for the lot)

        """
        if time_keep is not None:
            self._time_keep = time_keep
            # Ensure that sensor cache gets updated time selection
            if self.sensor:
                self.sensor._set_keep(self._time_keep)
        if freq_keep is not None:
            self._freq_keep = freq_keep
        if corrprod_keep is not None:
            self._corrprod_keep = corrprod_keep
        if weights_keep is not None:
            self._weights_keep = weights_keep
        if flags_keep is not None:
            self._flags_keep = flags_keep

    def select(self, **kwargs):
        """Select subset of data, based on time / frequency / corrprod filters.

        This applies a set of selection criteria to the data set, which updates
        the data set properties and attributes to match the selection. In other
        words, the :meth:`timestamps` and :meth:`vis` methods will return the
        selected subset of the data, while attributes such as :attr:`ants`,
        :attr:`channel_freqs` and :attr:`shape` are updated. The sensor cache
        will also return the selected subset of sensor data via the __getitem__
        interface. This function returns nothing, but modifies the existing
        data set in-place.

        The selection criteria are divided into groups, based on whether they
        affect the time, frequency or correlation product dimension::

        * Time: `dumps`, `timerange`, `scans`, `compscans`, `targets`
        * Frequency: `channels`, `freqrange`
        * Correlation product: `corrprods`, `ants`, `inputs`, `pol`

        The `subarray` and `spw` criteria are special, as they affect multiple
        dimensions (time + correlation product and time + frequency,
        respectively), are always active and are forced to be a single index.

        If there are multiple criteria on the same dimension within a select()
        call, they are ANDed together, while multiple items within the same
        criterion (e.g. `targets=['Hyd A', 'Vir A']`) are ORed together. When a
        second select() call is done, all new selections replace previous
        selections on the same dimension, while existing selections on other
        dimensions are preserved. The `reset` parameter finetunes this behaviour.

        If :meth:`select` is called without any parameters the selection is
        reset to the original data set.

        In addition, the `weights` and `flags` criteria are lists of names that
        select which weights and flags to include in the corresponding data set
        property.

        Parameters
        ----------
        strict : {True, False}, optional
            True if select() raises TypeError if it encounters an unknown kwarg

        dumps : int or slice or sequence of ints or sequence of bools, optional
            Select dumps by index, slice or boolean mask of length *T*
            (keep dumps where mask is True)
        timerange : sequence of 2 :class:`katpoint.Timestamp` objects
                    or equivalent, optional
            Select range of times between given start and end times
        scans : int or string or sequence, optional
            Select scans by index or state (or negate state by prepending '~')
        compscans : int or string or sequence, optional
            Select compscans by index or label (or negate label by prepending '~')
        targets : int or string or :class:`katpoint.Target` object or sequence,
                  optional
            Select targets by index or name or description or object

        spw : int, optional
            Select spectral window by index (only one may be active)
        channels : int or slice or sequence of ints or sequence of bools, optional
            Select frequency channels by index, slice or boolean mask of length
            *F* (keep channels where mask is True)
        freqrange : sequence of 2 floats, optional
            Select range of frequencies between start and end frequencies, in Hz

        subarray : int, optional
            Select subarray by index (only one may be active)
        corrprods : int or slice or sequence of ints or sequence of bools or
                    sequence of string pairs or {'auto', 'cross'}, optional
            Select correlation products by index, slice or boolean mask of length
            *B* (keep products where mask is True). Alternatively, select by
            value via a sequence of string pairs, or select all autocorrelations
            via 'auto' or all cross-correlations via 'cross'.
        ants : string or :class:`katpoint.Antenna` object or sequence, optional
            Select antennas by name or object
        inputs : string or sequence of strings, optional
            Select inputs by label
        pol : string or sequence of strings
              {'H', 'V', 'HH', 'VV', 'HV', 'VH'}, optional
            Select polarisation terms

        weights : 'all' or string or sequence of strings, optional
            List of names of weights to be multiplied together, as a sequence
            or string of comma-separated names (combine all weights by default)
        flags : 'all' or string or sequence of strings, optional
            List of names of flags that will be OR'ed together, as a sequence
            or string of comma-separated names (use all flags by default). An
            empty string or sequence discards all flags.

        reset : {'auto', '', 'T', 'F', 'B', 'TF', 'TB', 'FB', 'TFB'}, optional
            Remove existing selections on specified dimensions before applying
            the new selections. The default 'auto' option clears those dimensions
            that will be modified by the new selections and leaves the selections
            on unaffected dimensions intact except if `select` is called without
            any parameters, in which case all selections are cleared. By setting
            `reset` to '', new selections apply on top of existing selections.

        Raises
        ------
        TypeError
            If a keyword argument is unknown and `strict` is enabled
        IndexError
            If `spw` or `subarray` is out of range

        """
        time_selectors = ['dumps', 'timerange', 'scans', 'compscans', 'targets']
        freq_selectors = ['channels', 'freqrange']
        corrprod_selectors = ['corrprods', 'ants', 'inputs', 'pol']
        # Check if keywords are valid and raise exception only if this is explicitly enabled
        valid_kwargs = time_selectors + freq_selectors + corrprod_selectors + \
            ['spw', 'subarray', 'weights', 'flags', 'reset', 'strict']
        # Check for definition of strict
        strict = kwargs.get('strict', True)
        if strict and set(kwargs.keys()) - set(valid_kwargs):
            raise TypeError("select() got unexpected keyword argument(s) %s, valid ones are %s" %
                            (list(set(kwargs.keys()) - set(valid_kwargs)), valid_kwargs))
        # If select() is called without arguments, reset all selections
        reset = 'TFB' if not kwargs else kwargs.pop('reset', 'auto')
        kwargs['spw'] = spw = kwargs.get('spw', self.spw)
        if spw >= len(self.spectral_windows):
            raise IndexError('Data set has %d spectral window(s): spw should be in range 0..%d, is %d instead' %
                             (len(self.spectral_windows), len(self.spectral_windows) - 1, spw))
        kwargs['subarray'] = subarray = kwargs.get('subarray', self.subarray)
        if subarray >= len(self.subarrays):
            raise IndexError('Data set has %d subarray(s): subarray should be in range 0..%d, is %d instead' %
                             (len(self.subarrays), len(self.subarrays) - 1, subarray))
        # In 'auto' mode, only reset flags for those dimensions that will be affected by selectors
        if reset == 'auto':
            reset = 'T' if set(kwargs.keys()).intersection(time_selectors) else ''
            reset += 'F' if set(kwargs.keys()).intersection(freq_selectors) else ''
            reset += 'B' if set(kwargs.keys()).intersection(corrprod_selectors) else ''
        # Change spectral window and/or subarray
        if spw != self.spw:
            reset += 'TF'
            self.spw = spw
        if subarray != self.subarray:
            reset += 'TB'
            self.subarray = subarray
        # Reset the selection flags on the appropriate dimensions
        if 'T' in reset:
            self._time_keep[:] = True
            self._time_keep &= (self.sensor.get('Observation/spw_index') == spw)
            self._time_keep &= (self.sensor.get('Observation/subarray_index') == subarray)
            for key in time_selectors:
                self._selection.pop(key, None)
        # Since the number of freqs / corrprods may change due to spw / subarray, create these flags afresh
        if 'F' in reset:
            self._freq_keep = np.ones(self.spectral_windows[self.spw].num_chans, dtype=np.bool)
            for key in freq_selectors:
                self._selection.pop(key, None)
        if 'B' in reset:
            self._corrprod_keep = np.ones(len(self.subarrays[self.subarray].corr_products), dtype=np.bool)
            for key in corrprod_selectors:
                self._selection.pop(key, None)
        # Now add the new selection criteria to the list (after the existing ones were kept or culled)
        self._selection.update(kwargs)

        for k, v in self._selection.items():
            # Selections that affect time axis
            if k == 'dumps':
                if np.asarray(v).dtype == np.bool:
                    self._time_keep &= v
                else:
                    dump_keep = np.zeros(len(self._time_keep), dtype=np.bool)
                    dump_keep[v] = True
                    self._time_keep &= dump_keep
            elif k == 'timerange':
                start_time = katpoint.Timestamp(v[0]).secs + 0.5 * self.dump_period
                end_time = katpoint.Timestamp(v[1]).secs - 0.5 * self.dump_period
                self._time_keep &= (self.sensor.timestamps[:] >= start_time)
                self._time_keep &= (self.sensor.timestamps[:] <= end_time)
            elif k in ('scans', 'compscans'):
                scans = v if is_iterable(v) else [l.strip() for l in v.split(',')] if isinstance(v, basestring) else [v]
                scan_keep = np.zeros(len(self._time_keep), dtype=np.bool)
                scan_sensor = self.sensor.get('Observation/scan_state' if k == 'scans' else 'Observation/label')
                scan_index_sensor = self.sensor.get('Observation/%s_index' % (k[:-1],))
                for scan in scans:
                    if isinstance(scan, numbers.Integral):
                        scan_keep |= (scan_index_sensor == scan)
                    elif scan[0] == '~':
                        scan_keep |= ~(scan_sensor == scan[1:])
                    else:
                        scan_keep |= (scan_sensor == scan)
                self._time_keep &= scan_keep
            elif k == 'targets':
                targets = v if is_iterable(v) else [v]
                target_keep = np.zeros(len(self._time_keep), dtype=np.bool)
                target_index_sensor = self.sensor.get('Observation/target_index')
                for t in targets:
                    if isinstance(t, numbers.Integral):
                        target_index = t
                    elif t not in self.catalogue:
                        # Warn here, in case the user gets the target subtly wrong and wonders why it is not selected
                        logger.warning("Skipping unknown selected target '%s'" % (t,))
                        continue
                    elif isinstance(t, katpoint.Target) or isinstance(t, basestring) and ',' in t:
                        target_index = self.catalogue.targets.index(t)
                    else:
                        target_index = self.catalogue.targets.index(self.catalogue[t])
                    target_keep |= (target_index_sensor == target_index)
                self._time_keep &= target_keep
            # Selections that affect frequency axis
            elif k == 'channels':
                if np.asarray(v).dtype == np.bool:
                    self._freq_keep &= v
                else:
                    chan_keep = np.zeros(len(self._freq_keep), dtype=np.bool)
                    chan_keep[v] = True
                    self._freq_keep &= chan_keep
            elif k == 'freqrange':
                start_freq = v[0] + 0.5 * self.spectral_windows[self.spw].channel_width
                end_freq = v[1] - 0.5 * self.spectral_windows[self.spw].channel_width
                self._freq_keep &= (self.spectral_windows[self.spw].channel_freqs >= start_freq)
                self._freq_keep &= (self.spectral_windows[self.spw].channel_freqs <= end_freq)
            # Selections that affect corrprod axis
            elif k == 'corrprods':
                if v == 'auto':
                    self._corrprod_keep &= [(inpA[:-1] == inpB[:-1])
                                            for inpA, inpB in self.subarrays[self.subarray].corr_products]
                elif v == 'cross':
                    self._corrprod_keep &= [(inpA[:-1] != inpB[:-1])
                                            for inpA, inpB in self.subarrays[self.subarray].corr_products]
                else:
                    v = np.asarray(v)
                    if v.ndim == 2 and v.shape[1] == 2:
                        all_corrprods = self.subarrays[self.subarray].corr_products
                        v = v.tolist()
                        v = np.array([list(cp) in v for cp in all_corrprods])
                    if np.asarray(v).dtype == np.bool:
                        self._corrprod_keep &= v
                    else:
                        cp_keep = np.zeros(len(self._corrprod_keep), dtype=np.bool)
                        cp_keep[v] = True
                        self._corrprod_keep &= cp_keep
            elif k == 'ants':
                ants = [a.strip() for a in v.split(',')] if isinstance(v, basestring) else v if is_iterable(v) else [v]
                ant_names = [(ant.name if isinstance(ant, katpoint.Antenna) else ant) for ant in ants]
                self._corrprod_keep &= [(inpA[:-1] in ant_names and inpB[:-1] in ant_names)
                                        for inpA, inpB in self.subarrays[self.subarray].corr_products]
            elif k == 'inputs':
                inps = [i.strip() for i in v.split(',')] if isinstance(v, basestring) else v if is_iterable(v) else [v]
                self._corrprod_keep &= [(inpA in inps and inpB in inps)
                                        for inpA, inpB in self.subarrays[self.subarray].corr_products]
            elif k == 'pol':
                pols = [i.strip() for i in v.split(',')] if isinstance(v, basestring) else v if is_iterable(v) else [v]
                # Lower case and strip out empty strings
                pols = [i.lower() for i in pols if i]

                # Proceed if we have a selection
                if len(pols) > 0:
                    # If given a selection assume we keep nothing
                    keep = np.zeros(self._corrprod_keep.shape, dtype=np.bool)

                    # or separate polarisation selections together
                    for polAB in pols:
                        polAB = polAB * 2 if polAB in ('h', 'v') else polAB
                        keep |= [(inpA[-1] == polAB[0] and inpB[-1] == polAB[1])
                                               for inpA, inpB in self.subarrays[self.subarray].corr_products]

                    # and into final corrprod selection
                    self._corrprod_keep &= keep

            # Selections that affect weights and flags
            elif k == 'weights':
                self._weights_keep = v
            elif k == 'flags':
                self._flags_keep = v

        # Ensure that updated selections make their way to sensor cache and potentially underlying datasets
        self._set_keep(self._time_keep, self._freq_keep, self._corrprod_keep, self._weights_keep, self._flags_keep)
        # Update the relevant data members based on selection made
        # These would all be more efficient as properties, but at the expense of extra lines of code...
        self.shape = (self._time_keep.sum(), self._freq_keep.sum(), self._corrprod_keep.sum())
        self.size = np.prod(self.shape, dtype=np.int64) * np.dtype('complex64').itemsize
        if not self.size:
            logger.warning('The selection criteria resulted in an empty data set')
        # This is quicker than indexing np.arange()
        self.dumps = self._time_keep.nonzero()[0]
        self.channels = self._freq_keep.nonzero()[0]
        self.freqs = self.channel_freqs = self.spectral_windows[self.spw].channel_freqs[self._freq_keep]
        self.channel_width = self.spectral_windows[self.spw].channel_width
        self.corr_products = self.subarrays[self.subarray].corr_products[self._corrprod_keep]
        self.inputs = sorted(set(np.ravel(self.corr_products)))
        input_ants = set([inp[:-1] for inp in self.inputs])
        self.ants = [ant for ant in self.subarrays[self.subarray].ants if ant.name in input_ants]
        # Figure out which scans, compscans and targets are included in selection
        self.scan_indices = sorted(set(self.sensor['Observation/scan_index']))
        self.compscan_indices = sorted(set(self.sensor['Observation/compscan_index']))
        self.target_indices = sorted(set(self.sensor['Observation/target_index']))

    def scans(self):
        """Generator that iterates through scans in data set.

        This iterates through the currently selected list of scans, returning
        the scan index, scan state and associated target object. In addition,
        after each iteration the data set will reflect the scan selection, i.e.
        the timestamps, visibilities, sensor values, etc. will be those of the
        current scan. The scan selection applies on top of any existing
        selection.

        Yields
        ------
        scan : int
            Scan index
        state : string
            Scan state
        target : :class:`katpoint.Target` object
            Target associated with scan

        """
        scans = self.scan_indices[:]
        # This is the active selection onto which scan selection will be added
        preselection = dict(self._selection.items())
        # This will ensure that the original selection is properly restored
        preselection['reset'] = 'T'
        old_timekeep = self._time_keep.copy()
        state_data = self.sensor.get('Observation/scan_state')
        for scan in scans:
            # Add scan selection on top of existing selection
            self.select(scans=scan, reset='')
            state = state_data.unique_values[state_data.indices[scan]]
            target = self.catalogue.targets[self.target_indices[0]]
            yield scan, state, target
            # A quick way to reset the time selection to the original one
            self._set_keep(old_timekeep.copy())
            self._selection.pop('scans', None)
        # Restore original selection more thoroughly
        self.select(**preselection)

    def compscans(self):
        """Generator that iterates through compound scans in data set.

        This iterates through the currently selected list of compound scans,
        returning the compound scan index, label and the first associated target
        object. In addition, after each iteration the data set will reflect the
        compound scan selection, i.e. the timestamps, visibilities, sensor
        values, etc. will be those of the current compound scan. The compound
        scan selection applies on top of any existing selection.

        Yields
        ------
        compscan : int
            Compound scan index
        label : string
            Compound scan label
        target : :class:`katpoint.Target` object
            First target associated with compound scan

        """
        compscans = self.compscan_indices[:]
        # This is the active selection onto which compscan selection will be added
        preselection = dict(list(self._selection.items()))
        # This will ensure that the original selection is properly restored
        preselection['reset'] = 'T'
        old_timekeep = self._time_keep.copy()
        for compscan in compscans:
            # Add scan selection on top of existing selection
            self.select(compscans=compscan, reset='')
            label_data = self.sensor.get('Observation/label')
            label = label_data.unique_values[label_data.indices[compscan]]
            target = self.catalogue.targets[self.target_indices[0]]
            yield compscan, label, target
            # A quick way to reset the time selection to the original one
            self._set_keep(old_timekeep.copy())
            self._selection.pop('compscans', None)
        # Restore original selection more thoroughly
        self.select(**preselection)

    # - - - - - - - - - - - - - - Format-specific properties - - - - - - - - - - - - - - - - - -

    @property
    def timestamps(self):
        """Visibility timestamps in UTC seconds since Unix epoch.

        The timestamps are returned as an array of float64, shape (*T*,), with
        one timestamp per integration aligned with the integration *midpoint*.

        """
        raise NotImplementedError

    @property
    def vis(self):
        r"""Complex visibility data as a function of time, frequency and corrprod.

        The visibility data are returned as an array of complex64, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The array always has all three dimensions, even for
        scalar (single) values. The number of integrations *T* matches the
        length of :meth:`timestamps`, the number of frequency channels *F*
        matches the length of :meth:`freqs` and the number of correlation
        products *B* matches the length of :meth:`corr_products`.

        The sign convention of the imaginary part is consistent with an
        electric field of :math:`e^{i(\omega t - jz)}` i.e. phase that
        increases with time.
        """
        raise NotImplementedError

    @property
    def weights(self):
        """Visibility weights as a function of time, frequency and baseline.

        The weights data are returned as an array indexer of float32, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`.

        """
        raise NotImplementedError

    @property
    def flags(self):
        """Visibility flags as a function of time, frequency and baseline.

        The flags data are returned as an array indexer of bool, shape
        (*T*, *F*, *B*), with time along the first dimension, frequency along the
        second dimension and correlation product ("baseline") index along the
        third dimension. The number of integrations *T* matches the length of
        :meth:`timestamps`, the number of frequency channels *F* matches the
        length of :meth:`freqs` and the number of correlation products *B*
        matches the length of :meth:`corr_products`.

        """
        raise NotImplementedError

    @property
    def temperature(self):
        """Air temperature in degrees Celsius."""
        raise NotImplementedError

    @property
    def pressure(self):
        """Barometric pressure in millibars."""
        raise NotImplementedError

    @property
    def humidity(self):
        """Relative humidity as a percentage."""
        raise NotImplementedError

    @property
    def wind_speed(self):
        """Wind speed in metres per second."""
        raise NotImplementedError

    @property
    def wind_direction(self):
        """Wind direction as an azimuth angle in degrees."""
        raise NotImplementedError

    # - - - - - - - - - - - - Virtual sensors exposed as properties - - - - - - - - - - - - - - -

    @property
    def mjd(self):
        """Visibility timestamps in Modified Julian Days (MJD).

        The timestamps are returned as an array of float64, shape (*T*,), with
        one timestamp per integration aligned with the integration *midpoint*.

        """
        return self.sensor['Timestamps/mjd']

    @property
    def lst(self):
        """Local sidereal time at the reference antenna in hours.

        The sidereal times are returned in an array of float, shape (*T*, *A*).

        """
        return self.sensor['Antennas/%s/lst' % self.ref_ant] * (12 / np.pi)

    def _sensor_per_ant(self, base_name):
        """Extract a single sensor per antenna and safely stack the results."""
        def sensor_data(ant_name):
            return self.sensor['Antennas/%s/%s' % (ant_name, base_name)]
        return np.column_stack([sensor_data(ant.name) for ant in self.ants]) \
            if self.ants else np.zeros((self.shape[0], 0))

    def _sensor_per_corrprod(self, base_name):
        """Extract a single sensor per corrprod and safely stack the results."""
        def sensor_data(antA, antB):
            return self.sensor['Antennas/%s/%s_%s' % (antA, base_name, antB)]
        return np.column_stack([sensor_data(inpA[:-1], inpB[:-1])
                                for inpA, inpB in self.corr_products]) \
            if len(self.corr_products) else np.zeros((self.shape[0], 0))

    @property
    def az(self):
        """Azimuth angle of each dish in degrees.

        The azimuth angles are returned in an array of float, shape (*T*, *A*).

        """
        return rad2deg(self._sensor_per_ant('az'))

    @property
    def el(self):
        """Elevation angle of each dish in degrees.

        The elevation angles are returned in an array of float, shape (*T*, *A*).

        """
        return rad2deg(self._sensor_per_ant('el'))

    @property
    def ra(self):
        """Right ascension of the actual pointing of each dish in J2000 degrees.

        The right ascensions are returned in an array of float, shape (*T*, *A*).

        """
        return rad2deg(self._sensor_per_ant('ra'))

    @property
    def dec(self):
        """Declination of the actual pointing of each dish in J2000 degrees.

        The declinations are returned in an array of float, shape (*T*, *A*).

        """
        return rad2deg(self._sensor_per_ant('dec'))

    @property
    def parangle(self):
        """Parallactic angle of the actual pointing of each dish in degrees.

        The parallactic angle is the position angle of the observer's vertical
        on the sky, measured from north toward east. This is the angle between
        the great-circle arc connecting the celestial North pole to the dish
        pointing direction, and the great-circle arc connecting the zenith above
        the antenna to the pointing direction, or the angle between the
        *hour circle* and *vertical circle* through the pointing direction.
        It is returned as an array of float, shape (*T*, *A*).

        """
        return rad2deg(self._sensor_per_ant('parangle'))

    @property
    def target_x(self):
        """Target *x* coordinate of each dish in degrees.

        The target coordinates are projections of the spherical coordinates of
        the dish pointing direction to a plane with the target position at the
        origin. The type of projection (e.g. ARC, SIN, etc.) and spherical
        pointing coordinate system (e.g. azel or radec) can be set via the
        :attr:`target_projection` and :attr:`target_coordsys` attributes,
        respectively. The target *x* coordinates are returned as an array of
        float, shape (*T*, *A*).

        """
        name = 'target_x_%s_%s' % (self.target_projection, self.target_coordsys)
        return rad2deg(self._sensor_per_ant(name))

    @property
    def target_y(self):
        """Target *y* coordinate of each dish in degrees.

        The target coordinates are projections of the spherical coordinates of
        the dish pointing direction to a plane with the target position at the
        origin. The type of projection (e.g. ARC, SIN, etc.) and spherical
        pointing coordinate system (e.g. azel or radec) can be set via the
        :attr:`target_projection` and :attr:`target_coordsys` attributes,
        respectively. The target *y* coordinates are returned as an array of
        float, shape (*T*, *A*).

        """
        name = 'target_y_%s_%s' % (self.target_projection, self.target_coordsys)
        return rad2deg(self._sensor_per_ant(name))

    @property
    def u(self):
        """U coordinate for each correlation product in metres.

        This calculates the *u* coordinate of the baseline vector of each
        correlation product as a function of time while tracking the target.
        It is returned as an array of float, shape (*T*, *B*). The sign
        convention is :math:`u_1 - u_2` for baseline (ant1, ant2).

        """
        return self._sensor_per_corrprod('u')

    @property
    def v(self):
        """V coordinate for each correlation product in metres.

        This calculates the *v* coordinate of the baseline vector of each
        correlation product as a function of time while tracking the target.
        It is returned as an array of float, shape (*T*, *B*). The sign
        convention is :math:`v_1 - v_2` for baseline (ant1, ant2).

        """
        return self._sensor_per_corrprod('v')

    @property
    def w(self):
        """W coordinate for each correlation product in metres.

        This calculates the *w* coordinate of the baseline vector of each
        correlation product as a function of time while tracking the target.
        It is returned as an array of float, shape (*T*, *B*).The sign
        convention is :math:`w_1 - w_2` for baseline (ant1, ant2).

        """
        return self._sensor_per_corrprod('w')
