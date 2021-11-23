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

import threading

import numpy as np


class SpectralWindow:
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
        band = self.band if self.band else 'unknown',
        product = repr(self.product) if self.product else 'unknown'
        return (f"<katdal.SpectralWindow {band}-band product={product} "
                f"centre={self.centre_freq/1e6:.3f} MHz bandwidth={self.bandwidth/1e6:.3f} MHz "
                f"channels={self.num_chans} at {id(self):#x}>")

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
