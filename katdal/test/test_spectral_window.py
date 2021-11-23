###############################################################################
# Copyright (c) 2018-2021, National Research Foundation (SARAO)
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
###############################################################################

"""Tests for :py:mod:`katdal.spectral_window`."""

import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

from katdal.spectral_window import SpectralWindow


class TestSpectralWindow:
    def setUp(self):
        self.lsb = SpectralWindow(1000.0, 10.0, 6, sideband=-1, product='lsb')
        self.usb = SpectralWindow(1000.0, 10.0, 6, sideband=1, band='X')
        self.odd = SpectralWindow(1000.0, 10.0, 5, sideband=1)
        # channel_width will not be an exact float. The values have been
        # chosen so that bandwidth / num_chans * num_chans does not quite
        # equal bandwidth.
        self.inexact = SpectralWindow(1000.0, None, 14, sideband=1,
                                      bandwidth=230.0)

    def test_width_properties(self):
        assert_equal(self.lsb.channel_width, 10.0)
        assert_equal(self.lsb.bandwidth, 60.0)
        assert_equal(self.inexact.channel_width, 230.0 / 14)
        assert_equal(self.inexact.bandwidth, 230.0)

    def test_channel_freqs(self):
        assert_array_equal(self.lsb.channel_freqs,
                           [1030.0, 1020.0, 1010.0, 1000.0, 990.0, 980.0])
        assert_array_equal(self.usb.channel_freqs,
                           [970.0, 980.0, 990.0, 1000.0, 1010.0, 1020.0])
        assert_array_equal(self.odd.channel_freqs,
                           [980.0, 990.0, 1000.0, 1010.0, 1020.0])
        assert_array_almost_equal(self.inexact.channel_freqs,
                                  np.arange(14) * 230.0 / 14 + 885.0)
        # Check that the exactly representable values are exact
        assert_equal(self.inexact.channel_freqs[0], 885.0)
        assert_equal(self.inexact.channel_freqs[7], 1000.0)

    def test_repr(self):
        # Just a smoke test to check that it doesn't crash
        repr(self.lsb)
        repr(self.usb)

    def test_subrange(self):
        lsb_sub = self.lsb.subrange(0, 3)
        assert_array_equal(lsb_sub.channel_freqs, [1030.0, 1020.0, 1010.0])
        assert_equal(lsb_sub.product, self.lsb.product)
        usb_sub = self.usb.subrange(2, 6)
        assert_array_equal(usb_sub.channel_freqs,
                           [990.0, 1000.0, 1010.0, 1020.0])
        assert_equal(usb_sub.band, self.usb.band)
        # Check that updated bandwidth doesn't have rounding errors
        inexact_sub = self.inexact.subrange(0, 7)
        assert_equal(inexact_sub.bandwidth, 115.0)

    def test_rechannelise_same(self):
        lsb = self.lsb.rechannelise(6)
        assert lsb == self.lsb

    def test_rechannelise_to_even(self):
        lsb = self.lsb.rechannelise(2)
        assert_array_equal(lsb.channel_freqs, [1020.0, 990.0])
        usb = self.usb.rechannelise(2)
        assert_array_equal(usb.channel_freqs, [980.0, 1010.0])

    def test_rechannelise_to_odd(self):
        lsb = self.lsb.rechannelise(3)
        assert_array_equal(lsb.channel_freqs, [1025.0, 1005.0, 985.0])
        usb = self.usb.rechannelise(3)
        assert_array_equal(usb.channel_freqs, [975.0, 995.0, 1015.0])
        odd = self.odd.rechannelise(1)
        assert_array_equal(odd.channel_freqs, [1000.0])
