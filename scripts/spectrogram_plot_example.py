#! /usr/bin/env python

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

#
# Plot spectrogram of entire dataset in an efficient way that only loads
# enough data that will fit onto the screen.
#
# Ludwig Schwardt
# 26 June 2012
#

import optparse
import time

import matplotlib.pyplot as plt
import numpy as np

import katdal


class ResampledImage:
    """Image that only loads enough data that will fit onto screen pixels.

    Parameters
    ----------
    data : array-like, shape at least (N, M)
        Data object with ndarray interface
    extract : function, signature ``xy_data = f(data, x, y)``, optional
        Function used to extract 2-D image array from data object given x and
        y indices, using getitem interface on data by default
    autoscale : {False, True}, optional
        True if image should be renormalised after each update or zoom
    ax : :class:`matplotlib.axes.Axes` object or None, optional
        Axes onto which to plot image
    kwargs : dict, optional
        Additional parameters are passed on to underlying imshow

    """
    def __init__(self, data, extract=None, autoscale=False, ax=None, **kwargs):
        self.data = data
        self.extract = extract if extract is not None else lambda d, x, y: d[y, x]
        self.autoscale = autoscale
        self.ax = ax if ax is not None else plt.gca()
        kwargs.update({'aspect': 'auto', 'origin': 'lower', 'interpolation': 'nearest',
                       'extent': (-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5)})
        self.image = self.ax.imshow([[0]], **kwargs)
        self.update()
        # Connect to all events that change the data limits or the number of pixels in image
        self.ax.callbacks.connect('xlim_changed', self.update)
        self.ax.callbacks.connect('ylim_changed', self.update)
        self.ax.figure.canvas.mpl_connect('resize_event', self.update)

    def update(self, param=None):
        """Load required data and update image."""
        data_limits, view_limits = self.ax.dataLim, self.ax.viewLim
        display_limits = self.ax.get_window_extent()
        # print "data =", data_limits.extents[[0, 2, 1, 3]].tolist()
        # print "view =", view_limits.extents[[0, 2, 1, 3]].tolist()
        # print "display =", display_limits.extents[[0, 2, 1, 3]].tolist()
        data_scale_x = self.data.shape[1] / data_limits.width
        data_scale_y = self.data.shape[0] / data_limits.height
        x_from = max(int(np.floor(data_scale_x * (view_limits.x0 - data_limits.x0))), 0)
        y_from = max(int(np.floor(data_scale_y * (view_limits.y0 - data_limits.y0))), 0)
        x_to = max(int(np.ceil(data_scale_x * (view_limits.x1 - data_limits.x0))), x_from + 1)
        y_to = max(int(np.ceil(data_scale_y * (view_limits.y1 - data_limits.y0))), y_from + 1)
        x_step = max(int(view_limits.width / display_limits.width), 1)
        y_step = max(int(view_limits.height / display_limits.height), 1)
        # print "range = %d:%d:%d, %d:%d:%d" % (x_from, x_to, x_step, y_from, y_to, y_step)
        x_slice = slice(x_from, x_to, x_step)
        y_slice = slice(y_from, y_to, y_step)
        x_inds = list(range(*x_slice.indices(self.data.shape[1])))
        y_inds = list(range(*y_slice.indices(self.data.shape[0])))
        im_left = x_inds[0] / data_scale_x + data_limits.x0
        im_right = (x_inds[-1] + 1) / data_scale_x + data_limits.x0
        im_bottom = y_inds[0] / data_scale_y + data_limits.y0
        im_top = (y_inds[-1] + 1) / data_scale_y + data_limits.y0
        # print "im =", (im_left, im_right, im_bottom, im_top)
        before = time.time()
        # Load and update image data and make it fill the view
        data = self.extract(self.data, x_slice, y_slice)
        extract_time = time.time() - before
        size_bytes = data.size * np.dtype('complex64').itemsize
        print("Loaded %d visibilities - x %s y %s - in %.2f seconds (%g MB/s)" %
              (data.size, x_slice, y_slice, extract_time, size_bytes * 1e-6 / extract_time))
        self.image.set_data(data)
        self.image._extent = (im_left, im_right, im_bottom, im_top)
        if self.autoscale:
            self.image.autoscale()
        else:
            # Keep the same normalisation as soon as the extreme data values are known
            self.image.norm.vmin = min(self.image.norm.vmin, data.min())
            self.image.norm.vmax = max(self.image.norm.vmax, data.max())
        self.ax.figure.canvas.draw_idle()


parser = optparse.OptionParser(usage="%prog [options] <data file> [<data file> ...]",
                               description='Waterfall plot from HDF5 data file(s)')
parser.add_option('-a', '--ant',
                  help="Antenna to plot (e.g. 'ant1'), default is first antenna")
parser.add_option('-p', '--pol', type='choice', choices=['H', 'V'], default='H',
                  help="Polarisation term to use ('H' or 'V'), default is %default")
parser.add_option('-s', '--autoscale', action='store_true', default=False,
                  help="Renormalise colour scale after each zoom or resize, default is %default")
(opts, args) = parser.parse_args()

if len(args) == 0:
    print('Please specify at least one HDF5 file to load')
else:
    d = katdal.open(args)
    ant = opts.ant if opts.ant is not None else d.ref_ant
    d.select(ants=ant, pol=opts.pol)

    plt.figure(1)
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    im = ResampledImage(d.vis, extract=lambda data, x, y: np.abs(data[y, x, 0]),
                        autoscale=opts.autoscale, ax=ax)
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Dump index')
    ax.set_title(f'Spectrogram {d.name} {ant} {opts.pol}')
    plt.show()
