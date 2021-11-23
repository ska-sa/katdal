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

import numba
import numpy as np


@numba.jit(nopython=True, parallel=True)
def _average_visibilities(vis, weight, flag, timeav, chanav, flagav):
    # Workaround for https://github.com/numba/numba/issues/2921
    flag_u8 = flag.view(np.uint8)

    # Compute shapes
    n_time, n_chans, n_bl = vis.shape
    av_n_time = n_time // timeav
    av_n_chans = n_chans // chanav
    av_shape = (av_n_time, av_n_chans, n_bl)

    # Allocate output buffers
    av_vis = np.empty(av_shape, vis.dtype)
    av_weight = np.empty(av_shape, weight.dtype)
    av_flag = np.empty(av_shape, flag.dtype)

    scale = weight.dtype.type(1.0 / (timeav * chanav))
    wzero = weight.dtype.type(0)   # Zero constant of correct type

    bl_step = 128      # Want a chunk to be multiple cache lines but into L1
    # We put channel as the outer loop just because it's more likely than
    # time to get parallel speedup with prange (since the time axis is often
    # short e.g. 1).
    for av_c in numba.prange(0, av_n_chans):
        cstart = av_c * chanav
        vis_sum = np.empty(bl_step, vis.dtype)
        vis_weight_sum = np.empty(bl_step, vis.dtype)
        weight_sum = np.empty(bl_step, weight.dtype)
        flag_any = np.empty(bl_step, np.bool_)
        flag_all = np.empty(bl_step, np.bool_)
        for av_t in range(0, av_n_time):
            tstart = av_t * timeav
            for bstart in range(0, n_bl, bl_step):
                bstop = min(n_bl, bstart + bl_step)
                vis_sum[:] = 0
                vis_weight_sum[:] = 0
                weight_sum[:] = 0
                flag_any[:] = False
                flag_all[:] = True
                for t in range(tstart, tstart + timeav):
                    for c in range(cstart, cstart + chanav):
                        for b in range(bstop - bstart):
                            b1 = b + bstart
                            v = vis[t, c, b1]
                            w = weight[t, c, b1]
                            f = (flag_u8[t, c, b1] != 0)
                            if f:
                                # Don't simply use 0 here: it causes numba's type
                                # inference to upgrade w from float32 to float64.
                                w = wzero
                            flag_any[b] |= f
                            flag_all[b] &= f
                            vis_sum[b] += v
                            vis_weight_sum[b] += w * v
                            weight_sum[b] += w
                for b in range(bstop - bstart):
                    b1 = b + bstart
                    w = np.float32(weight_sum[b])
                    # If everything is flagged/zero-weighted, use an unweighted average
                    if not w:
                        v = vis_sum[b] * scale
                    else:
                        v = vis_weight_sum[b] / w
                    f = flag_any[b] if flagav else flag_all[b]
                    av_vis[av_t, av_c, b1] = v
                    av_weight[av_t, av_c, b1] = w
                    av_flag[av_t, av_c, b1] = f
    return av_vis, av_weight, av_flag


def average_visibilities(vis, weight, flag, timestamps, channel_freqs, timeav=10, chanav=8, flagav=False):
    """Average visibilities, flags and weights.

    Visibilities are weight-averaged using the weights in the `weight` array
    with flagged data set to weight zero. The averaged weights are the sum of
    the input weights for each average block. An average flag is retained if
    all of the data in an averaging block is flagged (the averaged visibility
    in this case is the unweighted average of the input visibilities). In cases
    where the averaging size in channel or time does not evenly divide the size
    of the input data, the remaining channels or timestamps at the end of the
    array after averaging are discarded. Channels are averaged first and the
    timestamps are second. An array of timestamps and frequencies corresponding
    to each channel is also directly averaged and returned.

    Parameters
    ----------
    vis: array(numtimestamps,numchannels,numbaselines) of complex64.
          The input visibilities to be averaged.
    weight: array(numtimestamps,numchannels,numbaselines) of float32.
          The input weights (used for weighted averaging).
    flag: array(numtimestamps,numchannels,numbaselines) of boolean.
          Input flags (flagged data have weight zero before averaging).
    timestamps: array(numtimestamps) of int.
          The timestamps (in mjd seconds) corresponding to the input data.
    channel_freqs: array(numchannels) of int.
          The frequencies (in Hz) corresponding to the input channels.
    timeav: int.
          The desired averaging size in timestamps.
    chanav: int.
          The desired averaging size in channels.
    flagav: bool
          Flagged averaged data in when there is a single flag in the bin if true.
          Only flag averaged data when all data in the bin is flagged if false.

    Returns
    -------
    av_vis: array(int(numtimestamps/timeav),int(numchannels/chanav)) of complex64.
    av_weight: array(int(numtimestamps/timeav),int(numchannels/chanav)) of float32.
    av_flag: array(int(numtimestamps/timeav),int(numchannels/chanav)) of boolean.
    av_mjd: array(int(numtimestamps/timeav)) of int.
    av_freq: array(int(numchannels)/chanav) of int.

    """
    # Trim data to integer multiples of the averaging factors
    n_time, n_chans, n_bl = vis.shape
    timeav = min(timeav, n_time)
    flagav = min(flagav, n_chans)
    n_time = n_time // timeav * timeav
    n_chans = n_chans // chanav * chanav

    vis = vis[:n_time, :n_chans]
    weight = weight[:n_time, :n_chans]
    flag = flag[:n_time, :n_chans]
    timestamps = timestamps[:n_time]
    channel_freqs = channel_freqs[:n_chans]

    # Average the data (using a numba-accelerated function)
    av_vis, av_weight, av_flag = \
        _average_visibilities(vis, weight, flag, timeav, chanav, flagav)

    # Average the metadata
    av_freq = np.mean(channel_freqs.reshape(-1, chanav), axis=-1)
    av_timestamps = np.mean(timestamps.reshape(-1, timeav), axis=-1)

    return av_vis, av_weight, av_flag, av_timestamps, av_freq
