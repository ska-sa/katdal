###############################################################################
# Copyright (c) 2018, National Research Foundation (Square Kilometre Array)
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

"""Utilities for applying calibration solutions to visibilities and weights."""

from functools import partial
import copy

import numpy as np
import dask.array as da

from katdal.categorical import CategoricalData


def calc_delay_correction(cache, product, index, freqs):
    """Calculate correction sensor from delay calibration solutions.

    This extracts the cal solution sensor associated with `product` from
    `cache`, then extracts the delay time series of the input specified by
    `index` (in the form (pol, ant)) and builds a categorical sensor for
    the corresponding complex correction terms (channelised by `freqs`).

    Invalid delays (NaNs) are replaced by zeros, since bandpass calibration
    still has a shot at fixing any residual delay. An invalid `product` results
    in a :exc:`KeyError`.
    """
    all_delays = cache.get('cal_product_' + product)
    events = all_delays.events
    delays = [np.nan_to_num(all_delays[n][index]) for n in events[:-1]]
    # Delays returned by cal pipeline are already corrections (no minus needed)
    corrections = [np.exp(2j * np.pi * d * freqs).astype('complex64')
                   for d in delays]
    return CategoricalData(corrections, events)


def add_applycal_sensors(cache, cal_ants, cal_pols, freqs):
    """Add virtual sensors that store calibration corrections, to sensor cache.

    This maps receptor inputs to the relevant indices in each calibration
    product based on the `cal_ants` and `cal_pols` lists. It then registers
    a virtual sensor per input and per cal product in the SensorCache `cache`,
    with template 'Calibration/{inp}_correction_{product}'. The virtual sensor
    function picks the appropriate correction calculator based on the cal
    product name, which also uses auxiliary info like the channel frequencies,
    `freqs`.
    """
    cal_input_map = {ant + pol: (pol_idx, ant_idx)
                     for (pol_idx, pol) in enumerate(cal_pols)
                     for (ant_idx, ant) in enumerate(cal_ants)}
    if not cal_input_map:
        return

    def calc_correction_per_input(cache, name, inp, product):
        """Calculate correction sensor for input `inp` from cal solutions."""
        try:
            index = cal_input_map[inp]
        except KeyError:
            raise KeyError("No calibration solutions available for input "
                           "'{}' - available ones are {}"
                           .format(inp, sorted(cal_input_map.keys())))
        if product == 'K':
            sensor_data = calc_delay_correction(cache, product, index, freqs)
        else:
            raise KeyError("Unknown calibration product '{}'".format(product))
        cache[name] = sensor_data
        return sensor_data

    correction_sensor = 'Calibration/{inp}_correction_{product}'
    cache.virtual[correction_sensor] = calc_correction_per_input


def calc_correction_per_corrprod(dump, channels, cache, inputs,
                                 input1_index, input2_index, cal_products):
    """Gain correction per channel per correlation product for a given dump.

    This calculates an array of complex gain correction terms of shape
    (n_chans, n_corrprods) that can be directly applied to visibility data.
    This incorporates all requested calibration products at the specified
    dump and channels.

    Parameters
    ----------
    dump : int
        Dump index (applicable to full data set, i.e. absolute)
    channels : list of int, length n_chans
        Channel indices (applicable to full data set, i.e. absolute)
    cache : :class:`katdal.sensordata.SensorCache` object
        Sensor cache, used to look up individual correction sensors
    inputs : sequence of string
        Correlator input labels
    input1_index, input2_index : list of int, length n_corrprods
        Indices into `inputs` of first and second items of correlation product
    cal_products : sequence of string
        Calibration products that will contribute to corrections

    Returns
    -------
    gains : array of complex64, shape (n_chans, n_corrprods)
        Gain corrections per channel per correlation product

    Raises
    ------
    KeyError
        If input and/or cal product has no associated correction
    """
    g_per_input = np.ones((len(inputs), len(channels)), dtype='complex64')
    for product in cal_products:
        for n, inp in enumerate(inputs):
            sensor_name = 'Calibration/{}_correction_{}'.format(inp, product)
            g_product = cache.get(sensor_name)[dump]
            if np.shape(g_product) != ():
                g_product = g_product[channels]
            g_per_input[n] *= g_product
    g_per_cp = g_per_input[input1_index] * g_per_input[input2_index].conj()
    return g_per_cp.T


def apply_vis_correction(out, correction):
    """Clean up and apply `correction` in-place to visibility data in `out`."""
    correction[np.isnan(correction)] = np.complex64(1)
    out *= correction


def add_applycal_transform(indexer, cache, corrprods, cal_products,
                           apply_correction):
    """Add transform to indexer that applies calibration corrections.

    This adds a transform to the indexer which wraps the underlying data
    (visibilities, weights or flags). The transform will apply all calibration
    corrections specified in `cal_products` to each dask chunk individually.
    The actual application method is also user-specified, which allows most
    of the machinery to be reused between visibilities, weights and flags.
    The time and frequency selections are salvaged from `indexer` but the
    selected `corrprods` still needs to be passed in as a parameter to identify
    the relevant inputs in order to access correction sensors.

    Parameters
    ----------
    indexer : :class:`katdal.lazy_indexer.DaskLazyIndexer` object
        Indexer with underlying dask array that will be transformed
    cache : :class:`katdal.sensordata.SensorCache` object
        Sensor cache, used to look up individual correction sensors
    corrprods : sequence of (string, string)
        Selected correlation products as pairs of correlator input labels
    cal_products : sequence of string
        Calibration products that will contribute to corrections
    apply_correction : function, signature ``out = f(out, correction)``
        Function that will actually apply correction to data from indexer
    """
    stage1_indices = tuple(k.nonzero()[0] for k in indexer.keep)
    # Turn corrprods into a list of input labels and two lists of indices
    inputs = sorted(set(np.ravel(corrprods)))
    input1_index = [inputs.index(cp[0]) for cp in corrprods]
    input2_index = [inputs.index(cp[1]) for cp in corrprods]
    # Prevent cal_products from changing underneath us if caller changes theirs
    cal_products = copy.deepcopy(cal_products)

    def calibrate_chunk(chunk, block_info):
        """Apply all specified calibration corrections to chunk."""
        corrected_chunk = chunk.copy()
        # Tuple of slices that cuts out `chunk` from full array
        slices = tuple(slice(*l) for l in block_info[0]['array-location'])
        dumps, chans, _ = tuple(i[s] for i, s in zip(stage1_indices, slices))
        index1 = input1_index[slices[2]]
        index2 = input2_index[slices[2]]
        ccpc_args = (chans, cache, inputs, index1, index2, cal_products)
        for n, dump in enumerate(dumps):
            correction = calc_correction_per_corrprod(dump, *ccpc_args)
            apply_correction(corrected_chunk[n], correction)
        return corrected_chunk

    transform = partial(da.map_blocks, calibrate_chunk, dtype=indexer.dtype)
    transform.__name__ = 'applycal[{}]'.format(','.join(cal_products))
    indexer.add_transform(transform)
