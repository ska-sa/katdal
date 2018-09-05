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
        if product.startswith('K'):
            sensor_data = calc_delay_correction(cache, product, index, freqs)
        else:
            raise KeyError("Unknown calibration product '{}'".format(product))
        cache[name] = sensor_data
        return sensor_data

    correction_sensor = 'Calibration/{inp}_correction_{product}'
    cache.virtual[correction_sensor] = calc_correction_per_input
