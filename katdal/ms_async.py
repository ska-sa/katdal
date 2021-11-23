################################################################################
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
################################################################################

"""Write data to a Measurement Set asynchronously.

This uses multiprocessing, a queue, and a circular buffer in shared memory to
pass visibility data to a separate process that actually writes to the
measurement set.

This is largely an implementation detail of the mvftoms.py script, and might
not be suited to other use cases. It is put into a separate module as a
workaround for https://bugs.python.org/issue9914.
"""

import contextlib
import multiprocessing
import multiprocessing.sharedctypes
from collections import namedtuple

import katpoint
import numpy as np

from . import ms_extra


class RawArray:
    """Shared memory array, in representation that can be passed through multiprocessing queue"""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        size = self.dtype.itemsize * int(np.product(shape))
        self.storage = multiprocessing.sharedctypes.RawArray('c', size)

    def asarray(self):
        """Return numpy array representation"""
        return np.frombuffer(self.storage, self.dtype).reshape(self.shape)


QueueItem = namedtuple('QueueItem', ['slot', 'target', 'time_utc', 'dump_time_width',
                                     'field_id', 'state_id', 'scan_itr'])
ScanResult = namedtuple('ScanResult', ['scan_size'])
EndOfScan = namedtuple('EndOfScan', [])


def ms_writer_process(
        work_queue, result_queue, options, antennas, cp_info, ms_name,
        raw_vis_data, raw_weight_data, raw_flag_data):
    """
    Function to be run in a separate process for writing to a Measurement Set.
    The MS is assumed to have already been created with the appropriate
    columns.

    Incoming work is provided by submitting instances of :class:`QueueItem`
    to `work_queue`. The `slot` indexes the first dimension of the shared
    memory arrays. One may also submit an :class:`EndOfScan`, which will flush
    to disk and return a :class:`ScanResult` through the `result_queue` (these
    are not actually required to match katdal scans).

    To terminate the process, submit ``None`` to `work_queue`.

    If an exception occurs, it will be placed into `result_queue`, after which
    work_queue items will be fetched and discarded until ``None`` is received.
    When finished (either successfully or after an error), ``None`` is put in
    `result_queue`.

    Parameters
    ----------
    work_queue : :class:`multiprocessing.Queue`
        Incoming work. Note that this function gives no explicit indication
        when it is done with a piece of work, so the queue capacity needs to
        be bounded to prevent data races.
    result_queue : :class:`multiprocessing.Queue`
        Information about progress (see :class:`ScanResult`)
    options : :class:`argparse.Namespace`
        Command-line options to mvftoms
    antennas : list of :class:`katpoint.Antenna`
        Antennas (used to compute UVW coordinates)
    cp_info : namedtuple
        Correlation product info (see mvftoms.py)
    ms_name : str
        Name of the Measurement Set to write
    raw_vis_data, raw_weight_data, raw_flag_data : :class:`RawArray`
        Circular buffers for the data, with shape
        (slots, time, baseline, channel, pol).
    """

    none_seen = False
    try:
        vis_arrays = raw_vis_data.asarray()
        weight_arrays = raw_weight_data.asarray()
        flag_arrays = raw_flag_data.asarray()
        scan_size = 0
        tdiff = vis_arrays.shape[1]
        nbl = vis_arrays.shape[2]

        main_table = ms_extra.open_table(ms_name, verbose=options.verbose)
        with contextlib.closing(main_table):
            array_centre = antennas[0].array_reference_antenna()
            while True:
                item = work_queue.get()
                if item is None:
                    none_seen = True
                    break
                elif isinstance(item, EndOfScan):
                    main_table.flush()    # Mostly to get realistic throughput stats
                    result_queue.put(ScanResult(scan_size))
                    scan_size = 0
                else:
                    # Extract the slot, and flatten time and baseline into a single axis
                    new_shape = (-1, vis_arrays.shape[-2], vis_arrays.shape[-1])
                    vis_data = vis_arrays[item.slot].reshape(new_shape)
                    weight_data = weight_arrays[item.slot].reshape(new_shape)
                    flag_data = flag_arrays[item.slot].reshape(new_shape)

                    # Iterate through baselines, computing UVW coordinates
                    # for a chunk of timesteps. Note that we can't rely on the
                    # u, v, w properties of the dataset because those
                    # correspond to the original dumps, and we might be
                    # averaging in time.
                    uvw_ant = item.target.uvw(antennas, item.time_utc, array_centre)
                    # Permute from axis, time, antenna to time, antenna, axis
                    uvw_ant = np.transpose(uvw_ant, (1, 2, 0))
                    # Compute baseline UVW coordinates from per-antenna coordinates.
                    # The sign convention matches `CASA`_, rather than the
                    # Measurement Set `definition`_.
                    # .. _CASA: https://casa.nrao.edu/Memos/CoordConvention.pdf
                    # .. _definition: https://casa.nrao.edu/Memos/229.html#SECTION00064000000000000000
                    uvw_coordinates = (np.take(uvw_ant, cp_info.ant1_index, axis=1)
                                       - np.take(uvw_ant, cp_info.ant2_index, axis=1))
                    # Flatten time and baseline axes together
                    uvw_coordinates = uvw_coordinates.reshape(-1, 3)

                    # Convert averaged UTC timestamps to MJD seconds.
                    # Blow time up to (ntime*nbl,)
                    out_mjd = np.asarray([katpoint.Timestamp(t).to_mjd() * 24 * 60 * 60
                                          for t in item.time_utc])

                    out_mjd = np.broadcast_to(out_mjd[:, np.newaxis], (tdiff, nbl)).ravel()

                    # Repeat antenna indices to (ntime*nbl,)
                    a1 = np.broadcast_to(cp_info.ant1_index[np.newaxis, :], (tdiff, nbl)).ravel()
                    a2 = np.broadcast_to(cp_info.ant2_index[np.newaxis, :], (tdiff, nbl)).ravel()

                    # Blow field ID up to (ntime*nbl,)
                    big_field_id = np.full((tdiff * nbl,), item.field_id, dtype=np.int32)
                    big_state_id = np.full((tdiff * nbl,), item.state_id, dtype=np.int32)
                    big_scan_itr = np.full((tdiff * nbl,), item.scan_itr, dtype=np.int32)

                    # Setup model_data and corrected_data if required
                    model_data = None
                    corrected_data = None

                    if options.model_data:
                        # unity intensity zero phase model data set, same shape as vis_data
                        model_data = np.ones(vis_data.shape, dtype=np.complex64)
                        # corrected data set copied from vis_data
                        corrected_data = vis_data

                    # Populate dictionary for write to MS
                    main_dict = ms_extra.populate_main_dict(
                        uvw_coordinates, vis_data,
                        flag_data, weight_data, out_mjd, a1, a2,
                        item.dump_time_width, big_field_id, big_state_id,
                        big_scan_itr, model_data, corrected_data)

                    # Write data to MS.
                    ms_extra.write_rows(main_table, main_dict, verbose=options.verbose)

                    # Calculate bytes written from the summed arrays in the dict
                    scan_size += sum(a.nbytes for a in main_dict.values()
                                     if isinstance(a, np.ndarray))
    except Exception as error:
        result_queue.put(error)
        while not none_seen:
            item = work_queue.get()
            if item is None:
                none_seen = True
    finally:
        result_queue.put(None)
