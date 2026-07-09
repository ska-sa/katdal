################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

import queue
from unittest import mock

import numpy as np
from katpoint import Target
import mock_casacore
from minimal_dataset import ANTENNAS

# Now that casacore has been mocked we can import these
from katdal import ms_extra, ms_async
from katdal.scripts import mvftoms


def test_ms_extra_logic():
    """Test that we can call create_ms without real casacore."""
    desc, dminfo = ms_extra.kat_ms_desc_and_dminfo(nbl=6, nchan=16, ncorr=4)
    ms_extra.create_ms("test.ms", desc, dminfo)
    assert mock_casacore.tables.default_ms.called


def test_ms_async_writer(tmp_path):
    """Test the ms_async writer process logic."""
    ms_name = str(tmp_path / "test.ms")
    options = mock.MagicMock(verbose=True, model_data=False)
    antennas = ANTENNAS
    cp_info = mock.MagicMock(ant1_index=np.array([0, 0]), ant2_index=np.array([0, 1]))

    work_queue = mock.Mock()
    result_queue = mock.Mock()

    # Mocking work_queue.get to return one item then None
    target = Target("Sun, special")
    item = ms_async.QueueItem(
        slot=0,
        target=target,
        time_utc=np.array([100.0]),
        dump_time_width=1.0,
        field_id=0,
        state_id=0,
        scan_itr=1,
    )
    work_queue.get.side_effect = [item, None]

    raw_vis = ms_async.RawArray((4, 1, 2, 16, 4), np.complex64)
    raw_weight = ms_async.RawArray((4, 1, 2, 16, 4), np.float32)
    raw_flag = ms_async.RawArray((4, 1, 2, 16, 4), bool)

    # We need to mock ms_extra.open_table since it returns a mock that we want to control
    with mock.patch("katdal.ms_extra.open_table") as mock_open:
        table_mock = mock_open.return_value
        table_mock.nrows.return_value = 0
        table_mock.colnames.return_value = list(mock_casacore.REQUIRED_MS_DESC.keys()) + [
            "DATA",
            "WEIGHT_SPECTRUM",
            "SIGMA_SPECTRUM",
        ]
        ms_async.ms_writer_process(
            work_queue,
            result_queue,
            options,
            antennas,
            cp_info,
            ms_name,
            raw_vis,
            raw_weight,
            raw_flag,
            start_row=0,
        )

    # Check if an error was put in the result queue
    for call in result_queue.put.call_args_list:
        obj = call[0][0]
        if isinstance(obj, Exception):
            raise obj

    assert table_mock.addrows.called


def test_mvftoms_main(tmp_path, dataset):
    """End-to-end test of the mvftoms script."""
    ms_name = str(tmp_path / "test.ms")

    with mock.patch("katdal.open", return_value=dataset):
        with mock.patch("sys.argv", ["mvftoms.py", "dummy.rdb", "-o", ms_name]):
            # Mock multiprocessing.Process to run synchronously
            with mock.patch("multiprocessing.Process") as mock_proc:
                # Capture the target and args
                def side_effect(*args, **kwargs):
                    p = mock.MagicMock()
                    return p

                mock_proc.side_effect = side_effect

                # We need to avoid the infinite loop in result_queue.get()
                with mock.patch("multiprocessing.Queue") as mock_queue_cls:
                    res_queue = mock_queue_cls.return_value
                    # MinimalDataSet has 4 tracks (and 4 slews which are skipped)
                    # For each track, result_queue.get() is called once.
                    # Then in finally, it's called until it gets None.
                    res_queue.get.side_effect = [mock.MagicMock(scan_size=1024)] * 4 + [None]
                    # get_nowait is used to check for errors asynchronously
                    res_queue.get_nowait.side_effect = queue.Empty

                    try:
                        mvftoms.main()
                    except SystemExit:
                        pass

    assert mock_casacore.tables.default_ms.called
