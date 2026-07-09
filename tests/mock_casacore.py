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

import sys
from unittest import mock

casacore = mock.MagicMock()
casacore.__version__ = "3.5.0"
casacore.tables = tables = mock.MagicMock()

# The output of casacore.tables.required_ms_desc("MAIN")
REQUIRED_MS_DESC = {
    "UVW": {
        "valueType": "double",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 5,
        "maxlen": 0,
        "comment": "Vector with uvw coordinates (in meters)",
        "ndim": 1,
        "shape": [3],
        "_c_order": True,
        "keywords": {
            "QuantumUnits": ["m", "m", "m"],
            "MEASINFO": {"type": "uvw", "Ref": "ITRF"},
        },
    },
    "FLAG": {
        "valueType": "boolean",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The data flags, array of bools with same shape as data",
        "ndim": 2,
        "_c_order": True,
        "keywords": {},
    },
    "FLAG_CATEGORY": {
        "valueType": "boolean",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The flag category, NUM_CAT flags for each datum",
        "ndim": 3,
        "_c_order": True,
        "keywords": {},
    },
    "WEIGHT": {
        "valueType": "float",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Weight for each polarization spectrum",
        "ndim": 1,
        "_c_order": True,
        "keywords": {},
    },
    "SIGMA": {
        "valueType": "float",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Estimated rms noise for channel with unity bandpass response",
        "ndim": 1,
        "_c_order": True,
        "keywords": {},
    },
    "ANTENNA1": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "ID of first antenna in interferometer",
        "keywords": {},
    },
    "ANTENNA2": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "ID of second antenna in interferometer",
        "keywords": {},
    },
    "ARRAY_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "ID of array or subarray",
        "keywords": {},
    },
    "DATA_DESC_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The data description table index",
        "keywords": {},
    },
    "EXPOSURE": {
        "valueType": "double",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The effective integration time",
        "keywords": {"QuantumUnits": ["s"]},
    },
    "FEED1": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The feed index for ANTENNA1",
        "keywords": {},
    },
    "FEED2": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The feed index for ANTENNA2",
        "keywords": {},
    },
    "FIELD_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Unique id for this pointing",
        "keywords": {},
    },
    "FLAG_ROW": {
        "valueType": "boolean",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Row flag - flag all data in this row if True",
        "keywords": {},
    },
    "INTERVAL": {
        "valueType": "double",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "The sampling interval",
        "keywords": {"QuantumUnits": ["s"]},
    },
    "OBSERVATION_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "ID for this observation, index in OBSERVATION table",
        "keywords": {},
    },
    "PROCESSOR_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Id for backend processor, index in PROCESSOR table",
        "keywords": {},
    },
    "SCAN_NUMBER": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Sequential scan number from on-line system",
        "keywords": {},
    },
    "STATE_ID": {
        "valueType": "int",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "ID for this observing state",
        "keywords": {},
    },
    "TIME": {
        "valueType": "double",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Modified Julian Day",
        "keywords": {
            "QuantumUnits": ["s"],
            "MEASINFO": {"type": "epoch", "Ref": "UTC"},
        },
    },
    "TIME_CENTROID": {
        "valueType": "double",
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "StandardStMan",
        "option": 0,
        "maxlen": 0,
        "comment": "Modified Julian Day",
        "keywords": {
            "QuantumUnits": ["s"],
            "MEASINFO": {"type": "epoch", "Ref": "UTC"},
        },
    },
    "_define_hypercolumn_": {},
    "_keywords_": {"MS_VERSION": 2.0},
    "_private_keywords_": {},
}
tables.required_ms_desc.return_value = REQUIRED_MS_DESC


# Based on the output of casacore.tables.tablecreatearraycoldesc("DATA", 0+0j)
def mock_tablecreatearraycoldesc(
    columnname,
    value,
    ndim=0,
    shape=[],
    datamanagertype="",
    datamanagergroup="",
    options=0,
    maxlen=0,
    comment="",
    valuetype="",
    keywords={},
):
    return {
        "name": columnname,
        "desc": {
            "valueType": valuetype,
            "dataManagerType": datamanagertype,
            "dataManagerGroup": datamanagergroup,
            "ndim": ndim,
            "shape": shape,
            "_c_order": True,
            "option": options,
            "maxlen": maxlen,
            "comment": comment,
            "keywords": keywords,
        },
    }


tables.tablecreatearraycoldesc.side_effect = mock_tablecreatearraycoldesc

mock_table = mock.MagicMock()
mock_table.nrows.return_value = 0
tables.table.return_value = mock_table

sys.modules["casacore"] = casacore
