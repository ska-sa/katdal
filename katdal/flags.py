################################################################################
# Copyright (c) 2019-2021, National Research Foundation (SARAO)
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

"""Definitions of flag bits"""

NAMES = ('reserved0', 'static', 'cam', 'data_lost',
         'ingest_rfi', 'predicted_rfi', 'cal_rfi', 'postproc')
DESCRIPTIONS = ('reserved - bit 0',
                'predefined static flag list',
                'flag based on live CAM information',
                'no data was received',
                'RFI detected in ingest',
                'RFI predicted from space based pollutants',
                'RFI detected in calibration',
                'some correction/postprocessing step could not be applied')

STATIC_BIT = 1
CAM_BIT = 2
DATA_LOST_BIT = 3
INGEST_RFI_BIT = 4
PREDICTED_RFI_BIT = 5
CAL_RFI_BIT = 6
POSTPROC_BIT = 7

STATIC = 1 << STATIC_BIT
CAM = 1 << CAM_BIT
DATA_LOST = 1 << DATA_LOST_BIT
INGEST_RFI = 1 << INGEST_RFI_BIT
PREDICTED_RFI = 1 << PREDICTED_RFI_BIT
CAL_RFI = 1 << CAL_RFI_BIT
POSTPROC = 1 << POSTPROC_BIT
