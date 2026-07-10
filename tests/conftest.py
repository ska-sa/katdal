################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

import pytest
import numpy as np
from katpoint import Target
from minimal_dataset import MinimalDataSet

TEST_DURATION_TOLERANCE = 0.1


def pytest_addoption(parser):
    parser.addoption(
        "--check-durations",
        action="store_true",
        help="Verify how long some tests run (the ones with an `expected_duration` mark)",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Optionally override pytest report creation to verify test duration."""
    report = (yield).get_result()
    # Only continue if the user requests this and the test has an expected_duration mark
    check_durations = item.config.getoption("--check-durations", default=False)
    mark = item.get_closest_marker("expected_duration")
    if not check_durations or mark is None:
        return report
    # The test will take at least as long as the expected duration and probably a bit longer
    minimum = mark.args[0]
    maximum = minimum + TEST_DURATION_TOLERANCE
    # Only verify duration if the test itself passes (and we are in the 'call' phase of test)
    if (
        report.when == 'call'
        and report.passed
        and not minimum <= report.duration <= maximum
    ):
        # Mark test as failed and report the timing discrepancy
        report.outcome = 'failed'
        report.longrepr = (f"\nTest took {report.duration:g} seconds, "
                           f"which is outside the range [{minimum:g}, {maximum:g}]\n")
    return report


@pytest.fixture
def dataset():
    """A basic dataset used to test the selection mechanism."""
    targets = [
        # It would have been nice to have radec = 19:39, -63:42 but then
        # selection by description string does not work because the catalogue's
        # description string pads it out to radec = 19:39:00.00, -63:42:00.0.
        # (XXX Maybe fix Target comparison in katpoint to support this?)
        Target('J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6'),
        Target('J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6'),
        Target('J0408-6545 | PKS 0408-65, radec bpcal, 4:08:20.38, -65:45:09.1'),
        Target('J1346-6024 | Cen B, radec, 13:46:49.04, -60:24:29.4'),
    ]
    # Ensure that len(timestamps) is an integer multiple of len(targets)
    timestamps = 1234667890.0 + 1.0 * np.arange(3 * len(targets))
    return MinimalDataSet(targets, timestamps)
