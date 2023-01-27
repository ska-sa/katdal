import pytest

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
