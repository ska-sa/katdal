import pytest

check_durations = False


def pytest_addoption(parser):
    parser.addoption(
        "--check-durations",
        action="store_true",
        help="Verify how long some tests run (the ones with @duration decorator)",
    )


def pytest_configure(config):
    # Save option at module level to be imported into standard function decorator,
    # which seems much simpler than the fixture route.
    global check_durations
    check_durations = config.getoption("--check-durations", default=False)
