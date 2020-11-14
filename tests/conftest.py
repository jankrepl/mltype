"""Configuration of tests."""

import pytest


@pytest.fixture(scope="session")
def empty_config_path(tmp_path_factory):
    """Create an empty config.ini file."""
    config_path = tmp_path_factory.getbasetemp() / "config.ini"
    config_path.touch()

    return config_path
