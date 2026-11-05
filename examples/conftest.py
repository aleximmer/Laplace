import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: downloads datasets or pretrained weights; run on a schedule"
    )
