from typing import Any

import pytest


def _missing_packages(value: Any) -> set[str]:
    """Collect the packages missing for backend stand-ins in a test parameter.

    Parameters
    ----------
    value : Any
        A parameter value; tuples and lists are searched one level deep.

    Returns
    -------
    set[str]
        Names of the missing packages, empty if `value` holds no stand-in.
    """
    values = value if isinstance(value, (tuple, list)) else [value]

    return {
        v._missing_package
        for v in values
        if isinstance(v, type) and hasattr(v, "_missing_package")
    }


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip parametrized tests that use a backend whose package isn't installed."""
    for item in items:
        callspec = getattr(item, "callspec", None)

        if callspec is None:
            continue

        missing = set().union(*map(_missing_packages, callspec.params.values()))

        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"{', '.join(sorted(missing))} not installed")
            )
