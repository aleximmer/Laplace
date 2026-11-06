from collections.abc import Iterator
from typing import Any

import pytest
import torch

from laplace import LowRankLaplace
from laplace.curvature.asdfghjkl import AsdfghjklInterface


@pytest.fixture(autouse=True)
def _isolate_global_torch_state(request: pytest.FixtureRequest) -> Iterator[None]:
    """Set torch's default dtype and RNG seed for each test, then restore the dtype.

    Without this, a module that changes the dtype or seeds the RNG at import
    time affects every module after it, so results depend on file order.

    The dtype comes from the `default_dtype` marker and falls back to
    `torch.float32`. The RNG is seeded from the `seed` marker, or with 0, so
    each test draws the same numbers however it was selected. The RNG state
    from before the test is restored afterwards.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The requesting test, whose markers (or its module's `pytestmark`)
        are read.
    """
    dtype_marker = request.node.get_closest_marker("default_dtype")
    seed_marker = request.node.get_closest_marker("seed")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype_marker.args[0] if dtype_marker else torch.float32)

    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed_marker.args[0] if seed_marker else 0)
            yield
    finally:
        torch.set_default_dtype(previous_dtype)


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


def _uses_old_asdfghjkl(value: Any) -> bool:
    """Check whether a test parameter is `LowRankLaplace` or an asdfghjkl backend.

    Parameters
    ----------
    value : Any
        A parameter value; tuples and lists are searched one level deep.

    Returns
    -------
    bool
        True if `value` is, or holds, `LowRankLaplace` or a subclass of
        `AsdfghjklInterface`.
    """
    values = value if isinstance(value, (tuple, list)) else [value]

    return any(
        v is LowRankLaplace or issubclass(v, AsdfghjklInterface)
        for v in values
        if isinstance(v, type)
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark and skip parametrized tests by the backends their parameters use.

    Tests that use `LowRankLaplace` or an asdfghjkl backend get the
    `asdfghjkl` marker. Tests that use a backend whose package isn't
    installed are skipped.
    """
    for item in items:
        callspec = getattr(item, "callspec", None)

        if callspec is None:
            continue

        if any(map(_uses_old_asdfghjkl, callspec.params.values())):
            item.add_marker(pytest.mark.asdfghjkl)

        missing = set().union(*map(_missing_packages, callspec.params.values()))

        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"{', '.join(sorted(missing))} not installed")
            )
