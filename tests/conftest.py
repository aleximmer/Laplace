from collections.abc import Iterator

import pytest
import torch


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
