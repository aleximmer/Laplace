from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace import LowRankLaplace
from laplace.curvature.asdfghjkl import AsdfghjklInterface


def pytest_configure(config: pytest.Config) -> None:
    """Run torch single-threaded for every test, however they were selected.

    The test tensors are tiny, and one thread per xdist worker keeps the
    workers from oversubscribing the cores.
    """
    torch.set_num_threads(1)


@pytest.fixture
def large_model() -> nn.Module:
    """Return an MLP with about 1M parameters whose last layer is named `fc`.

    The large-init tests check that the Hessian is sized by the subnetwork or
    last layer, not the whole network. A full `n x n` Hessian here would need
    about 4 TB, so a mistake fails at once.
    """
    return nn.Sequential(
        OrderedDict(hidden=nn.Linear(1000, 1000), act=nn.ReLU(), fc=nn.Linear(1000, 10))
    )


@pytest.fixture
def model() -> nn.Module:
    """Return a two-layer MLP (3 -> 20 -> 2) with `output_size`, `n_layers` and `n_params` set."""
    model = nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    model_params = list(model.parameters())

    setattr(model, "output_size", 2)
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))

    return model


@pytest.fixture
def class_loader() -> DataLoader:
    """Return a loader of 10 random 3-d inputs with binary class labels."""
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))

    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader() -> DataLoader:
    """Return a loader of 10 random 3-d inputs with 2-d regression targets."""
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)

    return DataLoader(TensorDataset(X, y), batch_size=3)


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

    Tests with the `asdfghjkl` marker, set here or by hand, also ignore the
    power-iteration warning: the toy models in the tests rarely let it
    converge. `pytest.warns` still catches it where a test checks for it.
    """
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params.values() if callspec else []

        if any(map(_uses_old_asdfghjkl, params)):
            item.add_marker(pytest.mark.asdfghjkl)

        if item.get_closest_marker("asdfghjkl"):
            item.add_marker(
                pytest.mark.filterwarnings(
                    "ignore:asdfghjkl's power iteration didn't converge"
                )
            )

        missing = set().union(*map(_missing_packages, params))

        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"{', '.join(sorted(missing))} not installed")
            )
