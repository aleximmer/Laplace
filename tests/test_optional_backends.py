"""
Checks that laplace works without the optional ASDL and BackPACK packages.

Each test runs in a fresh interpreter where the missing package is blocked by
setting `sys.modules[name] = None`, so `find_spec(name)` returns `None` there.
"""

import subprocess
import sys
import textwrap
from importlib.util import find_spec

import pytest

from laplace.curvature import CurvlinopsGGN
from tests.utils import AsdlEF, BackPackGGN


def _run_without(blocked: list[str], code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter where the `blocked` packages can't be imported."""
    prelude = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in blocked)

    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("blocked", [["asdl"], ["backpack"], ["asdl", "backpack"]])
def test_fit_without_optional_backends(blocked):
    code = """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import laplace.curvature
    from laplace import Laplace
    from laplace.curvature import *

    print("ALL", laplace.curvature.__all__)

    X, y = torch.randn(16, 3), torch.randn(16, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    )

    for subset_of_weights in ["all", "last_layer"]:
        for hessian_structure in ["full", "kron", "diag"]:
            la = Laplace(
                model,
                "regression",
                subset_of_weights=subset_of_weights,
                hessian_structure=hessian_structure,
            )
            la.fit(loader)

    print("OK")
    """
    res = _run_without(blocked, code)

    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout

    all_line = next(line for line in res.stdout.splitlines() if line.startswith("ALL"))
    prefixes = {"asdl": "'Asdl", "backpack": "'BackPack"}
    for module in blocked:
        assert prefixes[module] not in all_line


@pytest.mark.parametrize(
    "module,name,extra",
    [("asdl", "AsdlGGN", "asdl"), ("backpack", "BackPackGGN", "backpack")],
)
@pytest.mark.parametrize(
    "statement",
    ["from laplace.curvature import {name}", "import laplace.curvature.{module}"],
)
def test_missing_backend_raises_install_hint(module, name, extra, statement):
    code = f"""
    try:
        {statement.format(name=name, module=module)}
    except ImportError as e:
        print(e)
    """
    res = _run_without([module], code)

    assert res.returncode == 0, res.stderr
    assert f'pip install "laplace-torch[{extra}]"' in res.stdout


@pytest.mark.skipif(
    find_spec("asdl") is None or find_spec("backpack") is None,
    reason="asdl and backpack needed",
)
def test_is_optional_backend():
    from laplace.curvature import _is_optional_backend

    assert _is_optional_backend(BackPackGGN, "BackPackInterface")
    assert _is_optional_backend(AsdlEF, "AsdlInterface")
    assert not _is_optional_backend(CurvlinopsGGN, "BackPackInterface")
    assert not _is_optional_backend(BackPackGGN, "AsdlInterface")


@pytest.mark.skipif(find_spec("asdl") is None, reason="asdl not installed")
def test_broken_installed_backend_is_not_skipped():
    # asdl is installed but one of its modules fails to import, e.g. after an
    # API change. The test helpers must raise instead of handing out a stand-in
    # that would turn every ASDL test into a skip.
    code = """
    import tests.utils

    print(getattr(tests.utils.AsdlGGN, "_missing_package", None))
    """
    res = _run_without(["asdl.matrices"], code)

    assert res.returncode != 0, res.stdout
    assert "asdl.matrices" in res.stderr


@pytest.mark.skipif(
    find_spec("asdl") is None or find_spec("backpack") is None,
    reason="asdl and backpack needed",
)
def test_dir_lists_installed_backends():
    import laplace.curvature

    assert set(laplace.curvature._OPTIONAL_BACKENDS) <= set(dir(laplace.curvature))


def test_dir_hides_missing_backends():
    code = """
    import laplace.curvature

    print(sorted(n for n in dir(laplace.curvature) if n.startswith("BackPack")))
    """
    res = _run_without(["backpack"], code)

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == "[]"
