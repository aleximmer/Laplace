"""Smoke tests that run every example script and check the ones rendered as doc pages.

Each script runs in a fresh subprocess, inside a temporary working directory, and must
exit cleanly. New scripts are picked up automatically. Run with:

    pytest examples/test_examples.py -m "not slow"

The `slow` ones run the scripts in `FULL` without arguments, with the large pretrained
models and the full data. CI runs them once a month, not on pull requests.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent
DOCS_DIR = EXAMPLES_DIR.parent / "docs"

TIMEOUT = 600
SLOW_TIMEOUT = 3600

# Command-line arguments that shrink the slow scripts: fewer iterations, data subsets,
# and smaller models.
ARGS = {
    "bayesopt_example.py": ["--exp_len", "3"],
    "calibration_example.py": ["--n_train", "256", "--n_test", "256", "--tiny_model"],
    "calibration_gp_example.py": [
        *("--n_train", "256", "--n_test", "256", "--tiny_model"),
        *("--subset_sizes", "50"),
    ],
    "huggingface_example.py": ["--model", "hf-internal-testing/tiny-random-gpt2"],
}

# Scripts that also run once a month without `ARGS`, at the size the doc page shows
FULL = {"calibration_example.py", "calibration_gp_example.py", "huggingface_example.py"}

MARKDOWN_CELL = "# %% [markdown]"


def _scripts() -> list[str]:
    return sorted(
        p.name
        for p in EXAMPLES_DIR.glob("*.py")
        if p.name not in ("conftest.py", "test_examples.py")
    )


def _rendered() -> list[str]:
    """Examples that mkdocs renders as doc pages: jupytext scripts opening with prose."""
    return sorted(
        name
        for name in _scripts()
        if (EXAMPLES_DIR / name).read_text().startswith(MARKDOWN_CELL)
    )


def markdown_cells(source: str) -> list[str]:
    """Return the text of each markdown cell in a jupytext percent script."""
    cells = []

    for cell in re.split(r"^(?=# %%)", source, flags=re.MULTILINE):
        header, _, body = cell.partition("\n")
        if header.strip() != MARKDOWN_CELL:
            continue

        lines = [re.sub(r"^# ?", "", line) for line in body.splitlines()]
        cells.append("\n".join(lines).strip())

    return cells


def _params() -> list:
    """Each script with its shrinking `ARGS`, plus a slow `[full]` run for `FULL`."""
    params = [
        pytest.param(name, ARGS.get(name, []), TIMEOUT, id=name) for name in _scripts()
    ]
    params += [
        pytest.param(name, [], SLOW_TIMEOUT, id=f"{name}[full]", marks=pytest.mark.slow)
        for name in sorted(FULL)
    ]

    return params


def run_example(
    script: Path, cwd: Path, args: Sequence[str] = (), timeout: float = TIMEOUT
) -> subprocess.CompletedProcess:
    """Run `script` in a subprocess with `examples/` importable and plots kept in `cwd`."""
    env = os.environ | {
        "MPLBACKEND": "Agg",
        "LAPLACE_EXAMPLES_ASSETS_DIR": str(cwd),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(EXAMPLES_DIR), os.environ.get("PYTHONPATH")])
        ),
    }

    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize(("script", "args", "timeout"), _params())
def test_example_runs(
    script: str, args: list[str], timeout: float, tmp_path: Path
) -> None:
    _check(run_example(EXAMPLES_DIR / script, tmp_path, args, timeout))


def test_all_examples_collected() -> None:
    scripts = _scripts()

    assert "regression_example.py" in scripts
    assert "test_examples.py" not in scripts and "conftest.py" not in scripts

    # No stale entries for renamed or deleted files
    assert set(ARGS) <= set(scripts)


def test_runner_reports_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(10)\n")

    with pytest.raises(subprocess.TimeoutExpired):
        run_example(script, tmp_path, timeout=1)


def test_runner_reports_failure(tmp_path: Path) -> None:
    script = tmp_path / "broken.py"
    script.write_text("import a_package_that_does_not_exist\n")

    result = run_example(script, tmp_path)

    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


def test_rendered_examples_are_in_nav() -> None:
    nav = (EXAMPLES_DIR.parent / "mkdocs.yml").read_text()

    assert "regression_example.py" in _rendered()
    assert all(f": {name}" in nav for name in _rendered())


@pytest.mark.parametrize("script", _rendered())
def test_rendered_markdown_is_portable(script: str) -> None:
    cells = markdown_cells((EXAMPLES_DIR / script).read_text())
    text = "\n".join(cells)

    assert cells, "needs at least one markdown cell"
    assert "\\(" not in text and "\\[" not in text
    assert ".md)" not in text and "!!!" not in text

    for img in re.findall(r"\]\(\.\./assets/([^)]+)\)", text):
        assert (DOCS_DIR / "assets" / img).exists(), img

    rendered = {Path(n).stem for n in _rendered()}
    for page in re.findall(r"(?<!!)\[[^\]]*\]\(\.\./([\w/-]+)/", text):
        assert page in rendered or (DOCS_DIR / f"{page}.md").exists(), page


@pytest.mark.parametrize("script", _rendered())
def test_rendered_example_runs_in_jupyter(script: str) -> None:
    assert ".parse_args()" not in (EXAMPLES_DIR / script).read_text()


def test_markdown_cells() -> None:
    src = "# %% [markdown]\n# # T\n#\n# a $x$\n\n# %%\nx = 1\n# %% [markdown]\n# b\n"

    assert markdown_cells(src) == ["# T\n\na $x$", "b"]


def test_full_variants_are_slow() -> None:
    slow = {p.id for p in _params() if p.marks}

    assert slow == {f"{name}[full]" for name in FULL}
    assert FULL <= set(ARGS)


def test_docs_server_watches_examples() -> None:
    config = (EXAMPLES_DIR.parent / "mkdocs.yml").read_text()

    assert re.search(r"^watch:\n  - examples$", config, flags=re.MULTILINE)
