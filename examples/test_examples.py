"""Smoke tests that run every example script and the Python snippets of every doc page.

Each one runs in a fresh subprocess, inside a temporary working directory, and must
exit cleanly. New scripts and pages are picked up automatically. Run with:

    pytest examples/test_examples.py -m "not slow"

The `slow` ones are doc pages that train or evaluate on full datasets. CI runs them
on a schedule.
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
    "calibration_example.py": ["--n_train", "256", "--n_test", "256"],
    "calibration_gp_example.py": [
        *("--n_train", "1024", "--n_test", "256"),
        *("--epochs", "1", "--subset_sizes", "50"),
    ],
    "huggingface_example.py": ["--model", "hf-internal-testing/tiny-random-gpt2"],
}

# Doc snippets take no arguments, so these run on the full data.
SLOW = {"calibration_example.md", "calibration_gp_example.md"}

# Doc pages whose snippets are excerpts, not a complete program.
SKIP = {
    "index.md": "excerpts that call placeholder functions",
    "expectation_example.md": "excerpts of examples/expectation_example.py",
}

PYTHON_BLOCK = re.compile(
    r"^```python[^\n]*\n(.*?)^```", flags=re.DOTALL | re.MULTILINE
)


def _scripts() -> list[str]:
    return sorted(
        p.name
        for p in EXAMPLES_DIR.glob("*.py")
        if p.name not in ("conftest.py", "test_examples.py")
    )


def _pages() -> list[str]:
    return sorted(
        p.name for p in DOCS_DIR.glob("*.md") if PYTHON_BLOCK.search(p.read_text())
    )


def _params(names: list[str]) -> list:
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in SLOW else name
        for name in names
    ]


def extract_snippets(markdown: str) -> str:
    """Join the top-level ```` ```python ```` blocks of a doc page into one script."""
    return "\n\n".join(PYTHON_BLOCK.findall(markdown))


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


@pytest.mark.parametrize("script", _params(_scripts()))
def test_example_runs(script: str, tmp_path: Path) -> None:
    timeout = SLOW_TIMEOUT if script in SLOW else TIMEOUT
    result = run_example(EXAMPLES_DIR / script, tmp_path, ARGS.get(script, ()), timeout)

    _check(result)


@pytest.mark.parametrize("page", _params(_pages()))
def test_doc_snippets_run(page: str, tmp_path: Path) -> None:
    if page in SKIP:
        pytest.skip(SKIP[page])

    script = tmp_path / page.replace(".md", ".py")
    script.write_text(extract_snippets((DOCS_DIR / page).read_text()))

    timeout = SLOW_TIMEOUT if page in SLOW else TIMEOUT
    _check(run_example(script, tmp_path, timeout=timeout))


def test_all_examples_collected() -> None:
    scripts, pages = _scripts(), _pages()

    assert "regression_example.py" in scripts
    assert "test_examples.py" not in scripts and "conftest.py" not in scripts
    assert "regression_example.md" in pages and "devs_guide.md" not in pages

    # No stale entries for renamed or deleted files
    assert set(ARGS) <= set(scripts)
    assert set(SKIP) <= set(pages)
    assert SLOW <= set(scripts) | set(pages)


def test_extract_snippets_skips_other_blocks() -> None:
    markdown = (
        "Text\n\n```python\nx = 1\n```\n\n```bash\nls\n```\n\n"
        '```python title="b.py"\ny = x + 1\n```\n\n'
        "!!! note\n\n    ```python\n    z = 2\n    ```\n"
    )

    assert extract_snippets(markdown) == "x = 1\n\n\ny = x + 1\n"


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
