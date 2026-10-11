"""Run the published function-space examples as integration checks."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["ella", "valla"])
def test_functional_example_runs(name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", f"examples.{name}"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert name.casefold() in result.stdout.casefold()
