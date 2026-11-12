"""Signature snapshot of the public API.

`python -m tests.public_api_snapshot` writes `public_api_signatures.json`. Run it
after adding a public callable. It appends the new entries and leaves existing ones
alone, so a breaking change still fails `tests/test_public_api.py`.
"""

import enum
import inspect
import json
from pathlib import Path

import laplace
import laplace.curvature
import laplace.utils

SNAPSHOT_PATH = Path(__file__).parent / "public_api_signatures.json"

CLASSES = [
    "FullLaplace",
    "KronLaplace",
    "DiagLaplace",
    "LowRankLaplace",
    "FunctionalLaplace",
    "FullLLLaplace",
    "KronLLLaplace",
    "DiagLLLaplace",
    "FunctionalLLLaplace",
    "FullSubnetLaplace",
    "DiagSubnetLaplace",
    "ELLA",
    "VaLLA",
]


# Prefix of the snapshot keys for each module's exports
MODULES = {"": laplace, "utils.": laplace.utils, "curvature.": laplace.curvature}


def resolve(qualname: str) -> object:
    """Look up `"Class.method"`, `"function"` or `"utils.Class"` on `laplace`."""
    obj = laplace

    for part in qualname.split("."):
        obj = getattr(obj, part)

    return obj


def _describe(obj: object) -> list[dict]:
    """Parameters of `obj` as dicts, without `self`."""
    params = list(inspect.signature(obj).parameters.values())

    if params and params[0].name == "self":
        params = params[1:]

    return [
        {
            "name": p.name,
            "kind": p.kind.name,
            "has_default": p.default is not inspect.Parameter.empty,
        }
        for p in params
    ]


def snapshot() -> dict[str, list[dict]]:
    """Signatures of the public functions and methods."""
    from tests.test_public_api import PUBLIC_COMMON, PUBLIC_EXTRA

    names = PUBLIC_COMMON.union(*PUBLIC_EXTRA.values())
    result = {}

    # Every exported function and constructor, except the enums
    for prefix, module in MODULES.items():
        for export in module.__all__:
            obj = getattr(module, export)

            if inspect.isfunction(obj):
                result[prefix + export] = _describe(obj)
            elif inspect.isclass(obj) and not issubclass(obj, enum.Enum):
                result[f"{prefix}{export}.__init__"] = _describe(obj.__init__)

    for cls_name in CLASSES:
        cls = getattr(laplace, cls_name)

        for name in sorted(names):
            static = inspect.getattr_static(cls, name, None)

            # Skips properties, plain attributes and instance-only names
            if inspect.isfunction(static):
                result[f"{cls_name}.{name}"] = _describe(getattr(cls, name))

    return result


if __name__ == "__main__":
    new = snapshot()
    old = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}

    SNAPSHOT_PATH.write_text(json.dumps({**new, **old}, indent=1) + "\n")
