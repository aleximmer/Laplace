import importlib
import logging
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING

from laplace.curvature.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.curvature.curvlinops import (
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
    CurvlinopsInterface,
)

if TYPE_CHECKING:
    # Let type checkers and IDEs see the lazily imported classes below.
    from laplace.curvature.asdl import AsdlEF as AsdlEF
    from laplace.curvature.asdl import AsdlGGN as AsdlGGN
    from laplace.curvature.asdl import AsdlHessian as AsdlHessian
    from laplace.curvature.asdl import AsdlInterface as AsdlInterface
    from laplace.curvature.backpack import BackPackEF as BackPackEF
    from laplace.curvature.backpack import BackPackGGN as BackPackGGN
    from laplace.curvature.backpack import BackPackInterface as BackPackInterface

# Backends from optional packages, mapped to the module defining them. They are
# imported on first access, so `import laplace` works without the packages.
_OPTIONAL_BACKENDS = {
    "AsdlInterface": "laplace.curvature.asdl",
    "AsdlGGN": "laplace.curvature.asdl",
    "AsdlEF": "laplace.curvature.asdl",
    "AsdlHessian": "laplace.curvature.asdl",
    "BackPackInterface": "laplace.curvature.backpack",
    "BackPackGGN": "laplace.curvature.backpack",
    "BackPackEF": "laplace.curvature.backpack",
}

# The third-party package each optional backend module needs.
_OPTIONAL_PACKAGES = {
    "laplace.curvature.asdl": "asdl",
    "laplace.curvature.backpack": "backpack",
}


def __getattr__(name: str) -> type:
    """Import an optional backend class when it is first accessed.

    Parameters
    ----------
    name : str
        Attribute name looked up on `laplace.curvature`.

    Returns
    -------
    type
        The backend class.

    Raises
    ------
    ImportError
        If the package the backend needs is not installed.
    AttributeError
        If `name` is not an optional backend.
    """
    if name not in _OPTIONAL_BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_OPTIONAL_BACKENDS[name])

    return getattr(module, name)


def __dir__() -> list[str]:
    """List the module's names, including optional backends that are installed.

    Returns
    -------
    list[str]
        Sorted attribute names. Missing optional backends are left out, so
        `"BackPackGGN" in dir(laplace.curvature)` checks whether it can be used.
    """
    return sorted(set(globals()) | set(__all__))


def _is_optional_backend(cls: type, name: str) -> bool:
    """Check whether `cls` subclasses an optional backend class without importing it.

    If the module defining `name` was never imported, nothing can subclass it,
    so the check needs no import and works when the package is missing.

    Parameters
    ----------
    cls : type
        Class to check, usually a curvature backend.
    name : str
        Name of an optional backend class, e.g. `"BackPackInterface"`.

    Returns
    -------
    bool
        Whether `cls` is a subclass of the class called `name`.
    """
    module = sys.modules.get(_OPTIONAL_BACKENDS[name])

    return module is not None and issubclass(cls, getattr(module, name))


__all__ = [
    "CurvatureInterface",
    "GGNInterface",
    "EFInterface",
    "CurvlinopsInterface",
    "CurvlinopsGGN",
    "CurvlinopsEF",
    "CurvlinopsHessian",
]

__all__.extend(
    name
    for name, module in _OPTIONAL_BACKENDS.items()
    if find_spec(_OPTIONAL_PACKAGES[module]) is not None
)

if find_spec("asdfghjkl") is None:
    logging.info(
        """Asdfghjkl backend not available since the old asdfghjkl dependency """
        """is not installed. If you want to use it, run: """
        """pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl"""
    )
else:
    try:
        from laplace.curvature.asdfghjkl import (
            AsdfghjklEF,  # noqa: F401
            AsdfghjklGGN,  # noqa: F401
            AsdfghjklHessian,  # noqa: F401
            AsdfghjklInterface,  # noqa: F401
        )

        __all__.extend(
            [
                "AsdfghjklInterface",
                "AsdfghjklGGN",
                "AsdfghjklEF",
                "AsdfghjklHessian",
            ]
        )
    except ModuleNotFoundError:
        logging.info("Asdfghjkl backend not available.")
