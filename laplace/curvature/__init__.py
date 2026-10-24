import logging
from importlib.util import find_spec

from laplace.curvature.curvature import CurvatureInterface, EFInterface, GGNInterface

try:
    from laplace.curvature.backpack import BackPackEF, BackPackGGN, BackPackInterface
except ModuleNotFoundError:
    logging.info("Backpack backend not available.")

try:
    from laplace.curvature.asdl import AsdlEF, AsdlGGN, AsdlHessian, AsdlInterface
except ModuleNotFoundError:
    logging.info("ASDL backend not available.")

try:
    from laplace.curvature.curvlinops import (
        CurvlinopsEF,
        CurvlinopsGGN,
        CurvlinopsHessian,
        CurvlinopsInterface,
    )
except ModuleNotFoundError:
    logging.info("Curvlinops backend not available.")

__all__ = [
    "CurvatureInterface",
    "GGNInterface",
    "EFInterface",
    "BackPackInterface",
    "BackPackGGN",
    "BackPackEF",
    "AsdlInterface",
    "AsdlGGN",
    "AsdlEF",
    "AsdlHessian",
    "CurvlinopsInterface",
    "CurvlinopsGGN",
    "CurvlinopsEF",
    "CurvlinopsHessian",
]

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
                "AsdlInterface",
            ]
        )
    except ModuleNotFoundError:
        logging.info("Asdfghjkl backend not available.")
