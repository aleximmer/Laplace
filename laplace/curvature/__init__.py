import logging

from laplace.curvature.curvature import CurvatureInterface, EFInterface, GGNInterface

try:
    from laplace.curvature.backpack import BackPackEF, BackPackGGN, BackPackInterface
except ModuleNotFoundError:
    logging.info("Backpack backend not available.")

try:
    from laplace.curvature.asdfghjkl import (
        AsdfghjklEF,
        AsdfghjklGGN,
        AsdfghjklHessian,
        AsdfghjklInterface,
    )
except ModuleNotFoundError:
    logging.info("Asdfghjkl backend not available.")

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
    "AsdfghjklInterface",
    "AsdfghjklGGN",
    "AsdfghjklEF",
    "AsdfghjklHessian",
    "AsdlInterface",
    "AsdlGGN",
    "AsdlEF",
    "AsdlHessian",
    "CurvlinopsInterface",
    "CurvlinopsGGN",
    "CurvlinopsEF",
    "CurvlinopsHessian",
]
