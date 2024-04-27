import logging

from laplace.curvature.curvature import CurvatureInterface, GGNInterface, EFInterface

try:
    from laplace.curvature.backpack import BackPackGGN, BackPackEF, BackPackInterface
except ModuleNotFoundError:
    logging.info('Backpack backend not available.')

try:
    from laplace.curvature.asdfghjkl import (
        AsdfghjklHessian,
        AsdfghjklGGN,
        AsdfghjklEF,
        AsdfghjklInterface,
    )
except ModuleNotFoundError:
    logging.info('Asdfghjkl backend not available.')

try:
    from laplace.curvature.asdl import AsdlHessian, AsdlGGN, AsdlEF, AsdlInterface
except ModuleNotFoundError:
    logging.info('ASDL backend not available.')

try:
    from laplace.curvature.curvlinops import (
        CurvlinopsHessian,
        CurvlinopsGGN,
        CurvlinopsEF,
        CurvlinopsInterface,
    )
except ModuleNotFoundError:
    logging.info('Curvlinops backend not available.')

__all__ = [
    'CurvatureInterface',
    'GGNInterface',
    'EFInterface',
    'BackPackInterface',
    'BackPackGGN',
    'BackPackEF',
    'AsdfghjklInterface',
    'AsdfghjklGGN',
    'AsdfghjklEF',
    'AsdfghjklHessian',
    'AsdlInterface',
    'AsdlGGN',
    'AsdlEF',
    'AsdlHessian',
    'CurvlinopsInterface',
    'CurvlinopsGGN',
    'CurvlinopsEF',
    'CurvlinopsHessian',
]
