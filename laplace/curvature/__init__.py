import logging

from laplace.curvature.curvature import CurvatureInterface, GGNInterface, EFInterface

try:
    from laplace.curvature.backpack import BackPackGGN, BackPackEF, BackPackInterface
    from laplace.curvature.augmented_backpack import AugBackPackInterface, AugBackPackGGN
except ModuleNotFoundError:
    logging.info('Backpack not available.')

try:
    from laplace.curvature.asdl import AsdlGGN, AsdlEF, AsdlInterface
    from laplace.curvature.augmented_asdl import AugAsdlInterface, AugAsdlGGN, AugAsdlEF
except ModuleNotFoundError:
    logging.info('asdfghjkl backend not available.')

__all__ = ['CurvatureInterface', 'GGNInterface', 'EFInterface',
           'BackPackInterface', 'BackPackGGN', 'BackPackEF',
           'AsdlInterface', 'AsdlGGN', 'AsdlEF',
           'AugBackPackInterface', 'AugBackPackGGN',
           'AugAsdlInterface', 'AugAsdlGGN', 'AugAsdlEF']
