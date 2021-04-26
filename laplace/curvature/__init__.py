import logging

from laplace.curvature.curvature import CurvatureInterface

try:
    from laplace.curvature.backpack import BackPackGGN, BackPackEF, BackPackInterface
except ModuleNotFoundError:
    logging.info('Backpack not available.')

try:
    from laplace.curvature.kazuki import KazukiGGN, KazukiEF, KazukiInterface
except ModuleNotFoundError:
    logging.info('Kazuki backend not available.')
