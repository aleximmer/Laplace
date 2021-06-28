"""
.. include:: ../README.md

.. include:: ../regression_example.md
"""
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

from laplace.baselaplace import BaseLaplace, FullLaplace, KronLaplace, DiagLaplace
from laplace.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace
from laplace.laplace import Laplace

__all__ = ['Laplace',  # direct access to all Laplace classes via unified interface
           'BaseLaplace',  # base-class
           'FullLaplace', 'KronLaplace', 'DiagLaplace',  # all-weights
           'LLLaplace',  # base-class last-layer
           'FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace'] # last-layer
