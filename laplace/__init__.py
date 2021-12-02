"""
.. include:: ../README.md

.. include:: ../regression_example.md
"""
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

from laplace.baselaplace import BaseLaplace, ParametricLaplace, FullLaplace, KronLaplace, DiagLaplace, FunctionalLaplace
from laplace.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace, FunctionalLLLaplace
from laplace.laplace import Laplace

__all__ = ['Laplace',  # direct access to all Laplace classes via unified interface
           'BaseLaplace', 'ParametricLaplace', 'FunctionalLaplace', # base-class and its (first-level) subclasses
           'FullLaplace', 'KronLaplace', 'DiagLaplace',  # all-weights
           'LLLaplace',  # base-class last-layer
           'FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace', 'FunctionalLLLaplace'] # last-layer
