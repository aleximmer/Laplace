"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
"""
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

from laplace.baselaplace import (
    BaseLaplace, ParametricLaplace, FullLaplaceBase, KronLaplaceBase, DiagLaplaceBase,
    ALLaplace, FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace)
from laplace.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace
from laplace.laplace import Laplace
from laplace.marglik_training import marglik_training

__all__ = ['Laplace',  # direct access to all Laplace classes via unified interface
           'BaseLaplace', 'ParametricLaplace',  # base-class and its (first-level) subclasses
           'FullLaplaceBase', 'KronLaplaceBase', 'DiagLaplaceBase',  # base-classes for different Hessian structures
           'ALLaplace',  # base-class all-weights
           'FullLaplace', 'KronLaplace', 'DiagLaplace', 'LowRankLaplace',  # all-weights
           'LLLaplace',  # base-class last-layer
           'FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace',  # last-layer
           'marglik_training']  # methods
