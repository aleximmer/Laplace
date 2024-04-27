"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
"""

from laplace.baselaplace import (
    BaseLaplace,
    ParametricLaplace,
    FullLaplace,
    KronLaplace,
    DiagLaplace,
    LowRankLaplace,
    SubsetOfWeights,
    HessianStructure,
    Likelihood,
    PredType,
    LinkApprox,
    TuningMethod,
    PriorStructure,
)
from laplace.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace
from laplace.subnetlaplace import SubnetLaplace, FullSubnetLaplace, DiagSubnetLaplace
from laplace.laplace import Laplace
from laplace.marglik_training import marglik_training

__all__ = [
    'Laplace',  # direct access to all Laplace classes via unified interface
    'BaseLaplace',
    'ParametricLaplace',  # base-class and its (first-level) subclasses
    'FullLaplace',
    'KronLaplace',
    'DiagLaplace',
    'LowRankLaplace',  # all-weights
    'LLLaplace',  # base-class last-layer
    'FullLLLaplace',
    'KronLLLaplace',
    'DiagLLLaplace',  # last-layer
    'SubnetLaplace',  # base-class subnetwork
    'FullSubnetLaplace',
    'DiagSubnetLaplace',  # subnetwork
    'marglik_training',
    # Enums
    'SubsetOfWeights',
    'HessianStructure',
    'Likelihood',
    'PredType',
    'LinkApprox',
    'TuningMethod',
    'PriorStructure',
]
