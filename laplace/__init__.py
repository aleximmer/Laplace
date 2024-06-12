"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
.. include:: ../examples/reward_modeling_example.md
"""

REGRESSION = "regression"
CLASSIFICATION = "classification"

from laplace.baselaplace import (
    BaseLaplace,
    DiagLaplace,
    FullLaplace,
    FunctionalLaplace,
    KronLaplace,
    LowRankLaplace,
)
from laplace.laplace import Laplace
from laplace.lllaplace import (
    DiagLLLaplace,
    FullLLLaplace,
    FunctionalLLLaplace,
    KronLLLaplace,
    LLLaplace,
)
from laplace.marglik_training import marglik_training
from laplace.subnetlaplace import DiagSubnetLaplace, FullSubnetLaplace, SubnetLaplace

__all__ = [
    "Laplace",  # direct access to all Laplace classes via unified interface
    "BaseLaplace",
    "ParametricLaplace",  # base-class and its (first-level) subclasses
    "FullLaplace",
    "KronLaplace",
    "DiagLaplace",
    "FunctionalLaplace",
    "LowRankLaplace",  # all-weights
    "LLLaplace",  # base-class last-layer
    "FullLLLaplace",
    "KronLLLaplace",
    "DiagLLLaplace",
    "FunctionalLLLaplace",  # last-layer
    "SubnetLaplace",  # base-class subnetwork
    "FullSubnetLaplace",
    "DiagSubnetLaplace",  # subnetwork
    "marglik_training",
]  # methods
