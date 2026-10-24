"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
.. include:: ../examples/huggingface_example.md
.. include:: ../examples/reward_modeling_example.md
"""

from laplace.baselaplace import (
    ELLA,
    BaseFunctionalLaplace,
    BaseLaplace,
    DiagLaplace,
    FullLaplace,
    FunctionalLaplace,
    KronLaplace,
    LowRankLaplace,
    ParametricLaplace,
    VaLLA,
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
from laplace.utils.enums import (
    FunctionalApproximation,
    HessianStructure,
    Likelihood,
    LinkApprox,
    PredType,
    PriorStructure,
    SubsetOfWeights,
    TuningMethod,
)

__all__ = [
    "ELLA",
    "VaLLA",
    "Laplace",  # direct access to all Laplace classes via unified interface
    "BaseLaplace",
    "BaseFunctionalLaplace",
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
    # Enums
    "SubsetOfWeights",
    "FunctionalApproximation",
    "HessianStructure",
    "Likelihood",
    "PredType",
    "LinkApprox",
    "TuningMethod",
    "PriorStructure",
]
