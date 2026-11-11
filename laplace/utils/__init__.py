import importlib
import warnings

from laplace.utils.feature_extractor import FeatureExtractor
from laplace.utils.matrix import Kron, KronDecomposed
from laplace.utils.metrics import (
    RunningBernoulliNLLMetric,
    RunningGaussianNLLMetric,
    RunningNLLMetric,
)
from laplace.utils.subnetmask import (
    LargestMagnitudeSubnetMask,
    LargestVarianceDiagLaplaceSubnetMask,
    LargestVarianceSWAGSubnetMask,
    LastLayerSubnetMask,
    ModuleNameSubnetMask,
    ParamNameSubnetMask,
    RandomSubnetMask,
    SubnetMask,
)

__all__ = [
    "FeatureExtractor",
    "Kron",
    "KronDecomposed",
    "SubnetMask",
    "RandomSubnetMask",
    "LargestMagnitudeSubnetMask",
    "LargestVarianceDiagLaplaceSubnetMask",
    "LargestVarianceSWAGSubnetMask",
    "ParamNameSubnetMask",
    "ModuleNameSubnetMask",
    "LastLayerSubnetMask",
    "RunningNLLMetric",
    "RunningBernoulliNLLMetric",
    "RunningGaussianNLLMetric",
]

# Names exported here before 0.6, mapped to the module that defines them. They
# still resolve, with a `DeprecationWarning`. The enums are public in `laplace`;
# the rest are internal.
_DEPRECATED = {
    **dict.fromkeys(
        [
            "FunctionalApproximation",
            "HessianStructure",
            "LinkApprox",
            "PredType",
            "PriorStructure",
            "SubsetOfWeights",
            "TuningMethod",
        ],
        "laplace.utils.enums",
    ),
    **dict.fromkeys(
        [
            "batch_size",
            "concatenate_rows",
            "draw_gaussian",
            "preserve_model_gradients",
            "regression_targets",
            "same_configuration",
            "select_rows",
            "split_batch",
            "subset_loader",
            "to_device",
            "training_indices",
        ],
        "laplace.utils.functional_utils",
    ),
    **dict.fromkeys(
        [
            "HeteroscedasticGaussianNLLLoss",
            "WhitenHeteroscedastic",
            "as_joined_outputs",
            "join_mean_variance",
            "join_outputs_hook",
            "joined_outputs",
            "joins_outputs",
        ],
        "laplace.utils.heteroscedastic_utils",
    ),
    "fit_diagonal_swag_var": "laplace.utils.swag",
    **dict.fromkeys(
        [
            "SoDSampler",
            "_is_batchnorm",
            "_is_valid_scalar",
            "block_diag",
            "diagonal_add_scalar",
            "expand_prior_precision",
            "fix_prior_prec_structure",
            "get_nll",
            "hyperparameters_equal",
            "invsqrt_precision",
            "kron",
            "noise_per_output_update",
            "normal_samples",
            "parameters_per_layer",
            "symeig",
            "validate",
        ],
        "laplace.utils.utils",
    ),
}


def __getattr__(name: str) -> object:
    """Resolve `Likelihood` lazily (see `laplace.utils.enums`) and the deprecated names."""
    if name == "Likelihood":
        from laplace.likelihoods import Likelihood

        return Likelihood

    if name in _DEPRECATED:
        module = _DEPRECATED[name]

        if module == "laplace.utils.enums":
            hint = "Import it from `laplace` instead."
        else:
            hint = "It is internal and may change without notice."

        warnings.warn(
            f"`laplace.utils.{name}` is deprecated and will be removed. {hint}",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(importlib.import_module(module), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
