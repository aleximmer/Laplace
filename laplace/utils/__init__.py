from laplace.utils.enums import (
    FunctionalApproximation,
    HessianStructure,
    LinkApprox,
    PredType,
    PriorStructure,
    SubsetOfWeights,
    TuningMethod,
)
from laplace.utils.feature_extractor import FeatureExtractor
from laplace.utils.functional_utils import (
    batch_size,
    concatenate_rows,
    draw_gaussian,
    preserve_model_gradients,
    regression_targets,
    same_configuration,
    select_rows,
    split_batch,
    subset_loader,
    to_device,
    training_indices,
)
from laplace.utils.heteroscedastic_utils import (
    HeteroscedasticGaussianNLLLoss,
    WhitenHeteroscedastic,
    as_joined_outputs,
    join_mean_variance,
    join_outputs_hook,
    joined_outputs,
    joins_outputs,
)
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
from laplace.utils.swag import fit_diagonal_swag_var
from laplace.utils.utils import (
    SoDSampler,
    _is_batchnorm,
    _is_valid_scalar,
    block_diag,
    diagonal_add_scalar,
    expand_prior_precision,
    fix_prior_prec_structure,
    get_nll,
    hyperparameters_equal,
    invsqrt_precision,
    kron,
    noise_per_output_update,
    normal_samples,
    parameters_per_layer,
    symeig,
    validate,
)

__all__ = [
    "get_nll",
    "validate",
    "parameters_per_layer",
    "invsqrt_precision",
    "kron",
    "diagonal_add_scalar",
    "symeig",
    "hyperparameters_equal",
    "block_diag",
    "normal_samples",
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
    "SoDSampler",
    "_is_batchnorm",
    "_is_valid_scalar",
    "expand_prior_precision",
    "fix_prior_prec_structure",
    "noise_per_output_update",
    "FeatureExtractor",
    "Kron",
    "KronDecomposed",
    "fit_diagonal_swag_var",
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
    "join_mean_variance",
    "as_joined_outputs",
    "join_outputs_hook",
    "joined_outputs",
    "joins_outputs",
    "WhitenHeteroscedastic",
    "HeteroscedasticGaussianNLLLoss",
    "SubsetOfWeights",
    "FunctionalApproximation",
    "HessianStructure",
    "Likelihood",
    "PredType",
    "LinkApprox",
    "TuningMethod",
    "PriorStructure",
]


def __getattr__(name: str) -> object:
    """Resolve `Likelihood` lazily; see `laplace.utils.enums`."""
    if name == "Likelihood":
        from laplace.likelihoods import Likelihood

        return Likelihood

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
