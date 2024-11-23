from enum import Enum


class SubsetOfWeights(str, Enum):
    """Valid options for `subset_of_weights`."""

    ALL = "all"
    """All-layer, all-parameter Laplace."""

    LAST_LAYER = "last_layer"
    """Last-layer Laplace."""

    SUBNETWORK = "subnetwork"
    """Subnetwork Laplace."""


class HessianStructure(str, Enum):
    """Valid options for `hessian_structure`."""

    FULL = "full"
    """Full Hessian (generally very expensive)."""

    KRON = "kron"
    """Kronecker-factored Hessian (preferrable)."""

    DIAG = "diag"
    """Diagonal Hessian."""

    LOWRANK = "lowrank"
    """Low-rank Hessian."""

    GP = "gp"
    """Functional Laplace."""


class Likelihood(str, Enum):
    """Valid options for `likelihood`."""

    REGRESSION = "regression"
    """Homoskedastic regression, assuming `loss_fn = nn.MSELoss()`."""

    CLASSIFICATION = "classification"
    """Classification, assuming `loss_fn = nn.CrossEntropyLoss()`."""

    REWARD_MODELING = "reward_modeling"
    """Bradley-Terry likelihood, for preference learning / reward modeling."""


class PredType(str, Enum):
    """Valid options for `pred_type`."""

    GLM = "glm"
    """Linearized, closed-form predictive."""

    NN = "nn"
    """Monte-Carlo predictive on the NN's weights."""

    GP = "gp"
    """Gaussian-process predictive, done by inverting the kernel matrix."""


class LinkApprox(str, Enum):
    """Valid options for `link_approx`.
    Only works with `likelihood = Likelihood.CLASSIFICATION`.
    """

    MC = "mc"
    """Monte-Carlo approximation in the function space on top of the GLM predictive."""

    PROBIT = "probit"
    """Closed-form multiclass probit approximation."""

    BRIDGE = "bridge"
    """Closed-form Laplace Bridge approximation."""

    BRIDGE_NORM = "bridge_norm"
    """Closed-form Laplace Bridge approximation with normalization factor.
    Preferable to `BRIDGE`."""


class TuningMethod(str, Enum):
    """Valid options for the `method` parameter in `optimize_prior_precision`."""

    MARGLIK = "marglik"
    """Marginal-likelihood loss via SGD. Does not require validation data."""

    GRIDSEARCH = "gridsearch"
    """Grid search. Requires validation data."""


class PriorStructure(str, Enum):
    """Valid options for the `prior_structure` in `optimize_prior_precision`."""

    SCALAR = "scalar"
    """Scalar prior precision \\( \\tau I, \\tau \\in \\mathbf{R} \\)."""

    DIAG = "diag"
    """Scalar prior precision \\( \\tau \\in \\mathbb{R}^p \\)."""

    LAYERWISE = "layerwise"
    """Layerwise prior precision, i.e. a single scalar prior precision for each block 
    (corresponding to each the NN's layer) of the diagonal prior-precision matrix.."""
