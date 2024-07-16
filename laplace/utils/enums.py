from enum import Enum


class SubsetOfWeights(str, Enum):
    ALL = "all"
    LAST_LAYER = "last_layer"
    SUBNETWORK = "subnetwork"


class HessianStructure(str, Enum):
    FULL = "full"
    KRON = "kron"
    DIAG = "diag"
    LOWRANK = "lowrank"
    GP = "gp"


class Likelihood(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    REWARD_MODELING = "reward_modeling"


class PredType(str, Enum):
    GLM = "glm"
    NN = "nn"
    GP = "gp"


class LinkApprox(str, Enum):
    MC = "mc"
    PROBIT = "probit"
    BRIDGE = "bridge"
    BRIDGE_NORM = "bridge_norm"


class TuningMethod(str, Enum):
    MARGLIK = "marglik"
    GRIDSEARCH = "gridsearch"


class PriorStructure(str, Enum):
    SCALAR = "scalar"
    DIAG = "diag"
    LAYERWISE = "layerwise"
