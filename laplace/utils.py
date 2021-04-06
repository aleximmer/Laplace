import numpy as np
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def parameters_per_layer(model):
    return [np.prod(p.shape) for p in model.parameters()]


def invsqrt_precision(M):
    return _precision_to_scale_tril(M)
