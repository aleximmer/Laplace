import numpy as np
import torch
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def parameters_per_layer(model):
    return [np.prod(p.shape) for p in model.parameters()]


def invsqrt_precision(M):
    return _precision_to_scale_tril(M)


def _is_valid_scalar(scalar):
    if np.isscalar(scalar) and np.isreal(scalar):
        return True
    elif torch.is_tensor(scalar) and scalar.ndim <= 1:
        return True
    return False


def kron(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2