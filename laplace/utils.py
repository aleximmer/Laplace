import logging
from typing import Union
import numpy as np
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def parameters_per_layer(model):
    return [np.prod(p.shape) for p in model.parameters()]


def invsqrt_precision(M):
    return _precision_to_scale_tril(M)


def is_batchnorm(module):
    if isinstance(module, BatchNorm1d) or \
        isinstance(module, BatchNorm2d) or \
            isinstance(module, BatchNorm3d):
        return True
    return False


def _is_valid_scalar(scalar: Union[float, int, torch.Tensor]) -> bool:
    if np.isscalar(scalar) and np.isreal(scalar):
        return True
    elif torch.is_tensor(scalar) and scalar.ndim <= 1:
        if scalar.ndim == 1 and len(scalar) != 1:
            return False
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


def diagonal_add_scalar(X, value):
    if not X.device == torch.device('cpu'):
        indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    else:
        indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)


def symeig(M):
    """Symetric eigendecomposition avoiding failure cases by
    adding and removing jitter to the diagonal

    returns eigenvalues (l) and eigenvectors (W)
    """
    # could make double to get more precise computation
    # M = M.double()
    # and then below return L.float(), W.float()
    try:
        L, W = torch.symeig(M, eigenvectors=True)
    except RuntimeError:  # did not converge
        logging.info('SYMEIG: adding jitter, did not converge.')
        # use W L W^T + I = W (L + I) W^T
        M = diagonal_add_scalar(M, value=1.)
        try:
            L, W = torch.symeig(M, eigenvectors=True)
            L -= 1.
        except RuntimeError:
            stats = f'diag: {M.diagonal()}, max: {M.abs().max()}, '
            stats = stats + f'min: {M.abs().min()}, mean: {M.abs().mean()}'
            logging.info(f'SYMEIG: adding jitter failed. Stats: {stats}')
            exit()
    # eigenvalues of symeig at least 0
    L = L.clamp(min=0.0)
    L[torch.isnan(L)] = 0.0
    W[torch.isnan(W)] = 0.0
    return L, W


def block_diag(blocks):
    P = sum([b.shape[0] for b in blocks])
    M = torch.zeros(P, P)
    p_cur = 0
    for block in blocks:
        p_block = block.shape[0]
        M[p_cur:p_cur+p_block, p_cur:p_cur+p_block] = block
        p_cur += p_block
    return M
