from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Sampler
from torchmetrics import Metric

import laplace
from laplace.utils.enums import LinkApprox, PredType, PriorStructure

__all__ = [
    "get_nll",
    "validate",
    "parameters_per_layer",
    "invsqrt_precision",
    "kron",
    "diagonal_add_scalar",
    "symeig",
    "block_diag",
    "expand_prior_precision",
    "SoDSampler",
]


def get_nll(out_dist: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(torch.log(out_dist), targets)


@torch.no_grad()
def validate(
    laplace: laplace.baselaplace.BaseLaplace,
    val_loader: DataLoader,
    loss: torchmetrics.Metric
    | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    pred_type: PredType | str = PredType.GLM,
    link_approx: LinkApprox | str = LinkApprox.PROBIT,
    n_samples: int = 100,
    dict_key_y: str = "labels",
) -> float:
    laplace.model.eval()
    assert callable(loss) or isinstance(loss, Metric)
    is_offline = not isinstance(loss, Metric)

    if is_offline:
        output_means, output_vars = list(), list()
        targets = list()

    for data in val_loader:
        if isinstance(data, MutableMapping):
            X, y = data, data[dict_key_y]
        else:
            X, y = data
            X = X.to(laplace._device)
        y = y.to(laplace._device)
        out = laplace(
            X,
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=n_samples,
            fitting=True,
        )

        if type(out) is tuple:
            if is_offline:
                output_means.append(out[0])
                output_vars.append(out[1])
                targets.append(y)
            else:
                try:
                    loss.update(*out, y)
                except TypeError:  # If the online loss only accepts 2 args
                    loss.update(out[0], y)
        else:
            if is_offline:
                output_means.append(out)
                targets.append(y)
            else:
                loss.update(out, y)

    if is_offline:
        if len(output_vars) == 0:
            preds, targets = torch.cat(output_means, dim=0), torch.cat(targets, dim=0)
            return loss(preds, targets).item()

        means, variances = torch.cat(output_means, dim=0), torch.cat(output_vars, dim=0)
        targets = torch.cat(targets, dim=0)
        return loss(means, variances, targets).item()
    else:
        # Aggregate since torchmetrics output n_classes values for the MSE metric
        return loss.compute().sum().item()


def parameters_per_layer(model: nn.Module) -> list[int]:
    """Get number of parameters per layer.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    params_per_layer : list[int]
    """
    return [np.prod(p.shape) for p in model.parameters()]


def invsqrt_precision(M: torch.Tensor) -> torch.Tensor:
    """Compute ``M^{-0.5}`` as a tridiagonal matrix.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    M_invsqrt : torch.Tensor
    """
    return _precision_to_scale_tril(M)


def _is_batchnorm(module: nn.Module) -> bool:
    if isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
        return True
    return False


def _is_valid_scalar(scalar: float | int | torch.Tensor) -> bool:
    if np.isscalar(scalar) and np.isreal(scalar):
        return True
    elif torch.is_tensor(scalar) and scalar.ndim <= 1:
        if scalar.ndim == 1 and len(scalar) != 1:
            return False
        return True
    return False


def kron(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Computes the Kronecker product between two tensors.

    Parameters
    ----------
    t1 : torch.Tensor
    t2 : torch.Tensor

    Returns
    -------
    kron_product : torch.Tensor
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


def diagonal_add_scalar(X: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Add scalar value `value` to diagonal of `X`.

    Parameters
    ----------
    X : torch.Tensor
    value : torch.Tensor or float

    Returns
    -------
    X_add_scalar : torch.Tensor
    """
    indices = torch.LongTensor([[i, i] for i in range(X.shape[0])], device=X.device)
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)


def symeig(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symetric eigendecomposition avoiding failure cases by
    adding and removing jitter to the diagonal.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    L : torch.Tensor
        eigenvalues
    W : torch.Tensor
        eigenvectors
    """
    try:
        L, W = torch.linalg.eigh(M, UPLO="U")
    except RuntimeError:  # did not converge
        logging.info("SYMEIG: adding jitter, did not converge.")
        # use W L W^T + I = W (L + I) W^T
        M = M + torch.eye(M.shape[0], device=M.device)
        try:
            L, W = torch.linalg.eigh(M, UPLO="U")
            L -= 1.0
        except RuntimeError:
            stats = f"diag: {M.diagonal()}, max: {M.abs().max()}, "
            stats = stats + f"min: {M.abs().min()}, mean: {M.abs().mean()}"
            logging.info(f"SYMEIG: adding jitter failed. Stats: {stats}")
            exit()
    # eigenvalues of symeig at least 0
    L = L.clamp(min=0.0)
    L = torch.nan_to_num(L)
    W = torch.nan_to_num(W)
    return L, W


def block_diag(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Compose block-diagonal matrix of individual blocks.

    Parameters
    ----------
    blocks : list[torch.Tensor]

    Returns
    -------
    M : torch.Tensor
    """
    P = sum([b.shape[0] for b in blocks])
    M = torch.zeros(P, P, dtype=blocks[0].dtype, device=blocks[0].device)
    p_cur = 0
    for block in blocks:
        p_block = block.shape[0]
        M[p_cur : p_cur + p_block, p_cur : p_cur + p_block] = block
        p_cur += p_block
    return M


class SoDSampler(Sampler):
    def __init__(self, N, M, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.indices = torch.tensor(rng.choice(list(range(N)), M, replace=False))

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def expand_prior_precision(prior_prec: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Expand prior precision to match the shape of the model parameters.

    Parameters
    ----------
    prior_prec : torch.Tensor 1-dimensional
        prior precision
    model : torch.nn.Module
        torch model with parameters that are regularized by prior_prec

    Returns
    -------
    expanded_prior_prec : torch.Tensor
        expanded prior precision has the same shape as model parameters
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    theta = parameters_to_vector(trainable_params)
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat(
            [
                delta * torch.ones_like(m).flatten()
                for delta, m in zip(prior_prec, trainable_params)
            ]
        )


def fix_prior_prec_structure(
    prior_prec_init: torch.Tensor,
    prior_structure: PriorStructure | str,
    n_layers: int,
    n_params: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a tensor of prior precision with the correct shape, depending on the
    choice of the prior structure type.

    Parameters
    ----------
    prior_prec_init: torch.Tensor
        the initial prior precision tensor (could be scalar)
    prior_structure: PriorStructure | str
        the choice of the prior structure type
    n_layers: int
    n_params: int
    device: torch.device

    Returns
    -------
    correct_prior_precision: torch.Tensor
    """
    if prior_structure == PriorStructure.SCALAR:
        prior_prec_init = torch.full((1,), prior_prec_init, device=device)
    elif prior_structure == PriorStructure.LAYERWISE:
        prior_prec_init = torch.full((n_layers,), prior_prec_init, device=device)
    elif prior_structure == PriorStructure.DIAG:
        prior_prec_init = torch.full((n_params,), prior_prec_init, device=device)
    else:
        raise ValueError(f"Invalid prior structure {prior_structure}.")
    return prior_prec_init


def normal_samples(
    mean: torch.Tensor,
    var: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Produce samples from a batch of Normal distributions either parameterized
    by a diagonal or full covariance given by `var`.

    Parameters
    ----------
    mean : torch.Tensor
        `(batch_size, output_dim)`
    var : torch.Tensor
        (co)variance of the Normal distribution
        `(batch_size, output_dim, output_dim)` or `(batch_size, output_dim)`
    generator : torch.Generator
        random number generator
    """
    assert mean.ndim == 2, "Invalid input shape of mean, should be 2-dimensional."
    _, output_dim = mean.shape
    randn_samples = torch.randn(
        (output_dim, n_samples),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )

    if mean.shape == var.shape:
        # diagonal covariance
        scaled_samples = var.sqrt().unsqueeze(-1) * randn_samples.unsqueeze(0)
        return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
    elif mean.shape == var.shape[:2] and var.shape[-1] == mean.shape[1]:
        # full covariance
        scale = torch.linalg.cholesky(var)
        scaled_samples = torch.matmul(
            scale, randn_samples.unsqueeze(0)
        )  # expand batch dim
        return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
    else:
        raise ValueError("Invalid input shapes.")
