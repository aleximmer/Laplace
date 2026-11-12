"""Helpers for networks that predict a mean and a variance, as trained with
`nn.GaussianNLLLoss`, see
[`HeteroscedasticGaussianLikelihood`][laplace.HeteroscedasticGaussianLikelihood].

The library computes Jacobians and curvatures of one output tensor, so the pair
`(mean, var)` of such a network is joined into one tensor `(batch, ..., 2 * K)`, the
mean followed by the variance.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from math import log, pi, sqrt
from typing import Any, Callable

import torch
from torch import nn

__all__ = [
    "join_mean_variance",
    "as_joined_outputs",
    "join_outputs_hook",
    "joined_outputs",
    "joins_outputs",
    "WhitenHeteroscedastic",
    "HeteroscedasticGaussianNLLLoss",
]


def join_mean_variance(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Concatenate the mean and the variance of a heteroscedastic Gaussian into the
    outputs `(batch, ..., 2 * K)` that `HeteroscedasticGaussianLikelihood` takes.

    As in `nn.GaussianNLLLoss`, the variance has the shape of the mean `(batch, ...,
    K)`, or it is shared by the `K` outputs of a datum, with the shape `(batch, ...,
    1)` or `(batch, ...)`. A shared variance is expanded to the shape of the mean.

    Parameters
    ----------
    mean : torch.Tensor
        mean `(batch, ..., K)`
    var : torch.Tensor
        variance `(batch, ..., K)`, `(batch, ..., 1)`, or `(batch, ...)`

    Returns
    -------
    f : torch.Tensor
        `(batch, ..., 2 * K)`, the mean followed by the variance.

    Raises
    ------
    ValueError
        if the variance has none of these shapes.
    """
    if var.shape != mean.shape:
        if var.shape == mean.shape[:-1]:
            var = var.unsqueeze(-1)
        elif var.shape[:-1] != mean.shape[:-1] or var.shape[-1] != 1:
            raise ValueError(
                f"The variance has shape {tuple(var.shape)}, which fits neither the "
                f"mean {tuple(mean.shape)} nor a variance shared by its outputs, "
                f"{tuple(mean.shape[:-1]) + (1,)} or {tuple(mean.shape[:-1])}."
            )

    return torch.cat([mean, var.expand_as(mean)], dim=-1)


def as_joined_outputs(f: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> Any:
    """Join a `(mean, var)` pair with `join_mean_variance`, and return anything else
    as it is, e.g. outputs that are already joined.

    Parameters
    ----------
    f : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        model outputs.

    Returns
    -------
    f : Any
        the joined outputs `(batch, ..., 2 * K)` for a pair, else `f`.
    """
    if isinstance(f, (tuple, list)) and len(f) == 2:
        return join_mean_variance(*f)

    return f


def join_outputs_hook(module: nn.Module, args: Any, output: Any) -> torch.Tensor | Any:
    """Forward hook that joins the `(mean, var)` of a heteroscedastic network into
    one tensor `(batch, ..., 2 * K)`, with the variance detached.

    The library computes Jacobians and curvatures of a single output tensor. The
    detached variance keeps the Laplace approximation on the mean: the variance has a
    zero Jacobian, and the loss has no gradient w.r.t. it.

    Parameters
    ----------
    module : nn.Module
        the model.
    args : Any
        the inputs of the forward call, unused.
    output : Any
        what the model returned, the pair or already joined outputs.

    Returns
    -------
    f : torch.Tensor
        the joined outputs.
    """
    f = as_joined_outputs(output)

    if not torch.is_tensor(f):
        return output

    mean, var = f.chunk(2, dim=-1)

    return torch.cat([mean, var.detach()], dim=-1)


@contextmanager
def joined_outputs(model: nn.Module) -> Iterator[None]:
    """Join the `(mean, var)` of `model` into one tensor while the context is active,
    with `join_outputs_hook`.

    The hook is removed on exit, so the model returns its pair again, and its
    parameter names never change. A nested context keeps the outer hook, which it
    finds by a flag on the model.

    Parameters
    ----------
    model : nn.Module
        a model that returns the pair `(mean, var)`.
    """
    if getattr(model, "_joins_outputs", False):
        yield
        return

    handle = model.register_forward_hook(join_outputs_hook)
    model._joins_outputs = True

    try:
        yield
    finally:
        handle.remove()
        del model._joins_outputs


def joins_outputs(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a method of a Laplace approximation so that it runs inside
    `joined_outputs(self.model)` if the likelihood is heteroscedastic.

    Parameters
    ----------
    method : Callable
        a method of `BaseLaplace` that runs the model, e.g. `fit`.

    Returns
    -------
    wrapper : Callable
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Imported here, since the likelihoods import this module
        from laplace.likelihoods import _is_heteroscedastic

        if not _is_heteroscedastic(self.likelihood):
            return method(self, *args, **kwargs)

        with joined_outputs(self.model):
            return method(self, *args, **kwargs)

    return wrapper


class WhitenHeteroscedastic(nn.Module):
    """Divide the mean of a heteroscedastic network by its predicted noise std.

    The noise std is detached, so the plain squared error of the whitened mean is
    twice the negative log likelihood of the mean with the variance held fixed, up to
    a constant. Its GGN, Hessian, and empirical Fisher are then those of the Laplace
    approximation over the mean, see `HeteroscedasticGaussianLikelihood`.

    Parameters
    ----------
    model : nn.Module
        the model, which returns the pair `(mean, var)` or the joined outputs.
    eps : float
        lower bound of the variance.
    """

    def __init__(self, model: nn.Module, eps: float) -> None:
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, x: Any) -> torch.Tensor:
        """Compute the whitened mean `(batch, ..., K)`."""
        mean, var = as_joined_outputs(self.model(x)).chunk(2, dim=-1)

        return mean / var.detach().clamp(min=self.eps).sqrt()


class HeteroscedasticGaussianNLLLoss(nn.Module):
    """`nn.GaussianNLLLoss` of the pair `(mean, var)` or of the joined outputs.

    The pair is what the network returns, e.g. while `marglik_training` trains it.
    The joined outputs `(batch, ..., 2 * K)` are what `Laplace` sees. They carry the
    variance at the MAP, which `Laplace` detaches, so their loss has no gradient
    w.r.t. the variance.

    Parameters
    ----------
    eps : float
        lower bound of the variance.
    full : bool
        whether the loss includes the constant.
    reduction : str, default='sum'
        reduction of the loss, as in `nn.GaussianNLLLoss`.
    """

    def __init__(self, eps: float, full: bool, reduction: str = "sum") -> None:
        super().__init__()

        self.eps = eps
        self.full = full
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss of the outputs `input` w.r.t. the targets `target`.

        This is `nn.functional.gaussian_nll_loss` without its check for negative
        variances, which is data-dependent control flow that `torch.func.vmap` can't
        trace, e.g. for the per-sample gradients of the empirical Fisher. Negative
        variances are clamped at `eps` like small ones.
        """
        mean, var = as_joined_outputs(input).chunk(2, dim=-1)

        # As in torch, the clamp changes the value but passes the gradient through
        var = var + (var.clamp(min=self.eps) - var).detach()
        loss = 0.5 * (var.log() + (mean - target).square() / var)

        if self.full:
            loss = loss + log(sqrt(2 * pi))

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()

        return loss
