"""Likelihood objects that bundle everything a Laplace approximation needs to know
about the observation model: the loss, the output-space Hessian, the normalizer, and
how to sample targets.
"""

from abc import ABC, abstractmethod
from math import pi, sqrt
from typing import ClassVar

import torch
from torch import nn

from laplace.utils.enums import Likelihood


class BaseLikelihood(ABC):
    """Base class for likelihoods.

    A likelihood compares the model output `f` with the targets `y`. All leading
    dims of `f` but the last are batch dims, so `f` has shape `(batch, ..., outputs)`.

    To add a new likelihood, subclass this and implement the abstract members:
    `name`, `loss_fn`, `loss_factor`, `check_targets`, `output_hessian`,
    `inverse_link`, `sample`, and `output_gradient`. The following members have
    defaults that you can override: `loss` (calls `check_targets`, then `loss_fn`),
    `n_observations` (the number of leading dims of `f`), `log_normalizer` (zero),
    `is_gaussian` (`False`), and `fit_likelihood` and `predictive_likelihood` (both
    `self`).

    Instances compare equal to other instances of the same type and to the string
    or `Likelihood` member that equals `name`.

    A custom subclass can only be fit if its `name` is `"regression"` or
    `"classification"`, because the curvature backends pick their code path by name.
    """

    name: ClassVar[str]
    """Identifier of the likelihood, e.g. what is stored in `state_dict()`."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseLikelihood):
            return type(self) is type(other) and self.name == other.name

        if isinstance(other, (str, Likelihood)):
            return self.name == str(getattr(other, "value", other))

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @abstractmethod
    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        """Build the loss module.

        Parameters
        ----------
        reduction : str, default='sum'
            reduction of the loss, as in `torch.nn`.

        Returns
        -------
        loss_fn : nn.Module
            a new module on every call.
        """

    @property
    @abstractmethod
    def loss_factor(self) -> float:
        """Factor that turns the loss into the negative log likelihood, up to a
        constant."""

    @abstractmethod
    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Raise a `ValueError` if the targets `y` do not fit the outputs `f`."""

    @abstractmethod
    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        """Hessian of the negative log likelihood w.r.t. the outputs `f`.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`

        Returns
        -------
        H : torch.Tensor
            `(batch, ..., outputs, outputs)`, with the dtype of `f`.
        """

    @abstractmethod
    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        """Map the outputs `f` to the mean of the targets."""

    @abstractmethod
    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample targets from the likelihood given the outputs `f`.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`
        generator : torch.Generator, optional
            source of randomness. The global generator is used if `None`.

        Returns
        -------
        y : torch.Tensor
            samples with the shape of `f`.
        """

    @abstractmethod
    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of the negative log likelihood w.r.t. the outputs `f`."""

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Summed loss of the outputs `f` w.r.t. the targets `y`, after checking
        the targets with `check_targets`.
        """
        self.check_targets(f, y)

        return self.loss_fn()(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        """Number of observations in a batch. By default, every leading dim of `f`
        is a batch dim.
        """
        return f.shape[:-1].numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        """Constant that turns the negative loss into the log likelihood. It is zero
        by default.

        Parameters
        ----------
        n_obs : int
            number of observations
        n_outputs : int
            number of outputs
        sigma_noise : torch.Tensor
            observation noise, ignored if the likelihood has none.
        """
        sigma_noise = torch.as_tensor(sigma_noise)

        return torch.zeros_like(sigma_noise)

    @property
    def is_gaussian(self) -> bool:
        """Whether the likelihood is Gaussian, i.e. has a closed-form predictive and
        allows `sigma_noise`."""
        return False

    @property
    def fit_likelihood(self) -> "BaseLikelihood":
        """Likelihood used for fitting the curvature."""
        return self

    @property
    def predictive_likelihood(self) -> "BaseLikelihood":
        """Likelihood of the predictive distribution."""
        return self


class GaussianLikelihood(BaseLikelihood):
    """Gaussian likelihood with unit variance for the loss, i.e. regression."""

    name: ClassVar[str] = "regression"

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        """MSE loss."""
        return nn.MSELoss(reduction=reduction)

    @property
    def loss_factor(self) -> float:
        """0.5, since the loss is the MSE."""
        return 0.5

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Require targets with as many dims as `f`."""
        if y.ndim != f.ndim:
            raise ValueError(
                f"The model's output has {f.ndim} dims but the target has {y.ndim} dims."
            )

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        """Every entry of `y` but the last dim (the outputs) is an observation."""
        return y.shape[:-1].numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        """`n_obs * n_outputs * log(sigma_noise * sqrt(2 pi))`."""
        return (
            n_obs * n_outputs * torch.log(torch.as_tensor(sigma_noise) * sqrt(2 * pi))
        )

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        """Identity, expanded to the batch dims of `f`."""
        K = f.shape[-1]
        eye = torch.eye(K, device=f.device, dtype=f.dtype)

        return eye.expand(*f.shape[:-1], K, K)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        """Identity."""
        return f

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample `y ~ N(f, 1)`."""
        noise = torch.randn(
            f.shape, device=f.device, dtype=f.dtype, generator=generator
        )

        return f + noise

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`f - y`."""
        return f - y

    @property
    def is_gaussian(self) -> bool:
        """`True`."""
        return True


class CategoricalLikelihood(BaseLikelihood):
    """Categorical likelihood with a softmax link, i.e. classification.

    The targets are class indices `(batch, ...)` or class probabilities with the
    shape of the logits.
    """

    name: ClassVar[str] = "classification"

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        """Cross-entropy loss."""
        return nn.CrossEntropyLoss(reduction=reduction)

    @property
    def loss_factor(self) -> float:
        """1.0, since the cross-entropy is the negative log likelihood."""
        return 1.0

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Check that the targets `y` match the logits `f` `(batch, ..., classes)`:
        class indices `(batch, ...)` or class probabilities with the same shape as
        `f`.

        Flattening the batch dims would silently misalign targets of the wrong shape
        with the same number of elements, e.g. `(B, L, M)` instead of `(B, M, L)`.
        """
        if y.shape != f.shape[:-1] and y.shape != f.shape:
            raise ValueError(
                f"The targets have shape {tuple(y.shape)}, which matches neither the "
                f"class indices {tuple(f.shape[:-1])} nor the class probabilities "
                f"{tuple(f.shape)} expected from the logits."
            )

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Summed cross-entropy, where every leading dim of `f` but the last is a
        batch dim.

        `CrossEntropyLoss` expects the class axis at dim 1, while Laplace puts it
        last, e.g. logits `(B, L, C)` with targets `(B, L)`. So the batch dims are
        flattened, which makes every token an observation.
        """
        self.check_targets(f, y)

        if f.ndim > 2:
            # Class probabilities keep their class axis
            y = y.reshape(-1, f.shape[-1]) if y.ndim == f.ndim else y.reshape(-1)
            f = f.reshape(-1, f.shape[-1])

        return self.loss_fn()(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        """Every token is an observation. Class probabilities have a class dim that
        does not count.
        """
        return (y.shape[:-1] if y.ndim == f.ndim else y.shape).numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        """Zero, since the cross-entropy is already the negative log likelihood."""
        return torch.zeros_like(torch.as_tensor(sigma_noise))

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        """`diag(p) - p p^T`, where `p` is the softmax of `f`."""
        p = torch.softmax(f, dim=-1)

        return torch.diag_embed(p) - torch.einsum("...k,...c->...kc", p, p)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        """Softmax over the last dim."""
        return torch.softmax(f, dim=-1)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample one-hot targets from the categorical distribution given by the
        logits `f`.
        """
        p = torch.softmax(f, dim=-1).reshape(-1, f.shape[-1])
        idx = torch.multinomial(p, 1, generator=generator)
        y = torch.zeros_like(p).scatter_(1, idx, 1.0)

        return y.reshape(f.shape)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`softmax(f) - y`."""
        return torch.softmax(f, dim=-1) - y


class BradleyTerryLikelihood(BaseLikelihood):
    """Bradley-Terry likelihood for reward modeling.

    The curvature is fitted with a categorical likelihood over the compared items,
    while the predictive is a Gaussian over the reward. Loss, Hessian, and the other
    fitting quantities are those of the categorical likelihood.
    """

    name: ClassVar[str] = "reward_modeling"

    def __init__(self) -> None:
        self._fit = CategoricalLikelihood()
        self._predictive = GaussianLikelihood()

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        """Cross-entropy loss."""
        return self._fit.loss_fn(reduction)

    @property
    def loss_factor(self) -> float:
        """1.0."""
        return self._fit.loss_factor

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Check the targets as for the categorical likelihood."""
        self._fit.check_targets(f, y)

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Summed cross-entropy, as for the categorical likelihood."""
        return self._fit.loss(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        """Every entry of `y` is an observation."""
        return y.shape.numel()

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        """`diag(p) - p p^T`, as for the categorical likelihood."""
        return self._fit.output_hessian(f)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        """Softmax over the last dim."""
        return self._fit.inverse_link(f)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample one-hot targets, as for the categorical likelihood."""
        return self._fit.sample(f, generator)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`softmax(f) - y`."""
        return self._fit.output_gradient(f, y)

    @property
    def fit_likelihood(self) -> BaseLikelihood:
        """A `CategoricalLikelihood`."""
        return self._fit

    @property
    def predictive_likelihood(self) -> BaseLikelihood:
        """A `GaussianLikelihood`."""
        return self._predictive


def get_likelihood(likelihood: BaseLikelihood | Likelihood | str) -> BaseLikelihood:
    """Turn a likelihood name into a likelihood object.

    Parameters
    ----------
    likelihood : BaseLikelihood or Likelihood or str
        an instance, which is returned as is, or one of `'regression'`,
        `'classification'`, and `'reward_modeling'`.

    Returns
    -------
    likelihood : BaseLikelihood

    Raises
    ------
    ValueError
        if `likelihood` is neither a likelihood instance nor a known name.
    """
    if isinstance(likelihood, BaseLikelihood):
        return likelihood

    if isinstance(likelihood, (str, Likelihood)):
        for cls in (GaussianLikelihood, CategoricalLikelihood, BradleyTerryLikelihood):
            if cls.name == str(getattr(likelihood, "value", likelihood)):
                return cls()

    raise ValueError(f"Invalid likelihood type {likelihood}")
