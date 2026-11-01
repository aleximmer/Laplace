"""Likelihood objects that bundle everything a Laplace approximation needs to know
about the observation model: the loss, the output-space Hessian, the normalizer, and
how to sample targets.
"""

from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from math import pi, sqrt
from typing import Any, ClassVar

import torch
from torch import nn

_BUILTIN_KEYS = ("REGRESSION", "CLASSIFICATION", "REWARD_MODELING")


class _LikelihoodMeta(ABCMeta):
    """Gives `Likelihood` the class-level API of the enum it replaced: lookup by
    value with `Likelihood("regression")`, lookup by key with
    `Likelihood["REGRESSION"]`, iteration, `len`, and `in`.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Only the abstract base looks up by value, the way `Enum("value")` does. This
        # also unpickles members of the old enum, which pickled as `Likelihood(value)`.
        if vars(cls).get("_is_base", False) and len(args) == 1 and not kwargs:
            likelihood = get_likelihood(args[0])
            if likelihood is args[0]:
                return likelihood

            # Like an enum member, a built-in likelihood is a single shared instance
            return next(m for m in cls if m == likelihood)

        return super().__call__(*args, **kwargs)

    def _require_base(cls) -> None:
        """Keep the enum API on `Likelihood` itself, not on its subclasses."""
        if not vars(cls).get("_is_base", False):
            raise TypeError(f"{cls.__name__} does not list the built-in likelihoods.")

    def __getitem__(cls, key: str) -> Likelihood:
        cls._require_base()

        if key not in _BUILTIN_KEYS:
            raise KeyError(key)

        return getattr(cls, key)

    def __iter__(cls) -> Iterator[Likelihood]:
        cls._require_base()

        return (getattr(cls, key) for key in _BUILTIN_KEYS)

    def __len__(cls) -> int:
        cls._require_base()

        return len(_BUILTIN_KEYS)

    def __contains__(cls, item: object) -> bool:
        return any(item == member for member in cls)


class Likelihood(metaclass=_LikelihoodMeta):
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

    Likelihoods compare by `name`: an instance equals any likelihood or string with
    the same name. So a subclass of `CategoricalLikelihood` that keeps its name still
    equals `Likelihood.CLASSIFICATION`.

    A custom subclass can only be fit if its `name` is `"regression"` or
    `"classification"`, because the curvature backends pick their code path by name.
    FunctionalLaplace, ELLA, and VaLLA use the subclass's overrides, such as
    `output_hessian`. Most curvature methods of the Curvlinops, ASDL, and BackPACK
    backends use their own loss instead, so weight-space Laplace with these backends
    supports only the built-in likelihoods for now and raises a `ValueError` for a
    custom one.
    """

    name: ClassVar[str]
    """Identifier of the likelihood, e.g. what is stored in `state_dict()`."""

    REGRESSION: ClassVar[GaussianLikelihood]
    """The built-in `GaussianLikelihood`."""

    CLASSIFICATION: ClassVar[CategoricalLikelihood]
    """The built-in `CategoricalLikelihood`."""

    REWARD_MODELING: ClassVar[BradleyTerryLikelihood]
    """The built-in `BradleyTerryLikelihood`."""

    _is_base: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Require a `name` on every concrete subclass.

        Raises
        ------
        TypeError
            if a subclass with no abstract members left has no `name`.
        """
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls) and not isinstance(
            getattr(cls, "name", None), str
        ):
            raise TypeError(f"Likelihood {cls.__name__} must define a `name`.")

    def __eq__(self, other: object) -> bool:
        # By name only, so that a subclass of a built-in matches checks such as
        # `likelihood == Likelihood.CLASSIFICATION`, as it did against the old enum
        if isinstance(other, Likelihood):
            return self.name == other.name

        if isinstance(other, str):
            return self.name == str(getattr(other, "value", other))

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @property
    def value(self) -> str:
        """Same as `name`."""
        return self.name

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

        Returns
        -------
        c : torch.Tensor
            a 0-dim zero with the dtype and device of `sigma_noise`.
        """
        return _zero_like(sigma_noise)

    @property
    def is_gaussian(self) -> bool:
        """Whether the likelihood is Gaussian, i.e. has a closed-form predictive and
        allows `sigma_noise`."""
        return False

    @property
    def fit_likelihood(self) -> Likelihood:
        """Likelihood used for fitting the curvature."""
        return self

    @property
    def predictive_likelihood(self) -> Likelihood:
        """Likelihood of the predictive distribution."""
        return self


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood, i.e. regression.

    The loss is the MSE, so the negative log likelihood is half the loss plus the
    normalizer `n_obs * n_outputs * log(sigma_noise * sqrt(2 pi))`. The targets have
    as many dims as the outputs `f`, and every entry of the targets but the last dim
    is an observation.

    With unit noise, the output Hessian is the identity, the inverse link is the
    identity, samples are `y ~ N(f, 1)`, and the output gradient is `f - y`. A float
    `sigma_noise` gives a normalizer with the default dtype.
    """

    name: ClassVar[str] = "regression"

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return nn.MSELoss(reduction=reduction)

    @property
    def loss_factor(self) -> float:
        return 0.5

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        if y.ndim != f.ndim:
            raise ValueError(
                f"The model's output has {f.ndim} dims but the target has {y.ndim} dims."
            )

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        return y.shape[:-1].numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        sigma_noise = torch.as_tensor(sigma_noise, dtype=_float_dtype(sigma_noise))

        return n_obs * n_outputs * torch.log(sigma_noise * sqrt(2 * pi))

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        K = f.shape[-1]
        eye = torch.eye(K, device=f.device, dtype=f.dtype)

        return eye.expand(*f.shape[:-1], K, K)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return f

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        noise = torch.randn(
            f.shape, device=f.device, dtype=f.dtype, generator=generator
        )

        return f + noise

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return f - y

    @property
    def is_gaussian(self) -> bool:
        return True


class CategoricalLikelihood(Likelihood):
    """Categorical likelihood with a softmax link, i.e. classification.

    The loss is the cross-entropy, which is already the negative log likelihood, so
    the normalizer is zero. The targets are class indices `(batch, ...)` or class
    probabilities with the shape of the logits `f`.

    For logits with extra batch dims, e.g. `(B, L, C)` for sequences, `loss` flattens
    the batch dims, since `CrossEntropyLoss` expects the class axis at dim 1. Every
    token is then an observation; the class dim of probability targets does not
    count.

    With `p = softmax(f)`, the output Hessian is `diag(p) - p p^T`, the inverse link
    is the softmax, samples are one-hot draws from `p`, and the output gradient is
    `p - y`.
    """

    name: ClassVar[str] = "classification"

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return nn.CrossEntropyLoss(reduction=reduction)

    @property
    def loss_factor(self) -> float:
        return 1.0

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        # Check the exact shape: flattening the batch dims would silently misalign
        # targets of the wrong shape with as many elements, e.g. `(B, L, M)` instead
        # of `(B, M, L)`
        if y.shape != f.shape[:-1] and y.shape != f.shape:
            raise ValueError(
                f"The targets have shape {tuple(y.shape)}, which matches neither the "
                f"class indices {tuple(f.shape[:-1])} nor the class probabilities "
                f"{tuple(f.shape)} expected from the logits."
            )

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_targets(f, y)

        if f.ndim > 2:
            # Class probabilities keep their class axis
            y = y.reshape(-1, f.shape[-1]) if y.ndim == f.ndim else y.reshape(-1)
            f = f.reshape(-1, f.shape[-1])

        return self.loss_fn()(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        return (y.shape[:-1] if y.ndim == f.ndim else y.shape).numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        return _zero_like(sigma_noise)

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(f, dim=-1)

        return torch.diag_embed(p) - torch.einsum("...k,...c->...kc", p, p)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return torch.softmax(f, dim=-1)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        p = torch.softmax(f, dim=-1).reshape(-1, f.shape[-1])
        idx = torch.multinomial(p, 1, generator=generator)
        y = torch.zeros_like(p).scatter_(1, idx, 1.0)

        return y.reshape(f.shape)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.softmax(f, dim=-1) - y


class BradleyTerryLikelihood(Likelihood):
    """Bradley-Terry likelihood for reward modeling.

    The curvature is fitted with a `CategoricalLikelihood` over the compared items,
    so the loss, the target check, the output Hessian, the inverse link, samples,
    and the output gradient are all categorical. Unlike for classification, every
    entry of the targets is an observation. The predictive is a
    `GaussianLikelihood` over the reward.
    """

    name: ClassVar[str] = "reward_modeling"

    def __init__(self) -> None:
        self._fit = CategoricalLikelihood()
        self._predictive = GaussianLikelihood()

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return self._fit.loss_fn(reduction)

    @property
    def loss_factor(self) -> float:
        return self._fit.loss_factor

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        self._fit.check_targets(f, y)

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._fit.loss(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        return y.shape.numel()

    def output_hessian(self, f: torch.Tensor) -> torch.Tensor:
        return self._fit.output_hessian(f)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return self._fit.inverse_link(f)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._fit.sample(f, generator)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._fit.output_gradient(f, y)

    @property
    def fit_likelihood(self) -> Likelihood:
        return self._fit

    @property
    def predictive_likelihood(self) -> Likelihood:
        return self._predictive


def _float_dtype(x: torch.Tensor | float) -> torch.dtype:
    """Dtype of `x` if it is a floating tensor, else the default dtype."""
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.dtype

    return torch.get_default_dtype()


def _zero_like(x: torch.Tensor | float) -> torch.Tensor:
    """0-dim zero with the floating dtype and the device of `x`."""
    device = x.device if isinstance(x, torch.Tensor) else None

    return torch.zeros((), dtype=_float_dtype(x), device=device)


def get_likelihood(likelihood: Likelihood | str) -> Likelihood:
    """Turn a likelihood name into a likelihood object.

    Parameters
    ----------
    likelihood : Likelihood or str
        an instance, which is returned as is, or one of `'regression'`,
        `'classification'`, and `'reward_modeling'`.

    Returns
    -------
    likelihood : Likelihood

    Raises
    ------
    ValueError
        if `likelihood` is neither a likelihood instance nor a known name.
    """
    if isinstance(likelihood, Likelihood):
        return likelihood

    if isinstance(likelihood, str):
        for cls in (GaussianLikelihood, CategoricalLikelihood, BradleyTerryLikelihood):
            if cls.name == str(getattr(likelihood, "value", likelihood)):
                return cls()

    raise ValueError(f"Invalid likelihood type {likelihood}")


Likelihood.REGRESSION = GaussianLikelihood()
Likelihood.CLASSIFICATION = CategoricalLikelihood()
Likelihood.REWARD_MODELING = BradleyTerryLikelihood()
