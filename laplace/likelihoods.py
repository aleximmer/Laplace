"""Likelihood objects that bundle everything a Laplace approximation needs to know
about the observation model: the loss, the output-space Hessian, the normalizer, and
how to sample targets.
"""

from __future__ import annotations

import inspect
import warnings
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

    To add a new likelihood, subclass this, set `name`, and implement
    `distribution(f)`, which returns the model's predictive distribution over the
    targets as a `torch.distributions.Distribution`. Its `batch_shape + event_shape`
    is the shape of the targets. All other members have defaults built on it, and
    you can override any of them:

    - `loss_fn`: the negative log-probability of the targets, summed or averaged.
    - `loss_factor`: \\(1\\), since the loss is already the negative log likelihood.
    - `log_normalizer`: zero.
    - `loss`: calls `check_targets`, then `loss_fn`.
    - `check_targets`: raises a `ValueError` unless the targets have the
      distribution's shape.
    - `n_observations`: the number of entries in the batch dims of `f`.
    - `inverse_link`: the mean of the distribution, or the class probabilities of a
      `Categorical`.
    - `sample`: a draw from the distribution.
    - `output_gradient`: the gradient of the negative log likelihood, by autograd.
    - `output_hessian`: the Fisher of the distribution w.r.t. `f`, see below.
    - `is_gaussian`: `False`.
    - `fit_likelihood` and `predictive_likelihood`: `self`.

    The default `output_hessian` returns the Fisher

    $$
        F(f) = \\mathbb{E}_{y \\sim p(\\cdot \\mid f)}\\left[g_y g_y^\\top\\right],
        \\qquad g_y = \\nabla_f \\left(-\\log p(y \\mid f)\\right),
    $$

    in place of the Hessian of the negative log likelihood. The two agree for
    exponential families in their natural parameters, such as the Gaussian mean and
    the categorical logits. Elsewhere the Hessian can be indefinite, and the GGN
    needs the positive semidefinite \\(F\\). The Fisher is exact when the
    distribution has a finite support to enumerate, such as categorical, Bernoulli,
    or binomial. Otherwise it is a Monte Carlo estimate from `fisher_samples` draws,
    taken `fisher_chunk_size` at a time, and a `UserWarning` says so. If you know the
    Fisher in closed form, override `output_hessian` to make it exact and cheap.
    `output_hessian` also gets the targets `y` when the caller has them, for
    likelihoods whose curvature depends on them, e.g. through class weights or
    ignored tokens. It is `None` when the caller has no targets.

    The built-in likelihoods override all of these with closed forms and the exact
    torch losses. Their `distribution` is the model's predictive, so it does not
    change with options that only reweight or smooth the loss.

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

    fisher_samples: ClassVar[int] = 100
    """Number of samples for the Monte Carlo Fisher in the default `output_hessian`.
    With 1000 data and 10 outputs, 100 samples give about 1.5% error in the summed
    GGN and 25% per datum."""

    fisher_chunk_size: ClassVar[int] = 10
    """Number of Monte Carlo samples drawn at once, to bound memory."""

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
            return self.name == _name_of(other)

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
    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        """Predictive distribution of the targets given the outputs `f`.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`

        Returns
        -------
        dist : torch.distributions.Distribution
            a distribution whose `batch_shape + event_shape` is the shape of the
            targets.
        """

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        """Build the loss module. By default, it computes the negative
        log-probability of the targets under `distribution`.

        Parameters
        ----------
        reduction : str, default='sum'
            reduction of the loss, as in `torch.nn`.

        Returns
        -------
        loss_fn : nn.Module
            a new module on every call.
        """
        return _NegLogProbLoss(self, reduction)

    @property
    def loss_factor(self) -> float:
        """Factor that turns the loss into the negative log likelihood, up to a
        constant. It is \\(1\\) by default."""
        return 1.0

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Raise a `ValueError` if the targets `y` do not fit the outputs `f`. By
        default, `y` must have the shape `batch_shape + event_shape` of
        `distribution(f)`.
        """
        dist = self.distribution(f)
        shape = dist.batch_shape + dist.event_shape

        if y.shape != shape:
            raise ValueError(
                f"{type(self).__name__} expects targets of shape {tuple(shape)} "
                f"for outputs of shape {tuple(f.shape)}, but the targets have shape "
                f"{tuple(y.shape)}."
            )

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Curvature of the negative log likelihood w.r.t. the outputs `f`.

        By default, this is the Fisher \\(F(f)\\) of `distribution(f)`, exact for a
        distribution with enumerable support and a Monte Carlo estimate otherwise.
        For the built-in likelihoods, it equals the Hessian.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`
        y : torch.Tensor, optional
            targets, for likelihoods whose curvature depends on them. The default
            ignores them.

        Returns
        -------
        H : torch.Tensor
            `(batch, ..., outputs, outputs)`, with the dtype of `f`.
        """
        with torch.enable_grad():
            f = f.detach()
            dist = self.distribution(f)

            if dist.has_enumerate_support:
                return self._enumerated_fisher(f, dist)

            return self._monte_carlo_fisher(f)

    def _enumerated_fisher(
        self, f: torch.Tensor, dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        """Exact Fisher of a distribution with enumerable support.

        Parameters
        ----------
        f : torch.Tensor
            detached outputs `(batch, ..., outputs)`
        dist : torch.distributions.Distribution
            `distribution(f)`

        Returns
        -------
        F : torch.Tensor
            `(batch, ..., outputs, outputs)`
        """
        ys = dist.enumerate_support(expand=True)
        f_rep = f.expand(ys.shape[0], *f.shape).clone().requires_grad_(True)

        log_p = self.distribution(f_rep).log_prob(ys)
        (g,) = torch.autograd.grad(-log_p.sum(), f_rep)
        p_ys = log_p.detach().exp()

        # One distribution per output, e.g. a Bernoulli per logit: the outputs are
        # independent, so the Fisher is diagonal
        if p_ys.shape == g.shape:
            return torch.diag_embed((p_ys * g**2).sum(0))

        return torch.einsum("s...,s...k,s...c->...kc", p_ys, g, g)

    def _monte_carlo_fisher(self, f: torch.Tensor) -> torch.Tensor:
        """Monte Carlo estimate of the Fisher from `fisher_samples` draws.

        Parameters
        ----------
        f : torch.Tensor
            detached outputs `(batch, ..., outputs)`

        Returns
        -------
        F : torch.Tensor
            `(batch, ..., outputs, outputs)`
        """
        warnings.warn(
            f"{type(self).__name__} estimates its output Hessian as a Monte Carlo "
            f"Fisher from {self.fisher_samples} samples. Override `output_hessian` "
            "with a closed form to make it exact.",
            UserWarning,
            stacklevel=3,
        )

        F = f.new_zeros(*f.shape, f.shape[-1])
        remaining = self.fisher_samples

        while remaining > 0:
            n_samples = min(self.fisher_chunk_size, remaining)
            f_rep = f.expand(n_samples, *f.shape).clone().requires_grad_(True)

            dist = self.distribution(f_rep)
            ys = dist.sample()
            (g,) = torch.autograd.grad(-dist.log_prob(ys).sum(), f_rep)

            F += torch.einsum("s...k,s...c->...kc", g, g)
            remaining -= n_samples

        return F / self.fisher_samples

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        """Map the outputs `f` to the mean of the targets: by default the mean of
        `distribution(f)`, or its class probabilities for a `Categorical`."""
        dist = self.distribution(f)

        # The mean of a `Categorical` is NaN, since its classes are unordered
        if isinstance(dist, torch.distributions.Categorical):
            return dist.probs

        return dist.mean

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample targets from the likelihood given the outputs `f`.

        By default, this draws from `distribution(f)`. Since distributions take no
        generator, a given `generator` seeds the global one inside
        `torch.random.fork_rng`, which leaves the global state as it was.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`
        generator : torch.Generator, optional
            source of randomness. The global generator is used if `None`.

        Returns
        -------
        y : torch.Tensor
            samples with the shape of the targets.
        """
        if generator is None:
            return self.distribution(f).sample()

        seed = int(
            torch.randint(0, 2**62, (), generator=generator, device=generator.device)
        )

        with torch.random.fork_rng(devices=[f.device] if f.is_cuda else []):
            # Seed only the generators that `fork_rng` restores
            torch.default_generator.manual_seed(seed)

            if f.is_cuda:
                with torch.cuda.device(f.device):
                    torch.cuda.manual_seed(seed)

            return self.distribution(f).sample()

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient of the negative log likelihood w.r.t. the outputs `f`. By
        default, it comes from autograd through `distribution(f)`."""
        with torch.enable_grad():
            f = f.detach().requires_grad_(True)
            nll = -self.distribution(f).log_prob(y).sum()
            (g,) = torch.autograd.grad(nll, f)

        return g

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

    With observation noise \\(\\sigma\\), a target \\(y \\in \\mathbb{R}^K\\) given the
    outputs \\(f\\) has the likelihood

    $$
        p(y \\mid f) = \\mathcal{N}(y \\mid f, \\sigma^2 I).
    $$

    The loss is the summed MSE, so for \\(N\\) observations the negative log
    likelihood is

    $$
        -\\log p(\\mathcal{D} \\mid f)
        = \\frac{1}{\\sigma^2} \\cdot \\frac{1}{2} \\sum_{n=1}^N \\lVert y_n - f_n \\rVert^2
        + N K \\log\\left(\\sigma \\sqrt{2 \\pi}\\right),
    $$

    which gives the loss factor \\(\\tfrac{1}{2}\\) and the normalizer
    \\(N K \\log(\\sigma \\sqrt{2\\pi})\\). The targets have as many dims as the outputs,
    and every entry of the targets but the last dim is an observation.

    The noise enters through `sigma_noise` of the Laplace approximation, so the
    other quantities assume \\(\\sigma = 1\\): the output Hessian is \\(I\\), the inverse
    link is the identity, samples are \\(y \\sim \\mathcal{N}(f, I)\\), and the output
    gradient is \\(f - y\\). A float `sigma_noise` gives a normalizer with the default
    dtype.
    """

    name: ClassVar[str] = "regression"

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Normal(f, 1.0)

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

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    With logits \\(f \\in \\mathbb{R}^C\\) and class probabilities
    \\(p = \\operatorname{softmax}(f)\\), a class \\(y\\) has the likelihood
    \\(p(y \\mid f) = p_y\\). The loss is the cross-entropy \\(-\\log p_y\\), or
    \\(-\\sum_c y_c \\log p_c\\) for class probabilities \\(y\\), which is already the
    negative log likelihood. So the loss factor is \\(1\\) and the normalizer is zero.

    The targets are class indices `(batch, ...)` or class probabilities with the
    shape of the logits. For logits with extra batch dims, e.g. `(B, L, C)` for
    sequences, `loss` flattens the batch dims, since `CrossEntropyLoss` expects the
    class axis at dim 1. Every token is then an observation; the class dim of
    probability targets does not count.

    The output Hessian of the negative log likelihood is

    $$
        \\nabla^2_f \\left(-\\log p(y \\mid f)\\right) = \\operatorname{diag}(p) - p p^\\top,
    $$

    the inverse link is the softmax, samples are one-hot draws
    \\(y \\sim \\operatorname{Cat}(p)\\), and the output gradient is \\(p - y\\).
    """

    name: ClassVar[str] = "classification"

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=f)

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

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = self.inverse_link(f)

        return torch.diag_embed(p) - torch.einsum("...k,...c->...kc", p, p)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return torch.softmax(f, dim=-1)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        p = self.inverse_link(f).reshape(-1, f.shape[-1])
        idx = torch.multinomial(p, 1, generator=generator)
        y = torch.zeros_like(p).scatter_(1, idx, 1.0)

        return y.reshape(f.shape)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.inverse_link(f) - y


class BradleyTerryLikelihood(Likelihood):
    """Bradley-Terry likelihood for reward modeling.

    The model outputs a reward \\(r(x)\\) per item. For a pair of items \\(x_1\\) and
    \\(x_2\\), the probability that \\(x_1\\) is preferred is

    $$
        p(x_1 \\succ x_2) = \\frac{\\exp r(x_1)}{\\exp r(x_1) + \\exp r(x_2)},
    $$

    the softmax of the two rewards. The curvature is therefore fitted with a
    `CategoricalLikelihood` over the compared items, so the loss, the target check,
    the output Hessian, the inverse link, samples, and the output gradient are all
    categorical. Unlike for classification, every entry of the targets is an
    observation. The predictive is a `GaussianLikelihood` over the reward
    \\(r(x)\\).
    """

    name: ClassVar[str] = "reward_modeling"

    def __init__(self) -> None:
        self._fit = CategoricalLikelihood()
        self._predictive = GaussianLikelihood()

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return self._fit.distribution(f)

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

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._fit.output_hessian(f, y)

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


class _NegLogProbLoss(nn.Module):
    """Negative log-probability of the targets under a likelihood's
    `distribution`, the default `loss_fn`.

    Parameters
    ----------
    likelihood : Likelihood
        likelihood whose `distribution` scores the targets.
    reduction : str, default='sum'
        `'sum'` or `'mean'` over all entries of the log-probability.
    """

    def __init__(self, likelihood: Likelihood, reduction: str = "sum") -> None:
        super().__init__()

        if reduction not in ("sum", "mean"):
            raise ValueError(
                f"Reduction must be 'sum' or 'mean', but it is {reduction!r}."
            )

        self.likelihood = likelihood
        self.reduction = reduction

    def forward(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reduced negative log-probability of the targets `y` given the outputs
        `f`."""
        nll = -self.likelihood.distribution(f).log_prob(y)

        return nll.sum() if self.reduction == "sum" else nll.mean()


def _name_of(name: str) -> str:
    """The value of a string, or of a member of a `(str, Enum)` such as the old
    `Likelihood` enum, whose `str()` is not its value."""
    return str(getattr(name, "value", name))


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
            if cls.name == _name_of(likelihood):
                return cls()

    raise ValueError(f"Invalid likelihood type {likelihood}")


Likelihood.REGRESSION = GaussianLikelihood()
Likelihood.CLASSIFICATION = CategoricalLikelihood()
Likelihood.REWARD_MODELING = BradleyTerryLikelihood()
