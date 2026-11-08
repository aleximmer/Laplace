"""Likelihood objects that bundle everything a Laplace approximation needs to know
about the observation model: the loss, the output-space Hessian, the normalizer, and
how to sample targets.
"""

from __future__ import annotations

import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from math import log, pi, sqrt
from typing import Any, ClassVar

import torch
from torch import nn

from laplace.utils.heteroscedastic_utils import HeteroscedasticGaussianNLLLoss
from laplace.utils.utils import _float_dtype, _zero_like, hyperparameters_equal

_BUILTIN_KEYS = ("REGRESSION", "CLASSIFICATION", "REWARD_MODELING", "BERNOULLI")


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

            # Like an enum member, a built-in likelihood is a single shared instance.
            # A criterion with non-default options matches none of them.
            return next((m for m in cls if m == likelihood), likelihood)

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
    - `n_observations`: the number of entries in the batch dims of the targets, i.e.
      their leading dims that match the batch dims of `f`.
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
    ignored positions. It is `None` when the caller has no targets.

    The built-in likelihoods override all of these with closed forms and the exact
    torch losses. Their `distribution` is the model's predictive, so it does not
    change with options that only reweight or smooth the loss.

    Two likelihoods are equal if they have the same `name` and the same
    `hyperparameters()`, the options that differ from their defaults. A string equals
    a likelihood with that name. Internally, the library dispatches on `name`, so a
    subclass of `CategoricalLikelihood` that keeps its name and carries options still
    takes the classification code paths, even though it does not equal
    `Likelihood.CLASSIFICATION`.

    Weight-space Laplace fits a custom likelihood with the `diag` or `full`
    structure and a GGN or EF backend, which use its `output_hessian` and `loss_fn`.
    The `kron` structure and the Hessian backends raise a `ValueError` for it. The
    GLM predictive of a likelihood that is neither Gaussian nor categorical is the
    Monte Carlo average of `inverse_link` over the GLM predictive of the outputs.
    FunctionalLaplace, ELLA, and VaLLA only accept the names of the regression,
    classification, and reward-modeling likelihoods, but a subclass that keeps one of
    them may override members such as `output_hessian`.
    """

    name: ClassVar[str]
    """Identifier of the likelihood, e.g. what is stored in `state_dict()`."""

    REGRESSION: ClassVar[GaussianLikelihood]
    """The built-in `GaussianLikelihood`."""

    CLASSIFICATION: ClassVar[CategoricalLikelihood]
    """The built-in `CategoricalLikelihood`."""

    REWARD_MODELING: ClassVar[BradleyTerryLikelihood]
    """The built-in `BradleyTerryLikelihood`."""

    BERNOULLI: ClassVar[BernoulliLikelihood]
    """The built-in `BernoulliLikelihood`."""

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
        # A string names a likelihood type, so only the name counts. Two likelihoods
        # must also agree on their options.
        if isinstance(other, Likelihood):
            return self.name == other.name and hyperparameters_equal(
                self.hyperparameters(), other.hyperparameters()
            )

        if isinstance(other, str):
            return self.name == _name_of(other)

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        # Show the options, but only the shape of a tensor, which can be large
        options = ", ".join(
            f"{k}=tensor(shape={tuple(v.shape)})"
            if torch.is_tensor(v)
            else f"{k}={v!r}"
            for k, v in self.hyperparameters().items()
        )

        return f"{type(self).__name__}({options})"

    def hyperparameters(self) -> dict[str, Any]:
        """Options of the likelihood that differ from their defaults, by name.

        They are part of the likelihood's identity: two likelihoods are equal only if
        their names and hyperparameters match, and checkpoints store them. Override
        this in a subclass that takes options. Values are plain Python objects or
        tensors.

        Returns
        -------
        hyperparameters : dict
            empty by default.
        """
        return {}

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
        """Number of observations in a batch.

        By default, every leading dim of the outputs but the last is a batch dim,
        and the targets share them. So this counts the leading dims of `y`, e.g.
        `B * L` for outputs `(B, L, K)` and targets `(B, L)` or `(B, L, K)`. The
        batch size comes from the targets, so `f` may come from fewer data, as long
        as it has the same number of dims.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`, possibly of a single datum.
        y : torch.Tensor
            targets of the batch.

        Returns
        -------
        n_obs : int
        """
        return y.shape[: f.ndim - 1].numel()

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

    By default, the noise enters through `sigma_noise` of the Laplace approximation, so
    the other quantities assume \\(\\sigma = 1\\): the output Hessian is \\(I\\), the
    inverse link is the identity, samples are \\(y \\sim \\mathcal{N}(f, I)\\), and the
    output gradient is \\(f - y\\). A float `sigma_noise` gives a normalizer with the
    default dtype.

    For a known noise std per output, \\(\\sigma \\in \\mathbb{R}^K\\), pass a 1-D tensor
    as `sigma_noise`. The likelihood is then
    \\(\\mathcal{N}(y \\mid f, \\operatorname{diag}(\\sigma^2))\\) with the loss

    $$
        \\frac{1}{2} \\sum_{n=1}^N \\sum_{k=1}^K
        \\frac{(y_{nk} - f_{nk})^2}{\\sigma_k^2},
    $$

    the output Hessian \\(\\operatorname{diag}(\\sigma^{-2})\\), and the normalizer
    \\(N \\sum_k \\log(\\sigma_k \\sqrt{2\\pi})\\). The samples and the output gradient
    use it too. The vector is fixed when a Laplace approximation is fit, and the
    `sigma_noise` of the Laplace approximation must then stay 1. It is a hyperparameter
    of the likelihood, so checkpoints store it.

    Parameters
    ----------
    sigma_noise : torch.Tensor, optional
        known noise std per output `(outputs,)`. All entries must be positive and
        finite. `None` leaves the noise to the Laplace approximation.

    Raises
    ------
    ValueError
        if `sigma_noise` is not a floating 1-D tensor with positive, finite entries.
    """

    name: ClassVar[str] = "regression"

    # Class-level default keeps subclasses working that override `__init__` without
    # calling `super().__init__()`
    sigma_noise: torch.Tensor | None = None
    """Noise std per output `(outputs,)`, or `None` for the scalar noise of the Laplace
    approximation."""

    def __init__(self, *, sigma_noise: torch.Tensor | None = None) -> None:
        if sigma_noise is not None:
            if (
                not torch.is_tensor(sigma_noise)
                or sigma_noise.ndim != 1
                or not sigma_noise.is_floating_point()
            ):
                raise ValueError(
                    "The noise std per output must be a floating 1-D tensor "
                    f"`(outputs,)`, but it is {sigma_noise!r}."
                )

            if not bool((sigma_noise.isfinite() & (sigma_noise > 0)).all()):
                raise ValueError(
                    "The noise std per output must be positive and finite, but it is "
                    f"{sigma_noise}."
                )

        self.sigma_noise = sigma_noise

    def hyperparameters(self) -> dict[str, Any]:
        if self.sigma_noise is None:
            return {}

        return {"sigma_noise": self.sigma_noise}

    def _sigma_like(self, f: torch.Tensor) -> torch.Tensor | float:
        # The noise std per output with the dtype and device of `f`, or the float 1
        if self.sigma_noise is None:
            return 1.0

        return self.sigma_noise.to(dtype=f.dtype, device=f.device)

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Normal(f, self._sigma_like(f))

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        if self.sigma_noise is None:
            return nn.MSELoss(reduction=reduction)

        return _WhitenedMSELoss(self.sigma_noise, reduction)

    @property
    def loss_factor(self) -> float:
        return 0.5

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        if y.ndim != f.ndim:
            raise ValueError(
                f"The model's output has {f.ndim} dims but the target has {y.ndim} dims."
            )

        if self.sigma_noise is not None and len(self.sigma_noise) != f.shape[-1]:
            raise ValueError(
                f"The model has {f.shape[-1]} outputs, but the noise std per output "
                f"has {len(self.sigma_noise)} entries."
            )

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        return y.shape[:-1].numel()

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        # The std per output replaces the scalar one, which then stays at 1
        if self.sigma_noise is not None:
            sigma = self.sigma_noise.to(
                dtype=_float_dtype(sigma_noise),
                device=getattr(sigma_noise, "device", None),
            )

            return n_obs * torch.log(sigma * sqrt(2 * pi)).sum()

        sigma_noise = torch.as_tensor(sigma_noise, dtype=_float_dtype(sigma_noise))

        return n_obs * n_outputs * torch.log(sigma_noise * sqrt(2 * pi))

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        K = f.shape[-1]

        if self.sigma_noise is not None:
            precision = self._sigma_like(f).square().reciprocal()

            return torch.diag_embed(precision).expand(*f.shape[:-1], K, K)

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

        return f + self._sigma_like(f) * noise

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (f - y) / self._sigma_like(f) ** 2

    @property
    def is_gaussian(self) -> bool:
        return True


def _noise_per_output(likelihood: Likelihood) -> torch.Tensor | None:
    # The known noise std per output of a Gaussian likelihood, if it has one
    return likelihood.hyperparameters().get("sigma_noise")


class _WhitenedMSELoss(nn.Module):
    # `nn.MSELoss` of the outputs and targets divided by the noise std per output
    def __init__(self, sigma_noise: torch.Tensor, reduction: str = "sum") -> None:
        super().__init__()

        self.sigma_noise = sigma_noise.detach().clone()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # `Likelihood.loss` doesn't move the loss to the outputs, so cast here
        sigma = self.sigma_noise.to(dtype=input.dtype, device=input.device)

        return nn.functional.mse_loss(
            input / sigma, target / sigma, reduction=self.reduction
        )


class HeteroscedasticGaussianLikelihood(Likelihood):
    """Heteroscedastic Gaussian likelihood, i.e. regression where the network also
    predicts the noise, as trained with `nn.GaussianNLLLoss`.

    The network returns a mean \\(\\mu(x) \\in \\mathbb{R}^K\\) and a variance
    \\(\\sigma^2(x)\\), as the pair `(mean, var)` that `nn.GaussianNLLLoss` takes. A
    target then has the likelihood

    $$
        p(y \\mid x) = \\mathcal{N}\\left(y \\mid \\mu(x),
        \\operatorname{diag}(\\sigma^2(x))\\right).
    $$

    The variance has the shape of the mean, or it is shared by the outputs of a
    datum, see [`join_mean_variance`][laplace.utils.heteroscedastic_utils.join_mean_variance]. The
    targets have the shape of the mean, and every entry of the targets but the last
    dim is an observation. The options `eps` and `full` are those of
    `nn.GaussianNLLLoss`: the variance is clamped at `eps` from below, and `full`
    adds the constant \\(\\tfrac{1}{2} \\log 2\\pi\\) of each output to the loss.
    The loss is the summed `nn.GaussianNLLLoss`,

    $$
        \\frac{1}{2} \\sum_{n=1}^N \\sum_{k=1}^K \\left(\\log \\sigma_{nk}^2
        + \\frac{(y_{nk} - \\mu_{nk})^2}{\\sigma_{nk}^2}\\right),
    $$

    which is the negative log likelihood up to the normalizer
    \\(N K \\log \\sqrt{2\\pi}\\), or zero with `full=True`. So the loss factor is
    \\(1\\).

    The Laplace approximation is over the mean only, as in Immer et al. (2023): the
    variance is held at its value under the MAP parameters. The GGN is then

    $$
        \\sum_{n=1}^N J_\\mu(x_n)^\\top \\operatorname{diag}\\left(\\sigma^{-2}(x_n)
        \\right) J_\\mu(x_n),
    $$

    and parameters that only the variance depends on get no curvature from the data,
    only from the prior. The predictive `la(x)` returns `f_mean, f_alea, f_epis`: the
    mean, the network's variance, which is the aleatoric uncertainty, and the
    covariance of the mean under the posterior, which is the epistemic uncertainty.
    The predictive variance of the targets is `f_alea` plus the diagonal of
    `f_epis`.

    Inside the library, the outputs are `(batch, ..., 2 * K)`, the mean followed by
    the variance, and every method of this class takes them in that form.
    `Laplace` joins the pair of the network while it runs the model. The output
    Hessian and the output gradient are zero for the variance, and for the mean they
    are \\(\\operatorname{diag}(\\sigma^{-2})\\) and \\((\\mu - y) / \\sigma^2\\).
    `inverse_link` returns the mean, and the predictive likelihood is a
    `GaussianLikelihood` over it. `loss_fn` also takes the pair, so
    `marglik_training` can train the network with it.

    Weight-space Laplace supports the `diag`, `full`, and `kron` structures with the
    Curvlinops backends, and `diag` and `full` with any GGN or EF backend. Last-layer
    Laplace and the function-space methods don't support this likelihood.

    Parameters
    ----------
    eps : float, default=1e-6
        lower bound of the variance, as in `nn.GaussianNLLLoss`.
    full : bool, default=False
        whether the loss includes the constant, as in `nn.GaussianNLLLoss`.

    Raises
    ------
    ValueError
        if `eps` is not positive.

    References
    ----------
    Immer, A., Palumbo, E., Marx, A., & Vogt, J. E. [Effective Bayesian
    Heteroscedastic Regression with Deep Neural Networks](
    https://openreview.net/forum?id=A6EquH0enk). NeurIPS 2023.
    """

    name: ClassVar[str] = "heteroscedastic_gaussian"

    # Class-level defaults keep subclasses working that override `__init__` without
    # calling `super().__init__()`
    eps: float = 1e-6
    """Lower bound of the variance."""

    full: bool = False
    """Whether the loss includes the constant."""

    def __init__(self, eps: float = 1e-6, full: bool = False) -> None:
        if not eps > 0:
            raise ValueError(
                f"The variance bound eps must be positive, but it is {eps}."
            )

        self.eps = eps
        self.full = full

    def hyperparameters(self) -> dict[str, Any]:
        hyperparameters: dict[str, Any] = {}

        if self.eps != 1e-6:
            hyperparameters["eps"] = self.eps

        if self.full:
            hyperparameters["full"] = self.full

        return hyperparameters

    def split(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the joined outputs into the mean and the variance.

        Parameters
        ----------
        f : torch.Tensor
            `(batch, ..., 2 * K)`

        Returns
        -------
        mean : torch.Tensor
            `(batch, ..., K)`
        var : torch.Tensor
            `(batch, ..., K)`, as the network returns it, without the clamp at
            `eps`.
        """
        mean, var = f.chunk(2, dim=-1)

        return mean, var

    def _precision(self, f: torch.Tensor) -> torch.Tensor:
        # \(1 / \sigma^2\) with the variance clamped at `eps`, as in the loss
        return self.split(f)[1].clamp(min=self.eps).reciprocal()

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        mean, var = self.split(f)

        return torch.distributions.Normal(mean, var.clamp(min=self.eps).sqrt())

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return HeteroscedasticGaussianNLLLoss(self.eps, self.full, reduction)

    @property
    def loss_factor(self) -> float:
        return 1.0

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        if f.shape[-1] % 2 != 0:
            raise ValueError(
                f"{type(self).__name__} expects the joined mean and variance, with an "
                f"even number of outputs, but the outputs have shape "
                f"{tuple(f.shape)}. Return the pair `(mean, var)` from the model."
            )

        if f.ndim != y.ndim or y.shape[-1] != f.shape[-1] // 2:
            raise ValueError(
                f"{type(self).__name__} expects targets with the shape of the mean, "
                f"{tuple(f.shape[:-1]) + (f.shape[-1] // 2,)}, but the targets have "
                f"shape {tuple(y.shape)}."
            )

    def log_normalizer(
        self, n_obs: int, n_outputs: int, sigma_noise: torch.Tensor
    ) -> torch.Tensor:
        """Constant that turns the negative loss into the log likelihood.

        Parameters
        ----------
        n_obs : int
            number of observations
        n_outputs : int
            number of joined outputs \\(2 K\\), the mean and the variance.
        sigma_noise : torch.Tensor
            ignored, since the network predicts the noise. It sets the dtype and
            device of the result.

        Returns
        -------
        c : torch.Tensor
            \\(N K \\log \\sqrt{2\\pi}\\), or zero with `full=True`.
        """
        c = 0.0 if self.full else n_obs * (n_outputs // 2) * log(sqrt(2 * pi))

        return _zero_like(sigma_noise) + c

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        precision = self._precision(f)

        return torch.diag_embed(torch.cat([precision, torch.zeros_like(precision)], -1))

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return self.split(f)[0]

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        mean, var = self.split(f)
        noise = torch.randn(
            mean.shape, device=f.device, dtype=f.dtype, generator=generator
        )

        return mean + var.clamp(min=self.eps).sqrt() * noise

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = self.split(f)[0]
        g = (mean - y) * self._precision(f)

        return torch.cat([g, torch.zeros_like(g)], dim=-1)

    @property
    def predictive_likelihood(self) -> Likelihood:
        # The predictive is a Gaussian over the mean, see `__call__`
        return GaussianLikelihood()


def _is_heteroscedastic(likelihood: Likelihood) -> bool:
    # The library dispatches on the name, as for the other built-in likelihoods
    return likelihood.name == HeteroscedasticGaussianLikelihood.name


def _flatten_categorical_targets(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Flatten categorical targets to what `nn.CrossEntropyLoss` takes for the logits
    `f` of shape `(M, K)`, where `M` is the number of positions in `f`.

    The kind of the targets follows from their number of dims. Class probabilities
    have as many dims as `f` and keep their class dim, while class indices have one
    dim less. So `f` must still have its batch dims, or a class dim could be taken
    for a position.

    Parameters
    ----------
    f : torch.Tensor
        logits `(batch, ..., K)`, not flattened.
    y : torch.Tensor
        class indices `(batch, ...)` or class probabilities `(batch, ..., K)`.

    Returns
    -------
    y : torch.Tensor
        `(M, K)` for class probabilities, and `(M,)` for class indices, with `M` the
        product of the leading dims of `f`.
    """
    if y.ndim == f.ndim:
        return y.reshape(-1, f.shape[-1])

    return y.reshape(-1)


class CategoricalLikelihood(Likelihood):
    """Categorical likelihood with a softmax link, i.e. classification.

    With logits \\(f \\in \\mathbb{R}^C\\) and class probabilities
    \\(p = \\operatorname{softmax}(f)\\), a class \\(y\\) has the likelihood
    \\(p(y \\mid f) = p_y\\). The loss is the cross-entropy \\(-\\log p_y\\), or
    \\(-\\sum_c y_c \\log p_c\\) for class probabilities \\(y\\), which is already the
    negative log likelihood. So the loss factor is \\(1\\) and the normalizer is zero.

    The targets are class indices `(batch, ...)` or class probabilities with the
    shape of the logits, e.g. one-hot or soft labels. As in `nn.CrossEntropyLoss`,
    class probabilities must be floating point, and integer targets with the shape of
    the logits raise a `ValueError`: call `.float()` on one-hot targets. For logits
    with extra batch dims, e.g. `(B, L, C)` for sequences, `loss` flattens the batch
    dims, since `CrossEntropyLoss` expects the class axis at dim 1. Every position is
    then an observation; the class dim of probability targets does not count.

    The options `weight`, `ignore_index`, and `label_smoothing` are those of
    `nn.CrossEntropyLoss`, with the same defaults and meaning. Let \\(q\\) be the
    one-hot index target or the probability target, smoothed as
    \\(q \\leftarrow (1 - \\varepsilon)\\, q + \\varepsilon / C\\) with
    \\(\\varepsilon\\) the label smoothing. With class weights \\(w\\), the loss of a
    position is

    $$
        \\ell(f, y) = -m \\sum_c w_c q_c \\log p_c,
    $$

    where \\(m = [y \\neq \\text{ignore\\_index}]\\) for index targets and \\(m = 1\\)
    for probability targets, as in torch. Without weights, \\(w = 1\\). Only positions
    with \\(m = 1\\) count as observations. With the per-position scale
    \\(s = w^\\top q\\), or \\(s = 1\\) without weights, the output gradient is

    $$
        \\nabla_f \\ell = m \\left(s\\, p - w \\odot q\\right),
    $$

    which is \\(p - y\\) for one-hot \\(y\\) and default options, and the output
    Hessian is

    $$
        \\nabla^2_f \\ell = m\\, s \\left(\\operatorname{diag}(p) - p p^\\top\\right).
    $$

    Without targets, `output_hessian` returns \\(\\operatorname{diag}(p) - p p^\\top\\)
    without \\(m\\) and \\(s\\). The weight is cast to the dtype and device of the
    logits wherever it is used. Class weights and label smoothing make this a
    generalized likelihood, so `log_marginal_likelihood` is no longer a true model
    evidence.

    `distribution`, the inverse link (the softmax), and `sample` (one-hot draws
    \\(y \\sim \\operatorname{Cat}(p)\\)) ignore the options, since they describe the
    model's predictive, which the options leave unchanged.

    Parameters
    ----------
    weight : torch.Tensor, optional
        class weights of shape `(C,)`, as in `nn.CrossEntropyLoss`.
    ignore_index : int, default=-100
        index target to leave out, as in `nn.CrossEntropyLoss`.
    label_smoothing : float, default=0.0
        label smoothing \\(\\varepsilon \\in [0, 1]\\), as in `nn.CrossEntropyLoss`.

    Raises
    ------
    ValueError
        if `weight` is not 1-D or `label_smoothing` is outside \\([0, 1]\\).
    """

    name: ClassVar[str] = "classification"

    # Class-level defaults keep subclasses working that override `__init__` without
    # calling `super().__init__()`
    weight: torch.Tensor | None = None
    """Class weights `(C,)`, or `None` for equal weights."""

    ignore_index: int = -100
    """Index target to leave out."""

    label_smoothing: float = 0.0
    """Label smoothing \\(\\varepsilon\\)."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        if weight is not None and weight.ndim != 1:
            raise ValueError(
                f"The class weights must be 1-D, but they have shape "
                f"{tuple(weight.shape)}."
            )

        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(
                f"The label smoothing must be in [0, 1], but it is {label_smoothing}."
            )

        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def hyperparameters(self) -> dict[str, Any]:
        hyperparameters: dict[str, Any] = {}

        if self.weight is not None:
            hyperparameters["weight"] = self.weight

        if self.ignore_index != -100:
            hyperparameters["ignore_index"] = self.ignore_index

        if self.label_smoothing != 0.0:
            hyperparameters["label_smoothing"] = self.label_smoothing

        return hyperparameters

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=f)

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return self._criterion(reduction, self.weight)

    def _criterion(
        self, reduction: str, weight: torch.Tensor | None
    ) -> nn.CrossEntropyLoss:
        """`nn.CrossEntropyLoss` with the options of this likelihood and `weight`."""
        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=reduction,
        )

    def _weight_like(self, f: torch.Tensor) -> torch.Tensor | None:
        """The class weights with the dtype and device of the logits `f`."""
        if self.weight is None:
            return None

        return self.weight.to(dtype=f.dtype, device=f.device)

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

        # As in `nn.CrossEntropyLoss`, class probabilities are floats. Integer targets
        # with the shape of the logits are most likely a one-hot encoding that was not
        # converted, which torch rejects too.
        if y.ndim == f.ndim and not y.is_floating_point():
            raise ValueError(
                f"The targets have the shape {tuple(y.shape)} of the logits, so they "
                f"are read as class probabilities, which must be floating point, but "
                f"their dtype is {y.dtype}. Call `.float()` on one-hot targets, or "
                f"pass class indices of shape {tuple(f.shape[:-1])}."
            )

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_targets(f, y)
        y = _flatten_categorical_targets(f, y)
        f = f.reshape(-1, f.shape[-1])

        return self._criterion("sum", self._weight_like(f))(f, y)

    def n_observations(self, f: torch.Tensor, y: torch.Tensor) -> int:
        if y.ndim == f.ndim:
            return y.shape[:-1].numel()

        return int((y != self.ignore_index).sum())

    def _targets(
        self, f: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Smoothed target distribution \\(q\\) and mask \\(m\\) of the targets `y`.

        Parameters
        ----------
        f : torch.Tensor
            logits `(batch, ..., C)`
        y : torch.Tensor
            class indices `(batch, ...)` or class probabilities `(batch, ..., C)`

        Returns
        -------
        q : torch.Tensor
            `(batch, ..., C)`, with the dtype of `f`.
        m : torch.Tensor or None
            `(batch, ...)` mask of the positions that count, or `None` for probability
            targets, which have no ignored positions.

        Raises
        ------
        ValueError
            if the targets do not fit the logits, see `check_targets`.
        """
        self.check_targets(f, y)
        C = f.shape[-1]

        if y.ndim == f.ndim:
            q, m = y.to(f.dtype), None
        else:
            m = y != self.ignore_index
            # Ignored positions may hold an index outside the classes, e.g. -100
            idx = torch.where(m, y, torch.zeros_like(y))
            q = nn.functional.one_hot(idx, C).to(f.dtype)

        if self.label_smoothing != 0.0:
            q = (1 - self.label_smoothing) * q + self.label_smoothing / C

        return q, m

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = self.inverse_link(f)
        H = torch.diag_embed(p) - torch.einsum("...k,...c->...kc", p, p)

        if y is None:
            return H

        q, m = self._targets(f, y)
        weight = self._weight_like(f)

        if weight is not None:
            H = (q @ weight)[..., None, None] * H

        if m is not None:
            H = m[..., None, None] * H

        return H

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
        p = self.inverse_link(f)
        q, m = self._targets(f, y)
        weight = self._weight_like(f)

        if weight is None:
            g = p - q
        else:
            g = (q @ weight)[..., None] * p - weight * q

        if m is not None:
            g = m[..., None] * g

        return g


class BernoulliLikelihood(Likelihood):
    """Bernoulli likelihood with a sigmoid link, e.g. for binary or multi-label
    classification and per-pixel masks.

    With logits \\(f \\in \\mathbb{R}^K\\) and probabilities \\(p = \\sigma(f)\\), every
    output is an independent binary target \\(y_k \\in \\{0, 1\\}\\) with
    \\(p(y_k \\mid f_k) = p_k^{y_k} (1 - p_k)^{1 - y_k}\\). The loss is the binary
    cross-entropy of `nn.BCEWithLogitsLoss`, which is already the negative log
    likelihood. So the loss factor is \\(1\\) and the normalizer is zero.

    The targets are floats with the shape of the logits, `(batch, ..., outputs)`.
    As in torch, they may be soft labels in \\([0, 1]\\). The outputs are on the last
    dim, so an observation is one row of outputs. For logits `(B, L, K)`, there are
    \\(B L\\) observations.

    The options `weight` and `pos_weight` are those of `nn.BCEWithLogitsLoss`, with
    the same defaults and meaning, except that they are scalars or 1-D with one
    entry per output. With \\(w\\) the weight, \\(p_w\\) the positive weight, both
    \\(1\\) by default, and \\(a = p_w\\, y + 1 - y\\), the loss of an output is

    $$
        \\ell(f, y) = -w \\left(p_w\\, y \\log p + (1 - y) \\log (1 - p)\\right).
    $$

    Its gradient is

    $$
        \\nabla_f \\ell = w \\left(a\\, p - p_w\\, y\\right),
    $$

    which is \\(p - y\\) with the default options, and the output Hessian is the
    diagonal matrix

    $$
        \\nabla^2_f \\ell = \\operatorname{diag}\\left(w\\, a\\, p (1 - p)\\right).
    $$

    Without targets, `output_hessian` returns \\(\\operatorname{diag}(p (1 - p))\\)
    without \\(w\\) and \\(a\\). The options are cast to the dtype and device of the
    logits wherever they are used. They make this a generalized likelihood, so
    `log_marginal_likelihood` is no longer a true model evidence.

    `distribution`, the inverse link (the sigmoid), and `sample` (draws
    \\(y \\sim \\operatorname{Bernoulli}(p)\\)) ignore the options, since they describe
    the model's predictive, which the options leave unchanged. The `log_prob` of the
    distribution rejects soft labels, so the loss and the output gradient don't use
    it.

    Parameters
    ----------
    weight : torch.Tensor, optional
        rescaling weight of the loss, a scalar or of shape `(outputs,)`, as in
        `nn.BCEWithLogitsLoss`. A weight per example can't follow a shuffled data
        loader, so it is rejected.
    pos_weight : torch.Tensor, optional
        weight of the positive targets, a scalar or of shape `(outputs,)`, as in
        `nn.BCEWithLogitsLoss`.

    Raises
    ------
    ValueError
        if `weight` or `pos_weight` has more than one dim.
    """

    name: ClassVar[str] = "bernoulli"

    # Class-level defaults keep subclasses working that override `__init__` without
    # calling `super().__init__()`
    weight: torch.Tensor | None = None
    """Rescaling weight `()` or `(outputs,)`, or `None` for equal weights."""

    pos_weight: torch.Tensor | None = None
    """Weight of the positive targets `()` or `(outputs,)`, or `None` for one."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        for option, value in (("weight", weight), ("pos_weight", pos_weight)):
            if value is not None and value.ndim > 1:
                raise ValueError(
                    f"The option {option} must be a scalar or 1-D with one entry per "
                    f"output, but it has shape {tuple(value.shape)}."
                )

        self.weight = weight
        self.pos_weight = pos_weight

    def hyperparameters(self) -> dict[str, Any]:
        hyperparameters: dict[str, Any] = {}

        if self.weight is not None:
            hyperparameters["weight"] = self.weight

        if self.pos_weight is not None:
            hyperparameters["pos_weight"] = self.pos_weight

        return hyperparameters

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Bernoulli(logits=f)

    def loss_fn(self, reduction: str = "sum") -> nn.Module:
        return nn.BCEWithLogitsLoss(
            weight=self.weight, pos_weight=self.pos_weight, reduction=reduction
        )

    def _options_like(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """The weight \\(w\\) and the positive weight \\(p_w\\) with the dtype and
        device of the logits `f`, or the float \\(1\\) for an option that is unset.
        """
        w, pw = self.weight, self.pos_weight

        return (
            1.0 if w is None else w.to(dtype=f.dtype, device=f.device),
            1.0 if pw is None else pw.to(dtype=f.dtype, device=f.device),
        )

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        # Like torch, which doesn't broadcast the targets to the logits and rejects
        # integer targets
        if y.shape != f.shape:
            raise ValueError(
                f"{type(self).__name__} expects targets of shape {tuple(f.shape)} "
                f"like the outputs, but the targets have shape {tuple(y.shape)}."
            )

        if not y.is_floating_point():
            raise ValueError(
                f"{type(self).__name__} expects floating-point targets in [0, 1], "
                f"but they have dtype {y.dtype}."
            )

    def loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_targets(f, y)

        return self.loss_fn().to(f)(f, y.to(f.dtype))

    def output_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = torch.sigmoid(f) * torch.sigmoid(-f)

        if y is not None:
            self.check_targets(f, y)
            y = y.to(f.dtype)
            w, pw = self._options_like(f)
            h = w * (pw * y + 1 - y) * h

        return torch.diag_embed(h)

    def inverse_link(self, f: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(f)

    def sample(
        self, f: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.bernoulli(self.inverse_link(f), generator=generator)

    def output_gradient(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_targets(f, y)
        y = y.to(f.dtype)
        w, pw = self._options_like(f)

        return w * ((pw * y + 1 - y) * self.inverse_link(f) - pw * y)


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
    categorical, and so is the count of observations: one per probability target or
    per index target that isn't ignored. The predictive is a `GaussianLikelihood`
    over the reward \\(r(x)\\).
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
        """Count the observed pairs, as the categorical likelihood of the fit does.

        Parameters
        ----------
        f : torch.Tensor
            rewards of the pairs `(batch, ..., 2)`.
        y : torch.Tensor
            class indices `(batch, ...)` or class probabilities `(batch, ..., 2)`.

        Returns
        -------
        n_observations : int
        """
        return self._fit.n_observations(f, y)

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


def get_likelihood(likelihood: Likelihood | str | nn.Module) -> Likelihood:
    """Turn a likelihood name or a torch criterion into a likelihood object.

    Parameters
    ----------
    likelihood : Likelihood or str or nn.Module
        an instance, which is returned as is, one of `'regression'`,
        `'classification'`, `'reward_modeling'`, `'bernoulli'`, and
        `'heteroscedastic_gaussian'`, or a torch criterion. `nn.MSELoss` becomes a
        `GaussianLikelihood`, `nn.CrossEntropyLoss` a `CategoricalLikelihood` with the
        criterion's `weight`, `ignore_index`, and `label_smoothing`,
        `nn.BCEWithLogitsLoss` a `BernoulliLikelihood` with its `weight` and
        `pos_weight`, and `nn.GaussianNLLLoss` a `HeteroscedasticGaussianLikelihood`
        with its `eps` and `full`. The criterion's `reduction` is ignored, since
        Laplace always sums the loss.

    Returns
    -------
    likelihood : Likelihood

    Raises
    ------
    ValueError
        if `likelihood` is neither a likelihood instance, a known name, nor one of
        the four criteria above. Subclasses of the criteria are not accepted. Also
        if the `weight` or `pos_weight` of a `nn.BCEWithLogitsLoss` has more than one
        dim.
    """
    if isinstance(likelihood, Likelihood):
        return likelihood

    if isinstance(likelihood, str):
        for cls in (
            GaussianLikelihood,
            CategoricalLikelihood,
            BradleyTerryLikelihood,
            BernoulliLikelihood,
            HeteroscedasticGaussianLikelihood,
        ):
            if cls.name == _name_of(likelihood):
                return cls()

    if type(likelihood) is nn.MSELoss:
        return GaussianLikelihood()

    if type(likelihood) is nn.GaussianNLLLoss:
        return HeteroscedasticGaussianLikelihood(
            eps=likelihood.eps, full=likelihood.full
        )

    if type(likelihood) is nn.CrossEntropyLoss:
        return CategoricalLikelihood(
            weight=likelihood.weight,
            ignore_index=likelihood.ignore_index,
            label_smoothing=likelihood.label_smoothing,
        )

    if type(likelihood) is nn.BCEWithLogitsLoss:
        return BernoulliLikelihood(
            weight=likelihood.weight, pos_weight=likelihood.pos_weight
        )

    raise ValueError(f"Invalid likelihood type {likelihood}")


Likelihood.REGRESSION = GaussianLikelihood()
Likelihood.CLASSIFICATION = CategoricalLikelihood()
Likelihood.REWARD_MODELING = BradleyTerryLikelihood()
Likelihood.BERNOULLI = BernoulliLikelihood()
