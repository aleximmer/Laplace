from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from math import log, pi, sqrt
from typing import Any, Callable

import numpy as np
import torch
import torchmetrics
import tqdm
from torch import nn
from torch.linalg import LinAlgError
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from laplace.curvature.asdfghjkl import AsdfghjklHessian
from laplace.curvature.curvature import CurvatureInterface
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN
from laplace.utils.enums import (
    Likelihood,
    LinkApprox,
    PredType,
    PriorStructure,
    TuningMethod,
)
from laplace.utils.matrix import Kron, KronDecomposed
from laplace.utils.metrics import RunningNLLMetric
from laplace.utils.utils import (
    fix_prior_prec_structure,
    invsqrt_precision,
    normal_samples,
    validate,
)

__all__ = [
    "BaseLaplace",
    "ParametricLaplace",
    "FullLaplace",
    "KronLaplace",
    "DiagLaplace",
    "LowRankLaplace",
]


class BaseLaplace:
    """Baseclass for all Laplace approximations in this library.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression', 'reward_modeling'}
        determines the log likelihood Hessian approximation.
        In the case of 'reward_modeling', it fits Laplace using the classification likelihood,
        then does prediction as in regression likelihood. The model needs to be defined accordingly:
        The forward pass during training takes `x.shape == (batch_size, 2, dim)` with
        `y.shape = (batch_size,)`. Meanwhile, during evaluation `x.shape == (batch_size, dim)`.
        Note that 'reward_modeling' only supports `KronLaplace` and `DiagLaplace`.
    sigma_noise : torch.Tensor or float, default=1
        observation noise for the regression setting; must be 1 for classification
    prior_precision : torch.Tensor or float, default=1
        prior precision of a Gaussian prior (= weight decay);
        can be scalar, per-layer, or diagonal in the most general case
    prior_mean : torch.Tensor or float, default=0
        prior mean of a Gaussian prior, useful for continual learning
    temperature : float, default=1
        temperature of the likelihood; lower temperature leads to more
        concentrated posterior and vice versa.
    enable_backprop: bool, default=False
        whether to enable backprop to the input `x` through the Laplace predictive.
        Useful for e.g. Bayesian optimization.
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    backend : subclasses of `laplace.curvature.CurvatureInterface`
        backend for access to curvature/Hessian approximations. Defaults to CurvlinopsGGN if None.
    backend_kwargs : dict, default=None
        arguments passed to the backend on initialization, for example to
        set the number of MC samples for stochastic approximations.
    asdl_fisher_kwargs : dict, default=None
        arguments passed to the ASDL backend specifically on initialization.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        asdl_fisher_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if likelihood not in [lik.value for lik in Likelihood]:
            raise ValueError(f"Invalid likelihood type {likelihood}")

        self.model: nn.Module = model
        self.likelihood: Likelihood | str = likelihood

        # Only do Laplace on params that require grad
        self.params: list[torch.Tensor] = []
        self.is_subset_params: bool = False
        for p in model.parameters():
            if p.requires_grad:
                self.params.append(p)
            else:
                self.is_subset_params = True

        self.n_params: int = sum(p.numel() for p in self.params)
        self.n_layers: int = len(self.params)
        self.prior_precision: float | torch.Tensor = prior_precision
        self.prior_mean: float | torch.Tensor = prior_mean
        if sigma_noise != 1 and likelihood != Likelihood.REGRESSION:
            raise ValueError("Sigma noise != 1 only available for regression.")

        self.sigma_noise: float | torch.Tensor = sigma_noise
        self.temperature: float = temperature
        self.enable_backprop: bool = enable_backprop

        # For models with dict-like inputs (e.g. Huggingface LLMs)
        self.dict_key_x = dict_key_x
        self.dict_key_y = dict_key_y

        if backend is None:
            backend = CurvlinopsGGN
        else:
            if self.is_subset_params and (
                "backpack" in backend.__name__.lower()
                or "asdfghjkl" in backend.__name__.lower()
            ):
                raise ValueError(
                    "If some grad are switched off, the BackPACK and Asdfghjkl backends"
                    " are not supported."
                )

        self._backend: CurvatureInterface | None = None
        self._backend_cls: type[CurvatureInterface] = backend
        self._backend_kwargs: dict[str, Any] = (
            dict() if backend_kwargs is None else backend_kwargs
        )
        self._asdl_fisher_kwargs: dict[str, Any] = (
            dict() if asdl_fisher_kwargs is None else asdl_fisher_kwargs
        )

        # log likelihood = g(loss)
        self.loss: float = 0.0
        self.n_outputs: int = 0
        self.n_data: int = 0

        # Declare attributes
        self._prior_mean: torch.Tensor
        self._prior_precision: torch.Tensor
        self._sigma_noise: torch.Tensor
        self._posterior_scale: torch.Tensor | None

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def backend(self) -> CurvatureInterface:
        if self._backend is None:
            likelihood = (
                "classification"
                if self.likelihood == "reward_modeling"
                else self.likelihood
            )
            self._backend = self._backend_cls(
                self.model,
                likelihood,
                dict_key_x=self.dict_key_x,
                dict_key_y=self.dict_key_y,
                **self._backend_kwargs,
            )
        return self._backend

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fit(self, train_loader: DataLoader) -> None:
        raise NotImplementedError

    def log_marginal_likelihood(
        self,
        prior_precision: torch.Tensor | None = None,
        sigma_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def log_likelihood(self) -> torch.Tensor:
        """Compute log likelihood on the training data after `.fit()` has been called.
        The log likelihood is computed on-demand based on the loss and, for example,
        the observation noise which makes it differentiable in the latter for
        iterative updates.

        Returns
        -------
        log_likelihood : torch.Tensor
        """
        factor = -self._H_factor
        if self.likelihood == "regression":
            # loss used is just MSE, need to add normalizer for gaussian likelihood
            c = (
                self.n_data
                * self.n_outputs
                * torch.log(torch.tensor(self.sigma_noise) * sqrt(2 * pi))
            )
            return factor * self.loss - c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

    def __call__(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str,
        link_approx: LinkApprox | str,
        n_samples: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def predictive(
        self,
        x: torch.Tensor,
        pred_type: PredType | str,
        link_approx: LinkApprox | str,
        n_samples: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self(x, pred_type, link_approx, n_samples)

    def _check_jacobians(self, Js: torch.Tensor) -> None:
        if not isinstance(Js, torch.Tensor):
            raise ValueError("Jacobians have to be torch.Tensor.")
        if not Js.device == self._device:
            raise ValueError("Jacobians need to be on the same device as Laplace.")
        m, k, p = Js.size()
        if p != self.n_params:
            raise ValueError("Invalid Jacobians shape for Laplace posterior approx.")

    @property
    def prior_precision_diag(self) -> torch.Tensor:
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        prior_prec: torch.Tensor = (
            self.prior_precision
            if isinstance(self.prior_precision, torch.Tensor)
            else torch.tensor(self.prior_precision)
        )

        if prior_prec.ndim == 0 or len(prior_prec) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)
        elif len(prior_prec) == self.n_params:  # diagonal
            return prior_prec
        elif len(prior_prec) == self.n_layers:  # per layer
            n_params_per_layer = [p.numel() for p in self.params]
            return torch.cat(
                [
                    prior * torch.ones(n_params, device=self._device)
                    for prior, n_params in zip(prior_prec, n_params_per_layer)
                ]
            )
        else:
            raise ValueError(
                "Mismatch of prior and model. Diagonal, scalar, or per-layer prior."
            )

    @property
    def prior_mean(self) -> torch.Tensor:
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean: float | torch.Tensor) -> None:
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.tensor(prior_mean, device=self._device)
        elif isinstance(prior_mean, torch.Tensor):
            if prior_mean.ndim == 0:
                self._prior_mean = prior_mean.reshape(-1).to(self._device)
            elif prior_mean.ndim == 1:
                if len(prior_mean) not in [1, self.n_params]:
                    raise ValueError("Invalid length of prior mean.")
                self._prior_mean = prior_mean
            else:
                raise ValueError("Prior mean has too many dimensions!")
        else:
            raise ValueError("Invalid argument type of prior mean.")

    @property
    def prior_precision(self) -> torch.Tensor:
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision: float | torch.Tensor):
        self._posterior_scale = None

        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
        elif isinstance(prior_precision, torch.Tensor):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(self._device)
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError(
                        "Length of prior precision does not align with architecture."
                    )
                self._prior_precision = prior_precision.to(self._device)
            else:
                raise ValueError(
                    "Prior precision needs to be at most one-dimensional tensor."
                )
        else:
            raise ValueError(
                "Prior precision either scalar or torch.Tensor up to 1-dim."
            )

    def optimize_prior_precision(
        self,
        pred_type: PredType | str,
        method: TuningMethod | str = TuningMethod.MARGLIK,
        n_steps: int = 100,
        lr: float = 1e-1,
        init_prior_prec: float | torch.Tensor = 1.0,
        prior_structure: PriorStructure | str = PriorStructure.DIAG,
        val_loader: DataLoader | None = None,
        loss: torchmetrics.Metric
        | Callable[[torch.Tensor], torch.Tensor | float]
        | None = None,
        log_prior_prec_min: float = -4,
        log_prior_prec_max: float = 4,
        grid_size: int = 100,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> None:
        """Optimize the prior precision post-hoc using the `method`
        specified by the user.

        Parameters
        ----------
        pred_type : PredType or str in {'glm', 'nn'}
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictiv. The GLM predictive is consistent with the
            curvature approximations used here.
        method : TuningMethod or str in {'marglik', 'gridsearch'}, default=PredType.MARGLIK
            specifies how the prior precision should be optimized.
        n_steps : int, default=100
            the number of gradient descent steps to take.
        lr : float, default=1e-1
            the learning rate to use for gradient descent.
        init_prior_prec : float or tensor, default=1.0
            initial prior precision before the first optimization step.
        prior_structure : PriorStructure or str in {'scalar', 'layerwise', 'diag'}, default=PriorStructure.SCALAR
            if init_prior_prec is scalar, the prior precision is optimized with this structure.
            otherwise, the structure of init_prior_prec is maintained.
        val_loader : torch.data.utils.DataLoader, default=None
            DataLoader for the validation set; each iterate is a training batch (X, y).
        loss : callable or torchmetrics.Metric, default=None
            loss function to use for CV. If callable, the loss is computed offline (memory intensive).
            If torchmetrics.Metric, running loss is computed (efficient). The default
            depends on the likelihood: `RunningNLLMetric()` for classification and
            reward modeling, running `MeanSquaredError()` for regression.
        log_prior_prec_min : float, default=-4
            lower bound of gridsearch interval.
        log_prior_prec_max : float, default=4
            upper bound of gridsearch interval.
        grid_size : int, default=100
            number of values to consider inside the gridsearch interval.
        link_approx : LinkApprox or str in {'mc', 'probit', 'bridge'}, default=LinkApprox.PROBIT
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only `'mc'` is possible.
        n_samples : int, default=100
            number of samples for `link_approx='mc'`.
        verbose : bool, default=False
            if true, the optimized prior precision will be printed
            (can be a large tensor if the prior has a diagonal covariance).
        progress_bar : bool, default=False
            whether to show a progress bar; updated at every batch-Hessian computation.
            Useful for very large model and large amount of data, esp. when `subset_of_weights='all'`.
        """
        likelihood = (
            Likelihood.CLASSIFICATION
            if self.likelihood == Likelihood.REWARD_MODELING
            else self.likelihood
        )

        if likelihood == Likelihood.CLASSIFICATION:
            warnings.warn(
                "By default `link_approx` is `probit`. Make sure to set it equals to "
                "the way you want to call `la(test_data, pred_type=..., link_approx=...)`."
            )

        if method == TuningMethod.MARGLIK:
            if val_loader is not None:
                warnings.warn(
                    "`val_loader` will be ignored when `method` == 'marglik'. "
                    "Do you mean to set `method = 'gridsearch'`?"
                )

            self.prior_precision = (
                init_prior_prec
                if isinstance(init_prior_prec, torch.Tensor)
                else torch.tensor(init_prior_prec)
            )

            if (
                len(self.prior_precision) == 1
                and prior_structure != PriorStructure.SCALAR
            ):
                self.prior_precision = fix_prior_prec_structure(
                    self.prior_precision.item(),
                    prior_structure,
                    self.n_layers,
                    self.n_params,
                    self._device,
                )

            log_prior_prec = self.prior_precision.log()
            log_prior_prec.requires_grad = True
            optimizer = torch.optim.Adam([log_prior_prec], lr=lr)

            if progress_bar:
                pbar = tqdm.trange(n_steps)
                pbar.set_description("[Optimizing marginal likelihood]")
            else:
                pbar = range(n_steps)

            for _ in pbar:
                optimizer.zero_grad()
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -self.log_marginal_likelihood(
                    prior_precision=prior_prec
                )
                neg_log_marglik.backward()
                optimizer.step()

            self.prior_precision = log_prior_prec.detach().exp()
        elif method == TuningMethod.GRIDSEARCH:
            if val_loader is None:
                raise ValueError("gridsearch requires a validation set DataLoader")

            interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)

            if loss is None:
                loss = (
                    torchmetrics.MeanSquaredError(num_outputs=self.n_outputs).to(
                        self._device
                    )
                    if likelihood == Likelihood.REGRESSION
                    else RunningNLLMetric().to(self._device)
                )

            self.prior_precision = self._gridsearch(
                loss,
                interval,
                val_loader,
                pred_type=pred_type,
                link_approx=link_approx,
                n_samples=n_samples,
                progress_bar=progress_bar,
            )
        else:
            raise ValueError("For now only marglik and gridsearch is implemented.")

        if verbose:
            print(f"Optimized prior precision is {self.prior_precision}.")

    def _gridsearch(
        self,
        loss: torchmetrics.Metric | Callable[[torch.Tensor], torch.Tensor | float],
        interval: torch.Tensor,
        val_loader: DataLoader,
        pred_type: PredType | str,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        assert callable(loss) or isinstance(loss, torchmetrics.Metric)

        results: list[float] = list()
        prior_precs: list[torch.Tensor] = list()
        pbar = tqdm.tqdm(interval, disable=not progress_bar)

        for prior_prec in pbar:
            self.prior_precision = prior_prec

            try:
                result = validate(
                    self,
                    val_loader,
                    loss,
                    pred_type=pred_type,
                    link_approx=link_approx,
                    n_samples=n_samples,
                    dict_key_y=self.dict_key_y,
                )
            except LinAlgError:
                result = np.inf
            except RuntimeError as err:
                if "not positive definite" in str(err):
                    result = np.inf
                else:
                    raise err

            if progress_bar:
                pbar.set_description(
                    f"[Grid search | prior_prec: {prior_prec:.3e}, loss: {result:.3f}]"
                )

            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @property
    def sigma_noise(self) -> torch.Tensor:
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise: float | torch.Tensor) -> None:
        self._posterior_scale = None

        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.tensor(sigma_noise, device=self._device)
        elif isinstance(sigma_noise, torch.Tensor):
            if sigma_noise.ndim == 0:
                self._sigma_noise = sigma_noise.to(self._device)
            elif sigma_noise.ndim == 1:
                if len(sigma_noise) > 1:
                    raise ValueError("Only homoscedastic output noise supported.")
                self._sigma_noise = sigma_noise[0].to(self._device)
            else:
                raise ValueError("Sigma noise needs to be scalar or 1-dimensional.")
        else:
            raise ValueError(
                "Invalid type: sigma noise needs to be torch.Tensor or scalar."
            )

    @property
    def _H_factor(self) -> torch.Tensor:
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 / self.temperature


class ParametricLaplace(BaseLaplace):
    """
    Parametric Laplace class.

    Subclasses need to specify how the Hessian approximation is initialized,
    how to add up curvature over training data, how to sample from the
    Laplace approximation, and how to compute the functional variance.

    A Laplace approximation is represented by a MAP which is given by the
    `model` parameter and a posterior precision or covariance specifying
    a Gaussian distribution \\(\\mathcal{N}(\\theta_{MAP}, P^{-1})\\).
    The goal of this class is to compute the posterior precision \\(P\\)
    which sums as
    \\[
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    \\]
    Every subclass implements different approximations to the log likelihood Hessians,
    for example, a diagonal one. The prior is assumed to be Gaussian and therefore we have
    a simple form for \\(\\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}} = P_0 \\).
    In particular, we assume a scalar, layer-wise, or diagonal prior precision so that in
    all cases \\(P_0 = \\textrm{diag}(p_0)\\) and the structure of \\(p_0\\) can be varied.
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "inputs_id",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        asdl_fisher_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            dict_key_x,
            dict_key_y,
            backend,
            backend_kwargs,
            asdl_fisher_kwargs,
        )
        if not hasattr(self, "H"):
            self._init_H()
            # posterior mean/mode
            self.mean: float | torch.Tensor = self.prior_mean

    def _init_H(self) -> None:
        raise NotImplementedError

    def _check_H_init(self) -> None:
        if getattr(self, "H", None) is None:
            raise AttributeError("Laplace not fitted. Run fit() first.")

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch, either `(X, y)` tensors or a dict-like
            object containing keys as expressed by `self.dict_key_x` and
            `self.dict_key_y`. `train_loader.dataset` needs to be set to access
            \\(N\\), size of the data set.
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        progress_bar : bool, default=False
            whether to show a progress bar; updated at every batch-Hessian computation.
            Useful for very large model and large amount of data, esp. when `subset_of_weights='all'`.
        """
        if override:
            self._init_H()
            self.loss: float | torch.Tensor = 0
            self.n_data: int = 0

        self.model.eval()

        self.mean: torch.Tensor = parameters_to_vector(self.params)
        if not self.enable_backprop:
            self.mean = self.mean.detach()

        data: (
            tuple[torch.Tensor, torch.Tensor] | MutableMapping[str, torch.Tensor | Any]
        ) = next(iter(train_loader))

        with torch.no_grad():
            if isinstance(data, MutableMapping):  # To support Huggingface dataset
                if "backpack" in self._backend_cls.__name__.lower() or (
                    isinstance(self, DiagLaplace) and self._backend_cls == CurvlinopsEF
                ):
                    raise ValueError(
                        "Currently DiagEF is not supported under CurvlinopsEF backend "
                        + "for custom models with non-tensor inputs "
                        + "(https://github.com/pytorch/functorch/issues/159). Consider "
                        + "using AsdlEF backend instead. The same limitation applies "
                        + "to all BackPACK backend"
                    )

                out = self.model(data)
            else:
                X = data[0]
                try:
                    out = self.model(X[:1].to(self._device))
                except (TypeError, AttributeError):
                    out = self.model(X.to(self._device))
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)

        N = len(train_loader.dataset)

        pbar = tqdm.tqdm(train_loader, disable=not progress_bar)
        pbar.set_description("[Computing Hessian]")

        for data in pbar:
            if isinstance(data, MutableMapping):  # To support Huggingface dataset
                X, y = data, data[self.dict_key_y].to(self._device)
            else:
                X, y = data
                X, y = X.to(self._device), y.to(self._device)
            self.model.zero_grad()
            loss_batch, H_batch = self._curv_closure(X, y, N=N)
            self.loss += loss_batch
            self.H += H_batch

        self.n_data += N

    @property
    def scatter(self) -> torch.Tensor:
        """Computes the _scatter_, a term of the log marginal likelihood that
        corresponds to L-2 regularization:
        `scatter` = \\((\\theta_{MAP} - \\mu_0)^{T} P_0 (\\theta_{MAP} - \\mu_0) \\).

        Returns
        -------
        scatter: torch.Tensor
        """
        delta = self.mean - self.prior_mean
        return (delta * self.prior_precision_diag) @ delta

    @property
    def log_det_prior_precision(self) -> torch.Tensor:
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)

        Returns
        -------
        log_det : torch.Tensor
        """
        return self.prior_precision_diag.log().sum()

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        """Compute log determinant of the posterior precision
        \\(\\log \\det P\\) which depends on the subclasses structure
        used for the Hessian approximation.

        Returns
        -------
        log_det : torch.Tensor
        """
        raise NotImplementedError

    @property
    def log_det_ratio(self) -> torch.Tensor:
        """Compute the log determinant ratio, a part of the log marginal likelihood.
        \\[
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        \\]

        Returns
        -------
        log_det_ratio : torch.Tensor
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    def square_norm(self, value) -> torch.Tensor:
        """Compute the square norm under post. Precision with `value-self.mean` as 𝛥:
        \\[
            \\Delta^\top P \\Delta
        \\]
        Returns
        -------
        square_form
        """
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """Compute the log probability under the (current) Laplace approximation.

        Parameters
        ----------
        value: torch.Tensor
        normalized : bool, default=True
            whether to return log of a properly normalized Gaussian or just the
            terms that depend on `value`.

        Returns
        -------
        log_prob : torch.Tensor
        """
        if not normalized:
            return -self.square_norm(value) / 2
        log_prob = (
            -self.n_params / 2 * log(2 * pi) + self.log_det_posterior_precision / 2
        )
        log_prob -= self.square_norm(value) / 2
        return log_prob

    def log_marginal_likelihood(
        self,
        prior_precision: torch.Tensor | None = None,
        sigma_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Laplace approximation to the log marginal likelihood subject
        to specific Hessian approximations that subclasses implement.
        Requires that the Laplace approximation has been fit before.
        The resulting torch.Tensor is differentiable in `prior_precision` and
        `sigma_noise` if these have gradients enabled.
        By passing `prior_precision` or `sigma_noise`, the current value is
        overwritten. This is useful for iterating on the log marginal likelihood.

        Parameters
        ----------
        prior_precision : torch.Tensor, optional
            prior precision if should be changed from current `prior_precision` value
        sigma_noise : torch.Tensor, optional
            observation noise standard deviation if should be changed

        Returns
        -------
        log_marglik : torch.Tensor
        """
        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            if self.likelihood != Likelihood.REGRESSION:
                raise ValueError("Can only change sigma_noise for regression.")

            self.sigma_noise = sigma_noise

        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)

    def __call__(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str = PredType.GLM,
        joint: bool = False,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        fitting: bool = False,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior predictive on input data `x`.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            `(batch_size, input_shape)` if tensor. If MutableMapping, must contain
            the said tensor.

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here. When Laplace is done only
            on subset of parameters (i.e. some grad are disabled),
            only `nn` predictive is supported.

        link_approx : {'mc', 'probit', 'bridge', 'bridge_norm'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible.

        joint : bool
            Whether to output a joint predictive distribution in regression with
            `pred_type='glm'`. If set to `True`, the predictive distribution
            has the same form as GP posterior, i.e. N([f(x1), ...,f(xm)], Cov[f(x1), ..., f(xm)]).
            If `False`, then only outputs the marginal predictive distribution.
            Only available for regression and GLM predictive.

        n_samples : int
            number of samples for `link_approx='mc'`.

        diagonal_output : bool
            whether to use a diagonalized posterior predictive on the outputs.
            Only works for `pred_type='glm'` when `joint=False` in regression.
            In the case of last-layer Laplace with a diagonal or Kron Hessian,
            setting this to `True` makes computation much(!) faster for large
            number of outputs.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used).

        fitting : bool, default=False
            whether or not this predictive call is done during fitting. Only useful for
            reward modeling: the likelihood is set to `"regression"` when `False` and
            `"classification"` when `True`.

        Returns
        -------
        predictive: torch.Tensor or tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor
            is returned with the mean and the predictive covariance.
        """
        if pred_type not in [pred for pred in PredType]:
            raise ValueError("Only glm and nn supported as prediction types.")

        if link_approx not in [la for la in LinkApprox]:
            raise ValueError(f"Unsupported link approximation {link_approx}.")

        if pred_type == PredType.NN and link_approx != LinkApprox.MC:
            raise ValueError(
                "Only mc link approximation is supported for nn prediction type."
            )

        if generator is not None:
            if (
                not isinstance(generator, torch.Generator)
                or generator.device != self._device
            ):
                raise ValueError("Invalid random generator (check type and device).")

        likelihood = self.likelihood
        if likelihood == Likelihood.REWARD_MODELING:
            likelihood = Likelihood.CLASSIFICATION if fitting else Likelihood.REGRESSION

        if pred_type == PredType.GLM:
            f_mu, f_var = self._glm_predictive_distribution(
                x,
                joint=joint and likelihood == Likelihood.REGRESSION,
                diagonal_output=diagonal_output
                and self.likelihood == Likelihood.REGRESSION,
            )

            if likelihood == Likelihood.REGRESSION:
                return f_mu, f_var

            if link_approx == LinkApprox.MC:
                return self.predictive_samples(
                    x,
                    pred_type="glm",
                    n_samples=n_samples,
                    diagonal_output=diagonal_output,
                ).mean(dim=0)
            elif link_approx == LinkApprox.PROBIT:
                kappa = 1 / torch.sqrt(1.0 + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
                return torch.softmax(kappa * f_mu, dim=-1)
            elif "bridge" in link_approx:
                # zero mean correction
                f_mu -= (
                    f_var.sum(-1)
                    * f_mu.sum(-1).reshape(-1, 1)
                    / f_var.sum(dim=(1, 2)).reshape(-1, 1)
                )
                f_var -= torch.einsum(
                    "bi,bj->bij", f_var.sum(-1), f_var.sum(-2)
                ) / f_var.sum(dim=(1, 2)).reshape(-1, 1, 1)

                # Laplace Bridge
                _, K = f_mu.size(0), f_mu.size(-1)
                f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)

                # optional: variance correction
                if link_approx == LinkApprox.BRIDGE_NORM:
                    f_var_diag_mean = f_var_diag.mean(dim=1)
                    f_var_diag_mean /= torch.as_tensor(
                        [K / 2], device=self._device
                    ).sqrt()
                    f_mu /= f_var_diag_mean.sqrt().unsqueeze(-1)
                    f_var_diag /= f_var_diag_mean.unsqueeze(-1)

                sum_exp = torch.exp(-f_mu).sum(dim=1).unsqueeze(-1)
                alpha = (1 - 2 / K + f_mu.exp() / K**2 * sum_exp) / f_var_diag
                return torch.nan_to_num(alpha / alpha.sum(dim=1).unsqueeze(-1), nan=1.0)
            else:
                raise ValueError(
                    "Prediction path invalid. Check the likelihood, pred_type, link_approx combination!"
                )
        else:
            if likelihood == Likelihood.REGRESSION:
                samples = self._nn_predictive_samples(x, n_samples, **model_kwargs)
                return samples.mean(dim=0), samples.var(dim=0)
            else:  # classification; the average is computed online
                return self._nn_predictive_classification(x, n_samples, **model_kwargs)

    def predictive_samples(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from the posterior predictive on input data `x`.
        Can be used, for example, for Thompson sampling.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            input data `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        n_samples : int
            number of samples

        diagonal_output : bool
            whether to use a diagonalized glm posterior predictive on the outputs.
            Only applies when `pred_type='glm'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        if pred_type not in PredType.__members__.values():
            raise ValueError("Only glm and nn supported as prediction types.")

        if pred_type == PredType.GLM:
            f_mu, f_var = self._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size(
                [f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]]
            )

            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)

            f_samples = normal_samples(f_mu, f_var, n_samples, generator)

            if self.likelihood == Likelihood.REGRESSION:
                return f_samples
            else:
                return torch.softmax(f_samples, dim=-1)

        else:  # 'nn'
            return self._nn_predictive_samples(x, n_samples, generator)

    @torch.enable_grad()
    def _glm_predictive_distribution(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        joint: bool = False,
        diagonal_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "asdl" in self._backend_cls.__name__.lower():
            # Asdl's doesn't support backprop over Jacobians
            # falling back to functorch
            warnings.warn(
                "ASDL backend is used which does not support backprop through "
                "the functional variance, but `self.enable_backprop = True`. "
                "Falling back to using `self.backend.functorch_jacobians` "
                "which can be memory intensive for large models."
            )

            Js, f_mu = self.backend.functorch_jacobians(
                X, enable_backprop=self.enable_backprop
            )
        else:
            Js, f_mu = self.backend.jacobians(X, enable_backprop=self.enable_backprop)

        if joint:
            f_mu = f_mu.flatten()  # (batch*out)
            f_var = self.functional_covariance(Js)  # (batch*out, batch*out)
        else:
            f_var = self.functional_variance(Js)  # (batch, out, out)

            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=-2, dim2=-1)

        return (
            (f_mu.detach(), f_var.detach())
            if not self.enable_backprop
            else (f_mu, f_var)
        )

    def _nn_predictive_samples(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        fs = list()
        for sample in self.sample(n_samples, generator):
            vector_to_parameters(sample, self.params)
            logits = self.model(
                X.to(self._device) if isinstance(X, torch.Tensor) else X, **model_kwargs
            )
            fs.append(logits.detach() if not self.enable_backprop else logits)

        vector_to_parameters(self.mean, self.params)
        fs = torch.stack(fs)

        if self.likelihood == Likelihood.CLASSIFICATION:
            fs = torch.softmax(fs, dim=-1)

        return fs

    def _nn_predictive_classification(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        py = 0.0
        for sample in self.sample(n_samples):
            vector_to_parameters(sample, self.params)
            logits = self.model(
                X.to(self._device) if isinstance(X, torch.Tensor) else X, **model_kwargs
            ).detach()
            py += torch.softmax(logits, dim=-1) / n_samples

        vector_to_parameters(self.mean, self.params)

        return py

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        """Compute functional variance for the `'glm'` predictive:
        `f_var[i] = Js[i] @ P.inv() @ Js[i].T`, which is a output x output
        predictive covariance matrix.
        Mathematically, we have for a single Jacobian
        \\(\\mathcal{J} = \\nabla_\\theta f(x;\\theta)\\vert_{\\theta_{MAP}}\\)
        the output covariance matrix
        \\( \\mathcal{J} P^{-1} \\mathcal{J}^T \\).

        Parameters
        ----------
        Js : torch.Tensor
            Jacobians of model output wrt parameters
            `(batch, outputs, parameters)`

        Returns
        -------
        f_var : torch.Tensor
            output covariance `(batch, outputs, outputs)`
        """
        raise NotImplementedError

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        """Compute functional covariance for the `'glm'` predictive:
        `f_cov = Js @ P.inv() @ Js.T`, which is a batch*output x batch*output
        predictive covariance matrix.

        This emulates the GP posterior covariance N([f(x1), ...,f(xm)], Cov[f(x1), ..., f(xm)]).
        Useful for joint predictions, such as in batched Bayesian optimization.

        Parameters
        ----------
        Js : torch.Tensor
            Jacobians of model output wrt parameters
            `(batch*outputs, parameters)`

        Returns
        -------
        f_cov : torch.Tensor
            output covariance `(batch*outputs, batch*outputs)`
        """
        raise NotImplementedError

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample from the Laplace posterior approximation, i.e.,
        \\( \\theta \\sim \\mathcal{N}(\\theta_{MAP}, P^{-1})\\).

        Parameters
        ----------
        n_samples : int, default=100
            number of samples

        generator : torch.Generator, optional
            random number generator to control the samples

        Returns
        -------
        samples: torch.Tensor
        """
        raise NotImplementedError

    def optimize_prior_precision(
        self,
        pred_type: PredType | str = PredType.GLM,
        method: TuningMethod | str = TuningMethod.MARGLIK,
        n_steps: int = 100,
        lr: float = 1e-1,
        init_prior_prec: float | torch.Tensor = 1.0,
        prior_structure: PriorStructure | str = PriorStructure.SCALAR,
        val_loader: DataLoader | None = None,
        loss: torchmetrics.Metric
        | Callable[[torch.Tensor], torch.Tensor | float]
        | None = None,
        log_prior_prec_min: float = -4,
        log_prior_prec_max: float = 4,
        grid_size: int = 100,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> None:
        assert pred_type in PredType.__members__.values()

        super().optimize_prior_precision(
            pred_type,
            method,
            n_steps,
            lr,
            init_prior_prec,
            prior_structure,
            val_loader,
            loss,
            log_prior_prec_min,
            log_prior_prec_max,
            grid_size,
            link_approx,
            n_samples,
            verbose,
            progress_bar,
        )

    @property
    def posterior_precision(self) -> torch.Tensor:
        """Compute or return the posterior precision \\(P\\).

        Returns
        -------
        posterior_prec : torch.Tensor
        """
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        self._check_H_init()
        state_dict = {
            "mean": self.mean,
            "H": self.H,
            "loss": self.loss,
            "prior_mean": self.prior_mean,
            "prior_precision": self.prior_precision,
            "sigma_noise": self.sigma_noise,
            "n_data": self.n_data,
            "n_outputs": self.n_outputs,
            "likelihood": self.likelihood,
            "temperature": self.temperature,
            "enable_backprop": self.enable_backprop,
            "cls_name": self.__class__.__name__,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Dealbreaker errors
        if self.__class__.__name__ != state_dict["cls_name"]:
            raise ValueError(
                "Loading a wrong Laplace type. Make sure `subset_of_weights` and"
                + " `hessian_structure` are correct!"
            )
        if self.n_params is not None and len(state_dict["mean"]) != self.n_params:
            raise ValueError(
                "Attempting to load Laplace with different number of parameters than the model."
                + " Make sure that you use the same `subset_of_weights` value and the same `.requires_grad`"
                + " switch on `model.parameters()`."
            )
        if self.likelihood != state_dict["likelihood"]:
            raise ValueError("Different likelihoods detected!")

        # Ignorable warnings
        if self.prior_mean is None and state_dict["prior_mean"] is not None:
            warnings.warn(
                "Loading non-`None` prior mean into a `None` prior mean. You might get wrong results."
            )
        if self.temperature != state_dict["temperature"]:
            warnings.warn(
                "Different `temperature` parameters detected. Some calculation might be off!"
            )
        if self.enable_backprop != state_dict["enable_backprop"]:
            warnings.warn(
                "Different `enable_backprop` values. You might encounter error when differentiating"
                + " the predictive mean and variance."
            )

        self.mean = state_dict["mean"]
        self.H = state_dict["H"]
        self.loss = state_dict["loss"]
        self.prior_mean = state_dict["prior_mean"]
        self.prior_precision = state_dict["prior_precision"]
        self.sigma_noise = state_dict["sigma_noise"]
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        setattr(self.model, "output_size", self.n_outputs)
        self.likelihood = state_dict["likelihood"]
        self.temperature = state_dict["temperature"]
        self.enable_backprop = state_dict["enable_backprop"]


class FullLaplace(ParametricLaplace):
    """Laplace approximation with full, i.e., dense, log likelihood Hessian approximation
    and hence posterior precision. Based on the chosen `backend` parameter, the full
    approximation can be, for example, a generalized Gauss-Newton matrix.
    Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `BaseLaplace` for the full interface.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "full")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            dict_key_x,
            dict_key_y,
            backend,
            backend_kwargs,
        )
        self._posterior_scale: torch.Tensor | None = None

    def _init_H(self) -> None:
        self.H: torch.Tensor = torch.zeros(
            self.n_params, self.n_params, device=self._device
        )

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.full(X, y, N=N)

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        self._posterior_scale = None
        super().fit(train_loader, override=override, progress_bar=progress_bar)

    def _compute_scale(self) -> None:
        self._posterior_scale = invsqrt_precision(self.posterior_precision)

    @property
    def posterior_scale(self) -> torch.Tensor:
        """Posterior scale (square root of the covariance), i.e.,
        \\(P^{-\\frac{1}{2}}\\).

        Returns
        -------
        scale : torch.tensor
            `(parameters, parameters)`
        """
        if self._posterior_scale is None:
            self._compute_scale()
        return self._posterior_scale

    @property
    def posterior_covariance(self) -> torch.Tensor:
        """Posterior covariance, i.e., \\(P^{-1}\\).

        Returns
        -------
        covariance : torch.tensor
            `(parameters, parameters)`
        """
        scale = self.posterior_scale
        return scale @ scale.T

    @property
    def posterior_precision(self) -> torch.Tensor:
        """Posterior precision \\(P\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters, parameters)`
        """
        self._check_H_init()
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        return self.posterior_precision.logdet()

    def square_norm(self, value: torch.Tensor) -> torch.Tensor:
        delta = value - self.mean
        return delta @ self.posterior_precision @ delta

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncp,pq,nkq->nck", Js, self.posterior_covariance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)
        return torch.einsum("np,pq,mq->nm", Js, self.posterior_covariance, Js)

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        # (n_samples, n_params) x (n_params, n_params) -> (n_samples, n_params)
        samples = samples @ self.posterior_scale
        return self.mean.reshape(1, self.n_params) + samples


class KronLaplace(ParametricLaplace):
    """Laplace approximation with Kronecker factored log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have for each parameter group, e.g., torch.nn.Module,
    that \\P\\approx Q \\otimes H\\.
    See `BaseLaplace` for the full interface and see
    `laplace.utils.matrix.Kron` and `laplace.utils.matrix.KronDecomposed` for the structure of
    the Kronecker factors. `Kron` is used to aggregate factors by summing up and
    `KronDecomposed` is used to add the prior, a Hessian factor (e.g. temperature),
    and computing posterior covariances, marginal likelihood, etc.
    Damping can be enabled by setting `damping=True`.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "kron")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "inputs_id",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        damping: bool = False,
        backend_kwargs: dict[str, Any] | None = None,
        asdl_fisher_kwargs: dict[str, Any] | None = None,
    ):
        self.damping: bool = damping
        self.H_facs: Kron | None = None
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            dict_key_x,
            dict_key_y,
            backend,
            backend_kwargs,
            asdl_fisher_kwargs,
        )

    def _init_H(self) -> None:
        self.H: Kron | KronDecomposed | None = Kron.init_from_model(
            self.params, self._device
        )

    def _check_H_init(self):
        if getattr(self, "H_facs", None) is None:
            raise AttributeError("Laplace not fitted. Run fit() first.")

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.kron(X, y, N=N, **self._asdl_fisher_kwargs)

    @staticmethod
    def _rescale_factors(kron: Kron, factor: float) -> Kron:
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= factor
        return kron

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        if override:
            self.H_facs = None

        if self.H_facs is not None:
            n_data_old: int = self.n_data
            n_data_new: int = len(train_loader.dataset)
            self._init_H()  # re-init H non-decomposed
            # discount previous Kronecker factors to sum up properly together with new ones
            self.H_facs = self._rescale_factors(
                self.H_facs, n_data_old / (n_data_old + n_data_new)
            )

        super().fit(train_loader, override=override, progress_bar=progress_bar)

        if self.H_facs is None:
            self.H_facs = self.H
        else:
            # discount new factors that were computed assuming N = n_data_new
            self.H = self._rescale_factors(
                self.H, n_data_new / (n_data_new + n_data_old)
            )
            self.H_facs += self.H

        # Decompose to self.H for all required quantities but keep H_facs for further inference
        self.H = self.H_facs.decompose(damping=self.damping)

    @property
    def posterior_precision(self) -> KronDecomposed:
        """Kronecker factored Posterior precision \\(P\\).

        Returns
        -------
        precision : `laplace.utils.matrix.KronDecomposed`
        """
        self._check_H_init()
        return self.H * self._H_factor + self.prior_precision

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        if type(self.H) is Kron:  # Fall back to diag prior
            return self.prior_precision_diag.log().sum()
        return self.posterior_precision.logdet()

    def square_norm(self, value: torch.Tensor) -> torch.Tensor:
        delta = value - self.mean
        if type(self.H) is Kron:  # fall back to prior
            return (delta * self.prior_precision_diag) @ delta
        return delta @ self.posterior_precision.bmm(delta, exponent=1)

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        return self.posterior_precision.inv_square_form(Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params).unsqueeze(0)
        cov = self.posterior_precision.inv_square_form(Js).squeeze(0)
        assert cov.shape == (n_batch * n_outs, n_batch * n_outs)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.mean.reshape(1, self.n_params) + samples.reshape(
            n_samples, self.n_params
        )

    @BaseLaplace.prior_precision.setter
    def prior_precision(self, prior_precision: torch.Tensor) -> None:
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers]:
            raise ValueError("Prior precision for Kron either scalar or per-layer.")

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        assert isinstance(self.H_facs, Kron)
        state_dict["H"] = self.H_facs.kfacs
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        super().load_state_dict(state_dict)
        self._init_H()
        assert isinstance(self.H, Kron)
        self.H_facs = self.H
        self.H_facs.kfacs = state_dict["H"]
        self.H = self.H_facs.decompose(damping=self.damping)


class LowRankLaplace(ParametricLaplace):
    """Laplace approximation with low-rank log likelihood Hessian (approximation).
    The low-rank matrix is represented by an eigendecomposition (vecs, values).
    Based on the chosen `backend`, either a true Hessian or, for example, GGN
    approximation could be used.
    The posterior precision is computed as
    \\( P = V diag(l) V^T + P_0.\\)
    To sample, compute the functional variance, and log determinant, algebraic tricks
    are usedto reduce the costs of inversion to the that of a \\(K \times K\\) matrix
    if we have a rank of K.

    See `BaseLaplace` for the full interface.
    """

    _key = ("all", "lowrank")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        sigma_noise: float | torch.Tensor = 1,
        prior_precision: float | torch.Tensor = 1,
        prior_mean: float | torch.Tensor = 0,
        temperature: float = 1,
        enable_backprop: bool = False,
        dict_key_x: str = "inputs_id",
        dict_key_y: str = "labels",
        backend=AsdfghjklHessian,
        backend_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
            temperature=temperature,
            enable_backprop=enable_backprop,
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        self.backend: AsdfghjklHessian

    def _init_H(self):
        self.H: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def V(self) -> torch.Tensor:
        (U, eigvals), prior_prec_diag = self.posterior_precision
        return U / prior_prec_diag.reshape(-1, 1)

    @property
    def Kinv(self) -> torch.Tensor:
        (U, eigvals), _ = self.posterior_precision
        return torch.inverse(torch.diag(1 / eigvals) + U.T @ self.V)

    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        # override fit since output of eighessian not additive across batch
        if not override:
            # LowRankLA cannot be updated since eigenvalue representation not additive
            raise ValueError("LowRank LA does not support updating.")

        self.model.eval()
        self.mean = parameters_to_vector(self.model.parameters())

        if not self.enable_backprop:
            self.mean = self.mean.detach()

        X, _ = next(iter(train_loader))
        with torch.no_grad():
            try:
                out = self.model(X[:1].to(self._device))
            except (TypeError, AttributeError):
                out = self.model(X.to(self._device))
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)

        eigenvectors, eigenvalues, loss = self.backend.eig_lowrank(train_loader)
        self.H = (eigenvectors, eigenvalues)
        self.loss = loss

        self.n_data = len(train_loader.dataset)

    @property
    def posterior_precision(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return correctly scaled posterior precision that would be constructed
        as H[0] @ diag(H[1]) @ H[0].T + self.prior_precision_diag.

        Returns
        -------
        H : tuple(eigenvectors, eigenvalues)
            scaled self.H with temperature and loss factors.
        prior_precision_diag : torch.Tensor
            diagonal prior precision shape `parameters` to be added to H.
        """
        self._check_H_init()
        return (self.H[0], self._H_factor * self.H[1]), self.prior_precision_diag

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        prior_var = torch.einsum("ncp,nkp->nck", Js / self.prior_precision_diag, Js)
        Js_V = torch.einsum("ncp,pl->ncl", Js, self.V)
        info_gain = torch.einsum("ncl,nkl->nck", Js_V @ self.Kinv, Js_V)
        return prior_var - info_gain

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)
        prior_cov = torch.einsum("np,mp->nm", Js / self.prior_precision_diag, Js)
        Js_V = torch.einsum("np,pl->nl", Js, self.V)
        info_gain = torch.einsum("nl,ml->nm", Js_V @ self.Kinv, Js_V)
        cov = prior_cov - info_gain
        assert cov.shape == (n_batch * n_outs, n_batch * n_outs)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(self.n_params, n_samples, generator=generator)
        d = self.prior_precision_diag
        Vs = self.V * d.sqrt().reshape(-1, 1)
        VtV = Vs.T @ Vs
        Ik = torch.eye(len(VtV))
        A = torch.linalg.cholesky(VtV)
        B = torch.linalg.cholesky(VtV + Ik)
        A_inv = torch.inverse(A)
        C = torch.inverse(A_inv.T @ (B - Ik) @ A_inv)
        Kern_inv = torch.inverse(torch.inverse(C) + Vs.T @ Vs)
        dinv_sqrt = (d).sqrt().reshape(-1, 1)
        prior_sample = dinv_sqrt * samples
        gain_sample = dinv_sqrt * Vs @ Kern_inv @ (Vs.T @ samples)
        return self.mean + (prior_sample - gain_sample).T

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        (_, eigvals), prior_prec_diag = self.posterior_precision
        return (
            eigvals.log().sum() + prior_prec_diag.log().sum() - torch.logdet(self.Kinv)
        )


class DiagLaplace(ParametricLaplace):
    """Laplace approximation with diagonal log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have \\(P \\approx \\textrm{diag}(P)\\).
    See `BaseLaplace` for the full interface.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "diag")

    def _init_H(self) -> None:
        self.H: torch.Tensor = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.diag(X, y, N=N, **self._asdl_fisher_kwargs)

    @property
    def posterior_precision(self) -> torch.Tensor:
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        self._check_H_init()
        return self._H_factor * self.H + self.prior_precision_diag

    @property
    def posterior_scale(self) -> torch.Tensor:
        """Diagonal posterior scale \\(\\sqrt{p^{-1}}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision.sqrt()

    @property
    def posterior_variance(self) -> torch.Tensor:
        """Diagonal posterior variance \\(p^{-1}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision

    @property
    def log_det_posterior_precision(self) -> torch.Tensor:
        return self.posterior_precision.log().sum()

    def square_norm(self, value: torch.Tensor) -> torch.Tensor:
        delta = value - self.mean
        return delta @ (delta * self.posterior_precision)

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum("ncp,p,nkp->nck", Js, self.posterior_variance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        n_batch, n_outs, n_params = Js.shape
        Js = Js.reshape(n_batch * n_outs, n_params)
        cov = torch.einsum("np,p,mp->nm", Js, self.posterior_variance, Js)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples


class FunctionalLaplace(BaseLaplace):
    pass


class SoDLaplace(FunctionalLaplace):
    pass
