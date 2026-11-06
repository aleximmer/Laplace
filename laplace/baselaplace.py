from __future__ import annotations

import warnings
from collections import deque
from collections.abc import MutableMapping
from copy import deepcopy
from importlib.util import find_spec
from math import frexp, isfinite, log, pi, sqrt
from numbers import Integral
from typing import Any, Callable

import numpy as np
import torch
import torchmetrics
import tqdm
from scipy.cluster.vq import kmeans2
from torch import nn
from torch.linalg import LinAlgError
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from laplace.curvature import _is_optional_backend
from laplace.curvature.asdfghjkl import AsdfghjklHessian, AsdfghjklInterface
from laplace.curvature.curvature import CurvatureInterface
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.utils import SoDSampler
from laplace.utils.enums import (
    Likelihood,
    LinkApprox,
    PredType,
    PriorStructure,
    TuningMethod,
)
from laplace.utils.functional_utils import (
    _check_model_fingerprint,
    _individual_reward_inputs,
    _model_fingerprint,
    classification_targets,
    concatenate_rows,
    draw_gaussian,
    preserve_model_gradients,
    regression_targets,
    same_configuration,
    select_rows,
    split_batch,
    subset_loader,
    to_device,
    training_indices,
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
    "BaseFunctionalLaplace",
    "ParametricLaplace",
    "FunctionalLaplace",
    "ELLA",
    "VaLLA",
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
        Reward modeling is supported by `KronLaplace`, `DiagLaplace`, `ELLA`,
        and `VaLLA`.
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
                issubclass(backend, AsdfghjklInterface)
                or _is_optional_backend(backend, "BackPackInterface")
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
        self.loss: float | torch.Tensor = 0.0
        self.n_outputs: int = 0
        self.n_data: int = 0
        # Number of observations in the likelihood. Differs from `n_data` only when
        # each datum has extra (e.g. sequence) dims: then it counts all tokens.
        self.n_obs: int = 0

        # Declare attributes
        self._prior_mean: torch.Tensor
        self._prior_precision: torch.Tensor
        self._sigma_noise: torch.Tensor
        self._posterior_scale: torch.Tensor | None

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

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
                (self.n_obs or self.n_data)
                * self.n_outputs
                * torch.log(
                    torch.as_tensor(
                        self.sigma_noise, device=self._device, dtype=self._dtype
                    )
                    * sqrt(2 * pi)
                )
            )
            return factor * self.loss - c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

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
            the curvature approximations used here. It also works when Laplace
            is done on a subset of the parameters, either with subnetwork
            Laplace or by disabling the gradients of the other parameters.

        link_approx : {'mc', 'probit', 'bridge', 'bridge_norm'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible.

        joint : bool
            Whether to output a joint predictive distribution in regression with
            `pred_type='glm'`. If set to `True`, the predictive distribution
            has the same form as GP posterior, i.e. N([f(x1), ...,f(xm)], Cov[f(x1), ..., f(xm)]).
            If `False`, then only outputs the marginal predictive distribution.
            Only available for regression and GLM predictive.

        n_samples : int, default=100
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
            a distribution over classes (similar to a Softmax), with the same shape
            as the logits, e.g. `(batch, classes)` or `(batch, seq_len, classes)`.
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor
            is returned with the mean and the predictive covariance.
        """
        raise NotImplementedError

    def predictive(
        self,
        x: torch.Tensor,
        pred_type: PredType | str,
        link_approx: LinkApprox | str,
        n_samples: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self(
            x, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples
        )

    def _check_jacobians(self, Js: torch.Tensor) -> None:
        if not isinstance(Js, torch.Tensor):
            raise ValueError("Jacobians have to be torch.Tensor.")
        if not Js.device == self._device:
            raise ValueError("Jacobians need to be on the same device as Laplace.")
        if Js.ndim < 3 or Js.shape[-1] != self.n_params:
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
            else torch.as_tensor(self.prior_precision)
        )

        if prior_prec.ndim == 0 or len(prior_prec) == 1:  # scalar
            return self.prior_precision * torch.ones(
                self.n_params, device=self._device, dtype=self._dtype
            )
        elif len(prior_prec) == self.n_params:  # diagonal
            return prior_prec
        elif len(prior_prec) == self.n_layers:  # per layer
            n_params_per_layer = [p.numel() for p in self.params]
            return torch.cat(
                [
                    prior * torch.ones(n_params, device=self._device, dtype=self._dtype)
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
            self._prior_mean = torch.as_tensor(
                prior_mean, device=self._device, dtype=self._dtype
            )
        elif isinstance(prior_mean, torch.Tensor):
            if prior_mean.ndim == 0:
                self._prior_mean = prior_mean.reshape(-1).to(
                    device=self._device, dtype=self._dtype
                )
            elif prior_mean.ndim == 1:
                if len(prior_mean) not in [1, self.n_params]:
                    raise ValueError("Invalid length of prior mean.")
                self._prior_mean = prior_mean.to(device=self._device, dtype=self._dtype)
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
            self._prior_precision = torch.as_tensor(
                [prior_precision], device=self._device, dtype=self._dtype
            )
        elif isinstance(prior_precision, torch.Tensor):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(
                    device=self._device, dtype=self._dtype
                )
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError(
                        "Length of prior precision does not align with architecture."
                    )
                self._prior_precision = prior_precision.to(
                    device=self._device, dtype=self._dtype
                )
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
        log_prior_prec_base: float = 10.0,
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
            exponent of the lower bound of the gridsearch interval.
        log_prior_prec_max : float, default=4
            exponent of the upper bound of the gridsearch interval.
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
        log_prior_prec_base : float, default=10.0
            base of the logarithmic gridsearch interval. The bounds are
            `log_prior_prec_base` raised to `log_prior_prec_min` and
            `log_prior_prec_max`.
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
                else torch.as_tensor(init_prior_prec)
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
                    self._dtype,
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

            interval = torch.logspace(
                log_prior_prec_min,
                log_prior_prec_max,
                grid_size,
                base=log_prior_prec_base,
            )

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
        loss: torchmetrics.Metric
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
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
            self._sigma_noise = torch.as_tensor(
                sigma_noise, device=self._device, dtype=self._dtype
            )
        elif isinstance(sigma_noise, torch.Tensor):
            if sigma_noise.ndim == 0:
                self._sigma_noise = sigma_noise.to(
                    device=self._device, dtype=self._dtype
                )
            elif sigma_noise.ndim == 1:
                if len(sigma_noise) > 1:
                    raise ValueError("Only homoscedastic output noise supported.")
                self._sigma_noise = sigma_noise[0].to(
                    device=self._device, dtype=self._dtype
                )
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

    def _glm_forward_call(
        self,
        x: torch.Tensor | MutableMapping,
        likelihood: Likelihood | str,
        joint: bool = False,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        diagonal_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior predictive on input data `x` for "glm" pred type.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            `(batch_size, input_shape)` if tensor. If MutableMapping, must contain
            the said tensor.

        likelihood : Likelihood or str in {'classification', 'regression', 'reward_modeling'}
            determines the log likelihood Hessian approximation.

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
            Only works for `pred_type='glm'` and `link_approx='mc'`.

        Returns
        -------
        predictive: torch.Tensor or tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax), with the same shape
            as the logits, e.g. `(batch, classes)` or `(batch, seq_len, classes)`.
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor
            is returned with the mean and the predictive covariance.

        Warns
        -----
        RuntimeWarning
            For ``"bridge"`` and ``"bridge_norm"`` when gradient tracking is
            enabled and a covariance requiring gradients has subnormal,
            nonzero magnitude for an input row. Magnitude is the largest
            absolute covariance entry, compared with the smallest positive
            normal value of the covariance dtype.

        Notes
        -----
        The ``"bridge"`` and ``"bridge_norm"`` links use float64 intermediate
        calculations and return probabilities in the predictive mean's dtype.
        Mean corrections and normalization use scaled logit differences to
        avoid overflow.
        If corrected variances or the optional normalization scale are
        nonpositive, or the Bridge probabilities are nonfinite, prediction uses
        the probit approximation on centered moments before variance
        normalization. Corrected variances are clamped at zero for this fallback.

        For subnormal covariance magnitudes, covariance-entry derivatives can
        exceed their dtype's range. Covariance or upstream parameter gradients
        may therefore contain inf or NaN even when probabilities remain finite.
        Covariance backpropagation is not reliable in this regime. A
        RuntimeWarning is emitted when covariance gradients are tracked.
        """
        f_mu, f_var = self._glm_predictive_distribution(
            x, joint=joint and likelihood == Likelihood.REGRESSION
        )

        if likelihood == Likelihood.REGRESSION:
            if diagonal_output and not joint:
                f_var = torch.diagonal(f_var, dim1=-2, dim2=-1)
            return f_mu, f_var

        # Every leading dim of the logits but the last is a batch dim, e.g. `(B, L, C)`
        # for sequences. The link approximations work on `(N, C)`, so flatten them.
        batch_shape = f_mu.shape[:-1]
        f_mu = f_mu.reshape(-1, f_mu.shape[-1])
        f_var = f_var.reshape(-1, *f_var.shape[-2:])

        probs = self._glm_classification_link(
            f_mu, f_var, link_approx, n_samples, diagonal_output
        )
        return probs.reshape(*batch_shape, probs.shape[-1])

    def _glm_classification_link(
        self,
        f_mu: torch.Tensor,
        f_var: torch.Tensor,
        link_approx: LinkApprox | str,
        n_samples: int,
        diagonal_output: bool,
    ) -> torch.Tensor:
        """Approximate the predictive class probabilities from the GLM predictive
        `f_mu` `(batch, classes)` and `f_var` `(batch, classes, classes)`. See
        `_glm_forward_call` for the arguments.
        """
        if link_approx == LinkApprox.MC:
            return self._glm_predictive_samples(
                f_mu,
                f_var,
                n_samples=n_samples,
                diagonal_output=diagonal_output,
            ).mean(dim=0)
        elif link_approx == LinkApprox.PROBIT:
            kappa = 1 / torch.sqrt(1.0 + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
            return torch.softmax(kappa * f_mu, dim=-1)
        elif "bridge" in link_approx:
            _, K = f_mu.shape
            if K == 1:
                return torch.ones_like(f_mu)

            # Float32 reductions can overflow even when every input is finite.
            # Keep the link calculation in double precision, then restore the
            # predictive dtype at the return boundary.
            output_dtype, covariance_dtype = f_mu.dtype, f_var.dtype
            work_dtype = torch.promote_types(output_dtype, covariance_dtype)
            if work_dtype in (torch.float16, torch.bfloat16, torch.float32):
                work_dtype = torch.float64
            f_mu, f_var = f_mu.to(work_dtype), f_var.to(work_dtype)

            # Normalize covariance before reductions: even float64 sums can
            # overflow when each covariance entry is finite. The scale
            # cancels in the ratio, so detach it to avoid unstable gradients
            # through the normalization itself.
            covariance_scale = f_var.detach().abs().amax(dim=(1, 2), keepdim=True)
            if (
                torch.is_grad_enabled()
                and f_var.requires_grad
                and (
                    (covariance_scale > 0)
                    & (covariance_scale < torch.finfo(covariance_dtype).tiny)
                ).any()
            ):
                warnings.warn(
                    "Bridge covariance gradients may overflow for subnormal covariance, "
                    "even when probabilities are finite. Covariance and upstream "
                    "parameter gradients may contain inf or NaN; covariance "
                    "backpropagation is not reliable at this scale.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            safe_covariance_scale = torch.where(
                covariance_scale > 0, covariance_scale, 1
            )
            scaled_covariance = f_var / safe_covariance_scale
            row_sums = scaled_covariance.sum(dim=-1)
            sum_variance = row_sums.sum(dim=-1, keepdim=True).unsqueeze(-1)
            centered = sum_variance.abs() <= (
                torch.finfo(covariance_dtype).eps
                * scaled_covariance.abs().sum(dim=(1, 2), keepdim=True)
            )
            safe_variance = torch.where(centered, 1, sum_variance)
            ratio = torch.where(
                centered.squeeze(-1), 0, row_sums / safe_variance.squeeze(-1)
            )

            # Keep the mean correction scaled: even subtracting two finite
            # float64 means can overflow. Only restore the scale after removing
            # a common shift, since Bridge depends on logit differences.
            mean_magnitude = f_mu.detach().abs().amax(dim=-1, keepdim=True)
            _, mean_exponent = torch.frexp(mean_magnitude)
            # Bound both scales to leave headroom for normalization gradients.
            # A power of two preserves representable logit differences; exp2
            # with an explicit dtype is independent of the global default.
            logit_limit = sqrt(sqrt(torch.finfo(work_dtype).max))
            max_mean_exponent = frexp(logit_limit)[1] - 1
            mean_scale = torch.exp2(
                (mean_exponent - 1).clamp(0, max_mean_exponent).to(work_dtype)
            )
            scaled_mean = f_mu / mean_scale
            mean_logit = (scaled_mean / K).sum(dim=-1, keepdim=True)
            scaled_mean = scaled_mean - torch.where(
                centered.squeeze(-1),
                mean_logit,
                (K * ratio) * mean_logit,
            )
            f_mu = scaled_mean - scaled_mean.amax(dim=-1, keepdim=True)
            centered_covariance = scaled_covariance - ratio.unsqueeze(
                -1
            ) * row_sums.unsqueeze(-2)
            scaled_var_diag = centered_covariance.diagonal(dim1=-2, dim2=-1)
            centered_var_diag = scaled_var_diag * safe_covariance_scale.squeeze(-1)
            valid_variance = (centered_var_diag > 0).all(dim=-1, keepdim=True)
            normalization = torch.ones_like(mean_scale)

            if link_approx == LinkApprox.BRIDGE_NORM:
                variance_scale = (centered_var_diag / K).sum(
                    dim=-1, keepdim=True
                ) / sqrt(K / 2)
                valid_variance = valid_variance & (variance_scale > 0)
                normalization = torch.where(valid_variance, variance_scale, 1).sqrt()

            # Bound normalized differences before division, so saturated
            # branches cannot overflow in either forward or backward passes.
            # The detached bound is far beyond exponential underflow; large
            # variances still bring extreme gaps back into a meaningful range.
            minimum_logit = -(logit_limit / mean_scale) * normalization.detach()
            f_mu = (f_mu.clamp_min(minimum_logit) / normalization) * mean_scale

            # Normalize the Bridge's Dirichlet parameters in log space.
            # The log-sum-exp term is common to every class and cancels in
            # the final softmax. Remove it before combining the numerator
            # terms, avoiding overflow when finite logit gaps are enormous.
            log_sum_negative = torch.logsumexp(-f_mu, dim=-1, keepdim=True)
            log_base = log(1 - 2 / K) if K > 2 else -float("inf")
            log_numerator = torch.logaddexp(
                torch.full_like(f_mu, log_base) - log_sum_negative,
                f_mu - 2 * log(K),
            )
            # Remove the common offset before subtracting log-variance,
            # preserving variance-dependent weights for tied large logits.
            log_numerator = log_numerator - log_numerator.amax(dim=-1, keepdim=True)
            if K == 2:
                # Binary corrected variances are equal for symmetric covariance.
                # Cancel their common term before differentiation to avoid
                # overflowing gradients for subnormal covariance.
                log_alpha = log_numerator
            else:
                # Common covariance and optional variance scales cancel in alpha
                # normalization. Logging the scaled diagonal avoids reciprocals
                # of subnormal variances in backpropagation.
                log_alpha = (
                    log_numerator
                    - scaled_var_diag.clamp_min(torch.finfo(work_dtype).tiny).log()
                )
            probabilities = torch.softmax(log_alpha, dim=-1)

            # A common shift of every logit has no effect on softmax. The
            # fallback therefore uses the centered moments before Bridge's
            # optional variance normalization.
            fallback_logits = scaled_mean / torch.sqrt(
                1 + np.pi / 8 * centered_var_diag.clamp_min(0)
            )
            fallback_logits = fallback_logits - fallback_logits.amax(
                dim=-1, keepdim=True
            )
            fallback_logits = (
                fallback_logits.clamp_min(-logit_limit / mean_scale) * mean_scale
            )
            fallback = torch.softmax(fallback_logits, dim=-1)
            valid_variance = valid_variance & torch.isfinite(probabilities).all(
                dim=-1, keepdim=True
            )
            return torch.where(valid_variance, probabilities, fallback).to(output_dtype)
        else:
            raise ValueError(
                "Prediction path invalid. Check the likelihood, pred_type, link_approx combination!"
            )

    def sample(
        self, n_samples: int = 1, generator: torch.Generator | None = None
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

    def functional_samples(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 1,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        joint: bool = False,
    ) -> torch.Tensor:
        """Sample from the functional posterior on input data `x`.
        Pass `joint=True` for Thompson sampling or other uses that need
        samples consistent across inputs.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            input data `(batch_size, input_shape)`

        pred_type : {'glm'}, default='glm'
            type of posterior predictive, linearized GLM predictive.

        n_samples : int
            number of samples

        diagonal_output : bool
            whether to use a diagonalized glm posterior predictive on the outputs.
            Only applies when `pred_type='glm'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        joint : bool, default=False
            whether to sample all inputs in `x` jointly. With `joint=False`, each
            input's marginal is correct but the samples ignore the correlation
            between inputs. Use `joint=True` for Thompson sampling or batch
            acquisition functions such as qEI. The joint covariance has
            `(batch_size * outputs)**2` entries, so memory grows quadratically
            with the batch. Extra batch dimensions are flattened into
            `batch_size`. Ignored when `pred_type='nn'`, since one weight sample
            already gives one function over all inputs.

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        raise NotImplementedError

    def _glm_functional_samples(
        self,
        f_mu: torch.Tensor,
        f_var: torch.Tensor,
        n_samples: int,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from the posterior functional on input data `x` using "glm" prediction
        type.

        Parameters
        ----------
        f_mu : torch.Tensor or MutableMapping
            glm predictive mean `([batch_dims], output_shape)`

        f_var : torch.Tensor or MutableMapping
            glm predictive covariances `([batch_dims], output_shape, output_shape)`

        n_samples : int
            number of samples

        diagonal_output : bool
            whether to use a diagonalized glm posterior predictive on the outputs.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, [batch_dims], output_shape)`
        """
        # Flatten the batch dims (e.g. batch and sequence), sample, then restore them
        batch_shape = f_mu.shape[:-1]
        f_mu = f_mu.reshape(-1, f_mu.shape[-1])
        f_var = f_var.reshape(-1, *f_var.shape[-2:])
        assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])

        if diagonal_output:
            f_var = torch.diagonal(f_var, dim1=-2, dim2=-1)

        samples = normal_samples(f_mu, f_var, n_samples, generator)
        return samples.reshape(n_samples, *batch_shape, f_mu.shape[-1])

    def _glm_predictive_samples(
        self,
        f_mu: torch.Tensor,
        f_var: torch.Tensor,
        n_samples: int,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from the posterior predictive on input data `x` using "glm" prediction
        type. I.e., the inverse-link function correponding to the likelihood is applied
        on top of the functional sample.

        Parameters
        ----------
        f_mu : torch.Tensor or MutableMapping
            glm predictive mean `([batch_dims], output_shape)`

        f_var : torch.Tensor or MutableMapping
            glm predictive covariances `([batch_dims], output_shape, output_shape)`

        n_samples : int
            number of samples

        diagonal_output : bool
            whether to use a diagonalized glm posterior predictive on the outputs.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, [batch_dims], output_shape)`
        """
        f_samples = self._glm_functional_samples(
            f_mu, f_var, n_samples, diagonal_output, generator
        )

        # Reward models predict rewards, which are regression outputs
        if self.likelihood == Likelihood.CLASSIFICATION:
            return torch.softmax(f_samples, dim=-1)
        else:
            return f_samples


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

    $$
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    $$

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
        dict_key_x: str = "input_ids",
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
            self.n_obs: int = 0

        self.model.eval()

        self.mean: torch.Tensor = parameters_to_vector(self.params)
        if not self.enable_backprop:
            self.mean = self.mean.detach()

        data: (
            tuple[torch.Tensor, torch.Tensor] | MutableMapping[str, torch.Tensor | Any]
        ) = next(iter(train_loader))

        with torch.no_grad():
            if isinstance(data, MutableMapping):  # To support Huggingface dataset
                if "backpack" in self._backend_cls.__name__.lower():
                    raise ValueError(
                        "The BackPACK backends do not support custom models with "
                        + "non-tensor inputs. Consider using the Curvlinops or ASDL "
                        + "backends instead."
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

            if self.likelihood == Likelihood.REGRESSION and y.ndim != out.ndim:
                raise ValueError(
                    f"The model's output has {out.ndim} dims but "
                    f"the target has {y.ndim} dims."
                )

            if self.likelihood == Likelihood.CLASSIFICATION:
                self._check_classification_targets(out, y)

            if (
                self.likelihood != Likelihood.REWARD_MODELING
                and out.ndim > 2
                and "backpack" in self._backend_cls.__name__.lower()
            ):
                raise NotImplementedError(
                    "BackPack backends do not support outputs with more than one "
                    "batch dimension (e.g. sequences) yet."
                )

            self.model.zero_grad()
            loss_batch, H_batch = self._curv_closure(X, y, N=N)
            self.loss += loss_batch
            self.H += H_batch

            # Every token (extra dims between the batch and output dims) is an
            # observation. Count per batch since the sequence length can vary.
            # The last dim of regression targets and of class probabilities is the
            # output dim.
            has_output_dim = self.likelihood == Likelihood.REGRESSION or (
                self.likelihood == Likelihood.CLASSIFICATION and y.ndim == out.ndim
            )
            y_obs_shape = y.shape[:-1] if has_output_dim else y.shape
            self.n_obs += y_obs_shape.numel()

        self.n_data += N

    @staticmethod
    def _check_classification_targets(out: torch.Tensor, y: torch.Tensor) -> None:
        """Check that the targets match the logits `out` `(batch, ..., classes)`:
        either class indices `(batch, ...)` or, for logits `(batch, classes)`, class
        probabilities with the same shape as the logits.

        Only the number of dims is checked, since `out` is computed on a single
        datum and the sequence length can vary between batches. The backends check
        the exact shapes against the logits of each batch.
        """
        if out.ndim > 2:
            if y.ndim != out.ndim - 1:
                raise ValueError(
                    f"The model's logits have shape (batch, ..., classes) with "
                    f"{out.ndim} dims, so the targets must be class indices with "
                    f"{out.ndim - 1} dims, but they have {y.ndim} dims. Class "
                    "probabilities as targets are only supported for logits "
                    "(batch, classes)."
                )
            return

        is_index = y.ndim == out.ndim - 1
        is_prob = y.ndim == out.ndim and y.shape[-1] == out.shape[-1]
        if not (is_index or is_prob):
            raise ValueError(
                f"The model's logits have shape (batch, classes), but the targets "
                f"have {y.ndim} dims. The targets must be class indices (batch,) or "
                "class probabilities (batch, classes)."
            )

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

        $$
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        $$

        Returns
        -------
        log_det_ratio : torch.Tensor
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    def square_norm(self, value) -> torch.Tensor:
        """Compute the square norm under post. Precision with `value-self.mean` as 𝛥:

        $$
            \\Delta^\top P \\Delta
        $$

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
            the curvature approximations used here. It also works when Laplace
            is done on a subset of the parameters, either with subnetwork
            Laplace or by disabling the gradients of the other parameters.

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
            a distribution over classes (similar to a Softmax), with the same shape
            as the logits, e.g. `(batch, classes)` or `(batch, seq_len, classes)`.
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor
            is returned with the mean and the predictive covariance.

        Warns
        -----
        RuntimeWarning
            For ``"bridge"`` and ``"bridge_norm"`` when gradient tracking is
            enabled and a covariance requiring gradients has subnormal,
            nonzero magnitude for an input row. Magnitude is the largest
            absolute covariance entry, compared with the smallest positive
            normal value of the covariance dtype.

        Notes
        -----
        The ``"bridge"`` and ``"bridge_norm"`` links use float64 intermediate
        calculations and return probabilities in the predictive mean's dtype.
        Mean corrections and normalization use scaled logit differences to
        avoid overflow.
        If corrected variances or the optional normalization scale are
        nonpositive, or the Bridge probabilities are nonfinite, prediction uses
        the probit approximation on centered moments before variance
        normalization. Corrected variances are clamped at zero for this fallback.

        For subnormal covariance magnitudes, covariance-entry derivatives can
        exceed their dtype's range. Covariance or upstream parameter gradients
        may therefore contain inf or NaN even when probabilities remain finite.
        Covariance backpropagation is not reliable in this regime. A
        RuntimeWarning is emitted when covariance gradients are tracked.
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
            return self._glm_forward_call(
                x, likelihood, joint, link_approx, n_samples, diagonal_output
            )
        else:
            if likelihood == Likelihood.REGRESSION:
                samples = self._nn_predictive_samples(x, n_samples, **model_kwargs)
                return samples.mean(dim=0), samples.var(dim=0)
            else:  # classification; the average is computed online
                return self._nn_predictive_classification(x, n_samples, **model_kwargs)

    def functional_samples(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        joint: bool = False,
    ) -> torch.Tensor:
        """Sample from the function-space posterior on input data `x`.
        Can be used, for example, to compute an arbitrary expectation, or with
        `joint=True` for Thompson sampling.

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

        joint : bool, default=False
            whether to sample all inputs in `x` jointly. With `joint=False`, each
            input's marginal is correct but the samples ignore the correlation
            between inputs. Use `joint=True` for Thompson sampling or batch
            acquisition functions such as qEI. The joint covariance has
            `(batch_size * outputs)**2` entries, so memory grows quadratically
            with the batch. Extra batch dimensions are flattened into
            `batch_size`. Ignored when `pred_type='nn'`, since one weight sample
            already gives one function over all inputs.

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        if pred_type not in PredType.__members__.values():
            raise ValueError("Only glm and nn supported as prediction types.")

        if pred_type == PredType.GLM and joint:
            f_mu, f_cov = self._glm_predictive_distribution(x, joint=True)

            # Reward models are fitted on pairs but predict one reward per item
            n_outputs = (
                1 if self.likelihood == Likelihood.REWARD_MODELING else self.n_outputs
            )

            if diagonal_output:
                # Keep the covariance between inputs, drop it between outputs
                output_ids = torch.arange(f_mu.numel(), device=f_mu.device)
                output_ids = output_ids % n_outputs
                f_cov = f_cov * (output_ids[:, None] == output_ids[None, :])

            samples = draw_gaussian(f_mu, f_cov, n_samples, generator)
            return samples.reshape(n_samples, -1, n_outputs)
        elif pred_type == PredType.GLM:
            f_mu, f_var = self._glm_predictive_distribution(x)
            return self._glm_functional_samples(
                f_mu, f_var, n_samples, diagonal_output, generator
            )
        else:  # 'nn'
            return self._nn_functional_samples(x, n_samples, generator)

    def predictive_samples(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        joint: bool = False,
    ) -> torch.Tensor:
        """Sample from the posterior predictive on input data `x`. I.e., the respective
        inverse-link function (e.g. softmax) is applied on top of the functional
        sample.

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

        joint : bool, default=False
            whether to sample all inputs in `x` jointly. With `joint=False`, each
            input's marginal is correct but the samples ignore the correlation
            between inputs. Use `joint=True` for Thompson sampling or batch
            acquisition functions such as qEI. The joint covariance has
            `(batch_size * outputs)**2` entries, so memory grows quadratically
            with the batch. Extra batch dimensions are flattened into
            `batch_size`. Ignored when `pred_type='nn'`, since one weight sample
            already gives one function over all inputs.

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        if pred_type not in PredType.__members__.values():
            raise ValueError("Only glm and nn supported as prediction types.")

        if pred_type == PredType.GLM and joint:
            f_samples = self.functional_samples(
                x, pred_type, n_samples, diagonal_output, generator, joint=True
            )

            if self.likelihood == Likelihood.CLASSIFICATION:
                return torch.softmax(f_samples, dim=-1)
            else:
                return f_samples
        elif pred_type == PredType.GLM:
            f_mu, f_var = self._glm_predictive_distribution(x)
            return self._glm_predictive_samples(
                f_mu, f_var, n_samples, diagonal_output, generator
            )
        else:  # 'nn'
            return self._nn_predictive_samples(x, n_samples, generator)

    @torch.enable_grad()
    def _glm_predictive_distribution(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        joint: bool = False,
        diagonal_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In reward modeling, the output size differs between fitting (pairwise) and
        # prediction (single items), which ASDL's Jacobian can't handle
        if "asdl" in self._backend_cls.__name__.lower() and (
            self.enable_backprop or self.likelihood == Likelihood.REWARD_MODELING
        ):
            if self.enable_backprop:
                # Asdl's doesn't support backprop over Jacobians
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

    def _nn_functional_samples(
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

        return fs

    def _nn_predictive_samples(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        fs = self._nn_functional_samples(X, n_samples, generator, **model_kwargs)

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

        return torch.as_tensor(py)

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
            `([batch_dims], outputs, parameters)`

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
        log_prior_prec_base: float = 10.0,
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
            log_prior_prec_base=log_prior_prec_base,
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
            "n_obs": self.n_obs,
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
        self.n_obs = state_dict.get("n_obs", self.n_data)
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
            self.n_params, self.n_params, device=self._device, dtype=self._dtype
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
        return torch.einsum("...cp,pq,...kq->...ck", Js, self.posterior_covariance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        Js = Js.reshape(-1, Js.shape[-1])
        return torch.einsum("np,pq,mq->nm", Js, self.posterior_covariance, Js)

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples,
            self.n_params,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        # (n_samples, n_params) x (n_params, n_params) -> (n_samples, n_params)
        samples = samples @ self.posterior_scale.T
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
        dict_key_x: str = "input_ids",
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
            self.params, self._device, self._dtype
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

        n_obs_old: int = 0 if self.H_facs is None else self.n_obs
        n_data_new: int = len(train_loader.dataset)
        if self.H_facs is not None:
            self._init_H()  # re-init H non-decomposed

        super().fit(train_loader, override=override, progress_bar=progress_bar)

        # The backends sum the input factors over all tokens, e.g. `B * L` for
        # outputs `(B, L, K)`, and divide by `n_data_new`. Every token is an
        # observation, so divide by the number of tokens instead. This also holds
        # when the sequence length varies between batches.
        n_obs_new: int = self.n_obs - n_obs_old
        if n_obs_new > 0:
            self.H = self._rescale_factors(self.H, n_data_new / n_obs_new)

        if self.H_facs is None:
            self.H_facs = self.H
        else:
            # Weight previous and new Kronecker factors by their number of observations
            n_obs_total = n_obs_old + n_obs_new
            self.H_facs = self._rescale_factors(self.H_facs, n_obs_old / n_obs_total)
            self.H_facs += self._rescale_factors(self.H, n_obs_new / n_obs_total)

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
        # `inv_square_form` takes `(batch, outputs, parameters)`, so flatten the
        # batch dims, e.g. `(B, L)` for sequences
        batch_shape = Js.shape[:-2]
        f_var = self.posterior_precision.inv_square_form(Js.reshape(-1, *Js.shape[-2:]))
        return f_var.reshape(*batch_shape, *f_var.shape[-2:])

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        n_outs_total = Js.shape[:-1].numel()
        Js = Js.reshape(n_outs_total, Js.shape[-1]).unsqueeze(0)
        cov = self.posterior_precision.inv_square_form(Js).squeeze(0)
        assert cov.shape == (n_outs_total, n_outs_total)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples,
            self.n_params,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
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

    Note that only `AsdfghjklHessian` backend is supported. Install it via:
    pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl

    See `BaseLaplace` for the full interface.
    """

    _key = ("all", "lowrank")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        backend: type[CurvatureInterface] = AsdfghjklHessian
        if find_spec("asdfghjkl") is not None
        else CurvatureInterface,
        sigma_noise: float | torch.Tensor = 1,
        prior_precision: float | torch.Tensor = 1,
        prior_mean: float | torch.Tensor = 0,
        temperature: float = 1,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend_kwargs: dict[str, Any] | None = None,
    ):
        if find_spec("asdfghjkl") is None:
            raise ImportError(
                """To use LowRankLaplace, please install the old asdfghjkl dependency: """
                """pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl"""
            )

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

        X, y = next(iter(train_loader))
        with torch.no_grad():
            try:
                out = self.model(X[:1].to(self._device))
            except (TypeError, AttributeError):
                out = self.model(X.to(self._device))

        if self.likelihood == Likelihood.REGRESSION and y.ndim != out.ndim:
            raise ValueError(
                f"The model's output has {out.ndim} dims but "
                f"the target has {y.ndim} dims."
            )

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
        prior_var = torch.einsum(
            "...cp,...kp->...ck", Js / self.prior_precision_diag, Js
        )
        Js_V = torch.einsum("...cp,pl->...cl", Js, self.V)
        info_gain = torch.einsum("...cl,...kl->...ck", Js_V @ self.Kinv, Js_V)
        return prior_var - info_gain

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        Js = Js.reshape(-1, Js.shape[-1])
        prior_cov = torch.einsum("np,mp->nm", Js / self.prior_precision_diag, Js)
        Js_V = torch.einsum("np,pl->nl", Js, self.V)
        info_gain = torch.einsum("nl,ml->nm", Js_V @ self.Kinv, Js_V)
        cov = prior_cov - info_gain
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            self.n_params,
            n_samples,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        (U, eigvals), d = self.posterior_precision
        # The posterior covariance (U diag(eigvals) U^T + D)^-1 can be written as
        # D^-1/2 (I + W W^T)^-1 D^-1/2 with W = D^-1/2 U diag(eigvals)^1/2.
        # (I + W W^T)^-1/2 = I + W Q diag(c) Q^T W^T, where W^T W = Q diag(s) Q^T
        # and c = ((1 + s)^-1/2 - 1) / s, written in a form that is stable at s = 0.
        d_sqrt = d.sqrt().reshape(-1, 1)
        W = U * eigvals.sqrt() / d_sqrt
        s, Q = torch.linalg.eigh(W.T @ W)
        s = s.clamp(min=0)
        c = -1 / ((1 + s).sqrt() * (1 + (1 + s).sqrt()))
        WQ = W @ Q
        samples = samples + WQ @ (c.reshape(-1, 1) * (WQ.T @ samples))
        return self.mean + (samples / d_sqrt).T

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
        self.H: torch.Tensor = torch.zeros(
            self.n_params, device=self._device, dtype=self._dtype
        )

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
        return torch.einsum("...cp,p,...kp->...ck", Js, self.posterior_variance, Js)

    def functional_covariance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        # (prod(batch_dims) * out_dim, n_params)
        Js = Js.reshape(-1, Js.shape[-1])
        # (prod(batch_dims) * out_dim, prod(batch_dims) * out_dim)
        cov = torch.einsum("np,p,mp->nm", Js, self.posterior_variance, Js)
        return cov

    def sample(
        self, n_samples: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples,
            self.n_params,
            device=self._device,
            dtype=self._dtype,
            generator=generator,
        )
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples


class BaseFunctionalLaplace(BaseLaplace):
    """Provide prediction and sampling shared by function-space estimators.

    Subclasses supply latent predictive moments and refresh their own posterior
    state. This class handles output links, sampling, and checkpoint settings
    for ``FunctionalLaplace``, ``ELLA``, and ``VaLLA``.
    """

    @staticmethod
    def _check_output_dims(out: torch.Tensor) -> None:
        """Function-space estimators assume outputs `(batch, outputs)`."""
        if out.ndim > 2:
            raise NotImplementedError(
                "FunctionalLaplace, ELLA, and VaLLA do not support outputs with more "
                "than one batch dimension (e.g. sequences) yet. Use a parametric "
                "Laplace, e.g. `DiagLaplace` or `KronLaplace`, instead."
            )

    def _checkpoint_configuration(self) -> dict[str, Any]:
        """Record input keys and backend options required by a checkpoint.

        Returns
        -------
        dict[str, Any]
            Settings that must match when the checkpoint is restored.
        """
        return {
            "dict_key_x": self.dict_key_x,
            "dict_key_y": self.dict_key_y,
            "backend_class": (
                f"{self._backend_cls.__module__}.{self._backend_cls.__qualname__}"
            ),
            "backend_kwargs": deepcopy(self._backend_kwargs),
            "enable_backprop": self.enable_backprop,
        }

    def _validate_checkpoint_configuration(self, state_dict: dict[str, Any]) -> None:
        """Reject a checkpoint created with different input or backend settings.

        Parameters
        ----------
        state_dict : dict[str, Any]
            Saved functional posterior state.

        Raises
        ------
        ValueError
            If a required setting differs from the current estimator.
        """
        for key, current in self._checkpoint_configuration().items():
            if not same_configuration(state_dict.get(key), current):
                raise ValueError(f"Checkpoint requires matching {key}.")

    @property
    def _uses_backpack_backend(self) -> bool:
        """Whether the curvature backend extends modules using BackPACK."""
        return _is_optional_backend(self._backend_cls, "BackPackInterface")

    @property
    def _uses_asdl_backend(self) -> bool:
        """Whether the curvature backend computes Jacobians using ASDL."""
        return _is_optional_backend(self._backend_cls, "AsdlInterface")

    def _glm_predictive_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Provide latent moments from a concrete function-space estimator.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        joint : bool, default=False
            Whether to retain covariance between input rows.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean ``[B, O]`` and covariance ``[B, O, O]`` in model-output
            units. With ``joint=True``, the mean is ``[B*O]`` and covariance
            is ``[B*O, B*O]``. Observation noise is excluded.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement latent predictive moments.
        """
        raise NotImplementedError

    def _refresh_posterior(self) -> None:
        """Update cached posterior quantities after a hyperparameter change."""

    def _check_fitted(self) -> None:
        """Refresh a fitted posterior or reject predictions before fitting.

        Raises
        ------
        RuntimeError
            If ``fit`` has not completed.
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before making functional Laplace predictions.")
        self._refresh_posterior()

    def _check_nonempty_inputs(self, x: torch.Tensor | MutableMapping) -> None:
        """Require at least one input row for a function-space prediction.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.

        Raises
        ------
        ValueError
            If the input batch is empty.
        """
        inputs = x[self.dict_key_x] if isinstance(x, MutableMapping) else x
        if inputs.shape[0] == 0:
            raise ValueError("Function-space predictions require at least one input.")

    def predictive_moments(
        self, x: torch.Tensor | MutableMapping, *, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent predictive moments without observation noise.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        joint : bool, default=False
            Include covariance between input rows when ``True``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean ``[B, O]`` and covariance ``[B, O, O]`` by default. With
            ``joint=True``, the mean is flattened and covariance has shape
            ``[B*O, B*O]``.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.
        ValueError
            If the input batch is empty.
        """
        self._check_fitted()
        self._check_nonempty_inputs(x)
        return self._glm_predictive_distribution(x, joint=joint)

    def _classification_validation_nll(
        self, x: torch.Tensor | MutableMapping, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute summed classification NLL using the probit logit scale.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Validation inputs.
        targets : torch.Tensor
            Class targets for those inputs.

        Returns
        -------
        torch.Tensor
            Summed negative log likelihood for the batch.
        """
        mean, covariance = self.predictive_moments(x)
        scale = torch.sqrt(1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1))
        return torch.nn.functional.cross_entropy(
            mean / scale, classification_targets(targets), reduction="sum"
        )

    def __call__(
        self,
        x: torch.Tensor | MutableMapping,
        pred_type: PredType | str = PredType.GP,
        joint: bool = False,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        fitting: bool = False,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict from the fitted function-space posterior.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        pred_type : PredType or str, default=PredType.GP
            Prediction type. Function-space estimators support only ``"gp"``.
        joint : bool, default=False
            Retain cross-input covariance for regression predictions.
        link_approx : LinkApprox or str, default=LinkApprox.PROBIT
            Approximation for the classification probability link.
        n_samples : int, default=100
            Number of draws for a Monte Carlo classification link.
        diagonal_output : bool, default=False
            Return only output variances for regression when supported.
        generator : torch.Generator or None, default=None
            Random generator for Monte Carlo classification predictions.
        fitting : bool, default=False
            Evaluate reward-model preference pairs as classification inputs.
        **model_kwargs : Any
            Accepted for compatibility with the base prediction interface;
            these arguments are not forwarded to the model.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Classification probabilities ``[B, C]``, or regression/reward
            latent mean and covariance. With ``joint=True``, the regression
            mean is flattened and covariance has shape ``[B*O, B*O]``.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.
        ValueError
            If the input batch is empty, the prediction type or link is
            unsupported, or the generator has an incompatible type or device.

        Warns
        -----
        RuntimeWarning
            For ``"bridge"`` and ``"bridge_norm"`` when gradient tracking is
            enabled and a covariance requiring gradients has subnormal,
            nonzero magnitude for an input row. Magnitude is the largest
            absolute covariance entry, compared with the smallest positive
            normal value of the covariance dtype.

        Notes
        -----
        Regression and individual reward predictions exclude observation
        noise. Their marginal mean is ``[B, O]`` and covariance is
        ``[B, O, O]``, or variance ``[B, O]`` with ``diagonal_output=True``.

        The ``"bridge"`` and ``"bridge_norm"`` links use float64 intermediate
        calculations and return probabilities in the predictive mean's dtype.
        Mean corrections and normalization use scaled logit differences to
        avoid overflow.
        If corrected variances or the optional normalization scale are
        nonpositive, or the Bridge probabilities are nonfinite, prediction uses
        the probit approximation on centered moments before variance
        normalization. Corrected variances are clamped at zero for this fallback.

        For subnormal covariance magnitudes, covariance-entry derivatives can
        exceed their dtype's range. Covariance or upstream parameter gradients
        may therefore contain inf or NaN even when probabilities remain finite.
        Covariance backpropagation is not reliable in this regime. A
        RuntimeWarning is emitted when covariance gradients are tracked.
        """
        self._check_fitted()
        self._check_nonempty_inputs(x)
        if pred_type != PredType.GP:
            raise ValueError("Only gp is supported as a functional prediction type.")
        if link_approx not in list(LinkApprox):
            raise ValueError(f"Unsupported link approximation {link_approx}.")
        if generator is not None and (
            not isinstance(generator, torch.Generator)
            or generator.device != self._device
        ):
            raise ValueError("Invalid random generator (check type and device).")
        likelihood = self.likelihood
        if likelihood == Likelihood.REWARD_MODELING:
            likelihood = Likelihood.CLASSIFICATION if fitting else Likelihood.REGRESSION
        if likelihood == Likelihood.CLASSIFICATION and link_approx == LinkApprox.MC:
            return self.predictive_samples(
                x,
                n_samples=n_samples,
                diagonal_output=diagonal_output,
                generator=generator,
                fitting=fitting,
            ).mean(dim=0)
        return self._glm_forward_call(
            x, likelihood, joint, link_approx, n_samples, diagonal_output
        )

    def _glm_functional_samples(
        self,
        f_mu: torch.Tensor,
        f_var: torch.Tensor,
        n_samples: int,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw per-input latent samples from supplied Gaussian moments.

        Parameters
        ----------
        f_mu : torch.Tensor
            Latent means shaped ``[B, O]``.
        f_var : torch.Tensor
            Per-input covariance shaped ``[B, O, O]``.
        n_samples : int
            Number of draws.
        diagonal_output : bool, default=False
            Drop covariance between output coordinates before sampling.
        generator : torch.Generator or None, default=None
            Random generator for the draws.

        Returns
        -------
        torch.Tensor
            Latent draws shaped ``[n_samples, B, O]``.
        """
        covariance = (
            torch.diag_embed(f_var.diagonal(dim1=-2, dim2=-1))
            if diagonal_output
            else f_var
        )
        return draw_gaussian(f_mu, covariance, n_samples, generator)

    def functional_samples(
        self,
        x: torch.Tensor | MutableMapping,
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        joint: bool = False,
    ) -> torch.Tensor:
        """Draw latent outputs; use `joint=True` to retain cross-input covariance.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        pred_type : PredType or str, default=PredType.GLM
            Accepted for API compatibility. Every value draws from the
            function-space posterior; ``"nn"`` does not draw model weights.
        n_samples : int, default=100
            Number of draws.
        diagonal_output : bool, default=False
            Drop cross-output covariance before sampling.
        generator : torch.Generator or None, default=None
            Random generator on the model device.
        joint : bool, default=False
            Retain covariance between input rows.

        Returns
        -------
        torch.Tensor
            Latent draws shaped ``[n_samples, B, O]``.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.
        ValueError
            If the input batch is empty, the sampling type is unsupported,
            or ``n_samples`` is not positive.
        """
        self._check_fitted()
        self._check_nonempty_inputs(x)
        if pred_type not in list(PredType):
            raise ValueError(f"Unsupported functional sampling type {pred_type}.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        mean, covariance = self._glm_predictive_distribution(x, joint=joint)
        if joint:
            batch = (
                x[self.dict_key_x].shape[0]
                if isinstance(x, MutableMapping)
                else x.shape[0]
            )
            n_outputs = mean.numel() // batch
            if diagonal_output:
                output_ids = torch.arange(mean.numel(), device=mean.device) % n_outputs
                covariance = covariance * (output_ids[:, None] == output_ids[None, :])
            draws = draw_gaussian(mean, covariance, n_samples, generator)
            return draws.reshape(n_samples, batch, n_outputs)
        return self._glm_functional_samples(
            mean, covariance, n_samples, diagonal_output, generator
        )

    def predictive_samples(
        self,
        x: torch.Tensor | MutableMapping,
        pred_type: PredType | str = PredType.GLM,
        n_samples: int = 100,
        diagonal_output: bool = False,
        generator: torch.Generator | None = None,
        fitting: bool = False,
        joint: bool = False,
    ) -> torch.Tensor:
        """Draw outputs and apply the classification link when appropriate.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        pred_type : PredType or str, default=PredType.GLM
            Accepted for function-space sampling API compatibility.
        n_samples : int, default=100
            Number of draws.
        diagonal_output : bool, default=False
            Drop cross-output covariance before sampling.
        generator : torch.Generator or None, default=None
            Random generator on the model device.
        fitting : bool, default=False
            Interpret reward-model inputs as preference pairs.
        joint : bool, default=False
            Retain covariance between input rows.

        Returns
        -------
        torch.Tensor
            Predictive draws shaped ``[n_samples, B, O]``.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.
        ValueError
            If the input batch is empty, the sampling type is unsupported,
            or ``n_samples`` is not positive.

        Notes
        -----
        Classification draws are class probabilities obtained by applying
        softmax to latent samples. Regression and individual reward draws are
        latent function values without observation noise.
        """
        samples = self.functional_samples(
            x, pred_type, n_samples, diagonal_output, generator, joint
        )
        if self.likelihood == Likelihood.CLASSIFICATION or (
            self.likelihood == Likelihood.REWARD_MODELING and fitting
        ):
            return samples.softmax(dim=-1)
        return samples


class FunctionalLaplace(BaseFunctionalLaplace):
    """Applying the GGN (Generalized Gauss-Newton) approximation for the Hessian in the Laplace approximation of the posterior
    turns the underlying probabilistic model from a BNN into a GLM (generalized linear model).
    This GLM (in the weight space) is equivalent to a GP (in the function space), see
    [Approximate Inference Turns Deep Networks into Gaussian Processes (Khan et al., 2019)](https://arxiv.org/abs/1906.01930)

    This class implements the (approximate) GP inference through which
    we obtain the desired quantities (posterior predictive, marginal log-likelihood).
    See [Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)](https://arxiv.org/abs/2008.08400)
    for more details.

    Note that for `likelihood='classification'`, we approximate \\( L_{NN} \\) with a diagonal matrix
    ( \\( L_{NN} \\) is a block-diagonal matrix, where blocks represent Hessians of per-data-point log-likelihood w.r.t.
    neural network output \\( f \\), See Appendix [A.2.1](https://arxiv.org/abs/2008.08400) for exact definition). We
    resort to such an approximation because of the (possible) errors found in Laplace approximation for
    multiclass GP classification in Chapter 3.5 of [R&W 2006 GP book](http://www.gaussianprocess.org/gpml/),
    see the question
    [here](https://stats.stackexchange.com/questions/555183/gaussian-processes-multi-class-laplace-approximation)
    for more details. Alternatively, one could also resort to *one-vs-one* or *one-vs-rest* implementations
    for multiclass classification, however, that is not (yet) supported here.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression', 'reward_modeling'}
        determines the log likelihood Hessian approximation.
        In the case of 'reward_modeling', it fits Laplace using the classification likelihood,
        then does prediction as in regression likelihood. The model needs to be defined accordingly:
        The forward pass during training takes `x.shape == (batch_size, 2, dim)` with
        `y.shape = (batch_size,)`. Meanwhile, during evaluation `x.shape == (batch_size, dim)`.
        Reward modeling is supported by `KronLaplace`, `DiagLaplace`, `ELLA`,
        and `VaLLA`.
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
    n_subset : int
        number of data points for Subset-of-Data (SOD) approximate GP inference.
    independent_outputs : bool
        GP kernel here is product of Jacobians, which results in a \\( C \\times C\\) matrix where \\(C\\) is the output
        dimension. If `True`, only a diagonal of a GP kernel is used. This is (somewhat) equivalent to
        assuming independent GPs across output channels.
    seed: int, default=0
        Random seed for subset of data sampler.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "gp")

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        n_subset: int,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x="input_ids",
        dict_key_y="labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        independent_outputs: bool = False,
        seed: int = 0,
    ):
        assert (
            backend is None
            or issubclass(backend, CurvlinopsGGN)
            or _is_optional_backend(backend, "BackPackGGN")
            or _is_optional_backend(backend, "AsdlGGN")
        )
        self._check_prior_precision(prior_precision)
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
        self.enable_backprop = enable_backprop

        self.n_subset = n_subset
        self.independent_outputs = independent_outputs
        self.seed = seed

        self.K_MM = None
        self.Sigma_inv = None  # (K_{MM} + L_MM_inv)^{-1}
        self.train_loader = (
            None  # needed in functional variance and marginal log likelihood
        )
        self.batch_size = None
        self._prior_factor_sod = None
        self.mu = None  # mean in the scatter term of the log marginal likelihood
        self.L = None

        # Posterior mean (used in regression marginal likelihood)
        self.mean = parameters_to_vector(self.params).detach()

        self._fitted = False
        self._recompute_Sigma = True

    @staticmethod
    def _check_prior_precision(prior_precision: float | torch.Tensor):
        """Checks if the given prior precision is suitable for the GP interpretation of LLA.
        As such, only single value priors, i.e., isotropic priors are suitable.
        """
        if torch.is_tensor(prior_precision):
            if not (
                prior_precision.ndim == 0
                or (prior_precision.ndim == 1 and len(prior_precision) == 1)
            ):
                raise ValueError("Only isotropic priors supported in FunctionalLaplace")

    def _init_K_MM(self):
        """Allocates memory for the kernel matrix evaluated at the subset of the training
        data points. If the subset is of size \\(M\\) and the problem has \\(C\\) outputs,
        this is a list of C \\((M,M\\)) tensors for diagonal kernel and \\((M x C, M x C)\\)
        otherwise.
        """
        if self.independent_outputs:
            self.K_MM = [
                torch.empty(
                    size=(self.n_subset, self.n_subset),
                    device=self._device,
                    dtype=self._dtype,
                )
                for _ in range(self.n_outputs)
            ]
        else:
            self.K_MM = torch.empty(
                size=(self.n_subset * self.n_outputs, self.n_subset * self.n_outputs),
                device=self._device,
                dtype=self._dtype,
            )

    def _init_Sigma_inv(self):
        """Allocates memory for the cholesky decomposition of
        \\[
            K_{MM} + \\Lambda_{MM}^{-1}.
        \\]
        See See [Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)](https://arxiv.org/abs/2008.08400)
        Equation 15 for more information.
        """
        if self.independent_outputs:
            self.Sigma_inv = [
                torch.empty(
                    size=(self.n_subset, self.n_subset),
                    device=self._device,
                    dtype=self._dtype,
                )
                for _ in range(self.n_outputs)
            ]
        else:
            self.Sigma_inv = torch.empty(
                size=(self.n_subset * self.n_outputs, self.n_subset * self.n_outputs),
                device=self._device,
                dtype=self._dtype,
            )

    def _store_K_batch(self, K_batch: torch.Tensor, i: int, j: int):
        """Given the kernel matrix between the i-th and the j-th batch, stores it in the
        corresponding position in self.K_MM.
        """
        if self.independent_outputs:
            for c in range(self.n_outputs):
                self.K_MM[c][
                    i * self.batch_size : min((i + 1) * self.batch_size, self.n_subset),
                    j * self.batch_size : min((j + 1) * self.batch_size, self.n_subset),
                ] = K_batch[:, :, c]
                if i != j:
                    self.K_MM[c][
                        j * self.batch_size : min(
                            (j + 1) * self.batch_size, self.n_subset
                        ),
                        i * self.batch_size : min(
                            (i + 1) * self.batch_size, self.n_subset
                        ),
                    ] = torch.transpose(K_batch[:, :, c], 0, 1)
        else:
            bC = self.batch_size * self.n_outputs
            MC = self.n_subset * self.n_outputs
            self.K_MM[
                i * bC : min((i + 1) * bC, MC), j * bC : min((j + 1) * bC, MC)
            ] = K_batch
            if i != j:
                self.K_MM[
                    j * bC : min((j + 1) * bC, MC), i * bC : min((i + 1) * bC, MC)
                ] = torch.transpose(K_batch, 0, 1)

    def _build_L(self, lambdas: list[torch.Tensor]):
        """Given a list of the Hessians of per-batch log-likelihood w.r.t. neural network output \\( f \\),
        returns the contatenation of these hessians in a suitable format for the used kernel
        (diagonal or not).

        In this function the diagonal approximation is performed. Please refer to the introduction of the
        class for more details.

        Parameters
        ----------
        lambdas : list of torch.Tensor of shape (C, C)
                  Contains per-batch log-likelihood w.r.t. neural network output \\( f \\).

        Returns
        -------
        L : list with length C of tensors with shape M or tensor (MxC)
            Contains the given Hessians in a suitable format.
        """
        # Concatenate batch dimension and discard non-diagonal entries.
        L_diag = torch.diagonal(torch.cat(lambdas, dim=0), dim1=-2, dim2=-1).reshape(-1)

        if self.independent_outputs:
            return [L_diag[i :: self.n_outputs] for i in range(self.n_outputs)]
        else:
            return L_diag

    def _build_Sigma_inv(self):
        """Computes the cholesky decomposition of
        \\[
            K_{MM} + \\Lambda_{MM}^{-1}.
        \\]
        See See [Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)](https://arxiv.org/abs/2008.08400)
        Equation 15 for more information.

        As the diagonal approximation is performed with \\(\\Lambda_{MM}\\) (which is stored in self.L),
        the code is greatly simplified.
        """
        if self.independent_outputs:
            self.Sigma_inv = [
                torch.linalg.cholesky(
                    self.gp_kernel_prior_variance * self.K_MM[c]
                    + torch.diag(
                        torch.nan_to_num(1.0 / (self._H_factor * lambda_c), posinf=10.0)
                    )
                )
                for c, lambda_c in enumerate(self.L)
            ]
        else:
            self.Sigma_inv = torch.linalg.cholesky(
                self.gp_kernel_prior_variance * self.K_MM
                + torch.diag(
                    torch.nan_to_num(1 / (self._H_factor * self.L), posinf=10.0)
                )
            )
        self._recompute_Sigma = False

    def _batch_inputs(
        self, data: tuple[torch.Tensor, ...] | MutableMapping
    ) -> torch.Tensor | MutableMapping:
        """Get the model inputs from a training batch.

        Parameters
        ----------
        data : tuple or MutableMapping
            A batch from the training loader, either `(X, y)` or a mapping.

        Returns
        -------
        torch.Tensor or MutableMapping
            `X` on the model's device, or the whole mapping, which the model
            reads its inputs from.
        """
        if isinstance(data, MutableMapping):
            return data

        return data[0].to(self._device)

    def _get_SoD_data_loader(self, train_loader: DataLoader) -> DataLoader:
        """Subset-of-Datapoints data loader"""
        return DataLoader(
            dataset=train_loader.dataset,
            batch_size=train_loader.batch_size,
            collate_fn=train_loader.collate_fn,
            sampler=SoDSampler(
                N=len(train_loader.dataset), M=self.n_subset, seed=self.seed
            ),
            shuffle=False,
        )

    def fit(
        self, train_loader: DataLoader | MutableMapping, progress_bar: bool = False
    ):
        """Fit the Laplace approximation of a GP posterior.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
            `train_loader.batch_size` needs to be set to access \\(b\\) batch_size
        progress_bar : bool
            whether to show a progress bar during the fitting process.
        """
        # Set model to evaluation mode
        self.model.eval()

        data = next(iter(train_loader))
        with torch.no_grad():
            if isinstance(data, MutableMapping):  # To support Huggingface dataset
                if self._uses_backpack_backend:
                    raise ValueError(
                        "Currently BackPACK backend is not supported "
                        + "for custom models with non-tensor inputs "
                        + "(https://github.com/pytorch/functorch/issues/159). Use "
                        + "CurvlinopsGGN, the default backend, instead."
                    )

                out = self.model(data)
            else:
                X = data[0]
                try:
                    out = self.model(X[:1].to(self._device))
                except (TypeError, AttributeError):
                    out = self.model(X.to(self._device))
        self._check_output_dims(out)
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)
        self.batch_size = train_loader.batch_size

        if (
            self.likelihood == "regression"
            and self.n_outputs > 1
            and self.independent_outputs
        ):
            warnings.warn(
                "Using FunctionalLaplace with the diagonal approximation of a GP kernel is not recommended "
                "in the case of multivariate regression. Predictive variance will likely be overestimated."
            )

        N = len(train_loader.dataset)
        self.n_data = N

        assert self.n_subset <= N, (
            "`num_data` must be less than or equal to the original number of data points."
        )

        train_loader = self._get_SoD_data_loader(train_loader)
        self.train_loader = train_loader
        self._prior_factor_sod = self.n_subset / self.n_data

        self._init_K_MM()
        self._init_Sigma_inv()

        f, lambdas, mu = [], [], []

        if progress_bar:
            loader = enumerate(tqdm.tqdm(train_loader, desc="Fitting"))
        else:
            loader = enumerate(train_loader)

        for i, data in loader:
            if isinstance(data, MutableMapping):  # To support Huggingface dataset
                X, y = data, data[self.dict_key_y].to(self._device)
            else:
                X, y = data
                X, y = X.to(self._device), y.to(self._device)

            Js_batch, f_batch = self._jacobians(X, enable_backprop=False)

            if self.likelihood == Likelihood.REGRESSION and y.ndim != out.ndim:
                raise ValueError(
                    f"The model's output has {out.ndim} dims but "
                    f"the target has {y.ndim} dims."
                )

            with torch.no_grad():
                loss_batch = self.backend.factor * self.backend.lossfunc(f_batch, y)

            if self.likelihood == Likelihood.REGRESSION:
                b, C = f_batch.shape
                lambdas_batch = torch.unsqueeze(
                    torch.eye(C, device=self._device, dtype=self._dtype), 0
                ).repeat(b, 1, 1)
            else:
                # second derivative of log lik is diag(p) - pp^T
                ps = torch.softmax(f_batch, dim=-1)
                lambdas_batch = torch.diag_embed(ps) - torch.einsum(
                    "mk,mc->mck", ps, ps
                )

            self.loss += loss_batch
            lambdas.append(lambdas_batch)
            f.append(f_batch)
            mu.append(
                self._mean_scatter_term_batch(Js_batch, f_batch, y)
            )  # needed for marginal likelihood
            for j, data2 in enumerate(train_loader):
                if j >= i:
                    X2 = self._batch_inputs(data2)
                    K_batch = self._kernel_batch(Js_batch, X2)
                    self._store_K_batch(K_batch, i, j)

        self.L = self._build_L(lambdas)
        self.mu = torch.cat(mu, dim=0)
        self._build_Sigma_inv()
        self._fitted = True

    @torch.enable_grad()
    def _glm_predictive_distribution(self, X: torch.Tensor, joint: bool = False):
        Js, f_mu = self._jacobians(X)

        if joint:
            f_mu = f_mu.flatten()  # (batch*out)
            f_var = self.functional_covariance(Js)  # (batch*out, batch*out)
        else:
            f_var = self.functional_variance(Js)

        return (
            (f_mu.detach(), f_var.detach())
            if not self.enable_backprop
            else (f_mu, f_var)
        )

    def _refresh_posterior(self) -> None:
        """Rebuild the subset-of-data GP posterior after a prior change.

        Warns
        -----
        UserWarning
            If prior precision changed since fitting and the cached posterior
            must be recomputed.
        """
        if self._recompute_Sigma:
            warnings.warn(
                "The prior precision has changed since fit; recomputing the GP posterior."
            )
            self._build_Sigma_inv()

    @property
    def gp_kernel_prior_variance(self):
        return self._prior_factor_sod / self.prior_precision

    def functional_variance(self, Js_star: torch.Tensor) -> torch.Tensor:
        """GP posterior variance:

        $$
            k_{**} - K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}
        $$

        Parameters
        ----------
        Js_star : torch.Tensor of shape (N*, C, P)
                  Jacobians of test data points

        Returns
        -------
        f_var : torch.Tensor of shape (N*,C, C)
                Contains the posterior variances of N* testing points.
        """
        # Compute K_{**}
        K_star = self.gp_kernel_prior_variance * self._kernel_star(Js_star)

        # Compute K_{*M}
        K_M_star = []
        for data in self.train_loader:
            K_M_star_batch = self.gp_kernel_prior_variance * self._kernel_batch_star(
                Js_star, self._batch_inputs(data)
            )
            K_M_star.append(K_M_star_batch)

        # Build_K_star_M computes K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}
        f_var = K_star - self._build_K_star_M(K_M_star)

        # If the considered kernel is diagonal, embed the covariances.
        # from (N*, C) -> (N*, C, C)
        if self.independent_outputs:
            f_var = torch.diag_embed(f_var)

        return f_var

    def functional_covariance(self, Js_star: torch.Tensor) -> torch.Tensor:
        """GP posterior covariance:

        $$
            k_{**} - K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}
        $$

        Parameters
        ----------
        Js_star : torch.Tensor of shape (N*, C, P)
                  Jacobians of test data points

        Returns
        -------
        f_var : torch.Tensor of shape (N*xC, N*xC)
                Contains the posterior covariances of N* testing points.
        """
        # Compute K_{**}
        K_star = self.gp_kernel_prior_variance * self._kernel_star(Js_star, joint=True)

        # Compute K_{*M}
        K_M_star = []
        for data in self.train_loader:
            K_M_star_batch = self.gp_kernel_prior_variance * self._kernel_batch_star(
                Js_star, self._batch_inputs(data)
            )
            K_M_star.append(K_M_star_batch)

        # Build_K_star_M computes K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}
        f_var = K_star - self._build_K_star_M(K_M_star, joint=True)

        # If the considered kernel is diagonal, embed the covariances.
        # from (N*, N*, C) -> (N*, N*, C, C)
        if self.independent_outputs:
            f_var = torch.diag_embed(f_var)

        # Reshape from (N*, N*, C, C) to (N*xC, N*xC)
        f_var = f_var.permute(0, 2, 1, 3).flatten(0, 1).flatten(1, 2)

        return f_var

    def _build_K_star_M(
        self, K_M_star: torch.Tensor, joint: bool = False
    ) -> torch.Tensor:
        """Computes K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*} given K_{M*}.

        Parameters
        ----------
        K_M_star : list of torch.Tensor
                   Contains K_{M*}. Tensors have shape (N_test, C, C)
                   or (N_test, C) for diagonal kernel.

        joint : boolean
                Wether to compute cross covariances or not.

        Returns
        -------
        torch.tensor of shape (N_test, N_test, C) for joint diagonal,
        (N_test, C) for non-joint diagonal, (N_test, N_test, C, C) for
        joint non-diagonal and (N_test, C, C) for non-joint non-diagonal.
        """
        # Shape (N_test, N, C, C) or (N_test, N, C) for diagonal
        K_M_star = torch.cat(K_M_star, dim=1)

        if self.independent_outputs:
            prods = []
            for c in range(self.n_outputs):
                # Compute K_{*M}L^{-1}
                v = torch.squeeze(
                    torch.linalg.solve(
                        self.Sigma_inv[c], K_M_star[:, :, c].unsqueeze(2)
                    ),
                    2,
                )
                if joint:
                    prod = torch.einsum("bm,am->ba", v, v)
                else:
                    prod = torch.einsum("bm,bm->b", v, v)
                prods.append(prod.unsqueeze(-1))
            prods = torch.cat(prods, dim=-1)
            return prods
        else:
            # Reshape to (N_test, NxC, C) or (N_test, N, C)
            K_M_star = K_M_star.reshape(K_M_star.shape[0], -1, K_M_star.shape[-1])
            # Compute K_{*M}L^{-1}
            v = torch.linalg.solve(self.Sigma_inv, K_M_star)
            if joint:
                return torch.einsum("acm,bcn->abmn", v, v)
            else:
                return torch.einsum("bcm,bcn->bmn", v, v)

    @property
    def log_det_ratio(self) -> torch.Tensor:
        """Computes log determinant term in GP marginal likelihood

        For `classification` we use eq. (3.44) from Chapter 3.5 from
        [GP book R&W 2006](http://www.gaussianprocess.org/gpml/chapters/) with
        (note that we always use diagonal approximation \\(D\\) of the Hessian of log likelihood w.r.t. \\(f\\)):

        log determinant term := \\( \\log | I + D^{1/2}K D^{1/2} | \\)

        For `regression`, we use ["standard" GP marginal likelihood](https://stats.stackexchange.com/questions/280105/log-marginal-likelihood-for-gaussian-process):

        log determinant term := \\( \\log | K + \\sigma_2 I | \\)
        """
        if self.likelihood == Likelihood.REGRESSION:
            if self.independent_outputs:
                log_det = torch.tensor(
                    0.0, requires_grad=True, device=self._device, dtype=self._dtype
                )
                for c in range(self.n_outputs):
                    log_det = log_det + torch.logdet(
                        self.gp_kernel_prior_variance * self.K_MM[c]
                        + torch.eye(
                            n=self.K_MM[c].shape[0],
                            device=self._device,
                            dtype=self._dtype,
                        )
                        * self.sigma_noise.square()
                    )
                return log_det
            else:
                return torch.logdet(
                    self.gp_kernel_prior_variance * self.K_MM
                    + torch.eye(
                        n=self.K_MM.shape[0], device=self._device, dtype=self._dtype
                    )
                    * self.sigma_noise.square()
                )
        else:
            if self.independent_outputs:
                log_det = torch.tensor(
                    0.0, requires_grad=True, device=self._device, dtype=self._dtype
                )
                for c in range(self.n_outputs):
                    W = torch.sqrt(self._H_factor * self.L[c])
                    log_det = log_det + torch.logdet(
                        W[:, None] * self.gp_kernel_prior_variance * self.K_MM[c] * W
                        + torch.eye(
                            n=self.K_MM[c].shape[0],
                            device=self._device,
                            dtype=self._dtype,
                        )
                    )
                return log_det
            else:
                W = torch.sqrt(self._H_factor * self.L)
                return torch.logdet(
                    W[:, None] * self.gp_kernel_prior_variance * self.K_MM * W
                    + torch.eye(
                        n=self.K_MM.shape[0], device=self._device, dtype=self._dtype
                    )
                )

    @property
    def scatter(self, eps: float = 0.00001) -> torch.Tensor:
        """Compute scatter term in GP log marginal likelihood.

        For `classification` we use eq. (3.44) from Chapter 3.5 from
        [GP book R&W 2006](http://www.gaussianprocess.org/gpml/chapters/) with \\(\\hat{f} = f \\):

        scatter term := \\( f K^{-1} f^{T} \\)

        For `regression`, we use ["standard" GP marginal likelihood](https://stats.stackexchange.com/questions/280105/log-marginal-likelihood-for-gaussian-process):

        scatter term := \\( (y - m)K^{-1}(y -m )^T \\),
        where \\( m \\) is the mean of the GP prior, which in our case corresponds to
        \\( m := f + J (\\theta - \\theta_{MAP}) \\)

        """
        if self.likelihood == "regression":
            noise = self.sigma_noise.square()
        else:
            noise = eps
        if self.independent_outputs:
            scatter = torch.tensor(
                0.0, requires_grad=True, device=self._device, dtype=self._dtype
            )
            for c in range(self.n_outputs):
                m = self.K_MM[c].shape[0]
                mu_term = torch.linalg.solve(
                    torch.linalg.cholesky(
                        self.gp_kernel_prior_variance * self.K_MM[c]
                        + torch.diag(
                            torch.ones(m, device=self._device, dtype=self._dtype)
                            * noise
                        )
                    ),
                    self.mu[:, c],
                )
                scatter = scatter + torch.dot(mu_term, mu_term)
        else:
            m = self.K_MM.shape[0]
            mu_term = torch.linalg.solve(
                torch.linalg.cholesky(
                    self.gp_kernel_prior_variance * self.K_MM
                    + torch.diag(
                        torch.ones(m, device=self._device, dtype=self._dtype) * noise
                    )
                ),
                self.mu.reshape(-1),
            )
            scatter = torch.dot(mu_term, mu_term)
        return scatter

    def optimize_prior_precision(
        self,
        pred_type: PredType | str = PredType.GP,
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
        log_prior_prec_base: float = 10.0,
    ) -> None:
        """`optimize_prior_precision_base` from `BaseLaplace` with `pred_type='gp'`"""
        assert pred_type == PredType.GP  # only gp supported
        assert prior_structure == "scalar"  # only isotropic gaussian prior supported
        if method == "marglik":
            warnings.warn(
                "Use of method='marglik' in case of FunctionalLaplace is discouraged, rather use method='CV'."
            )
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
            log_prior_prec_base=log_prior_prec_base,
        )
        self._build_Sigma_inv()

    def _kernel_batch(
        self, jacobians: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute K_bb, which is part of K_MM kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b, C, P)
        batch : torch.Tensor (b, C)

        Returns
        -------
        kernel : torch.tensor
            K_bb with shape (b * C, b * C)
        """
        jacobians_2, _ = self._jacobians(batch)
        P = jacobians.shape[-1]  # nr model params
        if self.independent_outputs:
            kernel = torch.empty(
                (jacobians.shape[0], jacobians_2.shape[0], self.n_outputs),
                device=jacobians.device,
                dtype=self._dtype,
            )
            for c in range(self.n_outputs):
                kernel[:, :, c] = torch.einsum(
                    "bp,ep->be", jacobians[:, c, :], jacobians_2[:, c, :]
                )
        else:
            kernel = torch.einsum(
                "ap,bp->ab", jacobians.reshape(-1, P), jacobians_2.reshape(-1, P)
            )
        del jacobians_2
        return kernel

    def _kernel_star(
        self, jacobians: torch.Tensor, joint: bool = False
    ) -> torch.Tensor:
        """Compute K_star_star kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b, C, P)

        Returns
        -------
        kernel : torch.tensor
            K_star with shape (b, C, C)

        """
        if joint:
            if self.independent_outputs:
                kernel = torch.einsum("acp,bcp->abc", jacobians, jacobians)
            else:
                kernel = torch.einsum("acp,bep->abce", jacobians, jacobians)

        else:
            if self.independent_outputs:
                kernel = torch.empty(
                    (jacobians.shape[0], self.n_outputs),
                    device=jacobians.device,
                    dtype=self._dtype,
                )
                for c in range(self.n_outputs):
                    kernel[:, c] = torch.norm(jacobians[:, c, :], dim=1) ** 2
            else:
                kernel = torch.einsum("bcp,bep->bce", jacobians, jacobians)
        return kernel

    def _kernel_batch_star(
        self, jacobians: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute K_b_star, which is a part of K_M_star kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b1, C, P)
        batch : torch.Tensor (b2, C)

        Returns
        -------
        kernel : torch.tensor
            K_batch_star with shape (b1, b2, C, C)
        """
        jacobians_2, _ = self._jacobians(batch)
        if self.independent_outputs:
            kernel = torch.empty(
                (jacobians.shape[0], jacobians_2.shape[0], self.n_outputs),
                device=jacobians.device,
                dtype=self._dtype,
            )
            for c in range(self.n_outputs):
                kernel[:, :, c] = torch.einsum(
                    "bp,ep->be", jacobians[:, c, :], jacobians_2[:, c, :]
                )
        else:
            kernel = torch.einsum("bcp,dep->bdce", jacobians, jacobians_2)
        return kernel

    def _jacobians(self, X: torch.Tensor, enable_backprop: bool = None) -> tuple:
        """A wrapper function to compute jacobians - this enables reusing same
        kernel methods (kernel_batch etc.) in FunctionalLaplace and FunctionalLLLaplace
        by simply overwriting this method instead of all kernel methods.
        """
        if enable_backprop is None:
            enable_backprop = self.enable_backprop
        return self.backend.jacobians(X, enable_backprop=enable_backprop)

    def _mean_scatter_term_batch(
        self, Js: torch.Tensor, f: torch.Tensor, y: torch.Tensor
    ):
        """Compute mean vector in the scatter term in the log marginal likelihood

        See `scatter_lml` property above for the exact equations of mean vectors in scatter terms for
        both types of likelihood (regression, classification).

        Parameters
        ----------
        Js : torch.tensor
              Jacobians (batch, output_shape, parameters)
        f : torch.tensor
              NN output (batch, output_shape)
        y: torch.tensor
              data labels (batch, output_shape)

        Returns
        -------
        mu : torch.tensor
            K_batch_star with shape (batch, output_shape)
        """
        if self.likelihood == Likelihood.REGRESSION:
            return y - (f + torch.einsum("bcp,p->bc", Js, self.prior_mean - self.mean))
        elif self.likelihood == Likelihood.CLASSIFICATION:
            return -torch.einsum("bcp,p->bc", Js, self.prior_mean - self.mean)

    def log_marginal_likelihood(
        self,
        prior_precision: torch.Tensor | None = None,
        sigma_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Laplace approximation to the log marginal likelihood.
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

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        self._posterior_scale = None
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.as_tensor(
                [prior_precision], device=self._device, dtype=self._dtype
            )
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(
                    device=self._device, dtype=self._dtype
                )
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError(
                        "Length of prior precision does not align with architecture."
                    )
                self._prior_precision = prior_precision.to(
                    device=self._device, dtype=self._dtype
                )
            else:
                raise ValueError(
                    "Prior precision needs to be at most one-dimensional tensor."
                )
        else:
            raise ValueError(
                "Prior precision either scalar or torch.Tensor up to 1-dim."
            )
        # This is a change from BaseLaplace. If the prior precision is changed, the cholesky
        #  decomposition needs to be recomputed.
        self._recompute_Sigma = True

    def state_dict(self) -> dict:
        state_dict = {
            "mean": self.mean,
            "num_data": self.n_subset,
            "diagonal_kernel": self.independent_outputs,
            "seed": self.seed,
            "K_MM": self.K_MM,
            "Sigma_inv": self.Sigma_inv,
            "_prior_factor_sod": self._prior_factor_sod,
            "_fitted": self._fitted,
            "_recompute_Sigma": self._recompute_Sigma,
            "mu": self.mu,
            "L": self.L,
            "train_loader": self.train_loader,
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

    def load_state_dict(self, state_dict: dict):
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
        self.n_subset = state_dict["num_data"]
        self.independent_outputs = state_dict["diagonal_kernel"]
        self.seed = state_dict["seed"]
        self.K_MM = state_dict["K_MM"]
        self.Sigma_inv = state_dict["Sigma_inv"]
        self._prior_factor_sod = state_dict["_prior_factor_sod"]
        self.mu = state_dict["mu"]
        self.L = state_dict["L"]
        self._fitted = state_dict["_fitted"]
        self.train_loader = state_dict["train_loader"]

        self.loss = state_dict["loss"]
        self.prior_mean = state_dict["prior_mean"]
        self.prior_precision = state_dict["prior_precision"]
        self.sigma_noise = state_dict["sigma_noise"]
        # The property setters mark the factor stale; restore the saved state.
        self._recompute_Sigma = state_dict["_recompute_Sigma"]
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        setattr(self.model, "output_size", self.n_outputs)
        self.likelihood = state_dict["likelihood"]
        self.temperature = state_dict["temperature"]
        self.enable_backprop = state_dict["enable_backprop"]


class ELLA(BaseFunctionalLaplace):
    """Accelerated linearized Laplace using a rank-limited Nyström feature map.

    The pretrained model stays fixed. Fitting selects a subset of training
    inputs, builds parameter-Jacobian features, and accumulates projected GGN
    curvature. A numerically rank-deficient subset cannot supply the requested
    number of features.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained network to linearize.
    likelihood : Likelihood or str
        ``"classification"``, ``"regression"``, or ``"reward_modeling"``.
    subsample_size : int
        Number of training inputs used to construct the Nyström basis.
    n_eigenvalues : int
        Number of positive kernel directions retained, at most
        ``subsample_size``.
    sigma_noise : float or torch.Tensor, default=1.0
        Observation standard deviation for regression.
    prior_precision : float or torch.Tensor, default=1.0
        Scalar parameter-prior precision.
    prior_mean : float or torch.Tensor, default=0.0
        Prior mean; only zero is supported.
    temperature : float, default=1.0
        Positive scale of the likelihood curvature.
    enable_backprop : bool, default=False
        Retain derivatives through predictive moments.
    dict_key_x : str, default="input_ids"
        Input key for mapping-style batches.
    dict_key_y : str, default="labels"
        Target key for mapping-style batches.
    backend : type[CurvatureInterface] or None, default=None
        Curvature backend used to compute parameter Jacobians.
    backend_kwargs : dict[str, Any] or None, default=None
        Additional curvature backend options.
    seed : int, default=1234
        Seed for selecting basis inputs and output coordinates.

    Attributes
    ----------
    fit_history_ : dict[str, list[Any]]
        ``processed_examples`` records cumulative curvature-pass counts,
        ``val_nll`` records validation scores, and ``tuning`` records
        candidate hyperparameters and their validation scores.
    dual_directions : torch.Tensor or None
        Fitted parameter-space basis shaped ``[parameters, n_eigenvalues]``.
    feature_gram : torch.Tensor or None
        Accumulated projected curvature shaped
        ``[n_eigenvalues, n_eigenvalues]`` before noise and temperature scaling.
    chol_precision : torch.Tensor or None
        Lower Cholesky factor of the feature-space posterior precision.

    Notes
    -----
    ``fit`` returns ``None``. ``fit_history_`` stores processed example counts,
    validation NLL, and prior-grid scores when those options are used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Likelihood | str,
        subsample_size: int,
        n_eigenvalues: int,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        seed: int = 1234,
    ) -> None:
        """Initialize ELLA from the pretrained model and basis configuration."""
        if (
            not isinstance(subsample_size, Integral)
            or isinstance(subsample_size, bool)
            or not isinstance(n_eigenvalues, Integral)
            or isinstance(n_eigenvalues, bool)
            or subsample_size <= 0
            or n_eigenvalues <= 0
            or n_eigenvalues > subsample_size
        ):
            raise ValueError("Require 0 < n_eigenvalues <= subsample_size.")
        if torch.as_tensor(prior_precision).numel() != 1:
            raise ValueError("ELLA requires scalar prior_precision.")
        if torch.any(torch.as_tensor(prior_mean) != 0):
            raise ValueError("ELLA currently requires prior_mean=0.")
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
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
        self.subsample_size = subsample_size
        self.n_eigenvalues = n_eigenvalues
        self.seed = seed
        self.dual_directions: torch.Tensor | None = None
        self.feature_gram: torch.Tensor | None = None
        self.chol_precision: torch.Tensor | None = None
        self._posterior_dirty = False
        self._fitted = False
        self.fit_history_: dict[str, list[Any]] = {
            "processed_examples": [],
            "val_nll": [],
            "tuning": [],
        }

    @property
    def prior_precision(self) -> torch.Tensor:
        """Return the scalar parameter-prior precision."""
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, value: float | torch.Tensor) -> None:
        """Set a positive scalar precision and invalidate the posterior factor."""
        candidate = torch.as_tensor(value, device=self._device, dtype=self._dtype)
        if candidate.numel() != 1 or not torch.all(
            torch.isfinite(candidate) & (candidate > 0)
        ):
            raise ValueError("ELLA requires positive finite scalar prior_precision.")
        BaseLaplace.prior_precision.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    @property
    def prior_mean(self) -> torch.Tensor:
        """Return the zero parameter-prior mean."""
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, value: float | torch.Tensor) -> None:
        """Set the parameter-prior mean, which must be zero."""
        if torch.any(torch.as_tensor(value) != 0):
            raise ValueError("ELLA currently requires prior_mean=0.")
        BaseLaplace.prior_mean.fset(self, value)  # type: ignore[attr-defined]

    @property
    def sigma_noise(self) -> torch.Tensor:
        """Return the regression observation standard deviation."""
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, value: float | torch.Tensor) -> None:
        """Set observation noise and invalidate the posterior factor."""
        candidate = torch.as_tensor(value, device=self._device, dtype=self._dtype)
        if not torch.all(torch.isfinite(candidate) & (candidate > 0)):
            raise ValueError("sigma_noise must be positive and finite.")
        if self.likelihood != Likelihood.REGRESSION and torch.any(
            torch.as_tensor(value) != 1
        ):
            raise ValueError("sigma_noise != 1 is only available for regression.")
        BaseLaplace.sigma_noise.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    @property
    def temperature(self) -> float:
        """Return the positive likelihood temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set likelihood temperature and invalidate the posterior factor."""
        if not isfinite(value) or value <= 0:
            raise ValueError("temperature must be positive and finite.")
        self._temperature = value
        if hasattr(self, "feature_gram") and self.feature_gram is not None:
            self._posterior_dirty = True

    def _batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """Move a training batch and its targets to the model device.

        Parameters
        ----------
        batch : Any
            ``(inputs, targets)`` or a mapping containing ``dict_key_y``.

        Returns
        -------
        tuple[Any, torch.Tensor]
            Model inputs and targets on the model device. Floating-point
            inputs use the model dtype; target dtypes are preserved.
        """
        x, y = split_batch(batch, self.dict_key_y)
        return to_device(x, self._device, self._dtype), y.to(self._device)

    def _indices(self, loader: DataLoader, balanced: bool) -> torch.Tensor:
        """Select reproducible basis rows, optionally balanced across classes.

        Parameters
        ----------
        loader : DataLoader
            Training data from which basis rows are selected.
        balanced : bool
            Round-robin sample classes when true.

        Returns
        -------
        torch.Tensor
            Selected dataset indices of length ``subsample_size``.
        """
        available = training_indices(loader)
        n = len(available)
        if self.subsample_size > n:
            raise ValueError("subsample_size cannot exceed the training set size.")

        generator = torch.Generator().manual_seed(self.seed)
        if not balanced:
            chosen = torch.randperm(n, generator=generator)[: self.subsample_size]
            return available[chosen]

        if self.likelihood == Likelihood.REGRESSION:
            raise ValueError("balanced sampling is only defined for classification.")

        labels = []
        for index in available:
            _, target = split_batch(loader.dataset[int(index)], self.dict_key_y)
            labels.append(
                int(
                    torch.as_tensor(target).argmax()
                    if torch.as_tensor(target).numel() > 1
                    else target
                )
            )
        labels = torch.tensor(labels)
        classes = labels.unique(sorted=True)
        queues = []
        for cls in classes:
            candidates = torch.where(labels == cls)[0]
            order = torch.randperm(len(candidates), generator=generator)
            queues.append(deque(candidates[order].tolist()))

        selected: list[int] = []
        while len(selected) < self.subsample_size and any(queues):
            for queue in queues:
                if queue and len(selected) < self.subsample_size:
                    selected.append(queue.popleft())
        if len(selected) != self.subsample_size:
            raise ValueError("Not enough examples for balanced subsampling.")
        return available[torch.tensor(selected[: self.subsample_size])]

    def _build_basis(self, loader: DataLoader, indices: torch.Tensor) -> None:
        """Build the Nyström directions from selected parameter Jacobians.

        Parameters
        ----------
        loader : DataLoader
            Training loader used to retrieve selected inputs.
        indices : torch.Tensor
            Dataset indices selected for the subset kernel.

        Raises
        ------
        ValueError
            If the subset kernel has fewer usable directions than requested.
        """
        selected_rows = []
        generator = torch.Generator().manual_seed(self.seed + 1)
        for batch in subset_loader(loader, indices):
            x, _ = self._batch(batch)
            if (
                self.likelihood == Likelihood.REWARD_MODELING
                and not self._uses_backpack_backend
            ):
                jacobians, outputs = CurvatureInterface.jacobians(self.backend, x)
            else:
                jacobians, outputs = self.backend.jacobians(x)
            choices = torch.randint(
                outputs.shape[-1], (outputs.shape[0],), generator=generator
            ).to(jacobians.device)
            selected_rows.append(
                jacobians[torch.arange(len(choices), device=jacobians.device), choices]
            )

        rows = torch.cat(selected_rows).to(torch.float64)
        kernel = rows @ rows.T
        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        order = torch.argsort(eigenvalues, descending=True)[: self.n_eigenvalues]
        values = eigenvalues[order]
        tolerance = (
            torch.finfo(values.dtype).eps * max(rows.shape) * eigenvalues.abs().max()
        )

        if torch.any(values <= tolerance):
            usable_rank = int((eigenvalues > tolerance).sum().item())
            raise ValueError(
                "Nyström subset kernel has insufficient positive rank: "
                f"requested n_eigenvalues={self.n_eigenvalues}, but only "
                f"{usable_rank} of {len(eigenvalues)} sampled directions "
                "exceed the numerical tolerance. Reduce n_eigenvalues, "
                "increase subsample_size, or try a different seed."
            )

        self.dual_directions = (
            rows.T @ (eigenvectors[:, order] / values.sqrt().unsqueeze(0))
        ).to(self._dtype)

    def _features(
        self, x: torch.Tensor | MutableMapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project model Jacobians onto the Nyström basis and return outputs.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched inputs on the model device.

        Returns
        -------
        features : torch.Tensor
            Projected Jacobians shaped ``[batch, outputs, n_eigenvalues]``.
        outputs : torch.Tensor
            Network outputs shaped ``[batch, outputs]``.

        Raises
        ------
        RuntimeError
            If the basis is uninitialized, or differentiable Jacobians are
            requested but unavailable for the model and backend.

        Notes
        -----
        Uses Jacobian-vector products when supported, otherwise projects
        parameter Jacobians obtained from the curvature backend.
        """
        if self.dual_directions is None:
            raise RuntimeError("Call fit before requesting ELLA features.")
        # BackPACK extends modules in-place and is incompatible with torch.func.jvp.
        if not self._uses_backpack_backend:
            parameters = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            feature_columns = []
            for direction in self.dual_directions.T:
                offset = 0
                tangents = {}
                for name, param in parameters.items():
                    if param.requires_grad:
                        tangent = direction[offset : offset + param.numel()].reshape_as(
                            param
                        )
                        offset += param.numel()
                    else:
                        tangent = torch.zeros_like(param)
                    tangents[name] = tangent

                def model_fn(params):
                    return torch.func.functional_call(
                        self.model, (params, buffers), (x,)
                    )

                try:
                    output, product = torch.func.jvp(
                        model_fn, (parameters,), (tangents,)
                    )
                except (NotImplementedError, RuntimeError) as exc:
                    if "forward AD" not in str(exc) and "jvp" not in str(exc):
                        raise
                    break
                feature_columns.append(product)
            if len(feature_columns) == self.n_eigenvalues:
                return torch.stack(feature_columns, dim=-1), output
        if (
            self.likelihood == Likelihood.REWARD_MODELING
            and self._uses_backpack_backend
            and (query_shape := self.model(x).shape)[-1] == 1
        ):
            selectors = torch.zeros(
                query_shape[0], device=self._device, dtype=torch.long
            )
            jacobians, output = self.backend.selected_output_jacobians(
                x, selectors, enable_backprop=self.enable_backprop
            )
        elif (
            self.likelihood == Likelihood.REWARD_MODELING
            and not self._uses_backpack_backend
        ):
            jacobians, output = CurvatureInterface.jacobians(
                self.backend, x, enable_backprop=self.enable_backprop
            )
        elif self._uses_asdl_backend:
            # Reverse-mode torch.func can retain input derivatives when the
            # model does not support JVP; ASDL's Jacobians can detach them.
            try:
                jacobians, output = CurvatureInterface.jacobians(
                    self.backend, x, enable_backprop=self.enable_backprop
                )
            except (NotImplementedError, RuntimeError) as exc:
                if self.enable_backprop:
                    raise RuntimeError(
                        "This model cannot provide differentiable ELLA Jacobians."
                    ) from exc
                jacobians, output = self.backend.jacobians(x)
        else:
            jacobians, output = self.backend.jacobians(
                x, enable_backprop=self.enable_backprop
            )
        return jacobians @ self.dual_directions, output

    def _refresh_posterior(self) -> None:
        """Rebuild the posterior factor after a hyperparameter change."""
        if self._posterior_dirty:
            self._build_precision()

    def _build_precision(self) -> None:
        """Factor the feature-space posterior precision."""
        if self.feature_gram is None:
            return
        precision = self.feature_gram / self.temperature
        if self.likelihood == Likelihood.REGRESSION:
            precision = precision / self.sigma_noise.square()
        precision = precision + self.prior_precision * torch.eye(
            self.n_eigenvalues, device=self._device, dtype=self._dtype
        )
        self.chol_precision = torch.linalg.cholesky(precision)
        self._posterior_dirty = False

    @preserve_model_gradients
    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        val_steps: int | None = None,
        balanced: bool = False,
        progress_bar: bool = False,
    ) -> None:
        """Fit the ELLA posterior with projected training curvature.

        Parameters
        ----------
        train_loader : DataLoader
            Training batches for basis selection and curvature accumulation.
        val_loader : DataLoader or None, default=None
            Optional validation data for predictive negative log likelihood.
        val_steps : int or None, default=None
            Evaluate validation NLL every ``val_steps`` batches when given.
        balanced : bool, default=False
            Balance class labels in the Nyström subset.
        progress_bar : bool, default=False
            Show a fitting progress bar.

        Raises
        ------
        ValueError
            If sampling or validation-step settings are invalid, or the
            curvature pass processes no examples.

        Notes
        -----
        Basis selection and curvature accumulation sample the loader
        independently. Stochastic samplers may select different rows for each
        pass. ``n_data`` counts examples processed during curvature accumulation.
        """
        if val_steps is not None and (
            not isinstance(val_steps, Integral)
            or isinstance(val_steps, bool)
            or val_steps <= 0
        ):
            raise ValueError("val_steps must be positive integer when provided.")

        self.model.eval()
        self.fit_history_ = {"processed_examples": [], "val_nll": [], "tuning": []}
        self._fitted = False
        self.n_data = 0
        indices = self._indices(train_loader, balanced)
        first_x, _ = self._batch(next(iter(subset_loader(train_loader, indices))))
        if isinstance(first_x, MutableMapping) and self._uses_backpack_backend:
            raise ValueError(
                "BackPACK does not support mapping-style inputs; use AsdlGGN or CurvlinopsGGN."
            )

        out = self.model(first_x)
        self._check_output_dims(out)
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)
        self._build_basis(train_loader, indices)
        self.feature_gram = torch.zeros(
            self.n_eigenvalues,
            self.n_eigenvalues,
            device=self._device,
            dtype=self._dtype,
        )

        iterator = train_loader
        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(train_loader, desc="Fitting ELLA")

        for step, batch in enumerate(iterator, start=1):
            x, y = self._batch(batch)
            features, output = self._features(x)
            if self.likelihood == Likelihood.REGRESSION:
                regression_targets(y, output)
                self.feature_gram += torch.einsum(
                    "bck,bcl->kl", features, features
                ).detach()
            else:
                probs = output.softmax(dim=-1)
                hessian = torch.diag_embed(probs) - probs.unsqueeze(
                    -1
                ) * probs.unsqueeze(-2)
                self.feature_gram += torch.einsum(
                    "bck,bcd,bdl->kl", features, hessian, features
                ).detach()
            self.n_data += output.shape[0]
            self.fit_history_["processed_examples"].append(float(self.n_data))
            if val_loader is not None and val_steps and step % val_steps == 0:
                self._build_precision()
                self._fitted = True
                self.fit_history_["val_nll"].append(
                    float(self._validation_nll(val_loader))
                )

        if self.n_data == 0:
            raise ValueError("ELLA fit requires a nonempty training loader.")

        self._build_precision()
        self._fitted = True
        if val_loader is not None and (not val_steps or step % val_steps):
            self.fit_history_["val_nll"].append(float(self._validation_nll(val_loader)))

    def _feature_variance(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-input output covariance from projected features.

        Parameters
        ----------
        features : torch.Tensor
            Projected Jacobians shaped ``[batch, outputs, n_eigenvalues]``.

        Returns
        -------
        torch.Tensor
            Latent covariance shaped ``[batch, outputs, outputs]`` without
            observation noise, using the current posterior precision factor.
        """
        solved = torch.cholesky_solve(features.transpose(1, 2), self.chol_precision)
        return features @ solved

    def _feature_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """Return joint output covariance from projected features.

        Parameters
        ----------
        features : torch.Tensor
            Projected Jacobians shaped ``[batch, outputs, n_eigenvalues]``.

        Returns
        -------
        torch.Tensor
            Latent covariance shaped ``[batch * outputs, batch * outputs]``
            without observation noise. Coordinates are ordered by input,
            then output.
        """
        flat = features.reshape(-1, self.n_eigenvalues)
        solved = torch.cholesky_solve(flat.T, self.chol_precision)
        return flat @ solved

    def functional_variance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return per-input latent covariance from parameter Jacobians.

        Parameters
        ----------
        jacobians : torch.Tensor
            Parameter Jacobians shaped ``[batch, outputs, parameters]``.

        Returns
        -------
        torch.Tensor
            Per-input covariance shaped ``[batch, outputs, outputs]``.
        """
        self._check_fitted()
        self._check_jacobians(jacobians)
        return self._feature_variance(jacobians @ self.dual_directions)

    def functional_covariance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return joint latent covariance from parameter Jacobians.

        Parameters
        ----------
        jacobians : torch.Tensor
            Parameter Jacobians shaped ``[batch, outputs, parameters]``.

        Returns
        -------
        torch.Tensor
            Joint covariance shaped ``[batch * outputs, batch * outputs]``.
        """
        self._check_fitted()
        self._check_jacobians(jacobians)
        return self._feature_covariance(jacobians @ self.dual_directions)

    @preserve_model_gradients
    @torch.enable_grad()
    def _glm_predictive_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute model outputs and ELLA latent covariance for inputs.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        joint : bool, default=False
            Retain covariance between input rows.

        Returns
        -------
        mean : torch.Tensor
            Network outputs shaped ``[batch, outputs]``, flattened to
            ``[batch * outputs]`` when ``joint=True``.
        covariance : torch.Tensor
            Latent covariance shaped ``[batch, outputs, outputs]``, or
            ``[batch * outputs, batch * outputs]`` when ``joint=True``.
            Observation noise is excluded.
        """
        x = to_device(x, self._device, self._dtype)
        features, mean = self._features(x)
        covariance = (
            self._feature_covariance(features)
            if joint
            else self._feature_variance(features)
        )
        if joint:
            mean = mean.flatten()
        if not self.enable_backprop:
            return mean.detach(), covariance.detach()
        return mean, covariance

    def _validation_nll(self, loader: DataLoader) -> torch.Tensor:
        """Return mean predictive negative log likelihood on a loader.

        Parameters
        ----------
        loader : DataLoader
            Nonempty validation loader with regression targets or class labels.

        Returns
        -------
        torch.Tensor
            Scalar NLL averaged over examples, summing over regression output
            coordinates. Regression includes observation noise; classification
            uses the probit approximation.
        """
        total = torch.zeros((), device=self._device, dtype=self._dtype)
        count = 0
        for batch in loader:
            x, y = self._batch(batch)
            if self.likelihood == Likelihood.REGRESSION:
                mean, covariance = self.predictive_moments(x)
                y = regression_targets(y, mean)
                variance = (
                    covariance.diagonal(dim1=-2, dim2=-1) + self.sigma_noise.square()
                )
                total += (
                    0.5
                    * (
                        (y - mean).square() / variance
                        + variance.log()
                        + torch.log(torch.as_tensor(2 * torch.pi, device=self._device))
                    )
                ).sum()
            else:
                total += self._classification_validation_nll(x, y)
            count += y.shape[0]
        return total / count

    def optimize_prior_precision(
        self,
        pred_type: PredType | str = PredType.GP,
        method: TuningMethod | str = TuningMethod.GRIDSEARCH,
        n_steps: int = 100,
        lr: float = 1e-1,
        init_prior_prec: float | torch.Tensor = 1.0,
        prior_structure: PriorStructure | str = PriorStructure.SCALAR,
        val_loader: DataLoader | None = None,
        loss: Any = None,
        log_prior_prec_min: float = -4,
        log_prior_prec_max: float = 4,
        grid_size: int = 100,
        link_approx: LinkApprox | str = LinkApprox.PROBIT,
        n_samples: int = 100,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> None:
        """Choose scalar prior precision by validation grid search.

        Parameters
        ----------
        pred_type : PredType or str, default=PredType.GP
            Must be ``"gp"``.
        method : TuningMethod or str, default=TuningMethod.GRIDSEARCH
            Must be ``"gridsearch"``.
        n_steps : int, default=100
            Compatibility argument; only the default is supported and it does
            not determine grid size.
        lr : float, default=0.1
            Compatibility argument; only the default is supported.
        init_prior_prec : float or torch.Tensor, default=1.0
            Compatibility argument; must be the scalar default. Grid search
            evaluates the specified precision range instead.
        prior_structure : PriorStructure or str, default=PriorStructure.SCALAR
            Only scalar prior precision is supported.
        val_loader : DataLoader or None, default=None
            Validation data; required for grid search.
        loss : Any, default=None
            Must be ``None``; candidates are scored by predictive NLL.
        log_prior_prec_min : float, default=-4
            Base-10 logarithm of the smallest grid precision.
        log_prior_prec_max : float, default=4
            Base-10 logarithm of the largest grid precision.
        grid_size : int, default=100
            Number of candidate precisions.
        link_approx : LinkApprox or str, default=LinkApprox.PROBIT
            Classification validation link; only probit is supported.
        n_samples : int, default=100
            Compatibility argument; only the default is supported. Validation
            scoring does not use Monte Carlo samples.
        verbose : bool, default=False
            Compatibility argument; must be false.
        progress_bar : bool, default=False
            Compatibility argument; must be false.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.
        ValueError
            If the prediction type is not ``"gp"``, validation data are
            missing, or the grid size or candidate precisions are invalid.
        NotImplementedError
            If marginal-likelihood tuning or nondefault compatibility
            arguments are requested.

        Notes
        -----
        Other arguments are accepted for interface compatibility but only
        their defaults are supported by ELLA.
        """
        self._check_fitted()
        if pred_type != PredType.GP:
            raise ValueError("ELLA supports only pred_type='gp'.")
        if (
            n_steps != 100
            or lr != 1e-1
            or torch.as_tensor(init_prior_prec).numel() != 1
            or torch.as_tensor(init_prior_prec).item() != 1.0
            or prior_structure != PriorStructure.SCALAR
            or loss is not None
            or link_approx != LinkApprox.PROBIT
            or n_samples != 100
            or verbose
            or progress_bar
        ):
            raise NotImplementedError(
                "ELLA supports validation grid search without marginal-likelihood options."
            )
        if method != TuningMethod.GRIDSEARCH:
            raise NotImplementedError(
                "ELLA supports validation grid search, not marginal likelihood."
            )
        if val_loader is None:
            raise ValueError("gridsearch requires val_loader.")
        if (
            not isinstance(grid_size, Integral)
            or isinstance(grid_size, bool)
            or grid_size <= 0
        ):
            raise ValueError("grid_size must be positive integer.")
        grid = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.optimize_hyperparameters(val_loader, grid)

    def optimize_hyperparameters(self, val_loader: DataLoader, grid) -> None:
        """Select the prior (and optionally noise) with lowest validation NLL.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data used to score every candidate.
        grid : iterable
            Candidate prior precisions or ``(prior_precision, sigma_noise)``
            pairs for regression.

        Raises
        ------
        ValueError
            If the grid is empty or all candidate scores are nonfinite.
        """
        self._check_fitted()
        initial_prior = self.prior_precision.detach().clone()
        initial_noise = self.sigma_noise.detach().clone()
        best_score = float("inf")
        best = None
        has_candidates = False
        for candidate in grid:
            has_candidates = True
            if isinstance(candidate, (tuple, list)):
                self.prior_precision, self.sigma_noise = candidate
            else:
                self.prior_precision = candidate
            score = float(self._validation_nll(val_loader))
            self.fit_history_["tuning"].append(
                {
                    "prior_precision": float(self.prior_precision),
                    "sigma_noise": float(self.sigma_noise),
                    "val_nll": score,
                }
            )
            if isfinite(score) and score < best_score:
                best_score, best = score, candidate
        if best is None:
            self.prior_precision, self.sigma_noise = initial_prior, initial_noise
            self._refresh_posterior()
            if not has_candidates:
                raise ValueError("Hyperparameter grid is empty.")
            raise ValueError("No finite validation NLL was found in the grid.")
        if isinstance(best, (tuple, list)):
            self.prior_precision, self.sigma_noise = best
        else:
            self.prior_precision = best
        self._refresh_posterior()

    def log_marginal_likelihood(self, *args, **kwargs):
        """Reject marginal-likelihood evaluation for the ELLA approximation.

        Parameters
        ----------
        *args : Any
            Positional arguments accepted for compatibility with BaseLaplace.
        **kwargs : Any
            Keyword arguments accepted for compatibility with BaseLaplace.

        Raises
        ------
        NotImplementedError
            Always raised. ELLA does not define the subset-of-data GP
            marginal likelihood; use validation-based hyperparameter tuning.
        """
        raise NotImplementedError(
            "ELLA does not use FunctionalLaplace's subset-of-data marginal likelihood."
        )

    @property
    def log_likelihood(self) -> torch.Tensor:
        """Reject access to an untracked training log likelihood.

        Raises
        ------
        NotImplementedError
            Always raised because ELLA accumulates curvature without tracking
            the training log likelihood.
        """
        raise NotImplementedError("ELLA does not accumulate a training log likelihood.")

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint containing the fitted ELLA posterior.

        Returns
        -------
        dict[str, Any]
            Basis directions, projected curvature, hyperparameters, fitting
            history, seed, model fingerprint, and backend/input settings.
            The posterior precision factor is rebuilt when loading.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.

        Notes
        -----
        Loading validates the pretrained model against the saved fingerprint;
        it does not restore model parameters or buffers. Construct the
        receiving estimator with the same pretrained model and configuration.
        """
        self._check_fitted()
        return {
            "cls_name": type(self).__name__,
            "n_params": self.n_params,
            "model_fingerprint": _model_fingerprint(self.model),
            "likelihood": self.likelihood,
            "subsample_size": self.subsample_size,
            "n_eigenvalues": self.n_eigenvalues,
            "dual_directions": self.dual_directions.detach().clone()
            if self.dual_directions is not None
            else None,
            "feature_gram": self.feature_gram.detach().clone()
            if self.feature_gram is not None
            else None,
            "prior_precision": self.prior_precision,
            "sigma_noise": self.sigma_noise,
            "temperature": self.temperature,
            "n_data": self.n_data,
            "n_outputs": self.n_outputs,
            "fitted": self._fitted,
            "fit_history": deepcopy(self.fit_history_),
            "seed": self.seed,
            **self._checkpoint_configuration(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a fitted ELLA checkpoint for the same pretrained model.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State previously returned by :meth:`state_dict`.

        Raises
        ------
        ValueError
            If the checkpoint is unfitted or incompatible with this model.
        """
        if not state_dict.get("fitted", False):
            raise ValueError("ELLA checkpoint must contain a fitted posterior.")
        if (
            state_dict["cls_name"] != type(self).__name__
            or state_dict["n_params"] != self.n_params
        ):
            raise ValueError(
                "Checkpoint requires the same ELLA type and pretrained model."
            )
        if state_dict["likelihood"] != self.likelihood:
            raise ValueError("Checkpoint likelihood does not match.")
        self._validate_checkpoint_configuration(state_dict)
        _check_model_fingerprint(self.model, state_dict["model_fingerprint"])
        self.seed = state_dict["seed"]
        self.subsample_size = state_dict["subsample_size"]
        self.n_eigenvalues = state_dict["n_eigenvalues"]
        self.dual_directions = state_dict["dual_directions"].to(self._device)
        self.feature_gram = state_dict["feature_gram"].to(self._device)
        self.prior_precision = state_dict["prior_precision"]
        self.sigma_noise = state_dict["sigma_noise"]
        self.temperature = state_dict["temperature"]
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        setattr(self.model, "output_size", self.n_outputs)
        self._fitted = state_dict["fitted"]
        self.fit_history_ = deepcopy(state_dict["fit_history"])
        self._build_precision()
        self.model.eval()


class VaLLA(BaseFunctionalLaplace):
    """Variational linearized Laplace with inducing outputs and a fixed MAP model.

    The network weights remain fixed while ``fit`` optimizes the variational
    factor, inducing locations, prior precision, and regression noise.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained network to linearize.
    likelihood : Likelihood or str
        ``"classification"``, ``"regression"``, or ``"reward_modeling"``.
    inducing_locations : torch.Tensor, MutableMapping, or str
        Initial inputs, or ``"random"`` or ``"kmeans"`` to select them from
        the training loader. Floating-point tensor locations are optimized;
        mapping-style and integer tensor locations remain fixed.
    num_inducing : int or None, default=None
        Number of locations required with a selection strategy. If explicit
        locations are supplied, this must match their count when provided.
    inducing_classes : torch.Tensor or None, default=None
        Output coordinate selected at each inducing location, shaped
        ``[num_inducing]``. Explicit inputs default to coordinate zero.
        Strategies use training class labels for classification and coordinate
        zero for regression and individual rewards.
    sigma_noise : float or torch.Tensor, default=1.0
        Initial observation standard deviation for regression.
    prior_precision : float or torch.Tensor, default=1.0
        Initial scalar parameter-prior precision.
    prior_mean : float or torch.Tensor, default=0.0
        Prior mean; only zero is supported.
    temperature : float, default=1.0
        Positive scale of the likelihood contribution.
    enable_backprop : bool, default=False
        Retain derivatives through predictive moments.
    dict_key_x : str, default="input_ids"
        Input key for mapping-style batches.
    dict_key_y : str, default="labels"
        Target key for mapping-style batches.
    backend : type[CurvatureInterface] or None, default=None
        Curvature backend used for parameter Jacobians.
    backend_kwargs : dict[str, Any] or None, default=None
        Additional curvature backend options.
    seed : int, default=0
        Seed for inducing-point selection and Monte Carlo fitting.
    alpha : float, default=1.0
        Alpha-divergence value in ``[0, 1]``; zero uses the ELBO limit.
    mc_softmax_samples : int, default=0
        Monte Carlo draws for classification data terms when positive.
        Required for classification or reward modeling when ``alpha != 1``.

    Attributes
    ----------
    fit_history_ : dict[str, list[float]]
        ``objective`` records minibatch-scaled objective values computed
        before each update; ``val_nll`` records validation negative log likelihood.
    L : torch.nn.Parameter
        Packed lower-triangular variational factor containing
        ``num_inducing * (num_inducing + 1) // 2`` entries.
    log_prior_precision : torch.nn.Parameter
        Trainable logarithm of scalar parameter-prior precision.
    log_noise_variance : torch.nn.Parameter or None
        Trainable logarithm of observation variance for regression;
        ``None`` for classification and reward modeling.
    generator : torch.Generator
        Random generator for Monte Carlo data terms on the model device.

    Notes
    -----
    ``fit`` returns ``None`` and records objective and validation NLL in
    ``fit_history_``. Repeated fits reset variational and inducing state by
    default; ``fit(..., override=False)`` continues optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Likelihood | str,
        inducing_locations: torch.Tensor | MutableMapping | str,
        num_inducing: int | None = None,
        inducing_classes: torch.Tensor | None = None,
        sigma_noise: float | torch.Tensor = 1.0,
        prior_precision: float | torch.Tensor = 1.0,
        prior_mean: float | torch.Tensor = 0.0,
        temperature: float = 1.0,
        enable_backprop: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        backend: type[CurvatureInterface] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        mc_softmax_samples: int = 0,
    ) -> None:
        """Initialize VaLLA from a pretrained model and inducing setup."""
        if torch.as_tensor(prior_precision).numel() != 1:
            raise ValueError("VaLLA requires scalar prior_precision.")
        if torch.any(torch.as_tensor(prior_mean) != 0):
            raise ValueError("VaLLA currently requires prior_mean=0.")
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not isinstance(mc_softmax_samples, int) or mc_softmax_samples < 0:
            raise ValueError("mc_softmax_samples must be a nonnegative integer.")
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
        if (
            self.likelihood != Likelihood.REGRESSION
            and alpha != 1
            and mc_softmax_samples == 0
        ):
            raise ValueError(
                "Classification alpha != 1 requires mc_softmax_samples > 0; "
                "the deterministic probit approximation is alpha-independent."
            )
        self.seed = seed
        self.alpha = alpha
        self.mc_softmax_samples = mc_softmax_samples
        self.generator = torch.Generator(device=self._device).manual_seed(seed)
        self.log_prior_precision = torch.nn.Parameter(
            self._prior_precision.detach().log().clone()
        )
        self.log_noise_variance = (
            torch.nn.Parameter(2 * self._sigma_noise.detach().log().clone())
            if likelihood == Likelihood.REGRESSION
            else None
        )
        self._inducing_strategy = (
            inducing_locations if isinstance(inducing_locations, str) else None
        )
        if self._inducing_strategy is not None and self._inducing_strategy not in {
            "random",
            "kmeans",
        }:
            raise ValueError(
                "inducing_locations must be inputs, 'random', or 'kmeans'."
            )
        if isinstance(inducing_locations, str):
            if (
                not isinstance(num_inducing, Integral)
                or isinstance(num_inducing, bool)
                or num_inducing <= 0
            ):
                raise ValueError(
                    "num_inducing must be a positive integer for an inducing strategy."
                )
            self.num_inducing = num_inducing
            self.inducing_locations = None
        else:
            self.inducing_locations = self._make_inducing(inducing_locations)
            count = (
                self.inducing_locations[self.dict_key_x].shape[0]
                if isinstance(self.inducing_locations, MutableMapping)
                else self.inducing_locations.shape[0]
            )
            if num_inducing is not None and (
                not isinstance(num_inducing, Integral)
                or isinstance(num_inducing, bool)
                or num_inducing != count
            ):
                raise ValueError("num_inducing must match inducing_locations.")
            self.num_inducing = count
        if self.num_inducing <= 0:
            raise ValueError("At least one inducing location is required.")
        self.inducing_classes = (
            torch.zeros(self.num_inducing, device=self._device, dtype=torch.long)
            if inducing_classes is None
            else torch.as_tensor(
                inducing_classes, device=self._device, dtype=torch.long
            )
        )
        if self.inducing_classes.shape != (self.num_inducing,):
            raise ValueError("inducing_classes must have one entry per inducing input.")
        self._initial_inducing_locations = (
            None
            if self._inducing_strategy is not None
            else self._clone_fixed_inputs(self.inducing_locations)
        )
        self._initial_inducing_classes = self.inducing_classes.detach().clone()
        self._init_variational_factor()
        self.n_data = 0
        self._fitted = False
        self.fit_history_: dict[str, list[float]] = {"objective": [], "val_nll": []}

    @property
    def prior_precision(self) -> torch.Tensor:
        """Return the positive variational prior precision."""
        if hasattr(self, "log_prior_precision"):
            return self.log_prior_precision.exp()
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, value: float | torch.Tensor) -> None:
        """Set positive scalar precision and synchronize its log parameter."""
        candidate = torch.as_tensor(value, device=self._device, dtype=self._dtype)
        if candidate.numel() != 1 or not torch.all(
            torch.isfinite(candidate) & (candidate > 0)
        ):
            raise ValueError("VaLLA requires positive finite scalar prior_precision.")
        BaseLaplace.prior_precision.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, "log_prior_precision"):
            with torch.no_grad():
                self.log_prior_precision.copy_(self._prior_precision.log())

    @property
    def prior_mean(self) -> torch.Tensor:
        """Return the zero parameter-prior mean."""
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, value: float | torch.Tensor) -> None:
        """Set the parameter-prior mean, which must be zero."""
        if torch.any(torch.as_tensor(value) != 0):
            raise ValueError("VaLLA currently requires prior_mean=0.")
        BaseLaplace.prior_mean.fset(self, value)  # type: ignore[attr-defined]

    @property
    def temperature(self) -> float:
        """Return the positive likelihood temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the positive likelihood temperature."""
        if not isfinite(value) or value <= 0:
            raise ValueError("temperature must be positive and finite.")
        self._temperature = value

    @property
    def sigma_noise(self) -> torch.Tensor:
        """Return the regression observation standard deviation."""
        log_noise_variance = getattr(self, "log_noise_variance", None)
        if isinstance(log_noise_variance, torch.Tensor):
            return (0.5 * log_noise_variance).exp()
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, value: float | torch.Tensor) -> None:
        """Set positive noise and synchronize its log-variance parameter."""
        candidate = torch.as_tensor(value, device=self._device, dtype=self._dtype)
        if not torch.all(torch.isfinite(candidate) & (candidate > 0)):
            raise ValueError("sigma_noise must be positive and finite.")
        if self.likelihood != Likelihood.REGRESSION and torch.any(
            torch.as_tensor(value) != 1
        ):
            raise ValueError("sigma_noise != 1 is only available for regression.")
        BaseLaplace.sigma_noise.fset(self, value)  # type: ignore[attr-defined]
        log_noise_variance = getattr(self, "log_noise_variance", None)
        if isinstance(log_noise_variance, torch.Tensor):
            with torch.no_grad():
                log_noise_variance.copy_(2 * self._sigma_noise.log())

    @staticmethod
    def _clone_fixed_inputs(value: Any) -> Any:
        """Detach and recursively copy fixed inducing inputs.

        Parameters
        ----------
        value : Any
            Tensor, mapping, sequence, or metadata to copy.

        Returns
        -------
        Any
            Independent copy with detached tensors and the same nesting.
        """
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, MutableMapping):
            return {key: VaLLA._clone_fixed_inputs(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(VaLLA._clone_fixed_inputs(item) for item in value)
        return deepcopy(value)

    def _make_inducing(self, inputs: Any) -> torch.Tensor | dict:
        """Prepare inducing inputs and expose eligible locations for optimization.

        Parameters
        ----------
        inputs : Any
            Batched inducing tensor, mapping-style model input, or an
            array-like input convertible to a floating-point tensor.

        Returns
        -------
        torch.Tensor or dict
            Copied inputs on the model device. Floating-point tensor inputs
            become trainable parameters in the model dtype. Mappings and
            integer tensors remain detached and fixed.
        """
        converted = to_device(inputs, self._device, self._dtype)
        if isinstance(converted, MutableMapping):
            return self._clone_fixed_inputs(converted)
        if not isinstance(converted, torch.Tensor):
            converted = torch.as_tensor(
                converted, device=self._device, dtype=self._dtype
            )
        if converted.is_floating_point():
            return torch.nn.Parameter(converted.detach().clone())
        return converted.detach().clone()

    def _init_variational_factor(self) -> None:
        """Initialize the packed lower-triangular variational factor."""
        rows, cols = torch.tril_indices(self.num_inducing, self.num_inducing)
        initial = torch.eye(self.num_inducing, device=self._device, dtype=self._dtype)
        self.L = torch.nn.Parameter(initial[rows, cols].clone())

    def _factor_matrix(self) -> torch.Tensor:
        """Unpack the trainable lower-triangular variational factor.

        Returns
        -------
        torch.Tensor
            Lower-triangular matrix shaped ``[num_inducing, num_inducing]``
            with gradients retained through its packed entries in ``L``.
        """
        rows, cols = torch.tril_indices(
            self.num_inducing, self.num_inducing, device=self._device
        )
        matrix = torch.zeros(
            self.num_inducing, self.num_inducing, device=self._device, dtype=self._dtype
        )
        matrix[rows, cols] = self.L
        return matrix

    def _batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """Move a training batch and its targets to the model device.

        Parameters
        ----------
        batch : Any
            ``(inputs, targets)`` or a mapping containing ``dict_key_y``.

        Returns
        -------
        tuple[Any, torch.Tensor]
            Model inputs and targets on the model device. Floating-point
            inputs use the model dtype; target dtypes are preserved.
        """
        x, y = split_batch(batch, self.dict_key_y)
        return to_device(x, self._device, self._dtype), y.to(self._device)

    def _gather_inducing(self, loader: DataLoader) -> None:
        """Choose inducing inputs and output coordinates from training data.

        Parameters
        ----------
        loader : DataLoader
            Training loader used for random or k-means selection.

        Raises
        ------
        ValueError
            If the loader is empty, too small, or incompatible with k-means.
        """
        candidates = []
        classes = []
        source = loader
        if self._inducing_strategy == "random":
            available_indices = training_indices(loader)
            n_pairs = (self.num_inducing + 1) // 2
            needed = (
                n_pairs
                if self.likelihood == Likelihood.REWARD_MODELING
                else self.num_inducing
            )
            count = min(len(available_indices), needed)
            choices = torch.randperm(
                len(available_indices),
                generator=torch.Generator().manual_seed(self.seed),
            )[:count]
            indices = available_indices[choices]
            source = subset_loader(loader, indices)

        for batch in source:
            x, y = self._batch(batch)
            if self.likelihood == Likelihood.REWARD_MODELING:
                x = _individual_reward_inputs(x, self.dict_key_x)
                labels = torch.zeros(
                    x[self.dict_key_x].shape[0]
                    if isinstance(x, MutableMapping)
                    else x.shape[0],
                    device=self._device,
                    dtype=torch.long,
                )
            elif self.likelihood == Likelihood.CLASSIFICATION:
                labels = classification_targets(y)
            else:
                labels = torch.zeros(y.shape[0], device=self._device, dtype=torch.long)
            candidates.append(x)
            classes.append(labels)

        if not candidates:
            raise ValueError("Cannot initialize inducing inputs from an empty loader.")

        inputs = concatenate_rows(candidates)
        labels = torch.cat(classes)
        available = labels.shape[0]
        if available < self.num_inducing:
            raise ValueError("num_inducing exceeds available training inputs.")
        if self._inducing_strategy == "kmeans":
            if isinstance(inputs, MutableMapping) or not inputs.is_floating_point():
                raise ValueError("kmeans requires floating-point tensor inputs.")
            flat = inputs.detach().cpu().reshape(available, -1).numpy()
            centers, _ = kmeans2(
                flat, self.num_inducing, minit="points", seed=self.seed
            )
            distances = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            indices = torch.as_tensor(distances.argmin(0), device=self._device)
        else:
            indices = torch.randperm(
                available, generator=torch.Generator().manual_seed(self.seed)
            )[: self.num_inducing].to(self._device)

        self.inducing_locations = self._make_inducing(
            select_rows(inputs, indices, available)
        )
        self.inducing_classes = labels[indices].long()

    def _query_jacobians(
        self, x: torch.Tensor | MutableMapping, *, differentiable: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model Jacobians and outputs for query inputs.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched inputs.
        differentiable : bool, default=False
            Preserve gradients through Jacobian computation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Parameter Jacobians shaped ``[batch, outputs, parameters]`` and
            network outputs shaped ``[batch, outputs]``.
        """
        if self._uses_asdl_backend or (
            self.likelihood == Likelihood.REWARD_MODELING
            and not self._uses_backpack_backend
        ):
            return CurvatureInterface.jacobians(
                self.backend, x, enable_backprop=differentiable
            )
        output_size = self.model(x).shape[-1]
        previous = getattr(self.model, "output_size", None)
        setattr(self.model, "output_size", output_size)
        try:
            return self.backend.jacobians(x, enable_backprop=differentiable)
        finally:
            if previous is None:
                delattr(self.model, "output_size")
            else:
                setattr(self.model, "output_size", previous)

    def _inducing_jacobians(self) -> torch.Tensor:
        """Return selected-output Jacobians at all inducing locations.

        Returns
        -------
        torch.Tensor
            Parameter Jacobians shaped ``[num_inducing, parameters]``, with
            one output coordinate per location. Gradients through trainable
            inducing locations are retained.

        Raises
        ------
        RuntimeError
            If inducing locations have not been initialized.
        """
        if self.inducing_locations is None:
            raise RuntimeError("Inducing locations have not been initialized.")
        selected, _ = self.backend.selected_output_jacobians(
            self.inducing_locations,
            self.inducing_classes,
            enable_backprop=isinstance(self.inducing_locations, torch.nn.Parameter),
        )
        return selected[:, 0, :]

    def _inducing_factor(
        self, jacobians: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Factor the inducing system without forming its Gram matrix.

        Parameters
        ----------
        jacobians : torch.Tensor
            Selected-output inducing Jacobians shaped
            ``[num_inducing, parameters]``.

        Returns
        -------
        features : torch.Tensor
            Variationally transformed inducing Jacobians divided by the
            square root of prior precision, shaped
            ``[num_inducing, parameters]``.
        basis : torch.Tensor
            Reduced QR basis of the vertically stacked feature transpose
            and identity, shaped ``[parameters + num_inducing, num_inducing]``.
        triangular : torch.Tensor
            Upper-triangular QR factor shaped ``[num_inducing, num_inducing]``.

        Notes
        -----
        The factorization uses float64 arithmetic for numerical stability.
        """
        work_dtype = torch.float64
        prior_scale = self.prior_precision.to(work_dtype).sqrt()
        inducing = jacobians.to(work_dtype)
        factor = self._factor_matrix().to(work_dtype)
        features = factor.T @ inducing / prior_scale
        identity = torch.eye(
            self.num_inducing, device=features.device, dtype=work_dtype
        )
        basis, triangular = torch.linalg.qr(
            torch.cat((features.T, identity), dim=0), mode="reduced"
        )
        return features, basis, triangular

    def _posterior_covariance(
        self,
        jacobians: torch.Tensor,
        inducing_jacobians: torch.Tensor,
        *,
        joint: bool,
        basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Form latent covariance using the orthogonal inducing complement.

        Parameters
        ----------
        jacobians : torch.Tensor
            Query Jacobians shaped ``[batch, outputs, parameters]``.
        inducing_jacobians : torch.Tensor
            Selected-output inducing Jacobians shaped
            ``[num_inducing, parameters]``.
        joint : bool
            Include covariance between query inputs when true.
        basis : torch.Tensor or None, default=None
            Precomputed orthonormal inducing basis shaped
            ``[parameters + num_inducing, num_inducing]``. Computed from
            ``inducing_jacobians`` when omitted.

        Returns
        -------
        torch.Tensor
            Latent covariance shaped ``[batch, outputs, outputs]``, or
            ``[batch * outputs, batch * outputs]`` when ``joint=True``, in
            the query Jacobian dtype. Observation noise is excluded.
        """
        if basis is None:
            _, basis, _ = self._inducing_factor(inducing_jacobians)
        prior_scale = self.prior_precision.to(torch.float64).sqrt()
        query = jacobians.to(torch.float64).reshape(-1, self.n_params) / prior_scale
        padded = torch.nn.functional.pad(query, (0, self.num_inducing))
        roots = padded - (padded @ basis) @ basis.T
        if joint:
            covariance = roots @ roots.T
        else:
            batch, outputs = jacobians.shape[:2]
            roots = roots.reshape(batch, outputs, -1)
            covariance = torch.einsum("bcp,bdp->bcd", roots, roots)
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        return covariance.to(jacobians.dtype)

    def _latent_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latent moments and factors for the variational objective.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        joint : bool, default=False
            Retain covariance between input rows.

        Returns
        -------
        mean : torch.Tensor
            Network outputs shaped ``[batch, outputs]``, including when
            ``joint=True``.
        covariance : torch.Tensor
            Latent covariance shaped ``[batch, outputs, outputs]``, or
            ``[batch * outputs, batch * outputs]`` when ``joint=True``.
            Observation noise is excluded.
        features : torch.Tensor
            Float64 scaled variational features shaped
            ``[num_inducing, parameters]``.
        triangular : torch.Tensor
            Float64 upper-triangular inducing-system factor shaped
            ``[num_inducing, num_inducing]``.
        """
        x = to_device(x, self._device, self._dtype)
        jacobians, mean = self._query_jacobians(x, differentiable=self.enable_backprop)
        inducing_jacobians = self._inducing_jacobians()
        features, basis, triangular = self._inducing_factor(inducing_jacobians)
        covariance = self._posterior_covariance(
            jacobians, inducing_jacobians, joint=joint, basis=basis
        )
        return mean, covariance, features, triangular

    @preserve_model_gradients
    @torch.enable_grad()
    def _glm_predictive_distribution(
        self, x: torch.Tensor | MutableMapping, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute model outputs and VaLLA latent covariance for inputs.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Batched model inputs.
        joint : bool, default=False
            Retain covariance between input rows.

        Returns
        -------
        mean : torch.Tensor
            Network outputs shaped ``[batch, outputs]``, flattened to
            ``[batch * outputs]`` when ``joint=True``.
        covariance : torch.Tensor
            Latent covariance shaped ``[batch, outputs, outputs]``, or
            ``[batch * outputs, batch * outputs]`` when ``joint=True``.
            Observation noise is excluded.
        """
        mean, covariance, _, _ = self._latent_distribution(x, joint)
        if joint:
            mean = mean.flatten()
        if not self.enable_backprop:
            return mean.detach(), covariance.detach()
        return mean, covariance

    @preserve_model_gradients
    def functional_variance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return per-input latent covariance from parameter Jacobians.

        Parameters
        ----------
        jacobians : torch.Tensor
            Parameter Jacobians shaped ``[batch, outputs, parameters]``.

        Returns
        -------
        torch.Tensor
            Per-input covariance shaped ``[batch, outputs, outputs]``.
        """
        self._check_fitted()
        self._check_jacobians(jacobians)
        inducing = self._inducing_jacobians()
        return self._posterior_covariance(jacobians, inducing, joint=False)

    @preserve_model_gradients
    def functional_covariance(self, jacobians: torch.Tensor) -> torch.Tensor:
        """Return joint latent covariance from parameter Jacobians.

        Parameters
        ----------
        jacobians : torch.Tensor
            Parameter Jacobians shaped ``[batch, outputs, parameters]``.

        Returns
        -------
        torch.Tensor
            Joint covariance shaped ``[batch * outputs, batch * outputs]``.
        """
        self._check_fitted()
        self._check_jacobians(jacobians)
        inducing = self._inducing_jacobians()
        return self._posterior_covariance(jacobians, inducing, joint=True)

    def _objective(self, x: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the minibatch-scaled alpha objective plus inducing KL.

        Parameters
        ----------
        x : Any
            Batched model inputs.
        y : torch.Tensor
            Regression targets or class labels.

        Returns
        -------
        torch.Tensor
            Scalar loss minimized during fitting.
        """
        mean, covariance, features, triangular = self._latent_distribution(x)
        if self.likelihood == Likelihood.REGRESSION:
            y = regression_targets(y, mean)
            noise_variance = self.sigma_noise.square()
            latent_variance = covariance.diagonal(dim1=-2, dim2=-1)
            squared_error = (y - mean).square()
            if self.alpha == 0:
                energy = (squared_error + latent_variance) / noise_variance
            else:
                energy = (
                    squared_error / (noise_variance + self.alpha * latent_variance)
                    + torch.log1p(self.alpha * latent_variance / noise_variance)
                    / self.alpha
                )
            log_term = -0.5 * (energy + torch.log(2 * torch.pi * noise_variance)).sum()
        else:
            y = classification_targets(y)
            if self.mc_softmax_samples:
                draws = draw_gaussian(
                    mean, covariance, self.mc_softmax_samples, self.generator
                )
                log_probabilities = draws.log_softmax(dim=-1)
                selected = log_probabilities.gather(
                    -1,
                    y.long().view(1, -1, 1).expand(self.mc_softmax_samples, -1, 1),
                ).squeeze(-1)
                if self.alpha == 0:
                    log_term = selected.mean(dim=0).sum()
                else:
                    log_term = (
                        torch.logsumexp(self.alpha * selected, dim=0)
                        - torch.log(
                            torch.as_tensor(
                                self.mc_softmax_samples,
                                device=self._device,
                                dtype=self._dtype,
                            )
                        )
                    ).sum() / self.alpha
            else:
                scaled = mean / torch.sqrt(
                    1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1)
                )
                log_term = -torch.nn.functional.cross_entropy(
                    scaled, y.long(), reduction="sum"
                )

        solved = torch.linalg.solve_triangular(triangular.T, features, upper=False)
        kl = triangular.diagonal().abs().log().sum() - 0.5 * solved.square().sum()
        return -(self.n_data / y.shape[0]) * log_term / self.temperature + kl

    def _validation_nll(self, loader: DataLoader) -> torch.Tensor:
        """Return mean predictive negative log likelihood on a loader.

        Parameters
        ----------
        loader : DataLoader
            Nonempty validation loader with regression targets or class labels.

        Returns
        -------
        torch.Tensor
            Scalar NLL averaged over examples, summing over regression output
            coordinates. Regression includes observation noise; classification
            uses the probit approximation.
        """
        total = torch.zeros((), device=self._device, dtype=self._dtype)
        count = 0
        for batch in loader:
            x, y = self._batch(batch)
            if self.likelihood == Likelihood.REGRESSION:
                mean, covariance = self.predictive_moments(x)
                y = regression_targets(y, mean)
                variance = (
                    covariance.diagonal(dim1=-2, dim2=-1) + self.sigma_noise.square()
                )
                total += (
                    0.5
                    * (
                        (y - mean).square() / variance
                        + variance.log()
                        + torch.log(torch.as_tensor(2 * torch.pi, device=self._device))
                    )
                ).sum()
            else:
                total += self._classification_validation_nll(x, y)
            count += y.shape[0]
        return total / count

    @preserve_model_gradients
    def fit(
        self,
        train_loader: DataLoader,
        *,
        iterations: int = 100,
        lr: float = 1e-3,
        val_loader: DataLoader | None = None,
        val_steps: int | None = None,
        progress_bar: bool = False,
        override: bool = True,
    ) -> None:
        """Optimize the VaLLA variational posterior and hyperparameters.

        Parameters
        ----------
        train_loader : DataLoader
            Training batches used for inducing selection and optimization.
        iterations : int, default=100
            Number of minibatch optimizer steps.
        lr : float, default=1e-3
            Adam learning rate.
        val_loader : DataLoader or None, default=None
            Optional data for predictive negative log likelihood.
        val_steps : int or None, default=None
            Evaluate validation NLL every ``val_steps`` iterations when given.
        progress_bar : bool, default=False
            Show an optimization progress bar.
        override : bool, default=True
            Reset inducing and variational state; false continues a prior fit.

        Raises
        ------
        ValueError
            If optimization settings or inducing classes are invalid.
        """
        if (
            not isinstance(iterations, Integral)
            or isinstance(iterations, bool)
            or iterations <= 0
            or not isfinite(lr)
            or lr <= 0
        ):
            raise ValueError(
                "iterations must be a positive integer and lr positive and finite."
            )
        if val_steps is not None and (
            not isinstance(val_steps, Integral)
            or isinstance(val_steps, bool)
            or val_steps <= 0
        ):
            raise ValueError("val_steps must be positive integer when provided.")
        if not isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not isinstance(self.mc_softmax_samples, int) or self.mc_softmax_samples < 0:
            raise ValueError("mc_softmax_samples must be a nonnegative integer.")
        if (
            self.likelihood != Likelihood.REGRESSION
            and self.alpha != 1
            and self.mc_softmax_samples == 0
        ):
            raise ValueError(
                "Classification alpha != 1 requires mc_softmax_samples > 0."
            )

        self.model.eval()
        self.n_data = len(training_indices(train_loader))
        x_first, _ = self._batch(next(iter(train_loader)))
        if isinstance(x_first, MutableMapping) and self._uses_backpack_backend:
            raise ValueError(
                "BackPACK does not support mapping-style inputs; use AsdlGGN or CurvlinopsGGN."
            )

        out = self.model(x_first)
        self._check_output_dims(out)
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)
        if override or not self._fitted:
            if self._inducing_strategy is not None:
                self._gather_inducing(train_loader)
            else:
                self.inducing_locations = self._make_inducing(
                    self._initial_inducing_locations
                )
                self.inducing_classes = self._initial_inducing_classes.detach().clone()
            self._init_variational_factor()
            self.fit_history_ = {"objective": [], "val_nll": []}

        if (
            self.inducing_classes.shape != (self.num_inducing,)
            or torch.any(self.inducing_classes < 0)
            or torch.any(self.inducing_classes >= self.n_outputs)
        ):
            raise ValueError(
                "inducing_classes must contain one valid output per inducing input."
            )

        self._fitted = False
        parameters = [self.L, self.log_prior_precision]
        if self.log_noise_variance is not None:
            parameters.append(self.log_noise_variance)
        if isinstance(self.inducing_locations, torch.nn.Parameter):
            parameters.append(self.inducing_locations)

        optimizer = torch.optim.Adam(parameters, lr=lr)
        iterator = range(iterations)
        if progress_bar:
            from tqdm import trange

            iterator = trange(iterations, desc="Fitting VaLLA")

        batches = iter(train_loader)
        for step in iterator:
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                batch = next(batches)
            x, y = self._batch(batch)
            optimizer.zero_grad()
            objective = self._objective(x, y)
            objective.backward()
            optimizer.step()
            self.fit_history_["objective"].append(float(objective.detach()))
            if val_loader is not None and val_steps and (step + 1) % val_steps == 0:
                self._fitted = True
                with torch.no_grad():
                    self.fit_history_["val_nll"].append(
                        float(self._validation_nll(val_loader))
                    )

        self._fitted = True
        if val_loader is not None and (not val_steps or iterations % val_steps):
            with torch.no_grad():
                self.fit_history_["val_nll"].append(
                    float(self._validation_nll(val_loader))
                )

    def optimize_prior_precision(self, *args, **kwargs) -> None:
        """Reject separate prior tuning because :meth:`fit` learns the precision.

        Parameters
        ----------
        *args : Any
            Positional arguments accepted for compatibility with BaseLaplace.
        **kwargs : Any
            Keyword arguments accepted for compatibility with BaseLaplace.

        Raises
        ------
        NotImplementedError
            Always raised. VaLLA optimizes prior precision jointly with the
            variational parameters during fitting.
        """
        raise NotImplementedError("VaLLA optimizes prior precision during fit.")

    def log_marginal_likelihood(self, *args, **kwargs):
        """Reject marginal-likelihood evaluation for the VaLLA approximation.

        Parameters
        ----------
        *args : Any
            Positional arguments accepted for compatibility with BaseLaplace.
        **kwargs : Any
            Keyword arguments accepted for compatibility with BaseLaplace.

        Raises
        ------
        NotImplementedError
            Always raised. The optimized alpha-divergence objective is not a
            Laplace marginal likelihood.
        """
        raise NotImplementedError(
            "VaLLA's variational objective is not a Laplace marginal likelihood."
        )

    @property
    def log_likelihood(self) -> torch.Tensor:
        """Reject access to an untracked training log likelihood.

        Raises
        ------
        NotImplementedError
            Always raised because VaLLA records its alpha-divergence
            objective rather than a training log likelihood.
        """
        raise NotImplementedError("VaLLA optimizes an alpha-divergence objective.")

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint containing fitted variational and inducing state.

        Returns
        -------
        dict[str, Any]
            Variational factor, inducing locations and output coordinates,
            hyperparameters, fitting history, initialization settings, random
            generator state, model fingerprint, and backend/input settings.

        Raises
        ------
        RuntimeError
            If the posterior has not been fitted.

        Notes
        -----
        Loading validates the pretrained model against the saved fingerprint;
        it does not restore model parameters or buffers. Construct the
        receiving estimator with the same pretrained model and configuration.
        """
        self._check_fitted()
        return {
            "cls_name": type(self).__name__,
            "n_params": self.n_params,
            "model_fingerprint": _model_fingerprint(self.model),
            "likelihood": self.likelihood,
            "inducing_locations": self.inducing_locations.detach().clone()
            if isinstance(self.inducing_locations, torch.Tensor)
            else deepcopy(self.inducing_locations),
            "inducing_classes": self.inducing_classes.detach().clone(),
            "inducing_strategy": self._inducing_strategy,
            "initial_inducing_locations": self._clone_fixed_inputs(
                self._initial_inducing_locations
            ),
            "initial_inducing_classes": self._initial_inducing_classes.detach().clone(),
            "L": self.L.detach().clone(),
            "log_prior_precision": self.log_prior_precision.detach().clone(),
            "log_noise_variance": None
            if self.log_noise_variance is None
            else self.log_noise_variance.detach().clone(),
            "temperature": self.temperature,
            "alpha": self.alpha,
            "mc_softmax_samples": self.mc_softmax_samples,
            "n_data": self.n_data,
            "n_outputs": self.n_outputs,
            "fitted": self._fitted,
            "fit_history": deepcopy(self.fit_history_),
            "seed": self.seed,
            "generator_state": self.generator.get_state().clone(),
            **self._checkpoint_configuration(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a fitted VaLLA checkpoint for the same pretrained model.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State previously returned by :meth:`state_dict`.

        Raises
        ------
        ValueError
            If the checkpoint is unfitted or incompatible with this model.
        """
        if not state_dict.get("fitted", False):
            raise ValueError("VaLLA checkpoint must contain a fitted posterior.")
        if (
            state_dict["cls_name"] != type(self).__name__
            or state_dict["n_params"] != self.n_params
        ):
            raise ValueError(
                "Checkpoint requires the same VaLLA type and pretrained model."
            )
        if state_dict["likelihood"] != self.likelihood:
            raise ValueError("Checkpoint likelihood does not match.")
        self._validate_checkpoint_configuration(state_dict)
        _check_model_fingerprint(self.model, state_dict["model_fingerprint"])
        self.seed = state_dict["seed"]
        self.generator.set_state(state_dict["generator_state"])
        self.inducing_locations = self._make_inducing(state_dict["inducing_locations"])
        self.inducing_classes = state_dict["inducing_classes"].to(self._device)
        self.num_inducing = len(self.inducing_classes)
        self._inducing_strategy = state_dict["inducing_strategy"]
        self._initial_inducing_locations = self._clone_fixed_inputs(
            to_device(
                state_dict["initial_inducing_locations"], self._device, self._dtype
            )
        )
        self._initial_inducing_classes = state_dict["initial_inducing_classes"].to(
            self._device
        )
        self._init_variational_factor()
        with torch.no_grad():
            self.L.copy_(state_dict["L"].to(self._device))
            self.log_prior_precision.copy_(
                state_dict["log_prior_precision"].to(self._device)
            )
            if self.log_noise_variance is not None:
                self.log_noise_variance.copy_(
                    state_dict["log_noise_variance"].to(self._device)
                )
        self.n_data = state_dict["n_data"]
        self.n_outputs = state_dict["n_outputs"]
        self.temperature = state_dict["temperature"]
        self.alpha = state_dict["alpha"]
        self.mc_softmax_samples = state_dict["mc_softmax_samples"]
        setattr(self.model, "output_size", self.n_outputs)
        self._fitted = state_dict["fitted"]
        self.fit_history_ = deepcopy(state_dict["fit_history"])
        self.model.eval()
