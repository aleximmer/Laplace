from __future__ import annotations

import logging
import warnings
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Type

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Number
from torch.utils.data import DataLoader

from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from laplace.curvature import AsdlGGN
from laplace.curvature.curvature import CurvatureInterface
from laplace.utils import (
    HessianStructure,
    Likelihood,
    PriorStructure,
    SubsetOfWeights,
    expand_prior_precision,
    fix_prior_prec_structure,
)


def marglik_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    likelihood: Likelihood | str = Likelihood.CLASSIFICATION,
    hessian_structure: HessianStructure | str = HessianStructure.KRON,
    backend: Type[CurvatureInterface] = AsdlGGN,
    optimizer_cls: Type[Optimizer] = Adam,
    optimizer_kwargs: dict | None = None,
    scheduler_cls: Type[LRScheduler] | None = None,
    scheduler_kwargs: dict | None = None,
    n_epochs: int = 300,
    lr_hyp: float = 1e-1,
    prior_structure: PriorStructure | str = PriorStructure.LAYERWISE,
    n_epochs_burnin: int = 0,
    n_hypersteps: int = 10,
    marglik_frequency: int = 1,
    prior_prec_init: float = 1.0,
    sigma_noise_init: float = 1.0,
    temperature: float = 1.0,
    fix_sigma_noise: bool = False,
    progress_bar: bool = False,
    enable_backprop: bool = False,
    dict_key_x: str = "input_ids",
    dict_key_y: str = "labels",
) -> tuple[BaseLaplace, nn.Module, list[Number], list[Number]]:
    """Marginal-likelihood based training (Algorithm 1 in [1]).
    Optimize model parameters and hyperparameters jointly.
    Model parameters are optimized to minimize negative log joint (train loss)
    while hyperparameters minimize negative log marginal likelihood.

    This method replaces standard neural network training and adds hyperparameter
    optimization to the procedure.

    The settings of standard training can be controlled by passing `train_loader`,
    `optimizer_cls`, `optimizer_kwargs`, `scheduler_cls`, `scheduler_kwargs`, and `n_epochs`.
    The `model` should return logits, i.e., no softmax should be applied.
    With `likelihood=Likelihood.CLASSIFICATION` or `Likelihood.REGRESSION`, one can choose between
    categorical likelihood (CrossEntropyLoss) and Gaussian likelihood (MSELoss).

    As in [1], we optimize prior precision and, for regression, observation noise
    using the marginal likelihood. The prior precision structure can be chosen
    as `'scalar'`, `'layerwise'`, or `'diagonal'`. `'layerwise'` is a good default
    and available to all Laplace approximations. `lr_hyp` is the step size of the
    Adam hyperparameter optimizer, `n_hypersteps` controls the number of steps
    for each estimated marginal likelihood, `n_epochs_burnin` controls how many
    epochs to skip marginal likelihood estimation, `marglik_frequency` controls
    how often to estimate the marginal likelihood (default of 1 re-estimates
    after every epoch, 5 would estimate every 5-th epoch).

    References
    ----------
    [1] Immer, A., Bauer, M., Fortuin, V., Rätsch, G., Khan, EM.
    [*Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning*](https://arxiv.org/abs/2104.04975).
    ICML 2021.

    Parameters
    ----------
    model : torch.nn.Module
        torch neural network model (needs to comply with Backend choice)
    train_loader : DataLoader
        pytorch dataloader that implements `len(train_loader.dataset)` to obtain number of data points
    likelihood : str, default=Likelihood.CLASSIFICATION
        Likelihood.CLASSIFICATION or Likelihood.REGRESSION
    hessian_structure : {'diag', 'kron', 'full'}, default='kron'
        structure of the Hessian approximation
    backend : Backend, default=AsdlGGN
        Curvature subclass, e.g. AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    optimizer_cls : torch.optim.Optimizer, default=Adam
        optimizer to use for optimizing the neural network parameters togeth with `train_loader`
    optimizer_kwargs : dict, default=None
        keyword arguments for `optimizer_cls`, for example to change learning rate or momentum
    scheduler_cls : torch.optim.lr_scheduler._LRScheduler, default=None
        optionally, a scheduler to use on the learning rate of the optimizer.
        `scheduler.step()` is called after every batch of the standard training.
    scheduler_kwargs : dict, default=None
        keyword arguments for `scheduler_cls`, e.g. `lr_min` for CosineAnnealingLR
    n_epochs : int, default=300
        number of epochs to train for
    lr_hyp : float, default=0.1
        Adam learning rate for hyperparameters
    prior_structure : str, default='layerwise'
        structure of the prior. one of `['scalar', 'layerwise', 'diag']`
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int, default=10
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
        `marglik_frequency=1` would be every epoch,
        `marglik_frequency=5` would be every 5 epochs.
    prior_prec_init : float, default=1.0
        initial prior precision
    sigma_noise_init : float, default=1.0
        initial observation noise (for regression only)
    temperature : float, default=1.0
        factor for the likelihood for 'overcounting' data. Might be required for data augmentation.
    fix_sigma_noise: bool, default=False
        if False, optimize observation noise via marglik otherwise use `sigma_noise_init` throughout.
        Only works for regression.
    progress_bar: bool, default=False
        whether to show a progress bar (updated per epoch) or not
    enable_backprop : bool, default=False
        make the returned Laplace instance backpropable---useful for e.g. Bayesian optimization.
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.

    Returns
    -------
    lap : laplace
        fit Laplace approximation with the best obtained marginal likelihood during training
    model : torch.nn.Module
        corresponding model with the MAP parameters
    margliks : list
        list of marginal likelihoods obtained during training (to monitor convergence)
    losses : list
        list of losses (log joints) obtained during training (to monitor convergence)
    """
    if optimizer_kwargs is not None and "weight_decay" in optimizer_kwargs:
        warnings.warn("Weight decay is handled and optimized. Will be set to 0.")
        optimizer_kwargs["weight_decay"] = 0.0

    # get device, data set size N, number of layers H, number of parameters P
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    N = len(train_loader.dataset)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    H = len(trainable_params)
    P = len(parameters_to_vector(trainable_params))

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    log_prior_prec = fix_prior_prec_structure(
        log_prior_prec_init, prior_structure, H, P, device, dtype
    )
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == Likelihood.CLASSIFICATION:
        criterion = CrossEntropyLoss(reduction="mean")
        sigma_noise = 1.0
    elif likelihood == Likelihood.REGRESSION:
        criterion = MSELoss(reduction="mean")
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(
            1, device=device, dtype=dtype
        )
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # set up learning rate scheduler
    scheduler = None
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    losses = list()
    margliks = list()

    pbar = tqdm.trange(
        1,
        n_epochs + 1,
        disable=not progress_bar,
        position=1,
        leave=False,
        desc="[Training]",
        colour="blue",
    )

    for epoch in pbar:
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        for data in train_loader:
            if isinstance(data, MutableMapping):
                X, y = data, data[dict_key_y]
                y = y.to(device, non_blocking=True)
            else:
                X, y = data
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if likelihood == Likelihood.REGRESSION:
                sigma_noise = (
                    torch.exp(log_sigma_noise).detach()
                    if not fix_sigma_noise
                    else sigma_noise_init
                )
                crit_factor = temperature / (2 * sigma_noise**2)
            else:
                crit_factor = temperature

            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            )
            delta = expand_prior_precision(prior_prec, model)

            f = model(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() * len(y)

            if likelihood == Likelihood.REGRESSION:
                epoch_perf += (f.detach() - y).square().sum()
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item()

            if scheduler is not None:
                scheduler.step()

        losses.append(epoch_loss / N)

        # compute validation error to report during training
        logging.info(
            f"MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]:.3f}."
            + f"Perf={epoch_perf/N:.3f}"
        )

        # only update hyperparameters every marglik_frequency steps after burnin
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        if likelihood == Likelihood.CLASSIFICATION:
            sigma_noise = 1
        else:
            sigma_noise = (
                torch.exp(log_sigma_noise) if not fix_sigma_noise else sigma_noise_init
            )
        prior_prec = torch.exp(log_prior_prec)
        lap = Laplace(
            model,
            likelihood,
            hessian_structure=hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_prec,
            temperature=temperature,
            backend=backend,
            subset_of_weights="all",
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
        )
        lap.fit(train_loader)

        # 2. differentiate wrt. hyperparameters for n_hypersteps
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if likelihood == Likelihood.CLASSIFICATION or fix_sigma_noise:
                sigma_noise = None
            else:
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()
            margliks.append(marglik.item())

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            if likelihood == Likelihood.CLASSIFICATION:
                best_sigma = 1
            else:
                best_sigma = (
                    deepcopy(sigma_noise.detach())
                    if not fix_sigma_noise
                    else sigma_noise_init
                )
            best_marglik = margliks[-1]
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}. "
                + "Saving new best model."
            )
        else:
            logging.info(
                f"MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}."
                + f"No improvement over {best_marglik:.2f}"
            )

    logging.info("MARGLIK: finished training. Recover best model and fit Laplace.")

    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    logging.info(f"best params: {sigma_noise}, {prior_prec}")

    lap = Laplace(
        model,
        likelihood,
        hessian_structure=hessian_structure,
        sigma_noise=sigma_noise,
        prior_precision=prior_prec,
        temperature=temperature,
        backend=backend,
        subset_of_weights=SubsetOfWeights.ALL,
        enable_backprop=enable_backprop,
        dict_key_x=dict_key_x,
        dict_key_y=dict_key_y,
    )
    lap.fit(train_loader)
    return lap, model, margliks, losses
