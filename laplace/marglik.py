from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import logging

from laplace import (DiagLaplace, KronLaplace, FullLaplace,
                     DiagLLLaplace, KronLLLaplace, FullLLLaplace)
from laplace.curvature import KazukiGGN


def marglik_optimization(model,
                         train_loader,
                         likelihood='classification',
                         prior_structure='layerwise',
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='exp',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         laplace=KronLaplace,
                         backend=KazukiGGN,
                         backend_kwargs=None):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    temperature : float default=1
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    """
    # TODO: track deltas and sigmas
    if lr_min is None:
        lr_min = lr
    if backend_kwargs is None:
        backend_kwargs = dict()
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    last_layer = laplace in [DiagLLLaplace, KronLLLaplace, FullLLLaplace]
    if last_layer:  # specific to last layer
        assert prior_structure != 'layerwise', 'Not supported'
        lap = laplace(model, likelihood, sigma_noise=1., prior_precision=1.,
                      backend=backend, **backend_kwargs)
        X, _ = next(iter(train_loader))
        with torch.no_grad():
            lap.model.find_last_layer(X.to(device))
        last_layer_model = lap.model.last_layer
        P = len(parameters_to_vector(last_layer_model.parameters()))
    else:
        H = len(list(model.parameters()))
        P = len(parameters_to_vector(model.parameters()))

    hyperparameters = list()
    # set up prior precision
    if prior_structure == 'scalar':
        log_prior_prec = torch.zeros(1, requires_grad=True, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = torch.zeros(H, requires_grad=True, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = torch.zeros(P, requires_grad=True, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    hyperparameters.append(log_prior_prec)

    # set up loss
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='sum')
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='sum')
        log_sigma_noise = torch.zeros(1, requires_grad=True, device=device)
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer
    # stochastic optimizers has exponentially decaying learning rate to min_lr
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        lr = lr / (N * temperature)  # sgd does not normalize internally
        lr_min = lr_min / (N * temperature)
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')

    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)

    best_marglik = np.inf
    best_model = None
    best_precision = None
    losses = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            M = len(y)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = temperature / (2 * sigma_noise.square())
            else:
                crit_factor = temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            if last_layer:
                theta = parameters_to_vector(last_layer_model.parameters())
                delta = expand_prior_precision(prior_prec, last_layer_model)
            else:
                theta = parameters_to_vector(model.parameters())
                delta = expand_prior_precision(prior_prec, model)
            f = model(X)
            loss = N / M * crit_factor * criterion(f, y) + 0.5 * (delta * theta) @ theta
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += criterion(f.detach(), y).item() / len(train_loader)
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / M / len(train_loader)
            scheduler.step()
        losses.append(epoch_loss)

        logging.info(f'MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]}; '
                     + f'Perf={epoch_perf}; lr={scheduler.get_last_lr()}')

        # only update hyperparameters every "Frequency" steps
        if (epoch % marglik_frequency) != 0:
            continue

        sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                      temperature=temperature, backend=backend, **backend_kwargs)
        lap.fit(train_loader)
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if likelihood == 'classification':
                sigma_noise = None
            elif likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()
            margliks.append(marglik.item())

        if margliks[-1] < best_marglik:
            best_model = deepcopy(model)
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik}. '
                         + 'Saving new best model.')
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]}.'
                         + f'No improvement over {best_marglik}')

    logging.info('MARGLIK: finished training. Recover best model and fit Lapras.')
    model.load_state_dict(best_model.state_dict())
    lap = laplace(model, likelihood, sigma_noise=best_sigma, prior_precision=best_precision,
                  backend=backend, **backend_kwargs)
    lap.fit(train_loader)
    return lap, model, margliks, losses


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])
