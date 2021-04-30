from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import logging

from laplace import DiagLaplace, KronLaplace, FullLaplace
from laplace.curvature import KazukiGGN


def marglik_optimization(model, train_loader, likelihood='classification',
                         prior_structure='layerwise',
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
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
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    """
    # TODO: track deltas and sigmas
    if lr_min is None:
        lr_min = lr
    if backend_kwargs is None:
        backend_kwargs = dict()
    device = parameters_to_vector(model.parameters()).device
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    N = len(train_loader.dataset)

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
        log_sigma_noise = torch.zeros(1, requires_grad=False, device=device)
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='sum')
        log_sigma_noise = torch.zeros(1, requires_grad=True, device=device)
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer
    # stochastic optimizers has exponentially decaying learning rate to min_lr
    min_lr_factor = lr / lr_min
    gamma = np.exp(np.log(min_lr_factor) / n_epochs)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)

    best_marglik = np.inf
    best_model = None
    best_precision = None
    losses = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        for X, y in train_loader:
            M = len(y)
            optimizer.zero_grad()
            theta = parameters_to_vector(model.parameters())
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / (2 * sigma_noise.square())
            else:
                crit_factor = 1
            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)
            loss = N / M * crit_factor * criterion(model(X), y) + 0.5 * (delta * theta) @ theta
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
        losses.append(epoch_loss)
        scheduler.step()

        logging.info(f'MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]}; lr={scheduler.get_last_lr()}')

        # only update hyperparameters every "Frequency" steps
        if (epoch % marglik_frequency) != 0:
            continue

        sigma_noise = torch.exp(log_sigma_noise)  # == 1 (off) for classification 
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                      backend=backend, **backend_kwargs)
        lap.fit(train_loader)
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()
            margliks.append(marglik.item())

        logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]}')

        if margliks[-1] < best_marglik:
            best_model = deepcopy(model)
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = deepcopy(sigma_noise.detach())
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
    return lap, margliks, losses


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])
