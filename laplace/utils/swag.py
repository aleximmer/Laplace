from copy import deepcopy

import torch
from torch.nn.utils import parameters_to_vector


__all__ = ['fit_diagonal_swag_var']


def _param_vector(model):
    return parameters_to_vector(model.parameters()).detach()


def fit_diagonal_swag_var(model, train_loader, criterion, n_snapshots_total=40, snapshot_freq=1,
                          lr=0.01, momentum=0.9, weight_decay=3e-4, min_var=1e-30):
    """
    Fit diagonal SWAG [1], which estimates marginal variances of model parameters by
    computing the first and second moment of SGD iterates with a large learning rate.
    
    Implementation partly adapted from:
    - https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    - https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/run_swag.py

    References
    ----------
    [1] Maddox, W., Garipov, T., Izmailov, P., Vetrov, D., Wilson, AG. 
    [*A Simple Baseline for Bayesian Uncertainty in Deep Learning*](https://arxiv.org/abs/1902.02476). 
    NeurIPS 2019.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.data.utils.DataLoader
        training data loader to use for snapshot collection
    criterion : torch.nn.CrossEntropyLoss or torch.nn.MSELoss
        loss function to use for snapshot collection
    n_snapshots_total : int
        total number of model snapshots to collect
    snapshot_freq : int
        snapshot collection frequency (in epochs)
    lr : float
        SGD learning rate for collecting snapshots
    momentum : float
        SGD momentum
    weight_decay : float
        SGD weight decay
    min_var : float
        minimum parameter variance to clamp to (for numerical stability)

    Returns
    -------
    param_variances : torch.Tensor
        vector of marginal variances for each model parameter
    """

    # create a copy of the model to avoid undesired changes to the original model parameters
    _model = deepcopy(model)
    _model.train()
    device = next(_model.parameters()).device

    # initialize running estimates of first and second moment of model parameters
    mean = torch.zeros_like(_param_vector(_model))
    sq_mean = torch.zeros_like(_param_vector(_model))
    n_snapshots = 0

    # run SGD to collect model snapshots
    optimizer = torch.optim.SGD(
        _model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    n_epochs = snapshot_freq * n_snapshots_total
    for epoch in range(n_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(_model(inputs), targets)
            loss.backward()
            optimizer.step()

        if epoch % snapshot_freq == 0:
            # update running estimates of first and second moment of model parameters
            old_fac, new_fac = n_snapshots / (n_snapshots + 1), 1 / (n_snapshots + 1)
            mean = mean * old_fac + _param_vector(_model) * new_fac
            sq_mean = sq_mean * old_fac + _param_vector(_model) ** 2 * new_fac
            n_snapshots += 1

    # compute marginal parameter variances, Var[P] = E[P^2] - E[P]^2
    param_variances = torch.clamp(sq_mean - mean ** 2, min_var)
    return param_variances
