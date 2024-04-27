from math import sqrt
import pytest
from itertools import product
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Categorical
from torchvision.models import wide_resnet50_2

from laplace.laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace
from laplace.utils import KronDecomposed
from laplace.curvature import AsdlGGN, AsdlHessian, AsdlEF, BackPackEF, BackPackGGN
from tests.utils import jacobians_naive


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [KronLaplace, DiagLaplace, FullLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))

    # Subset of params
    for p in model.parameters():
        p.requires_grad = False
    model[0].weight.requires_grad = True

    return model


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.mark.parametrize(
    'laplace,lh', product(flavors, ['classification', 'regression'])
)
def test_incompatible_backend(laplace, lh, model):
    lap = laplace(model, lh, backend=AsdlEF)
    lap = laplace(model, lh, backend=AsdlGGN)
    lap = laplace(model, lh, backend=AsdlHessian)


@pytest.mark.parametrize(
    'laplace,lh', product(flavors, ['classification', 'regression'])
)
def test_incompatible_backend(laplace, lh, model):
    with pytest.raises(ValueError):
        lap = laplace(model, lh, backend=BackPackGGN)

    with pytest.raises(ValueError):
        lap = laplace(model, lh, backend=BackPackEF)


@pytest.mark.parametrize('laplace', flavors)
def test_mean_clf(laplace, model, class_loader):
    n_params = model[0].weight.numel()
    lap = laplace(model, 'classification')
    lap.fit(class_loader)
    assert lap.mean.shape == (n_params,)


@pytest.mark.parametrize('laplace', flavors)
def test_mean_reg(laplace, model, reg_loader):
    n_params = model[0].weight.numel()
    lap = laplace(model, 'regression')
    lap.fit(reg_loader)
    assert lap.mean.shape == (n_params,)


def test_post_precision_diag(model, class_loader):
    n_params = model[0].weight.numel()
    lap = DiagLaplace(model, 'classification')
    lap.fit(class_loader)
    assert lap.posterior_precision.shape == (n_params,)


def test_post_precision_kron(model, class_loader):
    n_params = model[0].weight.numel()
    lap = KronLaplace(model, 'classification')
    lap.fit(class_loader)
    assert lap.posterior_precision.to_matrix().shape == (n_params, n_params)


@pytest.mark.parametrize('laplace', flavors)
def test_predictive(laplace, model, class_loader):
    lap = laplace(model, 'classification')
    lap.fit(class_loader)
    lap(torch.randn(5, 3), pred_type='nn', link_approx='mc')


@pytest.mark.parametrize('laplace', flavors)
def test_marglik(laplace, model, class_loader):
    lap = laplace(model, 'classification')
    lap.fit(class_loader)
    lap.optimize_prior_precision(method='marglik')


@pytest.mark.parametrize('laplace', flavors)
def test_marglik(laplace, model, class_loader):
    lap = laplace(model, 'classification')
    lap.fit(class_loader)
    lap.optimize_prior_precision(
        method='gridsearch', val_loader=class_loader, pred_type='nn', link_approx='mc'
    )
