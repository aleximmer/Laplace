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
compatible_flavors = [KronLaplace, DiagLaplace]
noncompatible_flavors = [FullLaplace, LowRankLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def large_model():
    model = wide_resnet50_2()
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


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_compatible(laplace, model):
    lap = laplace(model, 'classification', subset_params=[model[0].weight])
    assert True


@pytest.mark.parametrize('laplace', noncompatible_flavors)
def test_noncompatible(laplace, model):
    with pytest.raises(TypeError):
        lap = laplace(model, 'classification', subset_params=[model[0].weight])


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_incompatible_backend(laplace, model):
    lap = laplace(model, 'classification', subset_params=[model[0].weight],
                  backend=AsdlEF)

    lap = laplace(model, 'classification', subset_params=[model[0].weight],
                  backend=AsdlGGN)

    lap = laplace(model, 'classification', subset_params=[model[0].weight],
                  backend=AsdlHessian)


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_incompatible_backend(laplace, model):
    with pytest.raises(ValueError):
        lap = laplace(model, 'classification', subset_params=[model[0].weight],
                      backend=BackPackGGN)

    with pytest.raises(ValueError):
        lap = laplace(model, 'classification', subset_params=[model[0].weight],
                      backend=BackPackEF)


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_wrong_value(laplace, model):
    with pytest.raises(ValueError):
        lap = laplace(model, 'classification', subset_params=model[0].weight)


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_mean(laplace, model, class_loader):
    n_params = model[0].weight.numel()
    lap = laplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)
    assert lap.mean.shape == (n_params,)


def test_post_precision_diag(model, class_loader):
    n_params = model[0].weight.numel()
    lap = DiagLaplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)
    assert lap.posterior_precision.shape == (n_params,)


def test_post_precision_kron(model, class_loader):
    n_params = model[0].weight.numel()
    lap = KronLaplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)
    assert lap.posterior_precision.to_matrix().shape == (n_params, n_params)


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_predictive(laplace, model, class_loader):
    lap = laplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)

    with pytest.raises(ValueError):
        lap(torch.randn(5, 3), pred_type='glm')

    lap(torch.randn(5, 3), pred_type='nn', link_approx='mc')


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_marglik(laplace, model, class_loader):
    lap = laplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)
    lap.optimize_prior_precision(method='marglik')


@pytest.mark.parametrize('laplace', compatible_flavors)
def test_marglik(laplace, model, class_loader):
    lap = laplace(model, 'classification', subset_params=[model[0].weight])
    lap.fit(class_loader)

    with pytest.raises(ValueError):
        lap.optimize_prior_precision(method='CV', val_loader=class_loader)

    lap.optimize_prior_precision(method='CV', val_loader=class_loader,
                                 pred_type='nn', link_approx='mc')
