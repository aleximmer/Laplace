from math import sqrt, prod
import pytest
from itertools import product
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
import os

from laplace.laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace, DiagSubnetLaplace, FullSubnetLaplace
from laplace.utils import KronDecomposed
from tests.utils import jacobians_naive


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace]
subnet_flavors = [DiagSubnetLaplace, FullSubnetLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Run after test
    if os.path.exists('state_dict.bin'):
        os.remove('state_dict.bin')


@pytest.mark.parametrize('laplace', flavors)
def test_serialize(laplace, model, reg_loader):
    la = laplace(model, 'regression')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model, 'regression')
    la2.load_state_dict(torch.load('state_dict.bin'))

    assert la.sigma_noise == la2.sigma_noise

    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X)
    f_mean2, f_var2 = la2(X)
    assert torch.allclose(f_mean, f_mean2)
    assert torch.allclose(f_var, f_var2)


@pytest.mark.parametrize('laplace', flavors)
def test_serialize_no_pickle(laplace, model, reg_loader):
    la = laplace(model, 'regression')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')
    state_dict = torch.load('state_dict.bin')

    # Make sure no pickle object
    for val in state_dict.values():
        assert isinstance(val, (list, tuple, int, float, torch.Tensor))


@pytest.mark.parametrize('laplace', subnet_flavors)
def test_serialize_subnetlaplace(laplace, model, reg_loader):
    subnetwork_indices = torch.LongTensor([1, 10, 104, 44])
    la = laplace(model, 'regression', subnetwork_indices=subnetwork_indices)
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model, 'regression', subnetwork_indices=subnetwork_indices)
    la2.load_state_dict(torch.load('state_dict.bin'))

    assert la.sigma_noise == la2.sigma_noise

    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X)
    f_mean2, f_var2 = la2(X)
    assert torch.allclose(f_mean, f_mean2)
    assert torch.allclose(f_var, f_var2)

