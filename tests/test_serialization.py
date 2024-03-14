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
from collections import OrderedDict

from laplace import Laplace
from laplace.laplace import FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace, DiagSubnetLaplace, FullSubnetLaplace
from laplace.utils import KronDecomposed
from tests.utils import jacobians_naive


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace]
flavors_no_llla = [FullLaplace, KronLaplace, DiagLaplace, LowRankLaplace]
flavors_llla = [FullLLLaplace, KronLLLaplace, DiagLLLaplace]
flavors_subnet = [DiagSubnetLaplace, FullSubnetLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model2():
    model = torch.nn.Sequential(nn.Linear(3, 25), nn.Linear(25, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model3():
    model = torch.nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(3, 20)),
            ('clf', nn.Linear(20, 2))
        ])
    )
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


@pytest.mark.parametrize('laplace', set(flavors_no_llla) - {LowRankLaplace})
def test_serialize_override(laplace, model, reg_loader):
    la = laplace(model, 'regression')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    H_orig = la.H_facs.to_matrix() if laplace == KronLaplace else la.H
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model, 'regression')
    la2.load_state_dict(torch.load('state_dict.bin'))

    # Emulating continual learning
    la2.fit(reg_loader, override=False)

    H_new = la2.H_facs.to_matrix() if laplace == KronLaplace else la2.H
    assert not torch.allclose(H_orig, H_new)


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
        if val is not None:
            assert isinstance(val, (list, tuple, int, float, str, bool, torch.Tensor))


@pytest.mark.parametrize('laplace', flavors_subnet)
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


@pytest.mark.parametrize('laplace', flavors_no_llla)
def test_serialize_fail_different_models(laplace, model, model2, reg_loader):
    la = laplace(model, 'regression')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model2, 'regression')

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load('state_dict.bin'))


def test_serialize_fail_different_hess_structures(model, reg_loader):
    la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='kron')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='diag')

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load('state_dict.bin'))


def test_serialize_fail_different_subset_of_weights(model, reg_loader):
    la = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='diag')

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load('state_dict.bin'))


@pytest.mark.parametrize('laplace', flavors)
def test_serialize_fail_different_liks(laplace, model, reg_loader):
    la = laplace(model, 'regression')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model, 'classification')

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load('state_dict.bin'))


@pytest.mark.parametrize('laplace', flavors_llla)
def test_serialize_fail_llla_different_last_layer_name(laplace, model, model3, reg_loader):
    print([n for n, _ in model.named_parameters()])
    la = laplace(model, 'regression', last_layer_name='1')
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), 'state_dict.bin')

    la2 = laplace(model3, 'classification', last_layer_name='clf')

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load('state_dict.bin'))
