import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace.laplace import Laplace, FullLaplace, KronLaplace, DiagLaplace

flavors = [FullLaplace, KronLaplace, DiagLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y))


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y))


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init(laplace, model):
    lap = laplace(model, 'classification')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_invalid_likelihood(laplace, model):
    with pytest.raises(ValueError):
        lap = laplace(model, 'otherlh')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_noise(laplace, model):
    # float
    sigma_noise = 1.2
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # torch.tensor 0-dim
    sigma_noise = torch.tensor(1.2)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # torch.tensor 1-dim
    sigma_noise = torch.tensor(1.2).reshape(-1)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)

    # for classification should fail
    sigma_noise = 1.2
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='classification', sigma_noise=sigma_noise)

    # other than that should fail
    # higher dim
    sigma_noise = torch.tensor(1.2).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # other datatype, only reals supported
    sigma_noise = '1.2'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_precision(laplace, model):
    # float
    precision = 10.6
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 0-dim
    precision = torch.tensor(10.6)
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim
    precision = torch.tensor(10.7).reshape(-1)
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim param-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim layer-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_layers)
    lap = laplace(model, likelihood='regression', prior_precision=precision)

    # other than that should fail
    # higher dim
    precision = torch.tensor(10.6).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)
    # unmatched dim 
    precision = torch.tensor(10.6).reshape(-1).repeat(17)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)
    # other datatype, only reals supported
    precision = '1.5'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_precision(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood='classification', temperature=T)
    assert lap.temperature == T


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_fit(laplace, model, reg_loader):
    lap = laplace(model, 'regression')
    lap.fit(reg_loader)