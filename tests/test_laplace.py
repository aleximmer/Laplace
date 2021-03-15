import pytest
import torch
from torch import nn

from laplace import Laplace


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    return model


def test_laplace_init(model):
    laplace = Laplace(model, 'classification')


def test_laplace_invalid_likelihood(model):
    with pytest.raises(ValueError):
        laplace = Laplace(model, 'otherlh')


def test_laplace_invalid_covtype(model):
    with pytest.raises(ValueError):
        laplace = Laplace(model, 'regression', cov_type='tridiag')

