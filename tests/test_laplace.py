import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.laplace import Laplace
from laplace.baselaplace import FullLaplace, KronLaplace, DiagLaplace
from laplace.lllaplace import FullLLLaplace, KronLLLaplace, DiagLLLaplace


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [FullLaplace, KronLaplace, DiagLaplace,
           FullLLLaplace, KronLLLaplace, DiagLLLaplace]
ids = [('all', 'full'), ('all', 'kron'), ('all', 'diag'),
       ('last-layer', 'full'), ('last-layer', 'kron'), ('last-layer', 'diag')]


@pytest.fixture
def model():
    model = nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    model_params = list(model.parameters())
    return model


def test_default_init(model, likelihood='classification'):
    # test if default initialization works, id=(last-layer, kron)
    lap = Laplace(model, likelihood)
    assert isinstance(lap, KronLLLaplace)


@pytest.mark.parametrize('laplace, id', zip(flavors, ids))
def test_all_init(laplace, id, model, likelihood='classification'):
    # test if all flavors are correctly initialized
    w, c = id
    lap = Laplace(model, likelihood, weights=w, cov_structure=c)
    assert isinstance(lap, laplace)


@pytest.mark.parametrize('id', ids)
def test_opt_keywords(id, model, likelihood='classification'):
    # test if optional keywords are correctly passed on
    w, c = id
    prior_mean = torch.zeros_like(parameters_to_vector(model.parameters()))
    lap = Laplace(model, likelihood, weights=w, cov_structure=c,
                  prior_precision=0.01, prior_mean=prior_mean, temperature=10.)
    assert torch.allclose(lap.prior_mean, prior_mean)
    assert lap.prior_precision == 0.01
    assert lap.temperature == 10.
