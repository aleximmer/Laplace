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
all_keys = [('all', 'full'), ('all', 'kron'), ('all', 'diag'),
            ('last_layer', 'full'), ('last_layer', 'kron'), ('last_layer', 'diag')]


@pytest.fixture
def model():
    model = nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    model_params = list(model.parameters())
    return model


def test_default_init(model, likelihood='classification'):
    # test if default initialization works, id=(last-layer, kron)
    lap = Laplace(model, likelihood)
    assert isinstance(lap, KronLLLaplace)


@pytest.mark.parametrize('laplace, key', zip(flavors, all_keys))
def test_all_init(laplace, key, model, likelihood='classification'):
    # test if all flavors are correctly initialized
    w, s = key
    lap = Laplace(model, likelihood, subset_of_weights=w, hessian_structure=s)
    assert isinstance(lap, laplace)


@pytest.mark.parametrize('key', all_keys)
def test_opt_keywords(key, model, likelihood='classification'):
    # test if optional keywords are correctly passed on
    w, s = key
    prior_mean = torch.zeros_like(parameters_to_vector(model.parameters()))
    lap = Laplace(model, likelihood, subset_of_weights=w, hessian_structure=s,
                  prior_precision=0.01, prior_mean=prior_mean, temperature=10.)
    assert torch.allclose(lap.prior_mean, prior_mean)
    assert lap.prior_precision == 0.01
    assert lap.temperature == 10.
