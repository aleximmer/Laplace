import pytest
from itertools import product
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Categorical

from laplace.laplace import Laplace, FullLaplace, KronLaplace, DiagLaplace
from tests.utils import Jacobians_naive

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
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


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
    if laplace == KronLaplace:
        # Kron should not accept per parameter prior precision
        with pytest.raises(ValueError):
            lap = laplace(model, likelihood='regression', prior_precision=precision)
    else:
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
def test_laplace_init_temperature(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood='classification', temperature=T)
    assert lap.temperature == T


@pytest.mark.parametrize('laplace,lh', product(flavors, ['classification', 'regression']))
def test_laplace_functionality(laplace, lh, model, reg_loader, class_loader):
    if lh == 'classification':
        loader = class_loader
        sigma_noise = 1.
    else:
        loader = reg_loader
        sigma_noise = 0.3
    lap = laplace(model, lh, sigma_noise=sigma_noise, prior_precision=0.7)
    lap.fit(loader)
    assert lap.n_data == len(loader.dataset)
    assert lap.n_outputs == model.output_size
    f = model(loader.dataset.tensors[0])
    y = loader.dataset.tensors[1]
    assert f.shape == torch.Size([10, 2])

    # Test log likelihood (Train)
    log_lik = lap.log_lik
    # compute true log lik
    if lh == 'classification':
        log_lik_true = Categorical(logits=f).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
    else:
        assert y.size() == f.size()
        log_lik_true = Normal(loc=f, scale=sigma_noise).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
        # change likelihood and test again
        lap.sigma_noise = 0.72
        log_lik = lap.log_lik
        log_lik_true = Normal(loc=f, scale=0.72).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)

    # Test marginal likelihood
    # lml = log p(y|f) - 1/2 theta @ prior_prec @ theta 
    #       + 1/2 logdet prior_prec - 1/2 log det post_prec
    lml = log_lik_true
    theta = parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(theta, lap.mean)
    prior_prec = torch.diag(lap.prior_precision_diag)
    assert prior_prec.shape == torch.Size([len(theta), len(theta)])
    lml = lml - 1/2 * theta @ prior_prec @ theta
    Sigma_0 = torch.inverse(prior_prec)
    if laplace == DiagLaplace:
        log_det_post_prec = lap.posterior_precision.log().sum()
    else:
        log_det_post_prec = lap.posterior_precision.logdet()
    lml = lml + 1/2 * (prior_prec.logdet() - log_det_post_prec)
    assert torch.allclose(lml, lap.marginal_likelihood())

    # test sampling
    torch.manual_seed(61)
    samples = lap.sample(n_samples=1)
    assert samples.shape == torch.Size([1, len(theta)])
    samples = lap.sample(n_samples=1000000)
    assert samples.shape == torch.Size([1000000, len(theta)])
    mu_comp = samples.mean(dim=0)
    mu_true = lap.mean
    assert torch.allclose(mu_comp, mu_true, rtol=1)

    # test functional variance
    if laplace == FullLaplace:
        Sigma = lap.posterior_covariance
    elif laplace == KronLaplace:
        Sigma = lap.posterior_precision.to_matrix(exponent=-1)
    elif laplace == DiagLaplace:
        Sigma = torch.diag(lap.posterior_variance)
    Js, f = Jacobians_naive(model, loader.dataset.tensors[0])
    true_f_var = torch.einsum('mkp,pq,mcq->mkc', Js, Sigma, Js)
    comp_f_var = lap.functional_variance(Js)
    assert torch.allclose(true_f_var, comp_f_var, rtol=1e-4)
