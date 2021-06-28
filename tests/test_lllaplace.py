import pytest
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Categorical

from laplace.lllaplace import LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace
from laplace.feature_extractor import FeatureExtractor
from tests.utils import jacobians_naive


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [FullLLLaplace, KronLLLaplace, DiagLLLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
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
    lap = laplace(model, 'classification', last_layer_name='1')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_invalid_likelihood(laplace, model):
    with pytest.raises(ValueError):
        lap = laplace(model, 'otherlh', last_layer_name='1')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_noise(laplace, model):
    # float
    sigma_noise = 1.2
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise,
                  last_layer_name='1')
    # torch.tensor 0-dim
    sigma_noise = torch.tensor(1.2)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise,
                  last_layer_name='1')
    # torch.tensor 1-dim
    sigma_noise = torch.tensor(1.2).reshape(-1)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise,
                  last_layer_name='1')

    # for classification should fail
    sigma_noise = 1.2
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='classification',
                      sigma_noise=sigma_noise, last_layer_name='1')

    # other than that should fail
    # higher dim
    sigma_noise = torch.tensor(1.2).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise,
                      last_layer_name='1')
    # other datatype, only reals supported
    sigma_noise = '1.2'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise,
                      last_layer_name='1')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_precision(laplace, model):
    feature_extractor = FeatureExtractor(model, last_layer_name='1')
    model_params = list(feature_extractor.last_layer.parameters())
    setattr(model, 'n_layers', 2)  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    # float
    precision = 10.6
    lap = laplace(model, likelihood='regression', prior_precision=precision,
                  last_layer_name='1')
    # torch.tensor 0-dim
    precision = torch.tensor(10.6)
    lap = laplace(model, likelihood='regression', prior_precision=precision,
                  last_layer_name='1')
    # torch.tensor 1-dim
    precision = torch.tensor(10.7).reshape(-1)
    lap = laplace(model, likelihood='regression', prior_precision=precision,
                  last_layer_name='1')
    # torch.tensor 1-dim param-shape
    if laplace == KronLLLaplace:  # kron only supports per layer
        with pytest.raises(ValueError):
            precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
            lap = laplace(model, likelihood='regression', prior_precision=precision,
                        last_layer_name='1')
    else:
        precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
        lap = laplace(model, likelihood='regression', prior_precision=precision,
                    last_layer_name='1')
    # torch.tensor 1-dim layer-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_layers)
    lap = laplace(model, likelihood='regression', prior_precision=precision,
                  last_layer_name='1')

    # other than that should fail
    # higher dim
    precision = torch.tensor(10.6).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision,
                      last_layer_name='1')
    # unmatched dim
    precision = torch.tensor(10.6).reshape(-1).repeat(17)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision,
                      last_layer_name='1')
    # other datatype, only reals supported
    precision = '1.5'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision,
                      last_layer_name='1')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_prior_mean_and_scatter(laplace, model):
    lap_scalar_mean = laplace(model, 'classification', last_layer_name='1',
                              prior_precision=1e-2, prior_mean=1.)
    assert torch.allclose(lap_scalar_mean.prior_mean, torch.tensor([1.]))
    lap_tensor_mean = laplace(model, 'classification', last_layer_name='1',
                              prior_precision=1e-2, prior_mean=torch.ones(1))
    assert torch.allclose(lap_tensor_mean.prior_mean, torch.tensor([1.]))
    lap_tensor_scalar_mean = laplace(model, 'classification', last_layer_name='1',
                                     prior_precision=1e-2, prior_mean=torch.ones(1)[0])
    assert torch.allclose(lap_tensor_scalar_mean.prior_mean, torch.tensor(1.))
    lap_tensor_full_mean = laplace(model, 'classification', last_layer_name='1',
                                   prior_precision=1e-2, prior_mean=torch.ones(20*2+2))
    assert torch.allclose(lap_tensor_full_mean.prior_mean, torch.ones(20*2+2))
    expected = lap_scalar_mean.scatter
    assert expected.ndim == 0
    assert torch.allclose(lap_tensor_mean.scatter, expected)
    assert lap_tensor_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_scalar_mean.scatter, expected)
    assert lap_tensor_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_full_mean.scatter, expected)
    assert lap_tensor_full_mean.scatter.shape == expected.shape

    # too many dims
    with pytest.raises(ValueError):
        prior_mean = torch.ones(20*2+2).unsqueeze(-1)
        laplace(model, 'classification', last_layer_name='1',
                prior_precision=1e-2, prior_mean=prior_mean)

    # unmatched dim
    with pytest.raises(ValueError):
        prior_mean = torch.ones(20*2-3)
        laplace(model, 'classification', last_layer_name='1',
                prior_precision=1e-2, prior_mean=prior_mean)

    # invalid argument type
    with pytest.raises(ValueError):
        laplace(model, 'classification', last_layer_name='1',
                prior_precision=1e-2, prior_mean='72')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_temperature(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood='classification', temperature=T,
                  last_layer_name='1')
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
    X = loader.dataset.tensors[0]
    f = model(X)
    y = loader.dataset.tensors[1]
    assert f.shape == torch.Size([10, 2])

    # Test log likelihood (Train)
    log_lik = lap.log_likelihood
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
        log_lik = lap.log_likelihood
        log_lik_true = Normal(loc=f, scale=0.72).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)

    # Test marginal likelihood
    # lml = log p(y|f) - 1/2 theta @ prior_prec @ theta
    #       + 1/2 logdet prior_prec - 1/2 log det post_prec
    lml = log_lik_true
    feature_extractor = FeatureExtractor(model, last_layer_name='1')
    theta = parameters_to_vector(feature_extractor.last_layer.parameters()).detach()
    assert torch.allclose(theta, lap.mean)
    prior_prec = torch.diag(lap.prior_precision_diag)
    assert prior_prec.shape == torch.Size([len(theta), len(theta)])
    lml = lml - 1/2 * theta @ prior_prec @ theta
    Sigma_0 = torch.inverse(prior_prec)
    if laplace == DiagLLLaplace:
        log_det_post_prec = lap.posterior_precision.log().sum()
    else:
        log_det_post_prec = lap.posterior_precision.logdet()
    lml = lml + 1/2 * (prior_prec.logdet() - log_det_post_prec)
    assert torch.allclose(lml, lap.log_marginal_likelihood())

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
    if laplace == FullLLLaplace:
        Sigma = lap.posterior_covariance
    elif laplace == KronLLLaplace:
        Sigma = lap.posterior_precision.to_matrix(exponent=-1)
    elif laplace == DiagLLLaplace:
        Sigma = torch.diag(lap.posterior_variance)
    _, phi = feature_extractor.forward_with_features(X)
    Js, f = jacobians_naive(feature_extractor.last_layer, phi)
    true_f_var = torch.einsum('mkp,pq,mcq->mkc', Js, Sigma, Js)
    # test last-layer Jacobians
    comp_Js, comp_f = lap.backend.last_layer_jacobians(lap.model, X)
    assert torch.allclose(Js, comp_Js)
    assert torch.allclose(f, comp_f)
    comp_f_var = lap.functional_variance(comp_Js)
    assert torch.allclose(true_f_var, comp_f_var, rtol=1e-4)


@pytest.mark.parametrize('laplace', flavors)
def test_regression_predictive(laplace, model, reg_loader):
    lap = laplace(model, 'regression', sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive, functional variance tested already above.
    f_mu, f_var = lap(X, pred_type='glm')
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(X, pred_type='nn')
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(X)


@pytest.mark.parametrize('laplace', flavors)
def test_classification_predictive(laplace, model, class_loader):
    lap = laplace(model, 'classification', prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    f_pred = lap(X, pred_type='glm', link_approx='mc', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='probit')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='bridge')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1

    # NN predictive
    f_pred = lap(X, pred_type='nn', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1


@pytest.mark.parametrize('laplace', flavors)
def test_regression_predictive_samples(laplace, model, reg_loader):
    lap = laplace(model, 'regression', sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive, functional variance tested already above.
    fsamples = lap.predictive_samples(X, pred_type='glm', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # NN predictive (only diagonal variance estimation)
    fsamples = lap.predictive_samples(X, pred_type='nn', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])


@pytest.mark.parametrize('laplace', flavors)
def test_classification_predictive_samples(laplace, model, class_loader):
    lap = laplace(model, 'classification', prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    fsamples = lap.predictive_samples(X, pred_type='glm', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1

    # NN predictive
    f_pred = lap.predictive_samples(X, pred_type='nn', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1
