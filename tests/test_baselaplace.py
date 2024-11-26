from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from importlib.util import find_spec
from itertools import product
from math import prod, sqrt

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2

from laplace import DiagLaplace, FullLaplace, KronLaplace, LowRankLaplace
from laplace.curvature import AsdlEF, AsdlGGN, BackPackGGN
from laplace.curvature.backpack import BackPackEF
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN
from laplace.utils import KronDecomposed
from tests.utils import ListDataset, dict_data_collator, jacobians_naive

torch.manual_seed(240)
torch.set_default_dtype(torch.double)

flavors = [FullLaplace, KronLaplace, DiagLaplace]
if find_spec("asdfghjkl") is not None:
    flavors.append(LowRankLaplace)

online_flavors = [FullLaplace, KronLaplace, DiagLaplace]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model_1d():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 1))
    setattr(model, "output_size", 1)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def large_model():
    model = wide_resnet50_2()
    return model


@pytest.fixture
def custom_model():
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 100), nn.ReLU(), nn.Linear(100, 2))

        def forward(self, data: MutableMapping | torch.Tensor):
            if isinstance(data, MutableMapping):
                x = data["test_input_key"].to(next(self.parameters()).device)
            else:
                x = data

            logits = self.net(x)
            return logits

    return CustomModel()


@pytest.fixture
def reward_model():
    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 1))

        def forward(self, x):
            """
            x: torch.Tensor
                If training == True then shape (batch_size, 2, dim)
                Else shape (batch_size, dim)
            """
            if len(x.shape) == 3:
                batch_size, _, dim = x.shape

                # Flatten to (batch_size*2, dim)
                flat_x = x.reshape(-1, dim)

                # Forward
                flat_logits = self.net(flat_x)  # (batch_size*2, 1)

                # Reshape back to (batch_size, 2)
                return flat_logits.reshape(batch_size, 2)
            else:
                logits = self.net(x)  # (batch_size, 1)
                return logits

    return RewardModel()


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


@pytest.fixture
def reg_loader_1d():
    torch.manual_seed(9999)
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader_1d_flat():
    torch.manual_seed(9999)
    X = torch.randn(10, 3)
    y = torch.randn((10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def custom_loader_clf():
    data = []
    for _ in range(10):
        datum = {
            "test_input_key": torch.randn(5),
            "test_label_key": torch.randint(2, (1,)),
        }
        data.append(datum)
    return DataLoader(ListDataset(data), batch_size=3, collate_fn=dict_data_collator)


@pytest.fixture
def custom_loader_reg():
    data = []
    for _ in range(10):
        datum = {
            "test_input_key": torch.randn(5),
            "test_label_key": torch.randn(2),
        }
        data.append(datum)
    return DataLoader(ListDataset(data), batch_size=3, collate_fn=dict_data_collator)


@pytest.fixture
def reward_loader():
    X = torch.randn(10, 2, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reward_test_X():
    X = torch.randn(10, 3)
    return X


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init(laplace, model):
    lap = laplace(model, "classification")
    assert torch.allclose(lap.mean, lap.prior_mean)
    if laplace in [FullLaplace, DiagLaplace]:
        H = lap.H.clone()
        lap._init_H()
        assert torch.allclose(H, lap.H)
    elif laplace == LowRankLaplace:
        assert lap.H is None
    else:
        H = [[k.clone() for k in kfac] for kfac in lap.H.kfacs]
        lap._init_H()
        for kfac1, kfac2 in zip(H, lap.H.kfacs):
            for k1, k2 in zip(kfac1, kfac2):
                assert torch.allclose(k1, k2)


@pytest.mark.skip(reason="Does not work well with Github actions")
def test_laplace_large_init(large_model):
    FullLaplace(large_model, "classification")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_invalid_likelihood(laplace, model):
    with pytest.raises(ValueError):
        laplace(model, "otherlh")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_noise(laplace, model):
    # float
    sigma_noise = 1.2
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # torch.tensor 0-dim
    sigma_noise = torch.tensor(1.2)
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # torch.tensor 1-dim
    sigma_noise = torch.tensor(1.2).reshape(-1)
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)

    # for classification should fail
    sigma_noise = 1.2
    with pytest.raises(ValueError):
        laplace(model, likelihood="classification", sigma_noise=sigma_noise)

    # other than that should fail
    # higher dim
    sigma_noise = torch.tensor(1.2).reshape(1, 1)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # other datatype, only reals supported
    sigma_noise = "1.2"
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", sigma_noise=sigma_noise)


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_precision(laplace, model):
    # float
    precision = 10.6
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 0-dim
    precision = torch.tensor(10.6)
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim
    precision = torch.tensor(10.7).reshape(-1)
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim param-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
    if laplace == KronLaplace:
        # Kron should not accept per parameter prior precision
        with pytest.raises(ValueError):
            laplace(model, likelihood="regression", prior_precision=precision)
    else:
        laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim layer-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_layers)
    laplace(model, likelihood="regression", prior_precision=precision)

    # other than that should fail
    # higher dim
    precision = torch.tensor(10.6).reshape(1, 1)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)
    # unmatched dim
    precision = torch.tensor(10.6).reshape(-1).repeat(17)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)
    # other datatype, only reals supported
    precision = "1.5"
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_prior_mean_and_scatter(laplace, model, class_loader):
    mean = parameters_to_vector(model.parameters())
    P = len(mean)
    lap_scalar_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=1.0
    )
    assert torch.allclose(lap_scalar_mean.prior_mean, torch.tensor([1.0]))
    lap_tensor_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(1)
    )
    assert torch.allclose(lap_tensor_mean.prior_mean, torch.tensor([1.0]))
    lap_tensor_scalar_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(1)[0]
    )
    assert torch.allclose(lap_tensor_scalar_mean.prior_mean, torch.tensor(1.0))
    lap_tensor_full_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(P)
    )
    assert torch.allclose(lap_tensor_full_mean.prior_mean, torch.ones(P))

    lap_scalar_mean.fit(class_loader)
    lap_tensor_mean.fit(class_loader)
    lap_tensor_scalar_mean.fit(class_loader)
    lap_tensor_full_mean.fit(class_loader)
    expected = torch.tensor(0).reshape(-1)
    # assert expected.ndim == 0
    expected = ((mean - 1) * 1e-2) @ (mean - 1)
    assert torch.allclose(lap_scalar_mean.scatter, expected)
    assert lap_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_mean.scatter, expected)
    assert lap_tensor_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_scalar_mean.scatter, expected)
    assert lap_tensor_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_full_mean.scatter, expected)
    assert lap_tensor_full_mean.scatter.shape == expected.shape

    # too many dims
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P).unsqueeze(-1)
        laplace(model, "classification", prior_precision=1e-2, prior_mean=prior_mean)

    # unmatched dim
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P - 3)
        laplace(model, "classification", prior_precision=1e-2, prior_mean=prior_mean)

    # invalid argument type
    with pytest.raises(ValueError):
        laplace(model, "classification", prior_precision=1e-2, prior_mean="72")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_temperature(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood="classification", temperature=T)
    assert lap.temperature == T


@pytest.mark.parametrize(
    "laplace,lh", product(flavors, ["classification", "regression"])
)
def test_laplace_functionality(laplace, lh, model, reg_loader, class_loader):
    if lh == "classification":
        loader = class_loader
        sigma_noise = 1.0
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
    log_lik = lap.log_likelihood
    # compute true log lik
    if lh == "classification":
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
    theta = parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(theta, lap.mean)
    prior_prec = torch.diag(lap.prior_precision_diag)
    assert prior_prec.shape == torch.Size([len(theta), len(theta)])
    lml = lml - 1 / 2 * theta @ prior_prec @ theta
    if laplace == DiagLaplace:
        log_det_post_prec = lap.posterior_precision.log().sum()
    elif laplace == LowRankLaplace:
        (U, eigval), p0 = lap.posterior_precision
        log_det_post_prec = (U @ torch.diag(eigval) @ U.T + p0.diag()).logdet()
    else:
        log_det_post_prec = lap.posterior_precision.logdet()
    lml = lml + 1 / 2 * (prior_prec.logdet() - log_det_post_prec)
    assert torch.allclose(lml, lap.log_marginal_likelihood())

    # test sampling
    torch.manual_seed(61)
    samples = lap.sample(n_samples=1)
    assert samples.shape == torch.Size([1, len(theta)])
    samples = lap.sample(n_samples=1000000)
    assert samples.shape == torch.Size([1000000, len(theta)])
    mu_comp = samples.mean(dim=0)
    mu_true = lap.mean
    assert torch.allclose(mu_comp, mu_true, atol=1e-2)

    # test functional variance
    if laplace == FullLaplace:
        Sigma = lap.posterior_covariance
    elif laplace == KronLaplace:
        Sigma = lap.posterior_precision.to_matrix(exponent=-1)
    elif laplace == LowRankLaplace:
        (U, eigval), p0 = lap.posterior_precision
        Sigma = (U @ torch.diag(eigval) @ U.T + p0.diag()).inverse()
    elif laplace == DiagLaplace:
        Sigma = torch.diag(lap.posterior_variance)
    Js, f = jacobians_naive(model, loader.dataset.tensors[0])
    true_f_var = torch.einsum("mkp,pq,mcq->mkc", Js, Sigma, Js)
    comp_f_var = lap.functional_variance(Js)
    assert torch.allclose(true_f_var, comp_f_var, rtol=1e-4)


@pytest.mark.parametrize("laplace", online_flavors)
def test_overriding_fit(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    if type(lap.posterior_precision) is KronDecomposed:
        P = lap.posterior_precision.to_matrix()
    else:
        P = lap.posterior_precision.clone()
    m = lap.mean.clone()
    marglik = lap.log_marginal_likelihood().detach().clone()
    assert lap.n_data == len(reg_loader.dataset)
    lap.fit(reg_loader, override=True)
    assert torch.allclose(lap.mean, m)
    if type(lap.posterior_precision) is KronDecomposed:
        assert torch.allclose(lap.posterior_precision.to_matrix(), P)
    else:
        assert torch.allclose(lap.posterior_precision, P)
    assert torch.allclose(marglik, lap.log_marginal_likelihood())
    assert lap.n_data == len(reg_loader.dataset)


@pytest.mark.parametrize("laplace", online_flavors)
def test_online_fit(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    if type(lap.H) is KronDecomposed:
        P = lap.H.to_matrix().clone()
    else:
        P = lap.H.clone()
    loss, n_data = deepcopy(lap.loss.item()), deepcopy(lap.n_data)
    # fit a second and third time but don't override
    lap.fit(reg_loader, override=False)
    lap.fit(reg_loader, override=False)
    # Hessian should be now roughly 3x the one before
    assert torch.allclose(3 * torch.tensor(loss), lap.loss)
    assert (3 * n_data) == lap.n_data
    if type(lap.H) is KronDecomposed:
        assert torch.allclose(lap.H.to_matrix(), 3 * P)
    else:
        assert torch.allclose(lap.H, 3 * P)


def test_log_prob_full(model, class_loader):
    lap = FullLaplace(model, "classification", prior_precision=0.7)
    theta = torch.randn_like(parameters_to_vector(model.parameters()))
    # posterior without fitting is just prior
    posterior = Normal(loc=torch.zeros_like(theta), scale=sqrt(1 / 0.7))
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta).sum())
    lap.fit(class_loader)
    posterior = MultivariateNormal(
        loc=lap.mean, precision_matrix=lap.posterior_precision
    )
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta))


def test_log_prob_kron(model, class_loader):
    lap = KronLaplace(model, "classification", prior_precision=0.24)
    theta = torch.randn_like(parameters_to_vector(model.parameters()))
    posterior = Normal(loc=lap.mean, scale=sqrt(1 / 0.24))
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta).sum())
    lap.fit(class_loader)
    posterior = MultivariateNormal(
        loc=lap.mean, precision_matrix=lap.posterior_precision.to_matrix()
    )
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta))


@pytest.mark.parametrize("laplace", flavors)
def test_regression_predictive(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    f_mu_glm, f_var_glm = lap(X, pred_type="glm")
    assert torch.allclose(f_mu_glm, f)
    assert f_var_glm.shape == torch.Size(
        [f_mu_glm.shape[0], f_mu_glm.shape[1], f_mu_glm.shape[1]]
    )
    assert len(f_mu_glm) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu_nn, f_var_nn = lap(X, pred_type="nn", link_approx="mc")
    assert f_mu_nn.shape == f_var_nn.shape
    assert f_var_nn.shape == torch.Size([f_mu_nn.shape[0], f_mu_nn.shape[1]])
    assert len(f_mu_nn) == len(X)

    # Test joint prediction
    f_mu_joint, f_cov_joint = lap(X, pred_type="glm", joint=True)
    assert len(f_mu_joint.shape) == 1
    assert f_mu_joint.shape[0] == prod(f_mu_glm.shape)
    assert len(f_cov_joint.shape) == 2
    assert f_cov_joint.shape == (f_mu_joint.shape[0], f_mu_joint.shape[0])

    # The "diagonal" of the joint cov should equal the non-joint var
    b, k = y.shape
    f_var_joint = torch.einsum("bkbl->bkl", f_cov_joint.reshape(b, k, b, k))
    assert torch.allclose(f_var_joint, f_var_glm)


@pytest.mark.parametrize("laplace", flavors)
def test_classification_predictive(laplace, model, class_loader):
    lap = laplace(model, "classification", prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    f_pred = lap(X, pred_type="glm", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="probit")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge_norm")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1

    # NN predictive
    f_pred = lap(X, pred_type="nn", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1


@pytest.mark.parametrize("laplace", flavors)
def test_regression_predictive_samples(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    fsamples = lap.predictive_samples(X, pred_type="glm", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # NN predictive (only diagonal variance estimation)
    fsamples = lap.predictive_samples(X, pred_type="nn", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])


@pytest.mark.parametrize("laplace", flavors)
def test_classification_predictive_samples(laplace, model, class_loader):
    lap = laplace(model, "classification", prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    fsamples = lap.predictive_samples(X, pred_type="glm", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1

    # NN predictive
    fsamples = lap.predictive_samples(X, pred_type="nn", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1


@pytest.mark.parametrize("laplace", flavors)
def test_functional_samples(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    generator = torch.Generator()

    fsamples_reg_glm = lap.functional_samples(
        X, pred_type="glm", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_reg_glm.shape == torch.Size([100, f.shape[0], f.shape[1]])

    fsamples_reg_nn = lap.functional_samples(
        X, pred_type="nn", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_reg_nn.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # The samples should not be affected by the likelihood
    lap.likelihood = "classification"

    fsamples_clf_glm = lap.functional_samples(
        X, pred_type="glm", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_clf_glm.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert torch.allclose(fsamples_clf_glm, fsamples_reg_glm)

    fsamples_clf_nn = lap.functional_samples(
        X, pred_type="nn", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_clf_nn.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert torch.allclose(fsamples_clf_nn, fsamples_reg_nn)


@pytest.mark.parametrize("laplace", [KronLaplace, DiagLaplace])
def test_reward_modeling(laplace, reward_model, reward_loader, reward_test_X):
    lap = laplace(reward_model, "reward_modeling")
    lap.fit(reward_loader)
    f = reward_model(reward_test_X)

    # error
    with pytest.raises(ValueError):
        lap(reward_test_X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    f_mu, f_var = lap(reward_test_X, pred_type="glm")
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(reward_test_X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(reward_test_X, pred_type="nn", link_approx="mc")
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(reward_test_X)


@pytest.mark.parametrize("laplace", [KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN, BackPackGGN, BackPackEF]
)
@pytest.mark.parametrize(
    "lik,custom_loader",
    [
        ("classification", "custom_loader_clf"),
        ("regression", "custom_loader_reg"),
        ("reward_modeling", "custom_loader_clf"),
    ],
)
def test_dict_data(laplace, backend, lik, custom_loader, custom_model, request):
    custom_loader = request.getfixturevalue(custom_loader)

    if (
        "backpack" not in backend.__name__.lower()
        and laplace != DiagLaplace
        and laplace != CurvlinopsEF
    ):
        with pytest.raises(KeyError):
            # Raises an error since custom_loader's input is under the key 'test_input_key'
            # but the default is 'input_ids'
            lap = laplace(custom_model, lik, backend=backend)
            lap.fit(custom_loader)

    lap = laplace(
        custom_model,
        lik,
        backend=backend,
        dict_key_x="test_input_key",
        dict_key_y="test_label_key",
    )

    if ("backpack" in backend.__name__.lower()) or (
        laplace == DiagLaplace and backend == CurvlinopsEF
    ):
        # Unsupported, thus raises an exception
        with pytest.raises(ValueError):
            lap.fit(custom_loader)

        return

    lap.fit(custom_loader)

    test_data = next(iter(custom_loader))
    f = custom_model(test_data)

    if lik == "classification":
        f_pred = lap(test_data, pred_type="glm")
        assert f_pred.shape == f.shape
        assert torch.allclose(
            f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
        )  # sum up to 1

        f_pred = lap(test_data, pred_type="nn", link_approx="mc")
        assert f_pred.shape == f.shape
        assert torch.allclose(
            f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
        )
    else:
        f_pred, f_var = lap(test_data, pred_type="glm")
        assert f_pred.shape == f.shape
        assert torch.allclose(f_pred, f)
        assert f_var.shape == (f_pred.shape[0], f_pred.shape[1], f_pred.shape[1])

        f_pred, f_var = lap(test_data, pred_type="nn", link_approx="mc")
        assert f_pred.shape == f.shape
        assert f_var.shape == (f_pred.shape[0], f_pred.shape[1])


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="glm")

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm_joint(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_cov = lap(X, pred_type="glm", joint=True)

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_cov.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm_mc(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="glm", link_approx="mc")

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_nn(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="nn", link_approx="mc", n_samples=10)

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
def test_reg_glm_predictive_correct_behavior(laplace, model, reg_loader):
    X, y = reg_loader.dataset.tensors
    n_batch = X.shape[0]
    n_outputs = y.shape[-1]

    lap = laplace(model, "regression")
    lap.fit(reg_loader)

    # Joint predictive ignores diagonal_output
    f_mean, f_var = lap(X, pred_type="glm", joint=True, diagonal_output=True)
    assert f_var.shape == (n_batch * n_outputs, n_batch * n_outputs)

    f_mean, f_var = lap(X, pred_type="glm", joint=True, diagonal_output=False)
    assert f_var.shape == (n_batch * n_outputs, n_batch * n_outputs)

    # diagonal_output affects non-joint
    f_mean, f_var = lap(X, pred_type="glm", joint=False, diagonal_output=True)
    assert f_var.shape == (n_batch, n_outputs)

    f_mean, f_var = lap(X, pred_type="glm", joint=False, diagonal_output=False)
    assert f_var.shape == (n_batch, n_outputs, n_outputs)


@pytest.mark.parametrize(
    "likelihood,custom_loader",
    [
        ("classification", "custom_loader_clf"),
        ("regression", "custom_loader_reg"),
        ("reward_modeling", "custom_loader_clf"),
    ],
)
def test_dict_data_diagEF_curvlinops_fails(
    custom_model, custom_loader, likelihood, request
):
    custom_loader = request.getfixturevalue(custom_loader)
    lap = DiagLaplace(custom_model, likelihood=likelihood, backend=CurvlinopsEF)

    with pytest.raises(ValueError):
        lap.fit(custom_loader)


@pytest.mark.parametrize(
    "likelihood", ["classification", "regression", "reward_modeling"]
)
@pytest.mark.parametrize("prior_prec_type", ["scalar", "layerwise", "diag"])
def test_gridsearch(model, likelihood, prior_prec_type, reg_loader, class_loader):
    if likelihood == "regression":
        dataloader = reg_loader
    else:
        dataloader = class_loader

    if prior_prec_type == "scalar":
        prior_prec = 1.0
    elif prior_prec_type == "layerwise":
        prior_prec = torch.ones(model.n_layers)
    else:
        prior_prec = torch.ones(model.n_params)

    lap = DiagLaplace(model, likelihood, prior_precision=prior_prec)
    lap.fit(dataloader)

    # Should not raise an error
    lap.optimize_prior_precision(method="gridsearch", val_loader=dataloader, n_steps=10)


@pytest.mark.parametrize("laplace", flavors)
def test_parametric_fit_y_shape(model_1d, reg_loader_1d, reg_loader_1d_flat, laplace):
    lap = laplace(model_1d, likelihood="regression")
    lap.fit(reg_loader_1d)  # OK

    lap2 = laplace(model_1d, likelihood="regression")

    with pytest.raises(ValueError):
        lap2.fit(reg_loader_1d_flat)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize(
    "backend", [AsdlEF, AsdlGGN, BackPackEF, BackPackGGN, CurvlinopsEF, CurvlinopsGGN]
)
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_dtype(laplace, backend, dtype, likelihood):
    X = torch.randn((10, 3), dtype=dtype)
    Y = torch.randn((10, 3), dtype=dtype)

    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=10)

    model = nn.Linear(3, 3, dtype=dtype)

    try:
        la = laplace(model, likelihood, backend=backend)
        la.fit(dataloader)

        assert la.H is not None

        if isinstance(la.H, torch.Tensor):
            assert la.H.dtype == dtype
        elif isinstance(la.H, KronDecomposed):
            assert la.H.eigenvalues[0][0].dtype == dtype
            assert la.H.eigenvectors[0][0].dtype == dtype

        assert la.log_marginal_likelihood().dtype == dtype

        y_pred, y_var = la(X, pred_type="glm")
        assert y_pred.dtype == dtype
        assert y_var.dtype == dtype

        y_pred = la(X, pred_type="nn", num_samples=3)
        assert y_pred.dtype == dtype
    except (ValueError, AttributeError, RuntimeError, SystemExit) as e:
        if "must have the same dtype" in str(e):
            assert False  # Fail the test
        else:
            pass  # Ignore
