"""Tests of the heteroscedastic Gaussian likelihood: parity with `nn.GaussianNLLLoss`,
the curvature of the mean against brute-force references, and the predictive
`f_mean, f_alea, f_epis = la(x)`.
"""

import copy
from math import log, pi, sqrt

import pytest
import torch
from torch import nn
from torch.func import functional_call, hessian, jacrev
from torch.utils.data import DataLoader, TensorDataset

from laplace import (
    HeteroscedasticGaussianLikelihood,
    Laplace,
    LowRankLaplace,
    get_likelihood,
    marglik_training,
)
from laplace.curvature import CurvlinopsEF, CurvlinopsGGN, CurvlinopsHessian
from laplace.utils import RunningGaussianNLLMetric
from laplace.utils.heteroscedastic_utils import join_mean_variance
from tests.utils import AsdlEF, AsdlGGN, BackPackEF, BackPackGGN

pytestmark = pytest.mark.default_dtype(torch.float64)

N_DATA = 13
N_IN = 3
K = 2


class MeanVarianceNet(nn.Module):
    """A network with a mean head and a variance head on a shared body.

    Parameters
    ----------
    shared : bool, default=False
        whether the variance is shared by the outputs, `(batch, 1)`.
    """

    def __init__(self, shared: bool = False) -> None:
        super().__init__()

        self.body = nn.Sequential(nn.Linear(N_IN, 4), nn.Tanh())
        self.mean = nn.Linear(4, K)
        self.var = nn.Sequential(nn.Linear(4, 1 if shared else K), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)

        return self.mean(h), self.var(h) + 1e-2


class MeanOnly(nn.Module):
    """The mean head of a `MeanVarianceNet`, for a Gaussian reference."""

    def __init__(self, net: MeanVarianceNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]


@pytest.fixture
def data() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(N_DATA, N_IN), torch.randn(N_DATA, K)


@pytest.fixture
def loader(data) -> DataLoader:
    return DataLoader(TensorDataset(*data), batch_size=5)


def _flat_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in model.named_parameters()}


def _to_matrix(blocks: dict, params: dict[str, torch.Tensor]) -> torch.Tensor:
    # A nested dict of second derivatives as one `(P, P)` matrix
    names = list(params)

    return torch.cat(
        [
            torch.cat(
                [
                    blocks[a][b].reshape(params[a].numel(), params[b].numel())
                    for b in names
                ],
                dim=1,
            )
            for a in names
        ],
        dim=0,
    )


def _nll_fixed_variance(model, params, X, y) -> torch.Tensor:
    # The NLL of the mean, with the variance held at its value under `model`
    mean = functional_call(model, params, (X,))[0]
    var = model(X)[1].detach()

    return nn.GaussianNLLLoss(reduction="sum")(mean, y, var)


def _references(model, X, y) -> dict[str, torch.Tensor]:
    """Brute-force GGN, Hessian, and empirical Fisher of the mean."""
    params = _flat_params(model)

    def mean_fn(p):
        return functional_call(model, p, (X,))[0]

    J = jacrev(mean_fn)(params)
    J = torch.cat([j.flatten(2) for j in J.values()], dim=-1)  # (N, K, P)
    var = model(X)[1].detach().expand(-1, K)
    ggn = torch.einsum("nkp,nk,nkq->pq", J, 1 / var, J)

    H = _to_matrix(
        hessian(lambda p: _nll_fixed_variance(model, p, X, y))(params), params
    )

    grads = []
    for i in range(len(X)):
        g = torch.func.grad(
            lambda p: _nll_fixed_variance(model, p, X[i : i + 1], y[i : i + 1])
        )(params)
        grads.append(torch.cat([v.flatten() for v in g.values()]))
    G = torch.stack(grads)

    return {"ggn": ggn, "hessian": H, "ef": G.T @ G, "J": J}


# Likelihood


def test_get_likelihood():
    assert type(get_likelihood("heteroscedastic_gaussian")) is (
        HeteroscedasticGaussianLikelihood
    )

    lik = get_likelihood(nn.GaussianNLLLoss(eps=1e-3, full=True))

    assert lik == HeteroscedasticGaussianLikelihood(eps=1e-3, full=True)
    assert lik.hyperparameters() == {"eps": 1e-3, "full": True}
    assert HeteroscedasticGaussianLikelihood().hyperparameters() == {}


def test_invalid_eps():
    with pytest.raises(ValueError, match="eps must be positive"):
        HeteroscedasticGaussianLikelihood(eps=0.0)


@pytest.mark.parametrize("var_shape", [(N_DATA, K), (N_DATA, 1), (N_DATA,)])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 0.5])
def test_loss_matches_torch(var_shape, full, eps):
    mean = torch.randn(N_DATA, K, requires_grad=True)
    var = torch.rand(var_shape, requires_grad=True)
    y = torch.randn(N_DATA, K)

    torch_loss = nn.GaussianNLLLoss(full=full, eps=eps, reduction="sum")
    lik = get_likelihood(torch_loss)

    expected = torch_loss(mean, y, var)
    loss = lik.loss_fn()((mean, var), y)
    assert torch.allclose(loss, expected)

    # The gradients agree too, also where the variance is clamped
    grads = torch.autograd.grad(loss, (mean, var))
    expected_grads = torch.autograd.grad(expected, (mean, var))
    for g, e in zip(grads, expected_grads):
        assert torch.allclose(g, e)

    f = join_mean_variance(mean, var)
    assert torch.allclose(lik.loss(f, y), expected)

    mean_reduction = lik.loss_fn("mean")((mean, var), y)
    torch_mean = nn.GaussianNLLLoss(full=full, eps=eps)(mean, y, var)
    assert torch.allclose(mean_reduction, torch_mean)


def test_join_mean_variance_rejects_other_shapes():
    with pytest.raises(ValueError, match="fits neither the mean"):
        join_mean_variance(torch.randn(4, 3), torch.rand(4, 2))


@pytest.mark.parametrize("full", [False, True])
def test_loss_and_normalizer_give_log_likelihood(full):
    mean, var = torch.randn(N_DATA, K), torch.rand(N_DATA, K) + 0.1
    y = torch.randn(N_DATA, K)

    lik = HeteroscedasticGaussianLikelihood(full=full)
    f = join_mean_variance(mean, var)
    c = lik.log_normalizer(N_DATA, 2 * K, torch.tensor(1.0))

    log_lik = torch.distributions.Normal(mean, var.sqrt()).log_prob(y).sum()

    assert torch.allclose(-lik.loss_factor * lik.loss(f, y) - c, log_lik)
    assert torch.allclose(
        c, torch.tensor(0.0 if full else N_DATA * K * log(sqrt(2 * pi)))
    )


def test_output_hessian_and_gradient_are_over_the_mean():
    mean, var = torch.randn(N_DATA, K), torch.rand(N_DATA, K) + 0.1
    y = torch.randn(N_DATA, K)

    lik = HeteroscedasticGaussianLikelihood()
    f = join_mean_variance(mean, var)

    H = lik.output_hessian(f)
    expected = torch.diag_embed(torch.cat([1 / var, torch.zeros_like(var)], -1))
    assert torch.allclose(H, expected)

    g = lik.output_gradient(f, y)
    expected = torch.cat([(mean - y) / var, torch.zeros_like(var)], -1)
    assert torch.allclose(g, expected)

    assert torch.equal(lik.inverse_link(f), mean)
    assert lik.sample(f).shape == y.shape
    assert lik.n_observations(f, y) == N_DATA


def test_check_targets():
    lik = HeteroscedasticGaussianLikelihood()

    with pytest.raises(ValueError, match="even number of outputs"):
        lik.check_targets(torch.randn(4, 3), torch.randn(4, 1))

    with pytest.raises(ValueError, match="shape of the mean"):
        lik.check_targets(torch.randn(4, 4), torch.randn(4, 3))


def test_running_gaussian_nll_metric():
    mean, var = torch.randn(N_DATA, K), torch.rand(N_DATA, K) + 0.1
    y = torch.randn(N_DATA, K)

    metric = RunningGaussianNLLMetric()
    metric.update(mean[:6], var[:6], y[:6])
    metric.update(mean[6:], var[6:], y[6:])

    nll = -torch.distributions.Normal(mean, var.sqrt()).log_prob(y).sum() / N_DATA
    assert torch.allclose(metric.compute(), nll)


# Curvature


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    "backend,reference",
    [
        (CurvlinopsGGN, "ggn"),
        (AsdlGGN, "ggn"),
        (BackPackGGN, "ggn"),
        (CurvlinopsEF, "ef"),
        (AsdlEF, "ef"),
        (BackPackEF, "ef"),
        (CurvlinopsHessian, "hessian"),
    ],
)
@pytest.mark.parametrize("structure", ["full", "diag"])
def test_curvature_matches_reference(
    shared, backend, reference, structure, data, loader
):
    if backend is CurvlinopsHessian and structure == "diag":
        pytest.skip("CurvlinopsHessian computes no diagonal.")

    model = MeanVarianceNet(shared)
    expected = _references(model, *data)[reference]

    la = Laplace(
        model,
        nn.GaussianNLLLoss(),
        subset_of_weights="all",
        hessian_structure=structure,
        backend=backend,
    )
    la.fit(loader)

    if structure == "diag":
        expected = expected.diagonal()

    assert torch.allclose(la.H, expected)

    # The parameters of the variance head get no curvature from the data
    n_var = sum(p.numel() for p in model.var.parameters())
    var_block = la.H[-n_var:] if structure == "diag" else la.H[-n_var:, -n_var:]
    assert torch.all(var_block == 0)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, CurvlinopsEF])
def test_kron_matches_gaussian_kfac(backend, loader):
    # With a constant variance, the KFAC of the mean is that of a Gaussian likelihood
    # with that noise, scaled by the precision (squared for the EF)
    model = MeanVarianceNet()

    with torch.no_grad():
        model.var[0].weight.zero_()
        model.var[0].bias.fill_(0.3)

    var = float(nn.functional.softplus(torch.tensor(0.3))) + 1e-2
    scale = 1 / var if backend is CurvlinopsGGN else 1 / var**2

    kwargs = dict(subset_of_weights="all", hessian_structure="kron", backend=backend)
    la = Laplace(model, "heteroscedastic_gaussian", **kwargs)
    la.fit(loader)

    # The Gaussian reference has no variance head to differentiate
    mean_only = copy.deepcopy(model)
    mean_only.var.requires_grad_(False)
    la_gauss = Laplace(MeanOnly(mean_only), "regression", **kwargs)
    la_gauss.fit(loader)

    H, H_gauss = la.H.to_matrix(), la_gauss.H.to_matrix()
    n = H_gauss.shape[0]

    assert torch.allclose(H[:n, :n], scale * H_gauss)
    assert torch.all(H[n:, n:] == 0)


@pytest.mark.parametrize("backend", [AsdlGGN, BackPackGGN])
def test_kron_raises_for_other_backends(backend, loader):
    la = Laplace(
        MeanVarianceNet(),
        "heteroscedastic_gaussian",
        subset_of_weights="all",
        hessian_structure="kron",
        backend=backend,
    )

    with pytest.raises(ValueError, match="does not support the likelihood"):
        la.fit(loader)


def test_subnetwork_matches_reference(data, loader):
    model = MeanVarianceNet()
    expected = _references(model, *data)["ggn"]
    indices = torch.arange(3, 30)

    la = Laplace(
        model,
        "heteroscedastic_gaussian",
        subset_of_weights="subnetwork",
        hessian_structure="full",
        subnetwork_indices=indices,
    )
    la.fit(loader)

    assert torch.allclose(la.H, expected[indices][:, indices])


def test_log_likelihood(data, loader):
    model = MeanVarianceNet()
    X, y = data

    la = Laplace(model, "heteroscedastic_gaussian", subset_of_weights="all")
    la.fit(loader)

    mean, var = model(X)
    expected = torch.distributions.Normal(mean, var.sqrt()).log_prob(y).sum()

    assert torch.allclose(la.log_likelihood, expected.detach())


# Predictive


def _posterior_covariance(la) -> torch.Tensor:
    # The dense posterior covariance of any Hessian structure
    if hasattr(la.posterior_precision, "to_matrix"):
        return torch.linalg.inv(la.posterior_precision.to_matrix())

    if la.posterior_precision.ndim == 1:
        return torch.diag(1 / la.posterior_precision)

    return la.posterior_covariance


@pytest.mark.parametrize("structure", ["full", "diag", "kron"])
def test_glm_predictive(structure, data, loader):
    model = MeanVarianceNet()
    X, y = data

    la = Laplace(
        model,
        nn.GaussianNLLLoss(),
        subset_of_weights="all",
        hessian_structure=structure,
    )
    la.fit(loader)

    f_mean, f_alea, f_epis = la(X)
    mean, var = model(X)

    assert torch.allclose(f_mean, mean)
    assert torch.allclose(f_alea, var)

    J = _references(model, X, y)["J"]
    expected = torch.einsum("nkp,pq,nlq->nkl", J, _posterior_covariance(la), J)
    assert torch.allclose(f_epis, expected)

    _, _, f_var = la(X, diagonal_output=True)
    assert torch.allclose(f_var, expected.diagonal(dim1=-2, dim2=-1))

    f_mean_joint, f_alea_joint, f_cov = la(X, joint=True)
    assert torch.allclose(f_mean_joint, mean.flatten())
    assert torch.allclose(f_alea_joint, var.flatten())
    assert f_cov.shape == (N_DATA * K, N_DATA * K)
    assert torch.allclose(
        f_cov.diagonal(), expected.diagonal(dim1=-2, dim2=-1).flatten()
    )


def test_shared_variance_is_expanded(data, loader):
    model = MeanVarianceNet(shared=True)
    X, _ = data

    la = Laplace(model, "heteroscedastic_gaussian", subset_of_weights="all")
    la.fit(loader)

    _, f_alea, _ = la(X)

    assert torch.allclose(f_alea, model(X)[1].expand(-1, K))


def test_nn_predictive_and_samples(data, loader):
    model = MeanVarianceNet()
    X, _ = data

    la = Laplace(model, "heteroscedastic_gaussian", subset_of_weights="all")
    la.fit(loader)

    f_mean, f_alea, f_var = la(X, pred_type="nn", link_approx="mc", n_samples=7)
    assert f_mean.shape == f_alea.shape == f_var.shape == (N_DATA, K)
    assert torch.allclose(f_alea, model(X)[1])

    for pred_type in ["glm", "nn"]:
        assert la.functional_samples(X, pred_type, n_samples=3).shape == (3, N_DATA, K)
        assert la.predictive_samples(X, pred_type, n_samples=3).shape == (3, N_DATA, K)

    joint = la.functional_samples(X, n_samples=3, joint=True)
    assert joint.shape == (3, N_DATA, K)


def test_model_is_left_unchanged(data, loader):
    model = MeanVarianceNet()
    X, _ = data
    names = [name for name, _ in model.named_parameters()]

    la = Laplace(model, "heteroscedastic_gaussian", subset_of_weights="all")
    la.fit(loader)
    la(X)
    la.predictive_samples(X, n_samples=2)

    # The hook that joins the outputs is gone, and the names are the user's
    out = model(X)
    assert isinstance(out, tuple) and len(out) == 2
    assert not model._forward_hooks
    assert not hasattr(model, "_joins_outputs")
    assert [name for name, _ in model.named_parameters()] == names


def test_aleatoric_variance_backprop(data, loader):
    model = MeanVarianceNet()
    X, _ = data

    la = Laplace(
        model, "heteroscedastic_gaussian", subset_of_weights="all", enable_backprop=True
    )
    la.fit(loader)

    X = X.clone().requires_grad_(True)
    f_mean, f_alea, f_epis = la(X)
    (f_mean.sum() + f_alea.sum() + f_epis.sum()).backward()

    assert X.grad is not None and torch.all(X.grad.isfinite())


def test_gridsearch_default_metric(data, loader):
    la = Laplace(MeanVarianceNet(), "heteroscedastic_gaussian", subset_of_weights="all")
    la.fit(loader)

    la.optimize_prior_precision(
        method="gridsearch", val_loader=loader, grid_size=3, n_samples=5
    )

    assert la.prior_precision.numel() == 1


# Restrictions


def test_sigma_noise_must_stay_one():
    with pytest.raises(ValueError, match="Sigma noise"):
        Laplace(
            MeanVarianceNet(),
            "heteroscedastic_gaussian",
            subset_of_weights="all",
            sigma_noise=2.0,
        )


# The parameter gives the test the `asdfghjkl` marker, see `tests/conftest.py`
@pytest.mark.parametrize("laplace_cls", [LowRankLaplace])
def test_low_rank_raises(laplace_cls, loader):
    pytest.importorskip("asdfghjkl")

    la = laplace_cls(MeanVarianceNet(), "heteroscedastic_gaussian")

    with pytest.raises(ValueError, match="LowRankLaplace does not support"):
        la.fit(loader)


@pytest.mark.parametrize("structure", ["kron", "diag", "full"])
def test_last_layer_raises(structure):
    with pytest.raises(ValueError, match="#254"):
        Laplace(
            MeanVarianceNet(), "heteroscedastic_gaussian", hessian_structure=structure
        )


@pytest.mark.parametrize(
    "approximation,kwargs",
    [
        ("sod", dict(n_subset=4)),
        ("nystrom", dict(subsample_size=4, n_eigenvalues=2)),
        ("variational", dict(inducing_locations=torch.randn(2, N_IN))),
    ],
)
def test_functional_methods_raise(approximation, kwargs):
    with pytest.raises(ValueError):
        Laplace(
            MeanVarianceNet(),
            "heteroscedastic_gaussian",
            subset_of_weights="all",
            hessian_structure="gp",
            functional_approximation=approximation,
            **kwargs,
        )


# Training and checkpoints


@pytest.mark.parametrize("structure", ["diag", "kron"])
def test_marglik_training_trains_both_heads(structure, loader):
    model = MeanVarianceNet()
    before = copy.deepcopy(model.state_dict())

    la, model, margliks, losses = marglik_training(
        model,
        loader,
        likelihood=nn.GaussianNLLLoss(),
        hessian_structure=structure,
        n_epochs=3,
        n_epochs_burnin=1,
        n_hypersteps=2,
    )

    assert len(margliks) > 0 and all(torch.isfinite(torch.tensor(margliks)))

    after = model.state_dict()
    for name in ["mean.weight", "var.0.weight"]:
        assert not torch.equal(before[name], after[name])

    X = next(iter(loader))[0]
    assert len(la(X)) == 3


def test_state_dict_round_trip(tmp_path, data, loader):
    model = MeanVarianceNet()
    X, _ = data

    lik = HeteroscedasticGaussianLikelihood(eps=1e-3)
    la = Laplace(model, lik, subset_of_weights="all", hessian_structure="diag")
    la.fit(loader)
    torch.save(la.state_dict(), tmp_path / "state_dict.bin")

    la2 = Laplace(model, lik, subset_of_weights="all", hessian_structure="diag")
    la2.load_state_dict(torch.load(tmp_path / "state_dict.bin", weights_only=False))

    for a, b in zip(la(X), la2(X)):
        assert torch.allclose(a, b)

    la3 = Laplace(
        model,
        "heteroscedastic_gaussian",
        subset_of_weights="all",
        hessian_structure="diag",
    )
    with pytest.raises(ValueError):
        la3.load_state_dict(torch.load(tmp_path / "state_dict.bin", weights_only=False))
