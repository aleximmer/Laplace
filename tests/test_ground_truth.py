"""Check Laplace quantities against brute-force references on tiny models.

The references come straight from `torch.func`: dense Jacobians, the exact GGN or
Hessian, and the textbook formulas for the log marginal likelihood and the GLM
predictive. The models have a few dozen parameters, so every matrix is dense.
"""

import copy
import math
import warnings
from collections.abc import Callable
from importlib.util import find_spec

import pytest
import torch
from torch import nn
from torch.func import functional_call, hessian, jacrev
from torch.utils.data import DataLoader, TensorDataset

from laplace import (
    CategoricalLikelihood,
    DiagLaplace,
    DiagLLLaplace,
    DiagSubnetLaplace,
    FullLaplace,
    FullLLLaplace,
    FullSubnetLaplace,
    FunctionalLaplace,
    GaussianLikelihood,
    KronLaplace,
    KronLLLaplace,
    Likelihood,
    LowRankLaplace,
)
from laplace.curvature import CurvlinopsEF, CurvlinopsGGN, CurvlinopsHessian
from laplace.likelihoods import BernoulliLikelihood
from laplace.marglik_training import marglik_training
from tests.utils import AsdlEF, AsdlGGN, AsdlHessian, BackPackEF, BackPackGGN, lik_id

pytestmark = pytest.mark.default_dtype(torch.float64)

N_DATA = 13
N_IN = 3

GGN_BACKENDS = [CurvlinopsGGN, AsdlGGN, BackPackGGN]
EF_BACKENDS = [CurvlinopsEF, AsdlEF, BackPackEF]
HESSIAN_BACKENDS = [CurvlinopsHessian, AsdlHessian]


def _mlp(n_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(N_IN, 4), nn.Tanh(), nn.Linear(4, n_out))


def _data(task: Likelihood, n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(N_DATA, N_IN)

    if isinstance(task, GaussianLikelihood):
        return X, torch.randn(N_DATA, n_out)

    return X, torch.randint(n_out, (N_DATA,))


def _loader(X: torch.Tensor, y: torch.Tensor) -> DataLoader:
    # Several batches, the last one partial, to check the accumulation over batches
    return DataLoader(TensorDataset(X, y), batch_size=5)


def _flat(model: nn.Module) -> tuple[Callable, torch.Tensor]:
    """The model as a function of its flat trainable parameters, and their vector.
    Frozen parameters stay fixed at their values in the model."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in trainable]
    shapes = [p.shape for _, p in trainable]

    def f(theta: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        params, i = {}, 0

        for name, shape in zip(names, shapes):
            params[name] = theta[i : i + shape.numel()].view(shape)
            i += shape.numel()

        return functional_call(model, params, (X,))

    theta = torch.cat([p.detach().flatten() for _, p in trainable])

    return f, theta


def _nll(
    task: Likelihood, out: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # Negative log likelihood without the Gaussian normalizer
    if isinstance(task, GaussianLikelihood):
        return 0.5 * ((out - y) ** 2).sum() / sigma**2

    return nn.functional.cross_entropy(out, y, reduction="sum")


def _log_normalizer(task: Likelihood, n_out: int, sigma: float) -> float:
    if isinstance(task, GaussianLikelihood):
        return N_DATA * n_out * math.log(sigma * math.sqrt(2 * math.pi))

    return 0.0


def _reference(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    task: Likelihood,
    curvature: str = "ggn",
    sigma: float = 1.0,
    prior: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Dense posterior precision and log marginal likelihood at the model's weights."""
    f, theta = _flat(model)
    out = f(theta, X)
    n_out = out.shape[-1]

    if curvature == "ggn":
        J = jacrev(f)(theta, X)

        if isinstance(task, GaussianLikelihood):
            L = torch.eye(n_out).expand(N_DATA, n_out, n_out) / sigma**2
        else:
            p = out.softmax(-1)
            L = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]

        H = torch.einsum("nkp,nkl,nlq->pq", J, L, J)
    elif curvature == "hessian":
        H = hessian(lambda t: _nll(task, f(t, X), y, sigma))(theta)
    else:
        # Per-datum gradients of the negative log likelihood
        G = jacrev(
            lambda t: torch.stack(
                [
                    _nll(task, f(t, X[i : i + 1]), y[i : i + 1], sigma)
                    for i in range(N_DATA)
                ]
            )
        )(theta)
        H = G.T @ G

    P = H + prior * torch.eye(len(theta))
    log_lik = -_nll(task, out, y, sigma) - _log_normalizer(task, n_out, sigma)
    log_det_ratio = torch.logdet(P) - len(theta) * math.log(prior)

    return {
        "P": P,
        "lml": log_lik - 0.5 * (log_det_ratio + prior * theta @ theta),
        "theta": theta,
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("backend", GGN_BACKENDS)
@pytest.mark.parametrize(
    "task,n_out,sigma,prior",
    [
        (GaussianLikelihood(), 1, 1.0, 1.0),
        (GaussianLikelihood(), 2, 0.7, 2.5),
        (CategoricalLikelihood(), 3, 1.0, 2.5),
    ],
    ids=lik_id,
)
def test_full_and_diag_ggn(backend, task, n_out, sigma, prior):
    model = _mlp(n_out)
    X, y = _data(task, n_out)
    ref = _reference(copy.deepcopy(model), X, y, task, "ggn", sigma, prior)

    la = FullLaplace(
        model, task, sigma_noise=sigma, prior_precision=prior, backend=backend
    )
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, ref["P"])
    _assert_close(la.log_marginal_likelihood(), ref["lml"])

    # The diagonal approximation changes only the log determinant
    la = DiagLaplace(
        model, task, sigma_noise=sigma, prior_precision=prior, backend=backend
    )
    la.fit(_loader(X, y))
    P_diag = ref["P"].diagonal()
    lml = ref["lml"] - 0.5 * (P_diag.log().sum() - torch.logdet(ref["P"]))

    _assert_close(la.posterior_precision, P_diag)
    _assert_close(la.log_marginal_likelihood(), lml)


@pytest.mark.parametrize("backend", HESSIAN_BACKENDS)
@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_full_hessian(backend, task, n_out):
    model = _mlp(n_out)
    X, y = _data(task, n_out)

    # A large prior keeps the posterior precision positive definite
    ref = _reference(copy.deepcopy(model), X, y, task, "hessian", prior=50.0)
    assert torch.linalg.eigvalsh(ref["P"]).min() > 0

    la = FullLaplace(model, task, prior_precision=50.0, backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, ref["P"])
    _assert_close(la.log_marginal_likelihood(), ref["lml"])


@pytest.mark.parametrize("backend", EF_BACKENDS)
def test_full_ef_classification(backend):
    model = _mlp(3)
    X, y = _data(CategoricalLikelihood(), 3)
    ref = _reference(copy.deepcopy(model), X, y, CategoricalLikelihood(), "ef")

    la = FullLaplace(model, CategoricalLikelihood(), backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, ref["P"])


@pytest.mark.parametrize("backend", EF_BACKENDS)
@pytest.mark.parametrize("sigma", [1.0, 0.5])
def test_full_ef_regression(backend, sigma):
    model = _mlp(2)
    X, y = _data(GaussianLikelihood(), 2)
    ref = _reference(copy.deepcopy(model), X, y, GaussianLikelihood(), "ef", sigma)

    la = FullLaplace(model, GaussianLikelihood(), sigma_noise=sigma, backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, ref["P"])

    la = DiagLaplace(model, GaussianLikelihood(), sigma_noise=sigma, backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, ref["P"].diagonal())


@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 1), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_temperature_scales_classification_and_curvature(task, n_out):
    # A temperature T divides the curvature by T. For classification it also divides
    # the whole log likelihood. Regression is checked below.
    T = 0.5
    model = _mlp(n_out)
    X, y = _data(task, n_out)
    ref = _reference(copy.deepcopy(model), X, y, task, "ggn")
    f, theta = _flat(model)

    la = FullLaplace(model, task, temperature=T)
    la.fit(_loader(X, y))

    _assert_close(
        la.posterior_precision,
        (ref["P"] - torch.eye(len(theta))) / T + torch.eye(len(theta)),
    )

    if isinstance(task, CategoricalLikelihood):
        _assert_close(la.log_likelihood, -_nll(task, f(theta, X), y, 1.0) / T)


def test_temperature_regression_log_likelihood():
    T, sigma = 0.5, 0.7
    model = _mlp(1)
    X, y = _data(GaussianLikelihood(), 1)
    f, theta = _flat(model)

    la = FullLaplace(model, GaussianLikelihood(), sigma_noise=sigma, temperature=T)
    la.fit(_loader(X, y))

    log_lik = -_nll(GaussianLikelihood(), f(theta, X), y, sigma) - _log_normalizer(
        GaussianLikelihood(), 1, sigma
    )

    _assert_close(la.log_likelihood, log_lik / T)


@pytest.mark.parametrize("backend", GGN_BACKENDS)
@pytest.mark.parametrize("n_out", [1, 3])
def test_kron_exact_on_single_linear_layer(backend, n_out):
    # Without a bias, a linear layer's GGN under the MSE is exactly A ⊗ I, so
    # KFAC is exact and so are all Kron quantities
    model = nn.Linear(N_IN, n_out, bias=False)
    X, y = _data(GaussianLikelihood(), n_out)
    ref_model = copy.deepcopy(model)
    ref = _reference(ref_model, X, y, GaussianLikelihood(), "ggn", sigma=0.7, prior=2.0)

    la = KronLaplace(
        model,
        GaussianLikelihood(),
        sigma_noise=0.7,
        prior_precision=2.0,
        backend=backend,
    )
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision.to_matrix(), ref["P"])
    _assert_close(la.log_marginal_likelihood(), ref["lml"])

    f, theta = _flat(ref_model)
    X_test = torch.randn(4, N_IN)
    Js = jacrev(f)(theta, X_test)
    var = torch.einsum("nkp,pq,nlq->nkl", Js, torch.linalg.inv(ref["P"]), Js)

    _assert_close(la(X_test, pred_type="glm")[1], var)


@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
@pytest.mark.parametrize(
    "prior,sigma,T",
    [(1.3, 1.0, 1.0), (torch.tensor([0.5, 1.0, 2.0, 4.0]), 1.0, 0.3), (0.9, 0.6, 1.0)],
)
def test_kron_decomposition_matches_dense(task, n_out, prior, sigma, T):
    # On a deep net KFAC is an approximation, but everything computed from its
    # eigendecomposition must match the dense matrix it represents
    if isinstance(task, CategoricalLikelihood) and sigma != 1.0:
        pytest.skip("classification has no noise")

    model = _mlp(n_out)
    X, y = _data(task, n_out)
    la = KronLaplace(
        model, task, prior_precision=prior, sigma_noise=sigma, temperature=T
    )
    la.fit(_loader(X, y))

    sizes = [p.numel() for p in la.params]
    prior_per_layer = torch.as_tensor(prior).expand(len(sizes))
    prior_diag = torch.cat([v.expand(s) for v, s in zip(prior_per_layer, sizes)])
    P = la.H.to_matrix() * la._H_factor + torch.diag(prior_diag)
    S = torch.linalg.inv(P)

    _assert_close(la.posterior_precision.to_matrix(), P)
    _assert_close(la.log_det_posterior_precision, torch.logdet(P))

    Js, _ = la.backend.jacobians(torch.randn(5, N_IN))
    cov = torch.einsum("nkp,pq,mlq->nkml", Js, S, Js)

    _assert_close(la.functional_covariance(Js), cov.reshape(5 * n_out, 5 * n_out))
    _assert_close(la.functional_variance(Js), torch.einsum("nknl->nkl", cov))

    value = la.mean + torch.randn(len(prior_diag))
    delta = value - la.mean

    _assert_close(la.square_norm(value), delta @ P @ delta)

    samples = la.sample(200_000)

    torch.testing.assert_close(samples.T.cov(), S, rtol=0, atol=3e-2 * S.abs().max())


@pytest.mark.parametrize("cls", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize("prior", [0.8, torch.tensor([1.5, 3.0]), "diag"])
@pytest.mark.parametrize("prior_mean", ["zero", "random"])
def test_prior_structures_in_marglik(cls, prior, prior_mean):
    if cls is KronLaplace and prior == "diag":
        pytest.skip("Kron takes a scalar or per-layer prior only")

    model = nn.Linear(N_IN, 2)
    X, y = _data(GaussianLikelihood(), 2)
    n_params = 8
    prior = (
        torch.rand(n_params) + 0.5
        if prior == "diag"
        else torch.as_tensor(prior, dtype=torch.float64)
    )
    mu = torch.randn(n_params) if prior_mean == "random" else torch.zeros(n_params)

    la = cls(model, GaussianLikelihood(), prior_precision=prior, prior_mean=mu)
    la.fit(_loader(X, y))

    if prior.numel() == 2:
        prior_diag = prior.repeat_interleave(torch.tensor([6, 2]))
    else:
        prior_diag = prior.expand(n_params)

    if cls is KronLaplace:
        H = la.H.to_matrix()
    elif cls is DiagLaplace:
        H = torch.diag(la.H)
    else:
        H = la.H

    P = H + torch.diag(prior_diag)
    log_det = P.diagonal().log().sum() if cls is DiagLaplace else torch.logdet(P)
    delta = la.mean - mu
    lml = la.log_likelihood - 0.5 * (
        log_det - prior_diag.log().sum() + delta @ (prior_diag * delta)
    )

    _assert_close(la.prior_precision_diag, prior_diag)
    _assert_close(la.log_marginal_likelihood(), lml)


@pytest.mark.parametrize("cls", [FullLaplace, KronLaplace, DiagLaplace])
def test_marglik_gradient_matches_finite_differences(cls):
    model = _mlp(2)
    X, y = _data(GaussianLikelihood(), 2)
    la = cls(model, GaussianLikelihood())
    la.fit(_loader(X, y))

    def lml(log_prior, log_sigma):
        return la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())

    log_prior = torch.tensor(0.3, requires_grad=True)
    log_sigma = torch.tensor(-0.2, requires_grad=True)
    grads = torch.stack(
        torch.autograd.grad(lml(log_prior, log_sigma), [log_prior, log_sigma])
    )

    eps = 1e-5

    with torch.no_grad():
        fd = torch.stack(
            [
                (lml(log_prior + eps, log_sigma) - lml(log_prior - eps, log_sigma))
                / (2 * eps),
                (lml(log_prior, log_sigma + eps) - lml(log_prior, log_sigma - eps))
                / (2 * eps),
            ]
        )

    torch.testing.assert_close(grads, fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "full_cls,ll_cls",
    [
        (FullLaplace, FullLLLaplace),
        (KronLaplace, KronLLLaplace),
        (DiagLaplace, DiagLLLaplace),
    ],
)
@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_last_layer_equals_frozen_body(full_cls, ll_cls, task, n_out):
    model = _mlp(n_out)
    X, y = _data(task, n_out)

    ll = ll_cls(copy.deepcopy(model), task, prior_precision=1.7)
    ll.fit(_loader(X, y))

    frozen = copy.deepcopy(model)

    for p in frozen[0].parameters():
        p.requires_grad_(False)

    full = full_cls(frozen, task, prior_precision=1.7)
    full.fit(_loader(X, y))

    _assert_close(ll.log_marginal_likelihood(), full.log_marginal_likelihood())

    X_test = torch.randn(4, N_IN)

    if isinstance(task, GaussianLikelihood):
        _assert_close(ll(X_test)[1], full(X_test)[1])
    else:
        _assert_close(ll(X_test), full(X_test))


@pytest.mark.parametrize(
    "subnet_cls,ll_cls",
    [(FullSubnetLaplace, FullLLLaplace), (DiagSubnetLaplace, DiagLLLaplace)],
)
def test_subnet_on_last_layer_equals_last_layer(subnet_cls, ll_cls):
    model = _mlp(3)
    X, y = _data(CategoricalLikelihood(), 3)

    # The first layer has 16 parameters, the last layer the remaining 15
    sub = subnet_cls(
        copy.deepcopy(model),
        CategoricalLikelihood(),
        subnetwork_indices=torch.arange(16, 31),
        prior_precision=1.7,
    )
    sub.fit(_loader(X, y))

    ll = ll_cls(copy.deepcopy(model), CategoricalLikelihood(), prior_precision=1.7)
    ll.fit(_loader(X, y))

    X_test = torch.randn(4, N_IN)

    _assert_close(sub.log_marginal_likelihood(), ll.log_marginal_likelihood())
    _assert_close(sub(X_test), ll(X_test))


@pytest.mark.parametrize("backend", GGN_BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_bernoulli_ggn(backend, weighted):
    model = _mlp(2)
    X = torch.randn(N_DATA, N_IN)
    y = torch.randint(2, (N_DATA, 2)).double()
    w = torch.tensor([0.5, 2.0]) if weighted else None
    pw = torch.tensor([3.0, 0.7]) if weighted else None

    f, theta = _flat(copy.deepcopy(model))
    out = f(theta, X)
    J = jacrev(f)(theta, X)
    s = out.sigmoid()
    h = s * (1 - s)

    if weighted:
        h = w * (pw * y + 1 - y) * h

    P = torch.einsum("nkp,nk,nkq->pq", J, h, J) + 1.3 * torch.eye(len(theta))
    nll = nn.functional.binary_cross_entropy_with_logits(
        out, y, weight=w, pos_weight=pw, reduction="sum"
    )
    lml = -nll - 0.5 * (
        torch.logdet(P) - len(theta) * math.log(1.3) + 1.3 * theta @ theta
    )

    likelihood = (
        nn.BCEWithLogitsLoss(weight=w, pos_weight=pw)
        if weighted
        else BernoulliLikelihood()
    )
    la = FullLaplace(model, likelihood, prior_precision=1.3, backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.posterior_precision, P)
    _assert_close(la.log_marginal_likelihood(), lml)


@pytest.mark.parametrize("n_out", [1, 2])
@pytest.mark.parametrize("sigma,prior", [(1.0, 1.0), (0.6, 2.3)])
def test_functional_laplace_predictive_equals_full(n_out, sigma, prior):
    # On all the training data, the GP view and the GGN weight-space view of the
    # linearized network give the same predictive
    model = _mlp(n_out)
    X, y = _data(GaussianLikelihood(), n_out)

    full = FullLaplace(
        copy.deepcopy(model),
        GaussianLikelihood(),
        sigma_noise=sigma,
        prior_precision=prior,
    )
    full.fit(_loader(X, y))

    gp = FunctionalLaplace(
        copy.deepcopy(model),
        GaussianLikelihood(),
        n_subset=N_DATA,
        sigma_noise=sigma,
        prior_precision=prior,
    )
    gp.fit(_loader(X, y))

    X_test = torch.randn(4, N_IN)

    for joint in [False, True]:
        mean_full, var_full = full(X_test, joint=joint)
        mean_gp, var_gp = gp(X_test, joint=joint)

        _assert_close(mean_gp, mean_full)
        _assert_close(var_gp, var_full)

    _assert_close(gp.log_det_ratio, full.log_det_ratio)


def _train_to_map(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    prior: float,
    T: float = 1.0,
) -> None:
    # The MAP of the tempered posterior p(D | theta)^(1/T) p(theta)
    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=5000,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = _nll(GaussianLikelihood(), model(X), y, sigma) / T
        loss = loss + 0.5 * prior * sum((p**2).sum() for p in model.parameters())
        loss.backward()

        return loss

    for _ in range(5):
        opt.step(closure)


@pytest.mark.parametrize("n_out", [1, 2])
@pytest.mark.parametrize("T", [1.0, 0.5])
def test_functional_laplace_marglik_equals_full_at_map(n_out, T):
    # At the MAP, the weight-space Laplace evidence of the linearized network is
    # exact, and so is the GP evidence on all the data, also when tempered
    sigma, prior = 0.6, 2.0
    model = _mlp(n_out)
    X, y = _data(GaussianLikelihood(), n_out)
    _train_to_map(model, X, y, sigma, prior, T)

    full = FullLaplace(
        copy.deepcopy(model),
        GaussianLikelihood(),
        sigma_noise=sigma,
        prior_precision=prior,
        temperature=T,
    )
    full.fit(_loader(X, y))

    gp = FunctionalLaplace(
        copy.deepcopy(model),
        GaussianLikelihood(),
        n_subset=N_DATA,
        sigma_noise=sigma,
        prior_precision=prior,
        temperature=T,
    )
    gp.fit(_loader(X, y))

    torch.testing.assert_close(
        gp.log_marginal_likelihood(),
        full.log_marginal_likelihood(),
        rtol=1e-5,
        atol=1e-6,
    )


def _bernoulli_data(n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(N_DATA, N_IN)

    return X, torch.randint(2, (N_DATA, n_out)).double()


@pytest.mark.parametrize("n_out", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_functional_laplace_bernoulli_predictive_equals_full(n_out, weighted):
    # The Bernoulli output Hessian is diagonal, so the diagonal noise of the GP is
    # exact, and on all the data the GP and the weight-space GGN views agree
    model = _mlp(n_out)
    X, y = _bernoulli_data(n_out)
    likelihood = (
        BernoulliLikelihood(pos_weight=torch.linspace(3.0, 0.7, n_out))
        if weighted
        else BernoulliLikelihood()
    )

    full = FullLaplace(copy.deepcopy(model), likelihood, prior_precision=1.3)
    full.fit(_loader(X, y))

    gp = FunctionalLaplace(
        copy.deepcopy(model),
        likelihood,
        n_subset=N_DATA,
        prior_precision=1.3,
        independent_outputs=False,
    )
    gp.fit(_loader(X, y))

    X_test = torch.randn(4, N_IN)

    for joint in [False, True]:
        mean_full, var_full = full._glm_predictive_distribution(X_test, joint=joint)
        mean_gp, var_gp = gp.predictive_moments(X_test, joint=joint)

        _assert_close(mean_gp, mean_full)
        _assert_close(var_gp, var_full)

    _assert_close(gp(X_test, link_approx="probit"), full(X_test, link_approx="probit"))
    _assert_close(gp.log_det_ratio, full.log_det_ratio)


@pytest.mark.parametrize("n_out", [1, 2])
@pytest.mark.parametrize("T", [1.0, 0.5])
def test_functional_laplace_bernoulli_marglik_equals_full_at_map(n_out, T):
    # At the MAP, the network's offset J theta is K times the output gradient, so
    # the GP scatter term equals the prior term theta' P0 theta of the weight space
    prior = 2.0
    model = _mlp(n_out)
    X, y = _bernoulli_data(n_out)

    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=5000,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(X), y, reduction="sum"
        )
        loss = loss / T + 0.5 * prior * sum((p**2).sum() for p in model.parameters())
        loss.backward()

        return loss

    for _ in range(5):
        opt.step(closure)

    full = FullLaplace(
        copy.deepcopy(model),
        BernoulliLikelihood(),
        prior_precision=prior,
        temperature=T,
    )
    full.fit(_loader(X, y))

    gp = FunctionalLaplace(
        copy.deepcopy(model),
        BernoulliLikelihood(),
        n_subset=N_DATA,
        prior_precision=prior,
        temperature=T,
        independent_outputs=False,
    )
    gp.fit(_loader(X, y))

    # The scatter term adds a small jitter to the kernel
    torch.testing.assert_close(
        gp.log_marginal_likelihood(),
        full.log_marginal_likelihood(),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [1.0, 0.25])
def test_marglik_training_reaches_the_tempered_map(T):
    # With the hyperparameters fixed, the training of a linear model ends at the
    # ridge solution of the tempered posterior p(D | theta)^(1/T) p(theta)
    sigma, prior = 0.5, 3.0
    model = nn.Linear(N_IN, 1, bias=False)
    X, y = _data(GaussianLikelihood(), 1)

    marglik_training(
        model,
        DataLoader(TensorDataset(X, y), batch_size=N_DATA),
        GaussianLikelihood(),
        hessian_structure="diag",
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1},
        n_epochs=2000,
        n_epochs_burnin=2001,
        prior_structure="scalar",
        prior_prec_init=prior,
        sigma_noise_init=sigma,
        temperature=T,
        fix_sigma_noise=True,
    )

    A = X.T @ X / (sigma**2 * T) + prior * torch.eye(N_IN)
    theta = torch.linalg.solve(A, X.T @ y.squeeze(-1) / (sigma**2 * T))

    torch.testing.assert_close(
        model.weight.detach().squeeze(0), theta, rtol=1e-5, atol=1e-7
    )


def _kfac_reference(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    task: Likelihood,
    curvature: str,
    sigma: float,
) -> torch.Tensor:
    """Dense KFAC of `_mlp`, built from its layers by hand.

    For each linear layer, KFAC is \\(G \\otimes A\\) with \\(A = \\sum_n a_n a_n^\\top\\)
    over the layer inputs \\(a_n\\) and \\(G\\) the average over the data of the
    curvature that reaches the layer's outputs. The bias is its own block, the sum of
    that curvature.
    """
    lin1, lin2 = model[0], model[2]

    with torch.no_grad():
        a2 = torch.tanh(lin1(X))
        out = lin2(a2)

    n, n_out = out.shape

    # Jacobians of the outputs with respect to the outputs of each layer
    B2 = torch.eye(n_out).expand(n, n_out, n_out)
    B1 = lin2.weight.detach()[None] * (1 - a2**2)[:, None, :]

    if curvature == "ggn":
        if isinstance(task, GaussianLikelihood):
            L = torch.eye(n_out).expand(n, n_out, n_out) / sigma**2
        else:
            p = out.softmax(-1)
            L = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]

        def per_example(B: torch.Tensor) -> torch.Tensor:
            return torch.einsum("nko,nkl,nlp->nop", B, L, B)
    else:
        if isinstance(task, GaussianLikelihood):
            r = (out - y) / sigma**2
        else:
            r = out.softmax(-1) - nn.functional.one_hot(y, n_out)

        def per_example(B: torch.Tensor) -> torch.Tensor:
            g = torch.einsum("nk,nko->no", r, B)
            return g[:, :, None] * g[:, None, :]

    blocks = []

    for a, B in [(X, B1), (a2, B2)]:
        C = per_example(B)
        blocks += [torch.kron(C.mean(0), a.T @ a), C.sum(0)]

    return torch.block_diag(*blocks)


@pytest.mark.parametrize(
    "backend,curvature",
    [(b, "ggn") for b in GGN_BACKENDS] + [(CurvlinopsEF, "ef"), (AsdlEF, "ef")],
)
@pytest.mark.parametrize(
    "task,n_out,sigma",
    [
        (GaussianLikelihood(), 1, 1.0),
        (GaussianLikelihood(), 3, 0.7),
        (CategoricalLikelihood(), 3, 1.0),
    ],
    ids=lik_id,
)
def test_kron_matches_kfac_reference(backend, curvature, task, n_out, sigma):
    # On a two-layer net KFAC is no longer the GGN, but every backend must compute
    # the same KFAC. BackPACK has no Kronecker-factored EF.
    model = _mlp(n_out)
    X, y = _data(task, n_out)
    ref = _kfac_reference(copy.deepcopy(model), X, y, task, curvature, sigma)

    la = KronLaplace(model, task, sigma_noise=sigma, backend=backend)
    la.fit(_loader(X, y))

    _assert_close(la.H.to_matrix() * la._H_factor, ref)


@pytest.mark.parametrize("backend", GGN_BACKENDS)
@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_single_layer_full_kron_diag_agree(backend, task, n_out):
    # On one linear layer with a bias, the diagonal is the diagonal of the GGN. For
    # the MSE, the output curvature is constant, so KFAC is exactly the GGN without
    # the weight-bias cross terms. For the cross-entropy it isn't.
    model = nn.Linear(N_IN, n_out)
    X, y = _data(task, n_out)

    full = FullLaplace(copy.deepcopy(model), task, sigma_noise=1.0, backend=backend)
    full.fit(_loader(X, y))
    H = full.H

    diag = DiagLaplace(copy.deepcopy(model), task, backend=backend)
    diag.fit(_loader(X, y))

    _assert_close(diag.H, H.diagonal())

    if isinstance(task, GaussianLikelihood):
        kron = KronLaplace(copy.deepcopy(model), task, backend=backend)
        kron.fit(_loader(X, y))

        n_weight = N_IN * n_out
        blocks = torch.block_diag(H[:n_weight, :n_weight], H[n_weight:, n_weight:])

        _assert_close(kron.H.to_matrix(), blocks)


@pytest.mark.parametrize("backend", HESSIAN_BACKENDS)
@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_last_layer_hessian(backend, task, n_out):
    # The Hessian of the last layer is that of the net with the body frozen
    model = _mlp(n_out)
    X, y = _data(task, n_out)

    ll = FullLLLaplace(
        copy.deepcopy(model), task, prior_precision=50.0, backend=backend
    )
    ll.fit(_loader(X, y))

    frozen = copy.deepcopy(model)

    for p in frozen[0].parameters():
        p.requires_grad_(False)

    ref = _reference(frozen, X, y, task, "hessian", prior=50.0)

    _assert_close(ll.posterior_precision, ref["P"])
    _assert_close(ll.log_marginal_likelihood(), ref["lml"])


@pytest.mark.asdfghjkl
@pytest.mark.skipif(
    find_spec("asdfghjkl") is None, reason="The old asdfghjkl is not installed"
)
@pytest.mark.parametrize(
    "task,n_out", [(GaussianLikelihood(), 2), (CategoricalLikelihood(), 3)], ids=lik_id
)
def test_lowrank_eigenvalues_sum_over_batches(task, n_out):
    # asdfghjkl averages its Hessian-vector products over the batches, so the
    # eigenvalues must be scaled back by the number of batches. The power iteration
    # starts from the same random vectors, but summing in another order moves where
    # it stops, so the eigenvalues agree only loosely.
    model = _mlp(n_out)
    X, y = _data(task, n_out)
    eigvals = []

    for batch_size in [N_DATA, 5]:
        torch.manual_seed(0)
        la = LowRankLaplace(copy.deepcopy(model), task, backend_kwargs={"low_rank": 2})
        la.fit(DataLoader(TensorDataset(X, y), batch_size=batch_size))
        eigvals.append(la.H[1])

    torch.testing.assert_close(eigvals[1], eigvals[0], rtol=5e-2, atol=0)

    # The power iteration stops while it still converges slowly, so the top
    # eigenvalue is off by a few percent. That still catches a wrong scale.
    P = _reference(model, X, y, task, "hessian", prior=1.0)["P"]
    top = torch.linalg.eigvalsh(P - torch.eye(len(P)))[-1]

    torch.testing.assert_close(eigvals[0][0], top, rtol=1e-1, atol=0)


@pytest.mark.asdfghjkl
@pytest.mark.skipif(
    find_spec("asdfghjkl") is None, reason="The old asdfghjkl is not installed"
)
def test_lowrank_warns_for_eigenvectors_that_are_not_orthonormal():
    from laplace.curvature.asdfghjkl import _warn_if_not_orthonormal

    Q, _ = torch.linalg.qr(torch.randn(10, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_not_orthonormal(Q)

    # An unconverged power iteration returns its last iterate unnormalized
    Q[:, 1] *= 2.0

    with pytest.warns(UserWarning, match="didn't converge"):
        _warn_if_not_orthonormal(Q)
