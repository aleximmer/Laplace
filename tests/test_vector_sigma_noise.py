"""A known noise std per output, set on `GaussianLikelihood(sigma_noise=...)`."""

import pytest
import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.curvature import CurvlinopsEF, CurvlinopsGGN
from laplace.likelihoods import GaussianLikelihood
from tests.utils import AsdlGGN, BackPackGGN

pytestmark = [pytest.mark.default_dtype(torch.float64), pytest.mark.seed(123)]

SIGMA = torch.tensor([0.5, 2.0], dtype=torch.float64)
PRIOR_PREC = 1.7


@pytest.fixture
def model():
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2))


@pytest.fixture
def data():
    torch.manual_seed(1)
    X, y = torch.randn(20, 3), torch.randn(20, 2)

    # The batches don't divide the data, so that sums over batches are exercised
    return X, y, DataLoader(TensorDataset(X, y), batch_size=7)


def _jacobians(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    # `(N, K, P)`, with the parameters in the order of `model.parameters()`
    params = {k: v.detach() for k, v in model.named_parameters()}

    def f(p, x):
        return functional_call(model, p, (x[None],))[0]

    Js = vmap(jacrev(f), in_dims=(None, 0))(params, X)

    return torch.cat([J.flatten(start_dim=2) for J in Js.values()], dim=2)


def _dense(H) -> torch.Tensor:
    if hasattr(H, "to_matrix"):
        return H.to_matrix()

    return torch.diag(H) if H.ndim == 1 else H


def _likelihood(sigma) -> tuple[GaussianLikelihood | str, dict]:
    # A vector goes into the likelihood, a scalar into `Laplace(sigma_noise=...)`
    if torch.is_tensor(sigma):
        return GaussianLikelihood(sigma_noise=sigma), {}

    return "regression", {"sigma_noise": sigma}


def _normalizer(n: int, sigma: torch.Tensor) -> torch.Tensor:
    return n * torch.log(sigma * (2 * torch.pi) ** 0.5).sum()


def _laplace(model, likelihood, structure, backend=CurvlinopsGGN, **kwargs):
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure=structure,
        prior_precision=PRIOR_PREC,
        backend=backend,
        **kwargs,
    )


def _fit(model, loader, sigma, structure, backend=CurvlinopsGGN):
    likelihood, options = _likelihood(sigma)
    la = _laplace(model, likelihood, structure, backend, **options)
    la.fit(loader)

    return la


def _gaussian_log_likelihood(model, X, y, sigma):
    with torch.no_grad():
        f = model(X)

    return torch.distributions.Normal(f, sigma).log_prob(y).sum()


# Equal entries give the scalar noise. Only Curvlinops honours the noise in KFAC.
STRUCTURE_BACKENDS = [
    (structure, backend)
    for structure in ("full", "diag")
    for backend in (CurvlinopsGGN, BackPackGGN, AsdlGGN)
] + [("kron", CurvlinopsGGN)]


@pytest.mark.parametrize("structure,backend", STRUCTURE_BACKENDS)
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_equal_vector_matches_scalar(model, data, structure, backend, temperature):
    _, _, loader = data
    s = 0.7

    scalar = _laplace(
        model, "regression", structure, backend, sigma_noise=s, temperature=temperature
    )
    vector = _laplace(
        model,
        GaussianLikelihood(sigma_noise=torch.full((2,), s)),
        structure,
        backend,
        temperature=temperature,
    )
    scalar.fit(loader)
    vector.fit(loader)

    # H * factor is the same, for the factor 1 / (T s^2) of the scalar noise
    assert torch.allclose(
        _dense(scalar.H) * scalar._H_factor,
        _dense(vector.H) * vector._H_factor,
        rtol=1e-5,
        atol=1e-8,
    )
    assert torch.allclose(
        _dense(scalar.posterior_precision),
        _dense(vector.posterior_precision),
        rtol=1e-5,
        atol=1e-8,
    )
    assert torch.allclose(
        scalar.log_marginal_likelihood(), vector.log_marginal_likelihood(), rtol=1e-6
    )

    X = torch.randn(5, 3)
    s_mean, s_var = scalar(X)
    v_mean, v_var = vector(X)
    assert torch.allclose(s_mean, v_mean)
    assert torch.allclose(s_var, v_var, rtol=1e-5)

    assert vector.sigma_noise == 1


@pytest.mark.parametrize(
    "subset,structure",
    [
        ("last_layer", "full"),
        ("last_layer", "diag"),
        ("last_layer", "kron"),
        ("subnetwork", "full"),
        ("subnetwork", "diag"),
    ],
)
def test_equal_vector_matches_scalar_for_subsets(model, data, subset, structure):
    _, _, loader = data
    s = 0.7
    kwargs = {"subset_of_weights": subset, "hessian_structure": structure}

    if subset == "subnetwork":
        kwargs["subnetwork_indices"] = torch.arange(0, 24, 2)

    scalar = Laplace(model, "regression", sigma_noise=s, **kwargs)
    vector = Laplace(
        model, GaussianLikelihood(sigma_noise=torch.full((2,), s)), **kwargs
    )

    for la in (scalar, vector):
        la.fit(loader)

    assert torch.allclose(
        _dense(scalar.posterior_precision),
        _dense(vector.posterior_precision),
        rtol=1e-5,
        atol=1e-8,
    )
    assert torch.allclose(
        scalar.log_marginal_likelihood(), vector.log_marginal_likelihood()
    )


# Different entries against brute force


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("backend", [CurvlinopsGGN, BackPackGGN, AsdlGGN])
def test_full_matches_bruteforce(model, data, backend, temperature):
    X, y, loader = data
    la = _laplace(
        model,
        GaussianLikelihood(sigma_noise=SIGMA),
        "full",
        backend,
        temperature=temperature,
    )
    la.fit(loader)

    Js = _jacobians(model, X)
    H = torch.einsum("nkp,k,nkq->pq", Js, 1 / SIGMA**2, Js)
    P = H.shape[0]

    assert torch.allclose(la.H, H, rtol=1e-5, atol=1e-8)

    posterior_precision = H / temperature + PRIOR_PREC * torch.eye(P)
    assert torch.allclose(la.posterior_precision, posterior_precision, rtol=1e-5)

    # The loss is the weighted one, and the likelihood the Gaussian one with the stds
    log_lik = _gaussian_log_likelihood(model, X, y, SIGMA)
    assert torch.allclose(la.loss, -(log_lik + _normalizer(len(X), SIGMA)))

    if temperature == 1.0:
        assert torch.allclose(la.log_likelihood, log_lik)

    # The marginal likelihood of the Laplace approximation
    log_det_ratio = torch.logdet(posterior_precision) - P * torch.log(
        torch.tensor(PRIOR_PREC)
    )
    scatter = PRIOR_PREC * la.mean.square().sum()
    expected = la.log_likelihood - 0.5 * (log_det_ratio + scatter)
    assert torch.allclose(la.log_marginal_likelihood(), expected, rtol=1e-5)

    # The predictive variance
    X_test = torch.randn(4, 3)
    J_test = _jacobians(model, X_test)
    f_var = torch.einsum(
        "nkp,pq,nlq->nkl", J_test, torch.linalg.inv(posterior_precision), J_test
    )
    _, la_var = la(X_test)
    assert torch.allclose(la_var, f_var, rtol=1e-5)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, BackPackGGN, AsdlGGN])
def test_diag_matches_bruteforce(model, data, backend):
    X, _, loader = data
    la = _fit(model, loader, SIGMA, "diag", backend)

    Js = _jacobians(model, X)
    H = torch.einsum("nkp,k,nkp->p", Js, 1 / SIGMA**2, Js)

    assert torch.allclose(la.H, H, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, BackPackGGN, AsdlGGN])
def test_subnetwork_matches_bruteforce(model, data, backend):
    X, _, loader = data
    indices = torch.arange(0, 24, 2)

    for structure in ("full", "diag"):
        la = Laplace(
            model,
            GaussianLikelihood(sigma_noise=SIGMA),
            "subnetwork",
            structure,
            subnetwork_indices=indices,
            backend=backend,
        )
        la.fit(loader)

        Js = _jacobians(model, X)[..., indices]
        pattern = "nkp,k,nkq->pq" if structure == "full" else "nkp,k,nkp->p"
        H = torch.einsum(pattern, Js, 1 / SIGMA**2, Js)

        assert torch.allclose(la.H, H, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("structure", ["full", "diag"])
def test_last_layer_matches_bruteforce(model, data, structure):
    X, y, loader = data
    la = Laplace(
        model,
        GaussianLikelihood(sigma_noise=SIGMA),
        "last_layer",
        structure,
        prior_precision=PRIOR_PREC,
    )
    la.fit(loader)

    # The Jacobians of the last layer: features for the weights, then the bias
    with torch.no_grad():
        phi = model[:-1](X)
    N = phi.shape[0]
    eye = torch.eye(2)
    Js = torch.cat(
        [torch.einsum("nd,kc->nkcd", phi, eye).reshape(N, 2, -1), eye.expand(N, 2, 2)],
        dim=2,
    )
    H = torch.einsum("nkp,k,nkq->pq", Js, 1 / SIGMA**2, Js)

    assert torch.allclose(
        la.H, H if structure == "full" else H.diagonal(), rtol=1e-5, atol=1e-8
    )
    assert torch.allclose(
        la.log_likelihood, _gaussian_log_likelihood(model, X, y, SIGMA)
    )


class _SelectOutput(nn.Module):
    # One output of the model: `(B, K)` -> `(B, 1)`
    def __init__(self, model: nn.Module, k: int) -> None:
        super().__init__()
        self.model = model
        self.k = k

    def forward(self, x):
        return self.model(x)[..., self.k : self.k + 1]


def _single_output_kron(model, X, y):
    # Dense KFAC of the plain Gaussian likelihood for the one output of `model`
    backend = CurvlinopsGGN(model, GaussianLikelihood())
    _, kron = backend.kron(X, y, N=X.shape[0])

    return kron.to_matrix()


def test_kron_matches_sum_of_per_output_krons(model, data):
    X, y, loader = data
    la = _fit(model, loader, SIGMA, "kron")

    # The input factor is shared, so the factors of the outputs add up in the output one
    total = sum(
        _single_output_kron(_SelectOutput(model, k), X, y[:, k : k + 1]) / SIGMA[k] ** 2
        for k in range(2)
    )

    assert torch.allclose(la.H.to_matrix(), total, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("structure", ["full", "diag"])
def test_ef_matches_bruteforce(model, data, structure):
    X, y, loader = data
    la = _fit(model, loader, SIGMA, structure, CurvlinopsEF)

    with torch.no_grad():
        residual = model(X) - y

    # One gradient per datum of the summed squared error, which the EF scales by the
    # loss factor, like for the scalar noise
    Gs = 2 * torch.einsum("nkp,nk->np", _jacobians(model, X), residual / SIGMA**2)
    H = 0.5 * Gs.T @ Gs

    expected = H if structure == "full" else H.diagonal()
    assert torch.allclose(la.H, expected, rtol=1e-5, atol=1e-8)


def test_mc_fisher_approaches_the_ggn(model, data):
    X, y, _ = data

    # The dense Curvlinops operator, and the torch.func path of a subnetwork
    for indices in (None, torch.arange(0, 24, 2)):
        backend = CurvlinopsGGN(
            model,
            GaussianLikelihood(sigma_noise=SIGMA),
            stochastic=True,
            subnetwork_indices=indices,
        )

        # Curvlinops takes `mc_samples`, the torch.func path `num_samples`
        backend.num_samples = 3000
        _, H_mc = backend.full(X, y, mc_samples=3000)

        Js = _jacobians(model, X)

        if indices is not None:
            Js = Js[..., indices]

        H = torch.einsum("nkp,k,nkq->pq", Js, 1 / SIGMA**2, Js)

        assert (H_mc - H).abs().max() < 0.15 * H.abs().max()


@pytest.mark.parametrize("structure", ["full", "diag", "kron"])
def test_loss_is_scaled_by_the_stds(model, data, structure):
    X, y, loader = data
    la = _fit(model, loader, SIGMA, structure)

    with torch.no_grad():
        f = model(X)

    assert torch.allclose(la.loss, 0.5 * (((f - y) / SIGMA) ** 2).sum())


# Errors


def test_constructor_redirects_a_vector_to_the_likelihood(model):
    with pytest.raises(ValueError, match="GaussianLikelihood"):
        Laplace(model, "regression", sigma_noise=SIGMA)

    # One entry is a scalar
    la = Laplace(model, "regression", sigma_noise=torch.tensor([0.5]))
    assert la.sigma_noise == 0.5


def test_noise_of_the_approximation_must_stay_one(model, data):
    _, _, loader = data
    lik = GaussianLikelihood(sigma_noise=SIGMA)

    with pytest.raises(ValueError, match="stay 1"):
        Laplace(model, lik, sigma_noise=0.5)

    la = _fit(model, loader, SIGMA, "full")

    with pytest.raises(ValueError, match="stay 1"):
        la.sigma_noise = 2.0

    with pytest.raises(ValueError, match="stay 1"):
        la.sigma_noise = torch.tensor(0.3)

    with pytest.raises(ValueError, match="stay 1"):
        la.log_marginal_likelihood(sigma_noise=torch.tensor(0.3))

    # The failed attempts left the approximation as it was
    assert la.sigma_noise == 1
    la.log_marginal_likelihood(sigma_noise=torch.tensor(1.0))

    # A vector for an approximation without one is redirected to the likelihood
    plain = _laplace(model, "regression", "full")

    with pytest.raises(ValueError, match="GaussianLikelihood"):
        plain.sigma_noise = SIGMA


@pytest.mark.parametrize("backend", [BackPackGGN, AsdlGGN])
def test_kron_of_other_backends_raises(model, data, backend):
    _, _, loader = data
    la = _laplace(model, GaussianLikelihood(sigma_noise=SIGMA), "kron", backend)

    with pytest.raises(ValueError, match="sigma_noise"):
        la.fit(loader)


def test_wrong_number_of_outputs_raises(model, data):
    _, _, loader = data

    with pytest.raises(ValueError, match="2 outputs"):
        _fit(model, loader, torch.ones(3), "diag")


# Function space


def _fs_problem(n=12):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    X, y = torch.randn(n, 3), torch.randn(n, 2)

    return model, X, y, DataLoader(TensorDataset(X, y), batch_size=5)


def _functional(model, sigma, independent_outputs, **kwargs):
    likelihood, options = _likelihood(sigma)

    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        n_subset=12,
        independent_outputs=independent_outputs,
        prior_precision=2.0,
        **options,
        **kwargs,
    )


@pytest.mark.parametrize("independent_outputs", [True, False])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_functional_matches_bruteforce(independent_outputs, temperature):
    model, X, y, loader = _fs_problem()
    la = _functional(model, SIGMA, independent_outputs, temperature=temperature)
    la.fit(loader)

    Js = _jacobians(model, X)
    N, K, _ = Js.shape
    s = la.gp_kernel_prior_variance
    mean = torch.cat([p.detach().flatten() for p in model.parameters()])

    with torch.no_grad():
        f = model(X)

    mu = y - (f + torch.einsum("nkp,p->nk", Js, -mean))

    if independent_outputs:
        kernels = [
            s * Js[:, k] @ Js[:, k].T + SIGMA[k] ** 2 * torch.eye(N) for k in range(K)
        ]
        log_det = sum(torch.logdet(k) for k in kernels)
        scatter = sum(
            mu[:, k] @ torch.linalg.solve(kernels[k], mu[:, k]) for k in range(K)
        )
    else:
        Jf = Js.reshape(N * K, -1)
        kernel = s * Jf @ Jf.T + torch.diag((SIGMA**2).repeat(N))
        log_det = torch.logdet(kernel)
        scatter = mu.reshape(-1) @ torch.linalg.solve(kernel, mu.reshape(-1))

    assert torch.allclose(la.log_det_ratio, log_det)
    assert torch.allclose(la.scatter, scatter)

    # The loss holds the stds, so at T = 1 the log likelihood is the Gaussian one
    assert torch.allclose(la.loss, 0.5 * (((f - y) / SIGMA) ** 2).sum())
    assert torch.allclose(
        la.log_likelihood, -la.loss / temperature - _normalizer(N, SIGMA)
    )

    if temperature == 1.0:
        assert torch.allclose(
            la.log_likelihood, _gaussian_log_likelihood(model, X, y, SIGMA)
        )


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_functional_predictive_matches_bruteforce(temperature):
    model, X, y, loader = _fs_problem()
    la = _functional(model, SIGMA, False, temperature=temperature)
    la.fit(loader)

    X_test = torch.randn(4, 3)
    Js, J_test = _jacobians(model, X), _jacobians(model, X_test)
    N, K, _ = Js.shape
    s = la.gp_kernel_prior_variance

    Jf, Jt = Js.reshape(N * K, -1), J_test.reshape(4 * K, -1)
    kernel = s * Jf @ Jf.T + torch.diag((SIGMA**2).repeat(N) * temperature)
    cross = s * Jt @ Jf.T
    expected = s * Jt @ Jt.T - cross @ torch.linalg.solve(kernel, cross.T)

    _, f_cov = la(X_test, joint=True)

    assert torch.allclose(f_cov, expected)


@pytest.mark.parametrize("independent_outputs", [True, False])
def test_functional_equal_vector_matches_scalar(independent_outputs):
    model, X, _, loader = _fs_problem()
    scalar = _functional(model, 0.7, independent_outputs)
    vector = _functional(model, torch.full((2,), 0.7), independent_outputs)

    for la in (scalar, vector):
        la.fit(loader)

    assert torch.allclose(
        scalar.log_marginal_likelihood(), vector.log_marginal_likelihood()
    )

    s_mean, s_var = scalar(X)
    v_mean, v_var = vector(X)
    assert torch.allclose(s_mean, v_mean)
    assert torch.allclose(s_var, v_var)


def test_functional_wrong_number_of_outputs_raises():
    model, _, _, loader = _fs_problem()

    with pytest.raises(ValueError, match="2 outputs"):
        _functional(model, torch.ones(3), False).fit(loader)


def _ella(model, sigma, **kwargs):
    likelihood, options = _likelihood(sigma)

    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="nystrom",
        subsample_size=6,
        n_eigenvalues=4,
        **options,
        **kwargs,
    )


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_ella_matches_bruteforce(temperature):
    model, X, _, loader = _fs_problem()
    ella = _ella(model, SIGMA, temperature=temperature, prior_precision=1.5)
    ella.fit(loader)

    features, _ = ella._features(X)
    gram = torch.einsum("nkd,k,nke->de", features, 1 / SIGMA**2, features)
    precision = gram / temperature + 1.5 * torch.eye(4)

    assert torch.allclose(ella.chol_precision @ ella.chol_precision.T, precision)

    _, covariance = ella.predictive_moments(X)
    solved = torch.linalg.solve(precision, features.transpose(1, 2))
    assert torch.allclose(covariance, features @ solved)


def test_ella_equal_vector_matches_scalar():
    model, X, _, loader = _fs_problem()
    scalar, vector = _ella(model, 0.7), _ella(model, torch.full((2,), 0.7))

    for ella in (scalar, vector):
        ella.fit(loader)

    assert torch.allclose(scalar(X)[1], vector(X)[1])
    assert torch.allclose(
        scalar._validation_nll(loader), vector._validation_nll(loader)
    )


def test_ella_validation_nll_uses_the_stds():
    model, X, y, loader = _fs_problem()
    ella = _ella(model, SIGMA)
    ella.fit(loader)

    mean, covariance = ella.predictive_moments(X)
    variance = covariance.diagonal(dim1=-2, dim2=-1) + SIGMA**2
    expected = 0.5 * (
        (y - mean) ** 2 / variance
        + variance.log()
        + torch.log(torch.tensor(2 * torch.pi))
    )

    assert torch.allclose(ella._validation_nll(loader), expected.sum() / len(X))


def test_ella_wrong_number_of_outputs_raises():
    model, _, _, loader = _fs_problem()

    with pytest.raises(ValueError, match="2 outputs"):
        _ella(model, torch.ones(3)).fit(loader)


def _valla(model, sigma, X, **kwargs):
    likelihood, options = _likelihood(sigma)

    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        inducing_locations=X[:4].clone(),
        **options,
        **kwargs,
    )


def test_valla_fixed_noise_is_not_trained():
    model, X, _, loader = _fs_problem()
    valla = _valla(model, SIGMA, X)

    assert valla.log_noise_variance is None

    valla.fit(loader, iterations=5, lr=0.05, val_loader=loader)

    assert valla.sigma_noise == 1
    assert torch.equal(valla.likelihood.sigma_noise, SIGMA)
    assert all(torch.isfinite(torch.tensor(valla.fit_history_["objective"])))
    assert all(torch.isfinite(torch.tensor(valla.fit_history_["val_nll"])))

    mean, covariance = valla(X)
    assert mean.shape == (12, 2)
    assert covariance.shape == (12, 2, 2)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_valla_objective_uses_the_stds(alpha):
    model, X, y, loader = _fs_problem()
    x_batch, y_batch = next(iter(loader))
    valla = _valla(model, SIGMA, X, alpha=alpha)
    valla.n_data, valla.n_outputs = 12, 2
    valla._init_variational_factor()

    mean, covariance, features, triangular = valla._latent_distribution(x_batch)
    noise = SIGMA**2
    latent = covariance.diagonal(dim1=-2, dim2=-1)
    error = (y_batch - mean).square()

    if alpha == 0:
        energy = (error + latent) / noise
    else:
        energy = (
            error / (noise + alpha * latent)
            + torch.log1p(alpha * latent / noise) / alpha
        )

    log_term = -0.5 * (energy + torch.log(2 * torch.pi * noise)).sum()
    solved = torch.linalg.solve_triangular(triangular.T, features, upper=False)
    kl = triangular.diagonal().abs().log().sum() - 0.5 * solved.square().sum()
    expected = -(12 / y_batch.shape[0]) * log_term + kl

    assert torch.allclose(valla._objective(x_batch, y_batch), expected)


def test_valla_equal_vector_objective_matches_scalar():
    model, X, _, loader = _fs_problem()
    x_batch, y_batch = next(iter(loader))
    objectives = []

    for sigma in (0.7, torch.full((2,), 0.7)):
        valla = _valla(model, sigma, X)
        valla.n_data, valla.n_outputs = 12, 2
        valla._init_variational_factor()
        objectives.append(valla._objective(x_batch, y_batch))

    assert torch.allclose(*objectives)


def test_valla_wrong_number_of_outputs_raises():
    model, X, _, loader = _fs_problem()

    with pytest.raises(ValueError, match="2 outputs"):
        _valla(model, torch.ones(3), X).fit(loader, iterations=1)
