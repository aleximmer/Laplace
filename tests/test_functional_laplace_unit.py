import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace.baselaplace import FunctionalLaplace
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.lllaplace import FunctionalLLLaplace
from laplace.utils.functional_utils import _class_labels
from tests.utils import AsdlGGN, BackPackGGN

pytestmark = [pytest.mark.default_dtype(torch.float64), pytest.mark.seed(240)]


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
def reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return X, y


def test_sod_data_loader(reg_loader, model):
    M = 5
    func_la = FunctionalLaplace(model, "regression", M)
    sod_data_loader = func_la._get_SoD_data_loader(reg_loader)

    first_iter = []
    for x, _ in sod_data_loader:
        first_iter.append(x)
    second_iter = []
    for x, y in sod_data_loader:
        second_iter.append(x)

    first_iter = torch.cat(first_iter, dim=0)
    second_iter = torch.cat(second_iter, dim=0)

    assert torch.allclose(first_iter, second_iter)
    assert len(first_iter) == M


def test_store_K_batch_full_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, "regression", M, independent_outputs=False)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    func_la._init_K_MM()
    # Right now K_MM is initialized with torch.empty. To run this tests we
    #  must set it to zero.
    func_la.K_MM.zero_()  # `*= 0` would keep any NaN or inf in the empty memory

    assert torch.allclose(torch.zeros(size=(M * C, M * C)), func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(4, 4)), 0, 0)
    assert torch.allclose(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        func_la.K_MM,
    )

    func_la._store_K_batch(torch.ones(size=(4, 2)), 0, 1)
    assert torch.allclose(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        ),
        func_la.K_MM,
    )

    func_la._store_K_batch(torch.ones(size=(2, 2)), 1, 1)
    assert torch.allclose(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        func_la.K_MM,
    )


def test_store_K_batch_block_diagonal_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, "regression", M, independent_outputs=True)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    def _check(expected_K_MM):
        for c in range(C):
            assert torch.allclose(expected_K_MM[c], func_la.K_MM[c])

    func_la._init_K_MM()
    # K_MM uses uninitialized storage; multiplying it by zero can preserve NaNs.
    func_la.K_MM = [torch.zeros_like(matrix) for matrix in func_la.K_MM]

    expected = [torch.zeros(size=(M, M)) for _ in range(C)]
    _check(expected)

    func_la._store_K_batch(
        torch.cat([torch.ones(size=(2, 2, 1)), 2 * torch.ones(size=(2, 2, 1))], dim=2),
        0,
        0,
    )
    expected = [
        (c + 1) * torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        for c in range(C)
    ]
    _check(expected)

    func_la._store_K_batch(
        torch.cat([torch.ones(size=(2, 1, 1)), 2 * torch.ones(size=(2, 1, 1))], dim=2),
        0,
        1,
    )
    expected = [
        (c + 1) * torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        for c in range(C)
    ]
    _check(expected)

    func_la._store_K_batch(
        torch.cat([torch.ones(size=(1, 1, 1)), 2 * torch.ones(size=(1, 1, 1))], dim=2),
        1,
        1,
    )
    expected = [
        (c + 1) * torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        for c in range(C)
    ]
    _check(expected)


@pytest.mark.parametrize(
    "kernel_type,jacobians,jacobians_2,expected_full_kernel,expected_block_diagonal_kernel",
    [
        (
            "kernel_batch",
            torch.tensor(
                [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]
            ),
            torch.tensor(
                [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]
            ),
            torch.tensor(
                [
                    [3.0, 6.0, 3.0, 9.0],
                    [6.0, 12.0, 6.0, 18.0],
                    [3.0, 6.0, 3.0, 9.0],
                    [9.0, 18.0, 9.0, 27.0],
                ]
            ),
            torch.tensor([[[3.0, 12.0], [3.0, 18.0]], [[3.0, 18.0], [3.0, 27.0]]]),
        ),
        (
            "kernel_star",
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
                ]
            ),
            torch.tensor(
                [
                    [[3.0, 3.0], [3.0, 3.0]],
                    [[3.0, 6.0], [6.0, 12.0]],
                    [[3.0, 9.0], [9.0, 27.0]],
                ]
            ),
            torch.tensor([[3.0, 3.0], [3.0, 12.0], [3.0, 27.0]]),
        ),
        (
            "kernel_batch_star",
            torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
            ),
            torch.tensor(
                [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]
            ),
            torch.tensor(
                [
                    [[[3.0, 6.0], [3.0, 6.0]], [[3.0, 9.0], [3.0, 9.0]]],
                    [[[1.5, 3.0], [1.5, 3.0]], [[1.5, 4.5], [1.5, 4.5]]],
                ]
            ),
            torch.tensor([[[3.0, 6.0], [3.0, 9.0]], [[1.5, 3.0], [1.5, 4.5]]]),
        ),
    ],
)
def test_gp_kernel(
    mocker,
    reg_Xy,
    model,
    kernel_type,
    jacobians,
    jacobians_2,
    expected_full_kernel,
    expected_block_diagonal_kernel,
):
    X, y = reg_Xy
    func_la = FunctionalLaplace(
        model,
        "regression",
        n_subset=X.shape[0],
        prior_precision=1.0,
        independent_outputs=False,
    )
    func_la.prior_factor_sod = 1.0
    func_la.n_outputs = y.shape[-1]
    if kernel_type == "kernel_batch":
        kernel = func_la._kernel_batch
    elif kernel_type == "kernel_star":
        kernel = func_la._kernel_star
    elif kernel_type == "kernel_batch_star":
        kernel = func_la._kernel_batch_star
    else:
        raise ValueError

    # mocking jacobians
    def mock_jacobians(self, x):
        return jacobians_2, None

    mocker.patch("laplace.baselaplace.FunctionalLaplace._jacobians", mock_jacobians)

    #  mocking prior precision
    mocker.patch(
        "laplace.baselaplace.FunctionalLaplace.prior_precision_diag", torch.ones(3)
    )

    # X does not have an impact since we mock jacobians in mock_jacobians above
    if kernel_type == "kernel_star":
        full_kernel = kernel(jacobians)
    else:
        full_kernel = kernel(jacobians, X)

    assert torch.allclose(
        expected_full_kernel, full_kernel.to(expected_full_kernel.dtype)
    )

    func_la.independent_outputs = True

    if kernel_type == "kernel_star":
        block_diag_kernel = kernel(jacobians)
    else:
        block_diag_kernel = kernel(jacobians, X)
    assert torch.allclose(
        expected_block_diagonal_kernel,
        block_diag_kernel.to(expected_block_diagonal_kernel.dtype),
    )


def test_joint_independent_outputs_predictive_moments(model, reg_loader):
    func_la = FunctionalLaplace(model, "regression", 5, independent_outputs=True)
    with pytest.warns(UserWarning, match="multivariate regression"):
        func_la.fit(reg_loader)

    x, _ = reg_loader.dataset.tensors
    mean, covariance = func_la.predictive_moments(x[:2], joint=True)
    _, variance = func_la.predictive_moments(x[:2])

    assert mean.shape == (4,)
    assert covariance.shape == (4, 4)
    torch.testing.assert_close(
        covariance.diagonal().reshape(2, 2), variance.diagonal(dim1=-2, dim2=-1)
    )
    torch.testing.assert_close(covariance[0::2, 1::2], torch.zeros(2, 2))
    torch.testing.assert_close(covariance[1::2, 0::2], torch.zeros(2, 2))


def test_functional_samples(model, reg_loader):
    lap = FunctionalLaplace(model, "regression", n_subset=5)
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


def test_functional_fit_y_shape(model_1d, reg_loader_1d, reg_loader_1d_flat):
    la = FunctionalLaplace(model_1d, "regression", 10, independent_outputs=False)
    la.fit(reg_loader_1d)

    la2 = FunctionalLaplace(model_1d, "regression", 10, independent_outputs=False)

    with pytest.raises(ValueError):
        la2.fit(reg_loader_1d_flat)


@pytest.mark.parametrize("laplace", [FunctionalLaplace, FunctionalLLLaplace])
@pytest.mark.parametrize("backend", [AsdlGGN, BackPackGGN, CurvlinopsGGN])
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
@pytest.mark.parametrize("independent_outputs", [True, False])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_dtype(laplace, backend, dtype, independent_outputs, likelihood):
    X = torch.randn((10, 3), dtype=dtype)
    Y = torch.randn((10, 3), dtype=dtype)

    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=10)

    model = nn.Linear(3, 3, dtype=dtype)

    try:
        la = laplace(
            model,
            likelihood,
            10,
            independent_outputs=independent_outputs,
            backend=backend,
        )
        la.fit(dataloader)

        assert la.L is not None
        assert la.Sigma_inv is not None

        if independent_outputs:
            assert la.L[0].dtype == dtype
            assert la.Sigma_inv[0].dtype == dtype
        else:
            assert la.L.dtype == dtype
            assert la.Sigma_inv.dtype == dtype

        assert la.log_marginal_likelihood().dtype == dtype

        y_pred, y_var = la(X)
        assert y_pred.dtype == dtype
        assert y_var.dtype == dtype
    except (ValueError, AttributeError, RuntimeError, SystemExit) as e:
        if "must have the same dtype" in str(e):
            assert False  # Fail the test
        else:
            pass  # Ignore


def test_sod_rejects_ignored_targets(model):
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    y[3] = -100
    loader = DataLoader(TensorDataset(X, y), batch_size=5)

    func_la = FunctionalLaplace(model, "classification", 10)

    with pytest.raises(ValueError, match="subset"):
        func_la.fit(loader)


def test_failed_refit_invalidates_posterior(model):
    X, y = torch.randn(8, 3), torch.randint(2, (8,))
    y_ignored = y.clone()
    y_ignored[0] = -100

    la = FunctionalLaplace(model, "classification", n_subset=8)
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4))

    with pytest.raises(ValueError, match="subset"):
        la.fit(DataLoader(TensorDataset(X, y_ignored), batch_size=4))

    with pytest.raises(RuntimeError, match="fit"):
        la(X)

    with pytest.raises(RuntimeError, match="fit"):
        la.log_likelihood

    with pytest.raises(RuntimeError, match="fit"):
        la.log_marginal_likelihood()


def test_refit_resets_loss(model, reg_loader):
    la = FunctionalLaplace(model, "regression", n_subset=10)

    la.fit(reg_loader)
    loss = la.loss.clone()
    log_marglik = la.log_marginal_likelihood().detach()

    la.fit(reg_loader)

    assert torch.allclose(la.loss, loss)
    assert torch.allclose(la.log_marginal_likelihood().detach(), log_marglik)


class _NoisyDataset(torch.utils.data.Dataset):
    """Returns a different input on every access, like random data augmentation."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, noise: float):
        self.X, self.y, self.noise = X, y, noise

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i] + self.noise * torch.randn_like(self.X[i]), self.y[i]


@pytest.mark.parametrize("independent_outputs", [True, False])
def test_stochastic_loader_gives_symmetric_K_MM(model, independent_outputs):
    X, y = torch.randn(40, 3), torch.randint(2, (40,))
    loader = DataLoader(_NoisyDataset(X, y, noise=0.1), batch_size=6)

    la = FunctionalLaplace(
        model,
        "classification",
        n_subset=20,
        independent_outputs=independent_outputs,
    )
    la.fit(loader)

    K_MMs = la.K_MM if independent_outputs else [la.K_MM]
    for K_MM in K_MMs:
        assert torch.allclose(K_MM, K_MM.T)

    # The predictive reuses the subset inputs that built K_MM, not a new draw
    X_subset = torch.cat([la._batch_inputs(batch) for batch in la.train_loader])
    X_subset_again = torch.cat([la._batch_inputs(batch) for batch in la.train_loader])
    assert torch.equal(X_subset, X_subset_again)

    f_var = la.functional_variance(la._jacobians(X_subset[:5])[0])
    assert torch.all(torch.diagonal(f_var, dim1=-2, dim2=-1) >= -1e-8)


class _FailingDataset(torch.utils.data.Dataset):
    """Raises after `n_ok` accesses, like a loader that breaks partway through."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, n_ok: int):
        self.X, self.y, self.n_ok, self.n_calls = X, y, n_ok, 0

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.n_calls += 1
        if self.n_calls > self.n_ok:
            raise OSError("Broken data")

        return self.X[i], self.y[i]


def test_refit_with_failing_loader_invalidates_posterior(model):
    X, y = torch.randn(8, 3), torch.randint(2, (8,))

    la = FunctionalLaplace(model, "classification", n_subset=8)
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4))

    # The first batch is read fine, the subset is not
    with pytest.raises(OSError, match="Broken data"):
        la.fit(DataLoader(_FailingDataset(X, y, n_ok=4), batch_size=4))

    with pytest.raises(RuntimeError, match="fit"):
        la(X)

    with pytest.raises(RuntimeError, match="fit"):
        la.log_likelihood

    with pytest.raises(RuntimeError, match="fit"):
        la.log_marginal_likelihood()


def test_class_labels_index_targets():
    labels = _class_labels(torch.tensor([2, 0, 1], dtype=torch.int32))

    assert labels.dtype == torch.long
    assert labels.tolist() == [2, 0, 1]


def test_class_labels_probability_targets():
    probs = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.2, 0.3]])

    assert _class_labels(probs).tolist() == [1, 0]


@pytest.mark.parametrize(
    "targets",
    [
        torch.eye(3, dtype=torch.long),
        torch.tensor([[1], [0]]),
        torch.tensor([0.0, 1.0]),
        torch.zeros(2, 3, 4),
    ],
    ids=["integer-one-hot", "column", "float-flat", "ndim-3"],
)
def test_class_labels_rejects_other_targets(targets):
    with pytest.raises(ValueError, match="class indices"):
        _class_labels(targets)


def test_functional_laplace_tunes_with_probability_val_loader():
    X, y = torch.randn(10, 3), torch.randint(2, (10,))
    one_hot = torch.nn.functional.one_hot(y, num_classes=2).to(X.dtype)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))

    priors = []
    for val_targets in (y, one_hot):
        la = FunctionalLaplace(
            model,
            "classification",
            n_subset=6,
        )
        la.fit(DataLoader(TensorDataset(X, y), batch_size=4))
        la.optimize_prior_precision(
            method="gridsearch",
            link_approx="probit",
            val_loader=DataLoader(TensorDataset(X, val_targets), batch_size=4),
            grid_size=5,
        )
        priors.append(la.prior_precision.clone())

    torch.testing.assert_close(priors[0], priors[1])
