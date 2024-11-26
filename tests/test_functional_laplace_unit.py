import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace.baselaplace import FunctionalLaplace
from laplace.curvature.asdl import AsdlGGN
from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.lllaplace import FunctionalLLLaplace

torch.manual_seed(240)
torch.set_default_dtype(torch.double)


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
    func_la.K_MM *= 0

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
    # Right now K_MM is initialized with torch.empty. To run this tests we
    #  must set it to zero.
    func_la.K_MM = [0 * func_la.K_MM[i] for i in range(func_la.n_outputs)]

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
