import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import (
    AsdlEF,
    AsdlGGN,
    AsdlHessian,
    BackPackGGN,
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
)


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.set_default_dtype(torch.float32)
    yield


@pytest.fixture
def model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def complex_model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(
        nn.Conv2d(3, 4, 2, 2),
        nn.Flatten(),
        nn.Tanh(),
        nn.Linear(16, 20),
        nn.Tanh(),
        nn.Linear(20, 2),
    )
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def complex_class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3, 5, 5)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return X, y


@pytest.fixture
def complex_reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3, 5, 5)
    y = torch.nn.functional.one_hot(torch.randint(2, (10,)), num_classes=2).float()
    return X, y


@pytest.mark.parametrize("loss_type", ["classification", "regression"])
def test_full_hess_curvlinops_vs_asdl(class_Xy, reg_Xy, model, loss_type):
    X, y = class_Xy if loss_type == "classification" else reg_Xy

    backend = CurvlinopsHessian(model, loss_type)
    loss, H = backend.full(X, y)

    backend = AsdlHessian(model, loss_type)
    loss_ref, H_ref = backend.full(X, y)

    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(H, H_ref, rtol=5e-4)


def test_full_ggn_curvlinops_vs_asdl(class_Xy, model):
    X, y = class_Xy

    backend = CurvlinopsGGN(model, "classification", stochastic=False)
    loss, H = backend.full(X, y)

    backend = AsdlGGN(model, "classification", stochastic=False)
    loss_ref, H_ref = backend.full(X, y)

    assert torch.allclose(loss, loss_ref, rtol=1e-4)
    assert torch.allclose(H, H_ref, rtol=1e-4)


def test_full_ggn_stochastic(class_Xy, model):
    torch.manual_seed(123)
    X, y = class_Xy

    backend = CurvlinopsGGN(model, "classification", stochastic=True)
    loss_mc1, H_mc1 = backend.full(X, y, mc_samples=1)

    backend = CurvlinopsGGN(model, "classification", stochastic=True)
    loss_mc100, H_mc100 = backend.full(X, y, mc_samples=100)

    backend = CurvlinopsGGN(model, "classification", stochastic=False)
    loss_exact, H_exact = backend.full(X, y)

    assert torch.allclose(loss_mc1, loss_exact)
    assert torch.allclose(loss_mc100, loss_exact)
    diff_mc1 = torch.norm(H_mc1 - H_exact)
    diff_mc100 = torch.norm(H_mc100 - H_exact)
    assert torch.norm(diff_mc1) > torch.norm(diff_mc100)


def test_full_ef_curvlinops_vs_asdl(class_Xy, model):
    X, y = class_Xy

    backend = CurvlinopsEF(model, "classification")
    loss, H = backend.full(X, y)

    backend = AsdlEF(model, "classification")
    loss_ref, H_ref = backend.full(X, y)

    assert torch.allclose(loss, loss_ref, rtol=1e-4)
    assert torch.allclose(H, H_ref, rtol=1e-4)


@pytest.mark.parametrize("loss_type", ["classification", "regression"])
def test_kron_ggn_curvlinops_vs_backpack(class_Xy, reg_Xy, model, loss_type):
    X, y = class_Xy if loss_type == "classification" else reg_Xy

    backend = CurvlinopsGGN(model, loss_type, stochastic=False)
    loss, kron = backend.kron(X, y, N=1)

    backend = BackPackGGN(model, loss_type, stochastic=False)
    loss_ref, kron_ref = backend.kron(X, y, N=1)

    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(kron.to_matrix(), kron_ref.to_matrix(), rtol=5e-5)


@pytest.mark.parametrize("loss_type", ["classification", "regression"])
def test_kron_ggn_stochastic(class_Xy, reg_Xy, model, loss_type):
    X, y = class_Xy if loss_type == "classification" else reg_Xy

    backend = CurvlinopsGGN(model, loss_type, stochastic=True)
    loss_mc1, kron_mc1 = backend.kron(X, y, N=1, mc_samples=1)

    backend = CurvlinopsGGN(model, loss_type, stochastic=True)
    loss_mc100, kron_mc100 = backend.kron(X, y, N=1, mc_samples=100)

    backend = CurvlinopsGGN(model, loss_type, stochastic=False)
    loss_ref, kron_exact = backend.kron(X, y, N=1)

    assert torch.allclose(loss_mc1, loss_ref)
    assert torch.allclose(loss_mc100, loss_ref)

    diff_mc1 = torch.norm(kron_mc1.to_matrix() - kron_exact.to_matrix())
    diff_mc100 = torch.norm(kron_mc100.to_matrix() - kron_exact.to_matrix())
    assert torch.norm(diff_mc1) > torch.norm(diff_mc100)


@pytest.mark.parametrize("loss_type", ["classification", "regression"])
def test_kron_ggn_set_kfac_approx(
    complex_class_Xy, complex_reg_Xy, complex_model, loss_type
):
    X, y = complex_class_Xy if loss_type == "classification" else complex_reg_Xy

    backend = CurvlinopsGGN(complex_model, loss_type, stochastic=False)
    loss_expand, kron_expand = backend.kron(X, y, N=1, kfac_approx="expand")

    backend = CurvlinopsGGN(complex_model, loss_type, stochastic=False)
    loss_reduce, kron_reduce = backend.kron(X, y, N=1, kfac_approx="reduce")

    assert torch.allclose(loss_expand, loss_reduce)
    assert not torch.allclose(kron_expand.to_matrix(), kron_reduce.to_matrix())


def test_kron_ef_cls_curvlinops_vs_backpack(class_Xy, model):
    X, y = class_Xy
    backend = CurvlinopsEF(model, "classification")
    loss, kron = backend.kron(X, y, N=1)

    backend = AsdlEF(model, "classification")
    loss_ref, kron_ref = backend.kron(X, y, N=1)

    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(kron.to_matrix(), kron_ref.to_matrix())


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_kron_batching_correction_cls_curvlinops(class_Xy, model, Backend):
    X, y = class_Xy
    backend = Backend(model, "classification")
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_kron_batching_correction_reg_curvlinops(reg_Xy, model, Backend):
    X, y = reg_Xy
    backend = Backend(model, "regression")
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_kron_summing_up_vs_diag_cls_curvlinops(class_Xy, model, Backend):
    X, y = class_Xy
    backend = Backend(model, "classification")
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


def test_complex_diag_ggn_stoch_cls_curvlinops(complex_class_Xy, complex_model):
    X, y = complex_class_Xy
    backend = CurvlinopsGGN(complex_model, "classification", stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params

    # same order of magnitude os non-stochastic.
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1)


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_complex_kron_curvlinops_vs_diag_curvlinops(
    complex_class_Xy, complex_model, Backend
):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = complex_class_Xy
    backend = Backend(complex_model, "classification")
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_complex_kron_batching_correction_curvlinops(
    complex_class_Xy, complex_model, Backend
):
    X, y = complex_class_Xy
    backend = Backend(complex_model, "classification")
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == complex_model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


@pytest.mark.parametrize("Backend", [CurvlinopsEF, CurvlinopsGGN])
def test_complex_kron_summing_up_vs_diag_class_curvlinops(
    complex_class_Xy, complex_model, Backend
):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = complex_class_Xy
    backend = Backend(complex_model, "classification")
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-2)


def test_kron_normalization_ggn_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = CurvlinopsGGN(model, "classification", stochastic=False)
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_kron_normalization_ef_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = CurvlinopsEF(model, "classification")
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)
