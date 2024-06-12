import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import (
    AsdlEF,
    AsdlGGN,
    BackPackEF,
    BackPackGGN,
    CurvatureInterface,
    EFInterface,
    GGNInterface,
)


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
def reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
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


def test_batchgrad_cls(class_Xy, model):
    X, y = class_Xy
    backend = CurvatureInterface(model, "classification")
    batchgrads, loss = backend.gradients(X, y)

    backend_asdl = AsdlEF(model, "classification")
    batchgrads_ref, loss_ref = backend_asdl.gradients(X, y)

    torch.allclose(batchgrads, batchgrads_ref)


def test_batchgrad_cls_complex(complex_class_Xy, complex_model):
    X, y = complex_class_Xy
    backend = CurvatureInterface(complex_model, "classification")
    batchgrads, loss = backend.gradients(X, y)

    backend_asdl = AsdlEF(complex_model, "classification")
    batchgrads_ref, loss_ref = backend_asdl.gradients(X, y)

    torch.allclose(batchgrads, batchgrads_ref)


def test_batchgrad_reg(reg_Xy, model):
    X, y = reg_Xy
    backend = CurvatureInterface(model, "regression")
    batchgrads, loss = backend.gradients(X, y)

    backend_asdl = BackPackEF(model, "regression")
    batchgrads_ref, loss_ref = backend_asdl.gradients(X, y)

    torch.allclose(batchgrads, batchgrads_ref)


def test_diag_ggn_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = GGNInterface(model, "classification", stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag())


def test_diag_ggn_complex_cls_curvlinops_against_backpack_full(
    complex_class_Xy, complex_model
):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = complex_class_Xy
    backend = GGNInterface(complex_model, "classification", stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params

    # check against manually computed full GGN:
    backend = AsdlGGN(complex_model, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag())


def test_diag_ggn_stoch_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = GGNInterface(model, "classification", stochastic=True, num_samples=10000)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag(), atol=0.01)


def test_diag_ef_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = EFInterface(model, "classification")
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model, "classification")
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag())


def test_diag_ggn_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = GGNInterface(model, "regression", stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "regression", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag())


def test_diag_ggn_stoch_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = GGNInterface(model, "regression", stochastic=True, num_samples=10000)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "regression", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag(), atol=0.1)


def test_diag_ef_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = EFInterface(model, "regression")
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model, "regression")
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diag())


def test_full_ggn_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = GGNInterface(model, "classification", stochastic=False)
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.1)


def test_full_ggn_stoch_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = GGNInterface(model, "classification", stochastic=True, num_samples=10000)
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.1)


def test_full_ef_cls_curvlinops_against_backpack_full(class_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = class_Xy
    backend = EFInterface(model, "classification")
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackEF(model, "classification")
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.0001)


def test_full_ggn_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = GGNInterface(model, "regression", stochastic=False)
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "regression", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.1)


def test_full_ggn_stoch_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = GGNInterface(model, "regression", stochastic=True, num_samples=10000)
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackGGN(model, "regression", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.1)


def test_full_ef_reg_curvlinops_against_backpack_full(reg_Xy, model):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = reg_Xy
    backend = EFInterface(model, "regression")
    loss, fggn = backend.full(X[:5], y[:5])
    loss2, fggn2 = backend.full(X[5:], y[5:])
    loss += loss2
    fggn += fggn2

    # sanity check size of diag ggn
    assert fggn.shape == (model.n_params, model.n_params)

    # check against manually computed full GGN:
    backend = BackPackEF(model, "regression")
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(fggn, H_ggn, atol=0.0001)
