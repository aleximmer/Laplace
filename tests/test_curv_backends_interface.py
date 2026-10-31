from collections import UserDict
from collections.abc import MutableMapping

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


class DictModel(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, data: MutableMapping) -> torch.Tensor:
        assert data["note"] == "not a tensor"
        return self.net(data["input_ids"])


@pytest.mark.parametrize("dict_cls", [dict, UserDict])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
@pytest.mark.parametrize("multidim", [False, True])
def test_batchgrad_dict_inputs(dict_cls, likelihood, multidim, model):
    torch.manual_seed(711)
    lead_dims = (10, 4) if multidim else (10,)
    X = torch.randn(*lead_dims, 3)
    y = (
        torch.randint(2, lead_dims)
        if likelihood == "classification"
        else torch.randn(*lead_dims, 2)
    )

    # Non-tensor entries must be passed through as they are
    data = dict_cls({"input_ids": X, "labels": y, "note": "not a tensor"})

    backend = CurvatureInterface(model, likelihood)
    batchgrads_ref, loss_ref = backend.gradients(X, y)

    backend_dict = CurvatureInterface(DictModel(model), likelihood)
    batchgrads, loss = backend_dict.gradients(data, y)

    assert torch.allclose(batchgrads, batchgrads_ref)
    assert torch.allclose(loss, loss_ref)


@pytest.mark.parametrize("dict_cls", [dict, UserDict])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
@pytest.mark.parametrize("multidim", [False, True])
def test_batchgrad_nested_dict_inputs(dict_cls, likelihood, multidim, model):
    class NestedDictModel(nn.Module):
        def __init__(self, net: nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, data: MutableMapping) -> torch.Tensor:
            assert data["payload"]["note"] == "not a tensor"
            return self.net(data["payload"]["input_ids"])

    torch.manual_seed(711)
    lead_dims = (10, 4) if multidim else (10,)
    X = torch.randn(*lead_dims, 3)
    y = (
        torch.randint(2, lead_dims)
        if likelihood == "classification"
        else torch.randn(*lead_dims, 2)
    )

    # Tensors in nested containers must be mapped over as well
    data = {"payload": dict_cls({"input_ids": X, "note": "not a tensor"})}

    backend = CurvatureInterface(model, likelihood)
    batchgrads_ref, loss_ref = backend.gradients(X, y)

    backend_dict = CurvatureInterface(NestedDictModel(model), likelihood)
    batchgrads, loss = backend_dict.gradients(data, y)

    assert torch.allclose(batchgrads, batchgrads_ref)
    assert torch.allclose(loss, loss_ref)


@pytest.mark.parametrize("backend_cls", [GGNInterface, EFInterface])
@pytest.mark.parametrize("structure", ["full", "diag"])
def test_multidim_classification_equivalent_to_flat(backend_cls, structure):
    # Logits `(B, L, C)` with the class axis last and targets `(B, L)`: every token is
    # an observation, so the curvature must equal that of the flat `(B * L, C)` data
    torch.manual_seed(711)
    B, L, D, C = 4, 3, 5, 6
    X, y = torch.randn(B, L, D), torch.randint(C, (B, L))
    net = nn.Sequential(nn.Linear(D, 4), nn.Tanh(), nn.Linear(4, C))

    loss_flat, H_flat = getattr(backend_cls(net, "classification"), structure)(
        X.reshape(-1, D), y.reshape(-1)
    )
    loss, H = getattr(backend_cls(net, "classification"), structure)(X, y)

    assert torch.allclose(loss, loss_flat)
    assert torch.allclose(H, H_flat)


def test_loss_multidim_classification_targets():
    B, L, C = 4, 3, 5
    f = torch.randn(B, L, C)
    y = torch.randint(C, (B, L))
    y_prob = nn.functional.one_hot(y, C).to(f.dtype)
    backend = GGNInterface(nn.Linear(2, C), "classification")

    loss_flat = nn.CrossEntropyLoss(reduction="sum")(f.reshape(-1, C), y.reshape(-1))
    assert torch.allclose(backend._loss(f, y), loss_flat)
    # Class probabilities keep their class axis
    assert torch.allclose(backend._loss(f, y_prob), loss_flat)

    with pytest.raises(ValueError, match="targets have shape"):
        backend._loss(f, y.T)  # `(L, B)` instead of `(B, L)`


def test_batchgrad_data_dependent_control_flow(class_Xy, model):
    class BranchingModel(nn.Module):
        def __init__(self, net: nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, data: MutableMapping) -> torch.Tensor:
            x = data["input_ids"]
            if x.sum() > 0:  # vmap cannot trace this
                x = 2 * x
            return self.net(x)

    X, y = class_Xy
    backend = CurvatureInterface(BranchingModel(model), "classification")

    with pytest.raises(ValueError, match="data-dependent control flow"):
        backend.gradients({"input_ids": X, "labels": y}, y)
