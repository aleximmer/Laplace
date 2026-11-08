import warnings
from collections import UserDict
from collections.abc import MutableMapping
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import (
    CurvatureInterface,
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
    EFInterface,
    GGNInterface,
)
from laplace.curvature.asdfghjkl import AsdfghjklEF, AsdfghjklGGN, AsdfghjklHessian
from laplace.likelihoods import (
    BernoulliLikelihood,
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
)
from tests.utils import (
    AsdlEF,
    AsdlGGN,
    AsdlHessian,
    BackPackEF,
    BackPackGGN,
    ScaledGaussian,
    loss_hessian,
    requires_backpack,
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
    # Logits `(B, L, C)` with the class axis last and targets `(B, L)`: every position is
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


@pytest.mark.parametrize("backend_cls", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
@pytest.mark.parametrize("structure", ["diag", "kron", "full"])
@pytest.mark.parametrize(
    "name,likelihood",
    [
        ("regression", GaussianLikelihood()),
        ("classification", CategoricalLikelihood()),
    ],
)
def test_backend_accepts_likelihood_object(
    backend_cls, structure, name, likelihood, model, class_Xy, reg_Xy
):
    X, y = reg_Xy if name == "regression" else class_Xy

    extra = (len(X),) if structure == "kron" else ()

    loss_str, H_str = getattr(backend_cls(model, name), structure)(X, y, *extra)
    loss_obj, H_obj = getattr(backend_cls(model, likelihood), structure)(X, y, *extra)

    assert torch.allclose(loss_str, loss_obj)

    if structure == "kron":
        H_str, H_obj = H_str.to_matrix(), H_obj.to_matrix()

    assert torch.allclose(H_str, H_obj)


@pytest.mark.parametrize("likelihood", [BradleyTerryLikelihood(), "reward_modeling"])
def test_backend_rejects_unsupported_likelihood(likelihood, model):
    with pytest.raises(ValueError, match="fit_likelihood"):
        CurvatureInterface(model, likelihood)


def test_backend_rejects_invalid_likelihood_name(model):
    with pytest.raises(ValueError, match="Invalid likelihood type"):
        CurvatureInterface(model, "foo")


@pytest.mark.parametrize("backend_cls", [CurvlinopsGGN, AsdlGGN])
def test_regression_rejects_target_with_fewer_dims(backend_cls):
    # f is `(B, 1)` and y is `(B,)`: MSELoss would silently broadcast to `(B, B)`
    torch.manual_seed(711)
    X, y = torch.randn(10, 3), torch.randn(10)
    net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))

    with pytest.raises(ValueError, match="dims but the target has"):
        backend_cls(net, "regression").diag(X, y)


@pytest.mark.parametrize("structure", ["diag", "full"])
@pytest.mark.parametrize(
    "name,likelihood",
    [
        ("regression", GaussianLikelihood()),
        ("classification", CategoricalLikelihood()),
    ],
)
def test_mc_fisher_str_vs_object(structure, name, likelihood, model, class_Xy, reg_Xy):
    # `stochastic=True` samples targets in `_get_mc_functional_fisher`
    X, y = reg_Xy if name == "regression" else class_Xy

    outs = []
    for lik in (name, likelihood):
        backend = GGNInterface(model, lik, stochastic=True, num_samples=3)

        torch.manual_seed(0)
        outs.append(getattr(backend, structure)(X, y))

    (loss_str, H_str), (loss_obj, H_obj) = outs

    assert torch.allclose(loss_str, loss_obj)
    assert torch.equal(H_str, H_obj)


def test_backend_likelihood_setter_coerces_str(model):
    backend = CurvlinopsGGN(model, "regression")

    backend.likelihood = "classification"
    assert type(backend.likelihood) is CategoricalLikelihood

    with pytest.raises(ValueError, match="fit_likelihood"):
        backend.likelihood = "reward_modeling"


def test_backend_legacy_pickle_state(model, class_Xy):
    X, y = class_Xy
    backend = CurvlinopsGGN(model, "classification")
    loss, H = backend.diag(X, y)

    # Pickles made before likelihood objects stored a plain string
    state = dict(vars(backend))
    state["likelihood"] = str(state.pop("_likelihood"))

    legacy = CurvlinopsGGN.__new__(CurvlinopsGGN)
    legacy.__setstate__(state)

    assert type(legacy.likelihood) is CategoricalLikelihood
    assert "likelihood" not in vars(legacy)

    loss_legacy, H_legacy = legacy.diag(X, y)
    assert torch.allclose(loss, loss_legacy)
    assert torch.allclose(H, H_legacy)


def test_backend_accepts_custom_name(model):
    backend = CurvlinopsGGN(model, ScaledGaussian())

    assert isinstance(backend.likelihood, ScaledGaussian)


@pytest.mark.parametrize("backend_cls", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
@pytest.mark.parametrize("structure", ["diag", "full"])
def test_custom_ggn_matches_closed_form(backend_cls, structure, model, reg_Xy):
    X, y = reg_Xy

    _, H_reg = getattr(backend_cls(model, "regression"), structure)(X, y)
    loss, H = getattr(backend_cls(model, ScaledGaussian(2.0)), structure)(X, y)

    # The output Hessian of a Gaussian with scale 2 is I / 4
    assert torch.allclose(H, H_reg / 4)

    f = model(X).detach()
    assert torch.allclose(loss, -torch.distributions.Normal(f, 2.0).log_prob(y).sum())


_ALL_OPTIONS = frozenset({"weight", "ignore_index", "label_smoothing"})
_TARGET_FREE = frozenset({"label_smoothing"})
_KRON = _TARGET_FREE | {"ignore_index"}

# Float32 weights for a float64 model: the backends must cast them
_OPTIONS_WEIGHT = torch.tensor([0.5, 2.0, 1.5])

_CURVATURE_OPTIONS = [
    {},
    {"weight": _OPTIONS_WEIGHT},
    {"ignore_index": 1},
    {"label_smoothing": 0.1},
    {"weight": _OPTIONS_WEIGHT, "ignore_index": 1, "label_smoothing": 0.1},
]

# Backend, kind of curvature, method, stochastic, and the options that the backend
# claims to honour natively (or, for `GGNInterface`/`EFInterface`, by itself).
# Deterministic GGN/EF `diag`/`full` must match the reference with any options,
# through the fallback if need be. Other cells raise for options they don't honour.
_CURVATURE_CELLS = [
    (CurvlinopsGGN, "ggn", "full", False, _ALL_OPTIONS),
    (CurvlinopsGGN, "ggn", "full", True, _TARGET_FREE),
    (CurvlinopsGGN, "ggn", "diag", False, _ALL_OPTIONS),
    (CurvlinopsGGN, "ggn", "kron", False, _KRON),
    # MC KFAC differentiates the loss at sampled labels, so label smoothing
    # changes it (downgraded). Ignored positions are dropped before sampling.
    (CurvlinopsGGN, "ggn", "kron", True, frozenset({"ignore_index"})),
    (CurvlinopsEF, "ef", "full", False, _ALL_OPTIONS),
    (CurvlinopsEF, "ef", "diag", False, _ALL_OPTIONS),
    (CurvlinopsEF, "ef", "kron", False, _KRON),
    (CurvlinopsHessian, "hessian", "full", False, _ALL_OPTIONS),
    (AsdlGGN, "ggn", "diag", False, _TARGET_FREE),
    (AsdlGGN, "ggn", "diag", True, _TARGET_FREE),
    (AsdlGGN, "ggn", "kron", False, _KRON),
    (AsdlGGN, "ggn", "kron", True, _KRON),
    (AsdlGGN, "ggn", "full", False, _ALL_OPTIONS),
    (AsdlEF, "ef", "diag", False, _ALL_OPTIONS),
    (AsdlEF, "ef", "kron", False, _KRON),
    (AsdlEF, "ef", "full", False, _ALL_OPTIONS),
    (AsdlHessian, "hessian", "full", False, _ALL_OPTIONS),
    (BackPackGGN, "ggn", "diag", False, _TARGET_FREE),
    (BackPackGGN, "ggn", "diag", True, _TARGET_FREE),
    (BackPackGGN, "ggn", "kron", False, _KRON),
    (BackPackGGN, "ggn", "kron", True, _KRON),
    (BackPackGGN, "ggn", "full", False, _ALL_OPTIONS),
    (BackPackEF, "ef", "diag", False, _ALL_OPTIONS),
    (BackPackEF, "ef", "full", False, _ALL_OPTIONS),
    (GGNInterface, "ggn", "full", False, _ALL_OPTIONS),
    (GGNInterface, "ggn", "diag", False, _ALL_OPTIONS),
    # The MC Fisher smooths the sampled labels in `output_gradient`, which changes
    # it (downgraded)
    (GGNInterface, "ggn", "full", True, frozenset()),
    (GGNInterface, "ggn", "diag", True, frozenset()),
    (EFInterface, "ef", "full", False, _ALL_OPTIONS),
    (EFInterface, "ef", "diag", False, _ALL_OPTIONS),
]


def _options_model(kron: bool) -> nn.Module:
    """Small float64 classifier with 3 classes. For Kronecker factors, a single
    linear layer, on which KFAC is exact for a single datum."""
    torch.manual_seed(711)

    if kron:
        model = nn.Linear(3, 3)
    else:
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3))

    setattr(model, "output_size", 3)

    return model.double()


def _options_data(kron: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and class targets, a single datum for Kronecker factors. The first two
    targets of the batch are class 1, which is ignored with `ignore_index=1`."""
    torch.manual_seed(711)

    if kron:
        return torch.randn(1, 3, dtype=torch.float64), torch.tensor([2])

    y = torch.randint(3, (8,))
    y[:2] = 1

    return torch.randn(8, 3, dtype=torch.float64), y


def _make_backend(backend_cls, model, likelihood, stochastic):
    """Build a backend, passing `stochastic` only to those that take it."""
    if stochastic:
        return backend_cls(model, likelihood, stochastic=True)

    return backend_cls(model, likelihood)


@pytest.mark.parametrize("opts", _CURVATURE_OPTIONS)
@pytest.mark.parametrize(
    "backend_cls,kind,method,stochastic,claimed",
    _CURVATURE_CELLS,
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_categorical_options_curvature(
    backend_cls, kind, method, stochastic, claimed, opts
):
    kron = method == "kron"
    X, y = _options_data(kron)
    likelihood = CategoricalLikelihood(**opts)

    # Fresh models, since BackPACK extends the model in place
    backend = _make_backend(backend_cls, _options_model(kron), likelihood, stochastic)
    args = (X, y, len(X)) if kron else (X, y)

    # Kronecker factors, Hessians, and Monte Carlo Fishers have no fallback
    must_honour = kron or kind == "hessian" or stochastic

    if must_honour and not set(opts) <= claimed:
        with pytest.raises(ValueError, match="diag"):
            getattr(backend, method)(*args)

        return

    torch.manual_seed(0)
    loss, H = getattr(backend, method)(*args)

    reference_model = _options_model(kron)
    f_ref = reference_model(X).detach()
    assert torch.allclose(loss, likelihood.loss(f_ref, y))

    if kron:
        H = H.diag()

    if stochastic:
        # The Monte Carlo Fisher ignores the targets, so the honoured options leave
        # it as it is without them, given the same samples
        default = _make_backend(
            backend_cls, _options_model(kron), CategoricalLikelihood(), True
        )
        torch.manual_seed(0)
        _, H_ref = getattr(default, method)(*args)

        if kron:
            H_ref = H_ref.diag()
    elif kind == "hessian":
        H_ref = loss_hessian(reference_model, likelihood.loss, X, y)
    else:
        reference_cls = GGNInterface if kind == "ggn" else EFInterface
        reference = reference_cls(reference_model, likelihood)
        _, H_ref = getattr(reference, "diag" if kron else method)(X, y)

    assert H.dtype == H_ref.dtype
    assert torch.allclose(H, H_ref)


def test_unhonoured_options_raise(model, class_Xy):
    X, y = class_Xy

    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="diag"):
        CurvlinopsGGN(model, weighted).kron(X, y, len(X))

    with pytest.raises(ValueError, match="diag"):
        CurvlinopsGGN(model, weighted, stochastic=True).full(X, y)

    # The GGN of a smoothed cross-entropy doesn't depend on the targets
    _, kron = CurvlinopsGGN(model, "classification").kron(X, y, len(X))
    _, kron_smooth = CurvlinopsGGN(
        model, CategoricalLikelihood(label_smoothing=0.1)
    ).kron(X, y, len(X))

    assert torch.allclose(kron.to_matrix(), kron_smooth.to_matrix())


def test_unsupported_error_names_ignore_index(model, class_Xy):
    X, y = class_Xy
    y = y.clone()
    y[0] = -100

    # The default `ignore_index` is not an option of the likelihood, but padding
    # makes it one. ASDL's diagonal GGN doesn't honour it.
    error = AsdlGGN(model, "classification")._unsupported_error("diag", y=y)

    assert isinstance(error, ValueError)
    assert "ignore_index" in str(error)


def test_custom_kron_raises(model, reg_Xy):
    X, y = reg_Xy

    with pytest.raises(ValueError, match="diag"):
        CurvlinopsGGN(model, ScaledGaussian()).kron(X, y, len(X))


@requires_backpack
def test_backpack_custom_construction(model, reg_Xy, class_Xy):
    X, y = reg_Xy
    _, H = BackPackGGN(model, ScaledGaussian()).diag(X, y)
    assert torch.isfinite(H).all()

    X, y = class_Xy
    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]))
    _, H = BackPackGGN(model, weighted).diag(X, y)
    assert torch.isfinite(H).all()


def _padded_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs, class targets whose first three entries are padding with the default
    `ignore_index=-100`, and the mask of the kept targets."""
    X, y = _options_data(kron=False)
    y[:3] = -100

    return X, y, y != -100


# Backend, method, and stochastic. None of these libraries masks padded positions in
# these methods, so the backends must fall back to torch.func, except for the
# deterministic Curvlinops and EF ones, which mask them natively.
_PADDED_CELLS = [
    (CurvlinopsGGN, "full", False),
    (CurvlinopsGGN, "full", True),
    (CurvlinopsEF, "full", False),
    (AsdlGGN, "diag", False),
    (AsdlGGN, "diag", True),
    (AsdlGGN, "full", False),
    (AsdlEF, "diag", False),
    (BackPackGGN, "diag", False),
    (BackPackGGN, "diag", True),
    (BackPackEF, "diag", False),
]


@pytest.mark.parametrize(
    "backend_cls,method,stochastic",
    _PADDED_CELLS,
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_padded_positions_match_reference(backend_cls, method, stochastic):
    X, y, keep = _padded_data()
    likelihood = CategoricalLikelihood()

    backend = _make_backend(backend_cls, _options_model(False), likelihood, stochastic)
    torch.manual_seed(0)
    _, H = getattr(backend, method)(X, y)

    # The torch.func backend with the same likelihood and the same samples
    is_ef = backend_cls in (CurvlinopsEF, AsdlEF, BackPackEF)
    reference_cls = EFInterface if is_ef else GGNInterface
    reference = _make_backend(
        reference_cls, _options_model(False), likelihood, stochastic
    )
    torch.manual_seed(0)
    _, H_ref = getattr(reference, method)(X, y)

    assert torch.allclose(H, H_ref)

    if not stochastic:
        # Padding counts for nothing, as if those data weren't there
        _, H_kept = getattr(reference, method)(X[keep], y[keep])
        assert torch.allclose(H, H_kept)


def test_mc_fisher_masks_padded_positions():
    X, y, keep = _padded_data()
    backend = GGNInterface(_options_model(False), "classification", stochastic=True)

    # A padded datum gets no curvature, whatever the samples
    _, H_pad = backend.diag(X[~keep], y[~keep])
    assert torch.all(H_pad == 0)

    # Probability targets are never ignored
    q = torch.nn.functional.one_hot(y.clamp(min=0), 3).double()
    _, H_prob = backend.diag(X, q)
    assert torch.all(H_prob[H_prob.abs() > 0].isfinite()) and H_prob.abs().sum() > 0


@pytest.mark.parametrize(
    "backend_cls",
    [CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF, BackPackGGN],
    ids=lambda v: v.__name__,
)
def test_kron_drops_padded_positions(backend_cls):
    X, y, keep = _padded_data()
    backend = backend_cls(_options_model(False), "classification")

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*ignored positions")
        _, kron = backend.kron(X, y, len(X))

    # Padding counts for nothing, as if those data weren't there
    _, kron_kept = backend.kron(X[keep], y[keep], len(X))
    assert torch.allclose(kron.to_matrix(), kron_kept.to_matrix())


def test_honours_counts_padding_as_ignore_index(model, class_Xy):
    X, y = class_Xy
    backend = CurvlinopsGGN(model, "classification", stochastic=True)

    # The MC Fisher of the dense operator honours label smoothing only
    assert backend._honours("full", y)

    y = y.clone()
    y[0] = -100
    assert not backend._honours("full", y)
    assert backend._honours("full")

    # Probability targets have no ignored positions
    q = torch.nn.functional.one_hot(y.clamp(min=0), 2).float()
    assert backend._honours("full", q)


def test_bernoulli_is_known_to_curvlinops_only(model):
    # Curvlinops has the loss of `nn.BCEWithLogitsLoss`, while BackPACK and ASDL
    # don't, so they must not treat it like a Gaussian or categorical likelihood
    assert CurvlinopsGGN(model, "bernoulli")._honours("full")
    assert CurvlinopsEF(model, "bernoulli")._honours("full")
    assert CurvlinopsHessian(model, "bernoulli")._honours("full")
    assert CurvlinopsGGN(model, "bernoulli")._honours("kron")

    for backend in (
        AsdlGGN(model, "bernoulli"),
        AsdlEF(model, "bernoulli"),
        AsdlHessian(model, "bernoulli"),
        GGNInterface(model, "bernoulli"),
    ):
        assert not any(
            backend._honours(method) for method in ("diag", "full", "kron")
        ), type(backend).__name__

    # The torch.func backends still compute it
    X, y = torch.randn(4, 3), torch.rand(4, 2)

    assert torch.isfinite(GGNInterface(model, "bernoulli").full(X, y)[1]).all()
    assert torch.isfinite(EFInterface(model, "bernoulli").diag(X, y)[1]).all()


def test_bernoulli_error_names_the_likelihood(model):
    X, y = torch.randn(4, 3), torch.rand(4, 2)

    with pytest.raises(ValueError, match=r"does not support the likelihood Bernoulli"):
        AsdlHessian(model, "bernoulli").full(X, y)

    # Options of a library likelihood are named instead
    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="the option weight of Categorical"):
        CurvlinopsGGN(model, weighted).kron(X, torch.tensor([0, 1, 0, 1]), 4)


def test_bernoulli_stochastic_fisher_raises_for_options(model):
    X, y = torch.randn(4, 3), torch.rand(4, 2)
    weighted = BernoulliLikelihood(pos_weight=torch.tensor([2.0, 3.0]))

    # The sampled labels are not the observed ones that the options refer to
    for method in ("diag", "full"):
        with pytest.raises(ValueError, match="stochastic=True"):
            getattr(GGNInterface(model, weighted, stochastic=True), method)(X, y)

    # Without options, soft targets are fine, since the samples ignore them
    for method in ("diag", "full"):
        _, H = getattr(GGNInterface(model, "bernoulli", stochastic=True), method)(X, y)
        assert torch.isfinite(H).all()


@pytest.mark.asdfghjkl
def test_bernoulli_is_rejected_by_asdfghjkl(model):
    X, y = torch.randn(4, 3), torch.rand(4, 2)

    for backend_cls in (AsdfghjklGGN, AsdfghjklEF):
        with pytest.raises(ValueError, match="only supports classification"):
            backend_cls(model, "bernoulli")

    # The Hessian-based backend of `LowRankLaplace` must not take the categorical
    # path either
    backend = AsdfghjklHessian(model, "bernoulli")

    with pytest.raises(ValueError, match="does not support the likelihood"):
        backend.full(X, y)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), 2)

    with pytest.raises(ValueError, match="does not support the likelihood"):
        backend.eig_lowrank(loader)


@pytest.mark.parametrize(
    "backend_cls,method",
    [
        (CurvlinopsHessian, "diag"),
        (CurvlinopsHessian, "kron"),
        (AsdlHessian, "diag"),
        (AsdlHessian, "kron"),
        (AsdfghjklHessian, "diag"),
        (AsdfghjklHessian, "kron"),
        (BackPackEF, "kron"),
    ],
)
def test_unavailable_structure_raises(backend_cls, method, model, reg_Xy):
    X, y = reg_Xy
    backend = backend_cls(model, "regression")
    args = (X, y, len(X)) if method == "kron" else (X, y)

    with pytest.raises(NotImplementedError, match=f"doesn't compute the `{method}`"):
        getattr(backend, method)(*args)


def _assert_no_backpack_state(model: nn.Module) -> None:
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._backward_hooks
        assert module._is_full_backward_hook is None
        assert not hasattr(module, "_backpack_extend")


@requires_backpack
@pytest.mark.parametrize(
    "first,structure",
    [
        (BackPackGGN, "diag"),
        (BackPackGGN, "kron"),
        (BackPackGGN, "full"),
        (BackPackEF, "diag"),
        (BackPackEF, "full"),
    ],
)
@pytest.mark.parametrize("then", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_backpack_leaves_the_model_as_it_was(model, reg_Xy, first, structure, then):
    # BackPACK must remove its hooks after each computation. Left on the model, they
    # break `torch.func`, ASDL, and later BackPACK computations on it.
    X, y = reg_Xy
    reference = deepcopy(model)
    args = (X, y, len(X)) if structure == "kron" else (X, y)
    getattr(first(model, "regression"), structure)(*args)

    _assert_no_backpack_state(model)

    _, H = then(model, "regression").diag(X, y)
    _, H_ref = then(reference, "regression").diag(X, y)

    torch.testing.assert_close(H, H_ref)


@requires_backpack
@pytest.mark.parametrize("asdl", [AsdlGGN, AsdlEF])
def test_backpack_after_asdl_raises(model, reg_Xy, asdl):
    # ASDL's regular backward hooks leave a mark on the modules that PyTorch checks
    # before it registers BackPACK's full ones
    X, y = reg_Xy
    asdl(model, "regression").diag(X, y)

    with pytest.raises(RuntimeError, match="regular backward hooks"):
        BackPackGGN(model, "regression").diag(X, y)

    # A fresh model with the same weights works, as the error says
    fresh = nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2))
    fresh.load_state_dict(model.state_dict())
    BackPackGGN(fresh, "regression").diag(X, y)
