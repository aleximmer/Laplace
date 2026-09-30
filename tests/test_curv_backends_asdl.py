from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import AsdlEF, AsdlGGN, BackPackEF, BackPackGGN


@pytest.fixture
def model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2))
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


def test_diag_ggn_cls_asdl_against_backpack_full(class_Xy, model):
    X, y = class_Xy
    model2 = deepcopy(model)
    backend = AsdlGGN(model, "classification", stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model2, "classification", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ggn_reg_asdl_against_backpack_full(reg_Xy, model):
    X, y = reg_Xy
    model2 = deepcopy(model)
    backend = AsdlGGN(model, "regression", stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model2, "regression", stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ef_cls_asdl_against_backpack_full(class_Xy, model):
    X, y = class_Xy
    model2 = deepcopy(model)
    backend = AsdlEF(model, "classification")
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model2, "classification")
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ef_reg_asdl_against_backpack_full(reg_Xy, model):
    X, y = reg_Xy
    model2 = deepcopy(model)
    backend = AsdlEF(model, "regression")
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model2, "regression")
    loss_bp, dggn_bp = backend.diag(X, y)
    assert torch.allclose(loss, loss_bp)
    assert torch.allclose(dggn, dggn_bp)  # H_ggn.diagonal())


def test_diag_ggn_stoch_cls_asdl(class_Xy, model):
    X, y = class_Xy
    backend = AsdlGGN(model, "classification", stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # same order of magnitude os non-stochastic.
    backend = AsdlGGN(model, "classification", stochastic=False)
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1e1)


def test_diag_ggn_asdl_subnetwork_index_order(class_Xy, model):
    X, y = class_Xy
    backend = AsdlGGN(model, "classification")
    _, full_diag = backend.diag(X, y)
    indices = torch.LongTensor([model.n_params - 1, 0, model.n_params // 2])

    backend.subnetwork_indices = indices
    _, subnet_diag = backend.diag(X, y)

    assert torch.allclose(subnet_diag, full_diag[indices])


@pytest.mark.parametrize("stochastic", [False, True])
def test_diag_ggn_asdl_subnetwork_with_frozen_module(class_Xy, stochastic):
    X, y = class_Xy
    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 2))
    for param in model[0].parameters():
        param.requires_grad_(False)
    indices = torch.LongTensor([7, 0])

    torch.manual_seed(711)
    _, full_diag = AsdlGGN(model, "classification", stochastic=stochastic).diag(X, y)
    torch.manual_seed(711)
    _, subnet_diag = AsdlGGN(
        model, "classification", subnetwork_indices=indices, stochastic=stochastic
    ).diag(X, y)

    assert full_diag.numel() == 8
    assert torch.allclose(subnet_diag, full_diag[indices])


@pytest.mark.parametrize("frozen_name", ["weight", "bias"])
def test_diag_ggn_asdl_subnetwork_with_partially_frozen_module(class_Xy, frozen_name):
    X, y = class_Xy
    model = nn.Linear(3, 2)
    getattr(model, frozen_name).requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    indices = torch.LongTensor([n_params - 1, 0])

    _, full_diag = AsdlGGN(model, "classification").diag(X, y)
    _, subnet_diag = AsdlGGN(model, "classification", subnetwork_indices=indices).diag(
        X, y
    )

    assert full_diag.numel() == n_params
    assert torch.allclose(subnet_diag, full_diag[indices])


def test_diag_ggn_asdl_uses_fisher_dtype(class_Xy):
    class CastOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2, dtype=torch.float64)

        def forward(self, x):
            return self.linear(x).float()

    X, y = class_Xy
    model = CastOutput()
    _, diag = AsdlGGN(model, "classification").diag(X.double(), y)
    fisher_diag = torch.cat(list(model.linear.fisher.to_vector()))

    assert diag.dtype == torch.float64
    assert torch.equal(diag, fisher_diag)


def test_diag_ggn_asdl_respects_parameter_registration_order(class_Xy):
    class BiasFirstLinear(nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            weight = self.weight
            del self._parameters["weight"]
            self.register_parameter("weight", weight)

    X, y = class_Xy
    standard = nn.Linear(3, 2)
    bias_first = BiasFirstLinear(3, 2)
    bias_first.load_state_dict(standard.state_dict())
    assert list(dict(bias_first.named_parameters())) == ["bias", "weight"]

    _, standard_diag = AsdlGGN(standard, "classification").diag(X, y)
    _, bias_first_diag = AsdlGGN(bias_first, "classification").diag(X, y)
    expected = torch.cat([standard_diag[-2:], standard_diag[:-2]])
    indices = torch.LongTensor([0, 2, 7])
    _, subnet_diag = AsdlGGN(
        bias_first, "classification", subnetwork_indices=indices
    ).diag(X, y)

    assert torch.allclose(bias_first_diag, expected)
    assert torch.allclose(subnet_diag, expected[indices])


def test_kron_ggn_cls_asdl_against_backpack(class_Xy, model):
    X, y = class_Xy
    model2 = deepcopy(model)
    backend = AsdlGGN(model, "classification", stochastic=False)
    loss, kron = backend.kron(X[:5], y[:5], len(y))
    loss2, kron2 = backend.kron(X[5:], y[5:], len(y))
    loss += loss2
    kron += kron2
    backend_bp = BackPackGGN(model2, "classification", stochastic=False)
    loss_bp, kron_bp = backend_bp.kron(X, y, len(y))

    # sanity check size of diag ggn
    assert len(kron.diag()) == model.n_params == len(kron_bp.diag())
    assert torch.allclose(loss, loss_bp)
    assert torch.allclose(kron.diag(), kron_bp.diag())


def test_kron_ggn_reg_asdl_against_backpack(reg_Xy, model):
    X, y = reg_Xy
    model2 = deepcopy(model)
    backend = AsdlGGN(model, "regression", stochastic=False)
    # loss, kron = backend.kron(X, y, len(y))
    loss, kron = backend.kron(X[:5], y[:5], len(y))
    loss2, kron2 = backend.kron(X[5:], y[5:], len(y))
    loss += loss2
    kron += kron2
    backend_bp = BackPackGGN(model2, "regression", stochastic=False)
    loss_bp, kron_bp = backend_bp.kron(X, y, len(y))

    # sanity check size of diag ggn
    assert len(kron.diag()) == model.n_params == len(kron_bp.diag())
    assert torch.allclose(loss, loss_bp)
    assert torch.allclose(kron.diag(), kron_bp.diag())


@pytest.mark.parametrize("Backend", [AsdlEF, AsdlGGN])
def test_kron_asdl_vs_diag_class(class_Xy, model, Backend):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = class_Xy
    backend = Backend(model, "classification")
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


@pytest.mark.parametrize("Backend", [AsdlEF, AsdlGGN])
def test_kron_asdl_vs_diag_reg(reg_Xy, model, Backend):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = reg_Xy
    backend = Backend(model, "regression")
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


@pytest.mark.parametrize("Backend", [AsdlEF, AsdlGGN])
def test_kron_batching_correction_asdl(class_Xy, model, Backend):
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


@pytest.mark.parametrize("Backend", [AsdlGGN, AsdlEF])
def test_kron_summing_up_vs_diag_asdl(class_Xy, model, Backend):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = class_Xy
    backend = Backend(model, "classification")
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


def test_complex_diag_ggn_stoch_cls_asdl(complex_class_Xy, complex_model):
    X, y = complex_class_Xy
    backend = AsdlGGN(complex_model, "classification", stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params

    # same order of magnitude os non-stochastic.
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1)


@pytest.mark.parametrize("Backend", [AsdlEF, AsdlGGN])
def test_complex_kron_asdl_vs_diag_asdl(complex_class_Xy, complex_model, Backend):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = complex_class_Xy
    backend = Backend(complex_model, "classification")
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


@pytest.mark.parametrize("Backend", [AsdlEF, AsdlGGN])
def test_complex_kron_batching_correction_asdl(
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


@pytest.mark.parametrize("Backend", [AsdlGGN, AsdlEF])
def test_complex_kron_summing_up_vs_diag_class_asdl(
    complex_class_Xy, complex_model, Backend
):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = complex_class_Xy
    backend = Backend(complex_model, "classification")
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-2)


def test_kron_summation_ggn_class(class_Xy, model):
    X, y = class_Xy
    X = torch.repeat_interleave(X[:1], 7, 0)
    y = torch.repeat_interleave(y[:1], 7, 0)
    backend = AsdlGGN(model, "classification", stochastic=False)
    loss_dg, diag = backend.diag(X, y)
    loss, kron = backend.kron(X, y, N=7)
    assert torch.allclose(loss_dg, loss)
    assert torch.allclose(kron.diag(), diag)


def test_kron_summation_ggn_reg(reg_Xy, model):
    X, y = reg_Xy
    X = torch.repeat_interleave(X[:1], 7, 0)
    y = torch.repeat_interleave(y[:1], 7, 0)
    backend = AsdlGGN(model, "regression", stochastic=False)
    loss_dg, diag = backend.diag(X, y)
    loss, kron = backend.kron(X, y, N=7)
    assert torch.allclose(loss_dg, loss)
    assert torch.allclose(kron.diag(), diag)


def test_kron_normalization_ggn_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = AsdlGGN(model, "classification", stochastic=False)
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
    backend = AsdlEF(model, "classification")
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_kron_normalization_ggn_reg(reg_Xy, model):
    X, y = reg_Xy
    xi, yi = X[:1], y[:1]
    backend = AsdlGGN(model, "regression", stochastic=False)
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)
