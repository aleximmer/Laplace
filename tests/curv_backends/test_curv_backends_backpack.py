import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace import CategoricalLikelihood, GaussianLikelihood
from laplace.likelihoods import BernoulliLikelihood
from tests.utils import (
    BackPackEF,
    BackPackGGN,
    DoubledCategorical,
    bce_data,
    bce_model,
    bce_options,
    requires_backpack,
)

pytestmark = requires_backpack


@pytest.fixture
def model_singleoutput():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 1))
    setattr(model, "output_size", 1)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


def test_full_ggn_backpack_reg_integration(reg_Xy, model):
    X, y = reg_Xy

    # cannot test, its implemented based on Jacobians.
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, H_ggn = backend.full(X, y)
    assert H_ggn.size() == torch.Size((model.n_params, model.n_params))


def test_full_ggn_backpack_cls_integration(class_Xy, model):
    X, y = class_Xy
    # cannot test, its implemented based on Jacobians.
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, H_ggn = backend.full(X, y)
    assert H_ggn.size() == torch.Size((model.n_params, model.n_params))


def test_diag_ggn_cls_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert loss == loss_f
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ggn_reg_backpack(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert loss == loss_f
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ggn_stoch_cls_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # same order of magnitude os non-stochastic.
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1e1)


def test_diag_ggn_stoch_reg_backpack(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # same order of magnitude os non-stochastic.
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1e1)


def test_kron_ggn_reg_backpack_vs_diag_reg(reg_Xy, model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = reg_Xy
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


def test_kron_batching_correction_reg(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = int(N / 2)
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


def test_kron_summing_up_vs_diag_reg(reg_Xy, model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = reg_Xy
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


def test_kron_ggn_reg_backpack_vs_diag_class(class_Xy, model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = class_Xy
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


def test_kron_batching_correction_class(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = int(N / 2)
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


def test_kron_summing_up_vs_diag_class(class_Xy, model):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = class_Xy
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


def test_full_vs_diag_ef_cls_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackEF(model, CategoricalLikelihood())
    loss, diag_ef = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(diag_ef) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model, CategoricalLikelihood())
    loss_f, H_ef = backend.full(X, y)
    assert loss == loss_f
    assert torch.allclose(diag_ef, H_ef.diagonal())


def test_full_vs_diag_ef_reg_backpack(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackEF(model, GaussianLikelihood())
    loss, diag_ef = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(diag_ef) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model, GaussianLikelihood())
    loss_f, H_ef = backend.full(X, y)
    assert loss == loss_f
    assert torch.allclose(diag_ef, H_ef.diagonal())


def test_kron_normalization_reg(reg_Xy, model):
    X, y = reg_Xy
    xi, yi = X[:1], y[:1]
    backend = BackPackGGN(model, GaussianLikelihood(), stochastic=False)
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_kron_normalization_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = BackPackGGN(model, CategoricalLikelihood(), stochastic=False)
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_custom_likelihood_rejected_by_kron(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, DoubledCategorical())

    with pytest.raises(ValueError, match="diag"):
        backend.kron(X, y, N=len(X))


# Bernoulli likelihood. BackPACK has no sigmoid loss with options, so the GGN and EF
# backends fall back to the torch.func code for `diag` and `full`, and `kron` raises.
_BCE = bce_options()


def test_bernoulli_does_not_extend_the_loss():
    # `extend` would turn the loss into a BackPACK module that only knows the
    # losses of Gaussian and categorical likelihoods
    backend = BackPackGGN(bce_model(), BernoulliLikelihood())

    assert not getattr(backend.lossfunc, "_backpack_extend", False)


@pytest.mark.parametrize("Backend", [BackPackGGN, BackPackEF])
@pytest.mark.parametrize("stochastic", [False, True])
def test_bernoulli_kron_raises_in_backpack(Backend, stochastic):
    X, y = bce_data((8,), soft=False)
    kwargs = {"stochastic": True} if stochastic and Backend is BackPackGGN else {}
    backend = Backend(bce_model(), BernoulliLikelihood(), **kwargs)

    if Backend is BackPackEF:
        with pytest.raises(NotImplementedError):
            backend.kron(X, y, len(X))
    else:
        with pytest.raises(ValueError, match="does not support the likelihood"):
            backend.kron(X, y, len(X))


def test_bernoulli_stochastic_diag_raises_for_options_in_backpack():
    X, y = bce_data((8,), soft=False)
    weighted = BernoulliLikelihood(pos_weight=torch.tensor([3.0, 0.5, 2.0]))

    with pytest.raises(ValueError, match="stochastic=True"):
        BackPackGGN(bce_model(), weighted, stochastic=True).diag(X, y)

    # Without options, the Monte Carlo Fisher is the generic one
    _, diag = BackPackGGN(bce_model(), BernoulliLikelihood(), stochastic=True).diag(
        X, y
    )
    assert torch.isfinite(diag).all()
