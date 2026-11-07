import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import (
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
    EFInterface,
    GGNInterface,
)
from laplace.likelihoods import BernoulliLikelihood, CategoricalLikelihood
from tests.utils import AsdlEF, AsdlGGN, AsdlHessian, BackPackGGN, DoubledCategorical


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


@pytest.mark.parametrize("structure", ["diag", "full"])
def test_custom_likelihood_honoured_by_curvlinops_ggn(class_Xy, model, structure):
    # The custom `output_hessian` is used through the torch.func fallback
    X, y = class_Xy

    _, H = getattr(CurvlinopsGGN(model, "classification"), structure)(X, y)
    _, H_doubled = getattr(CurvlinopsGGN(model, DoubledCategorical()), structure)(X, y)

    # Curvlinops and torch.func round differently in float32
    assert torch.allclose(H_doubled, 2 * H, atol=1e-6)


@pytest.mark.parametrize("structure", ["diag", "full"])
def test_custom_likelihood_curvlinops_ef_equals_builtin(class_Xy, model, structure):
    # The EF depends only on the loss, which `DoubledCategorical` keeps
    X, y = class_Xy

    _, H = getattr(CurvlinopsEF(model, "classification"), structure)(X, y)
    _, H_doubled = getattr(CurvlinopsEF(model, DoubledCategorical()), structure)(X, y)

    assert torch.allclose(H_doubled, H)


@pytest.mark.parametrize("Backend", [CurvlinopsGGN, CurvlinopsEF, CurvlinopsHessian])
def test_custom_likelihood_rejected_by_curvlinops_kron_and_hessian(
    class_Xy, model, Backend
):
    X, y = class_Xy
    backend = Backend(model, DoubledCategorical())

    # Equal to the built-in by name, but still not honoured, since it isn't one
    assert backend.likelihood == "classification"

    if Backend is CurvlinopsHessian:
        with pytest.raises(ValueError, match="diag"):
            backend.full(X, y)
    else:
        with pytest.raises(ValueError, match="diag"):
            backend.kron(X, y, N=len(X))


def test_custom_likelihood_used_by_curvlinops_ggn_diag(class_Xy, model):
    # The diagonal GGN goes through `GGNInterface`, which uses `output_hessian`
    X, y = class_Xy

    _, diag = CurvlinopsGGN(model, "classification").diag(X, y)
    _, diag_doubled = CurvlinopsGGN(model, DoubledCategorical()).diag(X, y)

    assert torch.allclose(diag_doubled, 2 * diag)


def test_custom_likelihood_jacobians_curvlinops(class_Xy, model):
    X, _ = class_Xy

    Js, f = CurvlinopsGGN(model, "classification").jacobians(X)
    Js_doubled, f_doubled = CurvlinopsGGN(model, DoubledCategorical()).jacobians(X)

    assert torch.allclose(Js, Js_doubled)
    assert torch.allclose(f, f_doubled)


# Bernoulli likelihood. Float32 options for float64 models, so the backends must
# cast them.
_BCE_OPTIONS = [
    {},
    {"weight": torch.tensor([0.5, 2.0, 1.5])},
    {"pos_weight": torch.tensor([3.0, 0.5, 2.0])},
    {
        "weight": torch.tensor([0.5, 2.0, 1.5]),
        "pos_weight": torch.tensor([3.0, 0.5, 2.0]),
    },
    {"weight": torch.tensor(0.7), "pos_weight": torch.tensor(2.5)},
]

# Outputs `(B, K)` and `(B, L, K)`, with `K = 3` outputs
_BCE_BATCH_SHAPES = [(8,), (4, 2)]


def _bce_model(kron: bool = False) -> nn.Module:
    """Small float64 model with 3 outputs. For Kronecker factors, a single linear
    layer, which is linear in its parameters."""
    torch.manual_seed(711)

    if kron:
        model = nn.Linear(3, 3)
    else:
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3))

    setattr(model, "output_size", 3)

    return model.double()


def _bce_data(
    batch_shape: tuple[int, ...], soft: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 inputs `(*batch_shape, 3)` and hard or soft targets of 3 outputs."""
    torch.manual_seed(711)
    X = torch.randn(*batch_shape, 3, dtype=torch.float64)
    y = torch.rand(*batch_shape, 3, dtype=torch.float64)

    return X, y if soft else (y > 0.5).to(y.dtype)


def _loss_hessian(
    model: nn.Module, criterion: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Hessian of the torch loss w.r.t. the flattened parameters, via torch.func."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    theta = torch.cat([v.flatten() for v in params.values()])

    def loss(theta: torch.Tensor) -> torch.Tensor:
        split, offset = {}, 0

        for k, v in params.items():
            split[k] = theta[offset : offset + v.numel()].reshape(v.shape)
            offset += v.numel()

        return criterion(torch.func.functional_call(model, split, (X,)), y)

    return torch.func.hessian(loss)(theta)


@pytest.mark.parametrize("opts", _BCE_OPTIONS)
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
@pytest.mark.parametrize(
    "Backend,Reference", [(CurvlinopsGGN, GGNInterface), (CurvlinopsEF, EFInterface)]
)
def test_bernoulli_full_matches_torch_func(Backend, Reference, batch_shape, soft, opts):
    X, y = _bce_data(batch_shape, soft)
    likelihood = BernoulliLikelihood(**opts)

    backend = Backend(_bce_model(), likelihood)

    # Curvlinops computes it, instead of the torch.func fallback that it is
    # compared with
    assert backend._honours("full", y)

    loss, H = backend.full(X, y)
    loss_ref, H_ref = Reference(_bce_model(), likelihood).full(X, y)

    assert H.dtype == torch.float64
    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(H, H_ref, atol=1e-5, rtol=1e-5)

    # The loss is the summed BCE
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()
    assert torch.allclose(loss, bce(_bce_model()(X), y))


@pytest.mark.parametrize("opts", _BCE_OPTIONS)
@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_full_ef_matches_autograd(batch_shape, opts):
    # The empirical Fisher is the sum of the outer products of the per-position
    # gradients of the torch loss
    X, y = _bce_data(batch_shape, soft=True)
    model = _bce_model()
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()

    def position_loss(params, x_t, y_t):
        f_t = torch.func.functional_call(model, params, (x_t[None],))
        return bce(f_t, y_t[None])

    params = {k: v.detach() for k, v in model.named_parameters()}
    grads = torch.func.vmap(torch.func.grad(position_loss), in_dims=(None, 0, 0))(
        params, X.reshape(-1, 3), y.reshape(-1, 3)
    )
    G = torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)

    _, H = CurvlinopsEF(model, BernoulliLikelihood(**opts)).full(X, y)

    assert torch.allclose(H, G.T @ G, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("opts", _BCE_OPTIONS)
@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
@pytest.mark.parametrize("Backend", [CurvlinopsGGN, CurvlinopsHessian])
def test_bernoulli_full_matches_autograd_hessian(Backend, batch_shape, opts):
    # A single linear layer is linear in its parameters, so the Hessian of the loss
    # is the GGN
    X, y = _bce_data(batch_shape, soft=True)
    model = _bce_model(kron=True)
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()

    _, H = Backend(model, BernoulliLikelihood(**opts)).full(X, y)

    assert torch.allclose(H, _loss_hessian(model, bce, X, y), atol=1e-5, rtol=1e-5)


def _kron_closed_form(
    X: torch.Tensor, B: torch.Tensor, N: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Factors that KFAC gives for a single linear layer with bias: the output
    factor `B` summed over positions and the input factor summed over positions
    and divided by `N` for the weight, and `B` for the bias."""
    X = X.reshape(-1, X.shape[-1])

    return [B, X.T @ X / N], [B]


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_kron_ggn_matches_closed_form(batch_shape, soft):
    X, y = _bce_data(batch_shape, soft)
    model = _bce_model(kron=True)
    N = len(X)

    loss, kron = CurvlinopsGGN(model, "bernoulli").kron(X, y, N)

    # The loss comes from the real targets, soft or not
    f = model(X).detach()
    assert torch.allclose(loss, nn.BCEWithLogitsLoss(reduction="sum").double()(f, y))

    # The output factor is the sum of diag(p (1 - p)) over positions
    p = torch.sigmoid(f).reshape(-1, 3)
    (exp_w_B, exp_w_A), (exp_b_B,) = _kron_closed_form(
        X, torch.diag_embed(p * (1 - p)).sum(0), N
    )
    weight, bias = kron.kfacs

    assert torch.allclose(weight[0], exp_w_B)
    assert torch.allclose(weight[1], exp_w_A)
    assert torch.allclose(bias[0], exp_b_B)


@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_kron_mc_input_factor_matches_closed_form(batch_shape):
    X, y = _bce_data(batch_shape, soft=True)
    model = _bce_model(kron=True)

    _, kron = CurvlinopsGGN(model, "bernoulli", stochastic=True).kron(X, y, len(X))

    # The input factor does not depend on the sampled labels
    _, exp_w_A = _kron_closed_form(X, torch.eye(3), len(X))[0]
    assert torch.allclose(kron.kfacs[0][1], exp_w_A)


@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_kron_ggn_matches_the_ggn_blocks_for_a_single_position(batch_shape):
    X, y = _bce_data((1,) * len(batch_shape), soft=True)
    model = _bce_model(kron=True)

    _, kron = CurvlinopsGGN(model, "bernoulli").kron(X, y, 1)
    _, H = CurvlinopsGGN(model, "bernoulli").full(X, y)
    K = kron.to_matrix()

    # KFAC is exact within the weight and within the bias, but drops their
    # cross-covariance
    assert torch.allclose(K[:9, :9], H[:9, :9], atol=1e-5, rtol=1e-5)
    assert torch.allclose(K[9:, 9:], H[9:, 9:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
@pytest.mark.parametrize("stochastic", [False, True])
def test_bernoulli_kron_ggn_ignores_soft_labels(batch_shape, stochastic):
    # The Fisher of the model does not depend on the targets, and Curvlinops
    # rejects soft ones, so they are rounded for it
    X, y = _bce_data(batch_shape, soft=True)
    model = _bce_model(kron=True)

    torch.manual_seed(0)
    _, kron_soft = CurvlinopsGGN(model, "bernoulli", stochastic=stochastic).kron(
        X, y, len(X)
    )
    torch.manual_seed(0)
    _, kron_hard = CurvlinopsGGN(model, "bernoulli", stochastic=stochastic).kron(
        X, (y >= 0.5).to(y.dtype), len(X)
    )

    assert torch.allclose(kron_soft.to_matrix(), kron_hard.to_matrix())


@pytest.mark.parametrize("opts", _BCE_OPTIONS[1:])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_kron_ef_matches_hand_computation(batch_shape, soft, opts):
    X, y = _bce_data(batch_shape, soft)
    model = _bce_model(kron=True)
    likelihood = BernoulliLikelihood(**opts)
    N = len(X)

    backend = CurvlinopsEF(model, likelihood)
    assert backend._honours("kron")

    loss, kron = backend.kron(X, y, N)

    # Output factor: sum over positions of g g^T, with g the gradient of the torch
    # loss w.r.t. the logits. Input factor: sum of x x^T over positions, over N.
    f = model(X).detach().requires_grad_(True)
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()
    (g,) = torch.autograd.grad(bce(f, y), f)
    g = g.reshape(-1, 3)
    (exp_w_B, exp_w_A), (exp_b_B,) = _kron_closed_form(X, g.T @ g, N)

    assert torch.allclose(loss, bce(f.detach(), y))

    weight, bias = kron.kfacs
    assert weight[0].dtype == torch.float64
    assert torch.allclose(weight[0], exp_w_B)
    assert torch.allclose(weight[1], exp_w_A)
    assert torch.allclose(bias[0], exp_b_B)

    # It is the same gradient that `output_gradient` gives
    g_lik = likelihood.output_gradient(f.detach(), y).reshape(-1, 3)
    assert torch.allclose(weight[0], g_lik.T @ g_lik)


@pytest.mark.parametrize("opts", _BCE_OPTIONS[1:])
@pytest.mark.parametrize("stochastic", [False, True])
def test_bernoulli_kron_options_raise_unless_ef_honours_them(opts, stochastic):
    X, y = _bce_data((8,), soft=False)
    model = _bce_model(kron=True)
    likelihood = BernoulliLikelihood(**opts)

    # KFAC Type-2 and MC KFAC silently ignore `weight` and `pos_weight`, so the GGN
    # backend must raise instead of returning the unweighted factors. See
    # https://github.com/f-dangel/curvlinops/issues/311
    with pytest.raises(ValueError, match="does not support the option"):
        CurvlinopsGGN(model, likelihood, stochastic=stochastic).kron(X, y, len(X))

    # The empirical Fisher honours them
    CurvlinopsEF(model, likelihood).kron(X, y, len(X))


def test_bernoulli_kron_without_options_runs_everywhere():
    X, y = _bce_data((8,), soft=False)
    model = _bce_model(kron=True)

    for backend in (
        CurvlinopsGGN(model, "bernoulli"),
        CurvlinopsGGN(model, "bernoulli", stochastic=True),
        CurvlinopsEF(model, "bernoulli"),
    ):
        _, kron = backend.kron(X, y, len(X))
        assert torch.isfinite(kron.to_matrix()).all()


def test_ef_kron_exemption_is_bernoulli_only(class_Xy, model):
    X, y = class_Xy
    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="does not support the option weight"):
        CurvlinopsEF(model, weighted).kron(X, y, len(X))


@pytest.mark.parametrize("batch_shape", _BCE_BATCH_SHAPES)
def test_bernoulli_stochastic_full_takes_soft_labels(batch_shape):
    X, y_soft = _bce_data(batch_shape, soft=True)
    _, y_hard = _bce_data(batch_shape, soft=False)

    backend = CurvlinopsGGN(_bce_model(), "bernoulli", stochastic=True)

    # Curvlinops' dense MC Fisher takes hard labels only, so soft ones are not
    # honoured and go to the generic Monte Carlo Fisher, which ignores the targets
    assert backend._honours("full", y_hard)
    assert not backend._honours("full", y_soft)
    assert backend._honours("full")

    reference = GGNInterface(_bce_model(), "bernoulli", stochastic=True)

    torch.manual_seed(0)
    loss, H = backend.full(X, y_soft)
    torch.manual_seed(0)
    loss_ref, H_ref = reference.full(X, y_soft)

    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(H, H_ref)

    # With more samples, both approach the exact GGN
    _, H_exact = CurvlinopsGGN(_bce_model(), "bernoulli").full(X, y_hard)
    torch.manual_seed(0)
    _, H_hard = backend.full(X, y_hard, mc_samples=400)

    backend.num_samples = 400
    torch.manual_seed(0)
    _, H_soft = backend.full(X, y_soft)

    for H_mc in (H_hard, H_soft):
        assert torch.norm(H_mc - H_exact) / torch.norm(H_exact) < 0.2


def test_bernoulli_stochastic_options_raise():
    X, y = _bce_data((8,), soft=False)
    model = _bce_model()
    weighted = BernoulliLikelihood(pos_weight=torch.tensor([3.0, 0.5, 2.0]))
    backend = CurvlinopsGGN(model, weighted, stochastic=True)

    for targets in (y, 0.9 * y):
        with pytest.raises(ValueError, match="stochastic=True"):
            backend.full(X, targets)

    with pytest.raises(ValueError, match="stochastic=True"):
        backend.diag(X, y)


def test_bernoulli_diag_and_jacobians():
    X, y = _bce_data((8,), soft=True)
    backend = CurvlinopsGGN(_bce_model(), "bernoulli")

    Js, f = backend.jacobians(X)
    _, diag = backend.diag(X, y)
    _, H = backend.full(X, y)

    p = torch.sigmoid(f)
    expected = torch.einsum("ncp,nc,ncp->p", Js, p * (1 - p), Js)

    assert torch.allclose(diag, expected)
    assert torch.allclose(diag, H.diagonal(), atol=1e-5, rtol=1e-5)
