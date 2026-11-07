"""Curvature backends with class probabilities as targets, e.g. one-hot or soft labels.

Class probabilities have the shape of the logits `(B, ..., K)` and a floating-point
dtype, like the targets of `nn.CrossEntropyLoss`. Every position of the batch dims is
an observation, and the class dim is not a position.
"""

from collections.abc import MutableMapping
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.curvature import (
    CurvatureInterface,
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
    GGNInterface,
)
from laplace.likelihoods import CategoricalLikelihood
from tests.utils import (
    AsdlEF,
    AsdlGGN,
    AsdlHessian,
    BackPackEF,
    BackPackGGN,
    ListDataset,
    _kron_closed_form,
    dict_data_collator,
    loss_hessian,
)

pytestmark = pytest.mark.default_dtype(torch.float64)

D, K = 4, 3

# Batch shapes of the logits `(B, K)`, `(B, L, K)`, and `(B, L, K)` with `L == K`,
# where the class dim could be mistaken for the position dim
SHAPES = [(8,), (8, 5), (4, 3)]
BATCH_SHAPES = [pytest.param(shape, id="x".join(map(str, shape))) for shape in SHAPES]


def mlp() -> nn.Module:
    """Small float64 model with `K` outputs, applied to each position on its own."""
    torch.manual_seed(711)
    model = nn.Sequential(nn.Linear(D, 5), nn.Tanh(), nn.Linear(5, K)).double()
    setattr(model, "output_size", K)

    return model


def linear() -> nn.Module:
    """A single linear layer, which is linear in its parameters."""
    torch.manual_seed(711)
    model = nn.Linear(D, K).double()
    setattr(model, "output_size", K)

    return model


def soft_data(batch_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs `(*batch_shape, D)` and class probabilities `(*batch_shape, K)`."""
    torch.manual_seed(5)
    X = torch.randn(*batch_shape, D, dtype=torch.float64)
    y = torch.softmax(torch.randn(*batch_shape, K, dtype=torch.float64), dim=-1)

    return X, y


def soft_ce(f: torch.Tensor, y: torch.Tensor, **options: Any) -> torch.Tensor:
    """Summed torch cross-entropy of the logits `f` `(..., K)` and the class
    probabilities `y`, where every leading dim is a position."""
    return F.cross_entropy(
        f.reshape(-1, K), y.reshape(-1, K), reduction="sum", **options
    )


def is_backpack(Backend: type[CurvatureInterface]) -> bool:
    return Backend.__name__.startswith("BackPack")


def cases(
    backends: list[type[CurvatureInterface]], structures: list[str]
) -> list[tuple[type[CurvatureInterface], str, tuple[int, ...]]]:
    """Backend, structure, and batch shape. BackPACK has no sequences."""
    return [
        pytest.param(
            Backend,
            structure,
            batch_shape,
            id=f"{Backend.__name__}-{structure}-{'x'.join(map(str, batch_shape))}",
        )
        for Backend in backends
        for structure in structures
        for batch_shape in SHAPES
        if not is_backpack(Backend) or len(batch_shape) == 1
    ]


def curvature(
    backend: CurvatureInterface, structure: str, X: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and curvature of `structure`, as a dense matrix for `kron`."""
    if structure == "kron":
        loss, H = backend.kron(X, y, len(X))

        return loss, H.to_matrix()

    return getattr(backend, structure)(X, y)


def position_gradients(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Gradients of the soft cross-entropy of each position w.r.t. the parameters,
    `(positions, parameters)`. The model must act on each position on its own."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def position_loss(
        params: dict[str, torch.Tensor], x_t: torch.Tensor, y_t: torch.Tensor
    ) -> torch.Tensor:
        f_t = torch.func.functional_call(model, params, (x_t[None],))

        return F.cross_entropy(f_t, y_t[None], reduction="sum")

    grads = torch.func.vmap(torch.func.grad(position_loss), in_dims=(None, 0, 0))(
        params, X.reshape(-1, D), y.reshape(-1, K)
    )

    return torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)


@pytest.mark.parametrize(
    "Backend,structure,batch_shape",
    cases([CurvlinopsEF, AsdlEF, BackPackEF], ["diag", "full"]),
)
def test_ef_matches_per_position_gradients(Backend, structure, batch_shape):
    # The library treats every position as an observation, so the empirical Fisher
    # sums the outer products of the gradients of the positions
    X, y = soft_data(batch_shape)
    model = mlp()
    G = position_gradients(model, X, y)

    loss, H = getattr(Backend(model, "classification"), structure)(X, y)

    expected = G.T @ G if structure == "full" else (G**2).sum(0)

    assert H.dtype == torch.float64
    assert torch.allclose(H, expected, atol=1e-8, rtol=1e-6)
    assert torch.allclose(loss, soft_ce(mlp()(X), y))


@pytest.mark.parametrize(
    "Backend,structure,batch_shape",
    cases([CurvlinopsGGN, AsdlGGN, BackPackGGN], ["diag", "full", "kron"]),
)
def test_ggn_equals_ggn_of_argmax_labels(Backend, structure, batch_shape):
    # The GGN depends on the model only, while the loss comes from the soft targets
    X, y = soft_data(batch_shape)
    backend = Backend(mlp(), "classification")

    loss, H = curvature(backend, structure, X, y)
    _, H_hard = curvature(backend, structure, X, y.argmax(-1))

    assert torch.allclose(H, H_hard)
    assert torch.allclose(loss, soft_ce(mlp()(X), y))

    if structure != "kron":
        # The generic torch.func code is an independent reference
        _, H_ref = curvature(GGNInterface(mlp(), "classification"), structure, X, y)
        assert torch.allclose(H, H_ref, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_weighted_soft_ggn_full_matches_torch_func(batch_shape):
    # Class weights scale the curvature of a position by `q @ w`, with `q` the soft
    # target. The weights are float32, the model float64.
    X, y = soft_data(batch_shape)
    likelihood = CategoricalLikelihood(weight=torch.tensor([0.5, 2.0, 1.5]))
    backend = CurvlinopsGGN(mlp(), likelihood)

    assert backend._honours("full", y)

    loss, H = backend.full(X, y)
    loss_ref, H_ref = GGNInterface(mlp(), likelihood).full(X, y)

    assert H.dtype == torch.float64
    assert torch.allclose(H, H_ref, atol=1e-8, rtol=1e-6)
    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(loss, soft_ce(mlp()(X), y, weight=likelihood.weight.double()))


@pytest.mark.parametrize(
    "Backend,structure,batch_shape",
    cases([CurvlinopsHessian, AsdlHessian], ["full"]),
)
def test_hessian_matches_autograd(Backend, structure, batch_shape):
    X, y = soft_data(batch_shape)
    model = mlp()

    loss, H = Backend(model, "classification").full(X, y)

    assert torch.allclose(H, loss_hessian(mlp(), soft_ce, X, y), atol=1e-8, rtol=1e-6)
    assert torch.allclose(loss, soft_ce(mlp()(X), y))


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("Backend", [CurvlinopsEF, AsdlEF])
def test_kron_ef_matches_hand_computation(Backend, batch_shape, label_smoothing):
    X, y = soft_data(batch_shape)
    model = linear()
    likelihood = CategoricalLikelihood(label_smoothing=label_smoothing)
    N = len(X)

    loss, kron = Backend(model, likelihood).kron(X, y, N)

    # Output factor: sum over positions of g g^T, with g the gradient of the torch
    # loss w.r.t. the logits. Input factor: sum of x x^T over positions, over N.
    f = linear()(X).detach().requires_grad_(True)
    ce = soft_ce(f, y, label_smoothing=label_smoothing)
    (g,) = torch.autograd.grad(ce, f)
    (exp_w_B, exp_w_A), (exp_b_B,) = _kron_closed_form(
        X, g.reshape(-1, K).T @ g.reshape(-1, K), N
    )

    assert torch.allclose(loss, ce.detach())

    weight, bias = kron.kfacs
    assert torch.allclose(weight[0], exp_w_B)
    assert torch.allclose(weight[1], exp_w_A)
    assert torch.allclose(bias[0], exp_b_B)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_kron_ggn_matches_hand_computation(batch_shape):
    X, y = soft_data(batch_shape)
    N = len(X)

    loss, kron = CurvlinopsGGN(linear(), "classification").kron(X, y, N)

    # The output factor is the sum of diag(p) - p p^T over positions, and it does not
    # depend on the targets
    p = torch.softmax(linear()(X).detach(), dim=-1).reshape(-1, K)
    (exp_w_B, exp_w_A), (exp_b_B,) = _kron_closed_form(
        X, (torch.diag_embed(p) - p[:, :, None] * p[:, None, :]).sum(0), N
    )

    assert torch.allclose(loss, soft_ce(linear()(X), y))

    weight, bias = kron.kfacs
    assert torch.allclose(weight[0], exp_w_B)
    assert torch.allclose(weight[1], exp_w_A)
    assert torch.allclose(bias[0], exp_b_B)


@pytest.mark.parametrize(
    "Backend,structure",
    [
        (CurvlinopsGGN, "kron"),
        (CurvlinopsGGN, "full"),
        (AsdlGGN, "kron"),
        (AsdlGGN, "diag"),
    ],
)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_stochastic_soft_loss_is_real_loss(Backend, structure, batch_shape):
    # The Monte Carlo Fisher samples its own labels. The loss is still the one of the
    # soft targets, and the curvature does not depend on them.
    X, y = soft_data(batch_shape)
    backend = Backend(mlp(), "classification", stochastic=True)

    torch.manual_seed(0)
    loss, H = curvature(backend, structure, X, y)
    torch.manual_seed(0)
    _, H_hard = curvature(backend, structure, X, y.argmax(-1))

    assert torch.isfinite(H).all()
    assert torch.allclose(loss, soft_ce(mlp()(X), y))
    assert torch.allclose(H, H_hard)


@pytest.mark.parametrize(
    "Backend,structure,batch_shape",
    cases([CurvlinopsGGN, AsdlGGN, BackPackGGN], ["diag", "full", "kron"])
    + cases([CurvlinopsEF, AsdlEF], ["diag", "full", "kron"])
    + cases([BackPackEF], ["diag", "full"])
    + cases([CurvlinopsHessian, AsdlHessian], ["full"]),
)
def test_float_one_hot_equals_index(Backend, structure, batch_shape):
    X, y = soft_data(batch_shape)
    y = y.argmax(-1)
    y_one_hot = F.one_hot(y, K).to(X.dtype)

    loss_index, H_index = curvature(Backend(mlp(), "classification"), structure, X, y)
    loss_one_hot, H_one_hot = curvature(
        Backend(mlp(), "classification"), structure, X, y_one_hot
    )

    assert torch.allclose(loss_one_hot, loss_index)
    assert torch.allclose(H_one_hot, H_index, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("Backend", [CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF])
def test_weight_kron_still_raises_with_soft_targets(Backend):
    X, y = soft_data((8,))
    likelihood = CategoricalLikelihood(weight=torch.tensor([0.5, 2.0, 1.5]))

    with pytest.raises(ValueError, match="does not support the option weight"):
        Backend(linear(), likelihood).kron(X, y, len(X))


class DictLinear(nn.Module):
    """Linear model that reads its input from a dict-like batch."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, data: MutableMapping) -> torch.Tensor:
        return self.linear(data["input_ids"])


@pytest.mark.parametrize("Backend", [CurvlinopsGGN, CurvlinopsEF])
def test_curvlinops_kron_dict_batch_soft_labels(Backend):
    X, y = soft_data((10,))
    data = [{"input_ids": x_i, "labels": y_i} for x_i, y_i in zip(X, y)]
    batch = dict_data_collator(data)
    N = len(X)

    loss, kron = Backend(DictLinear(linear()), "classification").kron(
        batch, batch["labels"], N
    )
    loss_ref, kron_ref = Backend(linear(), "classification").kron(X, y, N)

    assert torch.allclose(loss, loss_ref)
    assert torch.allclose(kron.to_matrix(), kron_ref.to_matrix())


@pytest.mark.parametrize("Backend", [CurvlinopsGGN, CurvlinopsEF])
def test_laplace_fit_dict_loader_soft_labels(Backend):
    X, y = soft_data((10,))
    data = [{"input_ids": x_i, "labels": y_i} for x_i, y_i in zip(X, y)]
    loader = DataLoader(ListDataset(data), batch_size=5, collate_fn=dict_data_collator)

    options = dict(subset_of_weights="all", hessian_structure="kron", backend=Backend)

    la = Laplace(DictLinear(linear()), "classification", **options)
    la.fit(loader)

    la_ref = Laplace(linear(), "classification", **options)
    la_ref.fit(DataLoader(TensorDataset(X, y), batch_size=5))

    assert la.n_obs == la_ref.n_obs == 10
    assert torch.allclose(la.loss, la_ref.loss)
    assert torch.allclose(la.H.to_matrix(), la_ref.H.to_matrix())


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("soft", [False, True])
def test_last_layer_ef_matches_hand_computation(structure, soft):
    # The empirical Fisher of the torch.func code must not vanish for a last layer
    X, q = soft_data((8,))
    model = mlp()
    y = q if soft else q.argmax(dim=-1)

    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure=structure,
        backend=CurvlinopsEF,
    )
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4))

    # The weight of the last layer comes before its bias
    features = model[:2](X).detach()
    target = q if soft else F.one_hot(y, K).double()
    g = model[2](features).softmax(dim=-1).detach() - target
    G = torch.cat([torch.einsum("bk,bd->bkd", g, features).flatten(1), g], dim=1)
    H = G.T @ G

    assert H.abs().max() > 0
    assert torch.allclose(la.H, H if structure == "full" else H.diagonal())


@pytest.mark.parametrize("structure", ["full", "diag"])
def test_last_layer_ef_with_buffers_in_the_features(structure):
    # The buffers of the feature extractor must not reach the last layer's call
    X, q = soft_data((8,))
    torch.manual_seed(711)
    model = nn.Sequential(
        nn.Linear(D, 5), nn.BatchNorm1d(5), nn.Tanh(), nn.Linear(5, K)
    ).double()
    model.eval()

    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure=structure,
        backend=CurvlinopsEF,
    )
    la.fit(DataLoader(TensorDataset(X, q), batch_size=4))

    features = model[:3](X).detach()
    g = model[3](features).softmax(dim=-1).detach() - q
    G = torch.cat([torch.einsum("bk,bd->bkd", g, features).flatten(1), g], dim=1)
    H = G.T @ G

    assert torch.allclose(la.H, H if structure == "full" else H.diagonal())


@pytest.mark.parametrize("structure", ["full", "diag"])
@pytest.mark.parametrize("subset", ["last_layer", "subnetwork"])
def test_lllaplace_and_subnet_soft_targets(subset, structure):
    torch.manual_seed(3)
    X, y = soft_data((12,))
    y_index = y.argmax(-1)
    y_one_hot = F.one_hot(y_index, K).to(X.dtype)
    kwargs = {}

    if subset == "subnetwork":
        kwargs["subnetwork_indices"] = torch.LongTensor([0, 3, 8, 20, 30, 31])

    def fit(targets: torch.Tensor) -> Any:
        la = Laplace(
            mlp(),
            "classification",
            subset_of_weights=subset,
            hessian_structure=structure,
            **kwargs,
        )
        la.fit(DataLoader(TensorDataset(X, targets), batch_size=4))

        return la

    la_soft, la_one_hot, la_index, la_hard = (
        fit(y),
        fit(y_one_hot),
        fit(y_index),
        fit(y.argmax(-1)),
    )

    assert la_soft.n_obs == la_one_hot.n_obs == la_index.n_obs == len(X)

    # One-hot probabilities are the indices
    assert torch.allclose(la_one_hot.loss, la_index.loss)
    assert torch.allclose(la_one_hot.H, la_index.H)

    # The loss is that of the soft targets, the GGN does not depend on them
    assert torch.allclose(la_soft.loss, soft_ce(mlp()(X), y))
    assert torch.allclose(la_soft.H, la_hard.H)
    assert torch.isfinite(la_soft.log_marginal_likelihood())


@pytest.mark.asdfghjkl
def test_asdfghjkl_soft_targets():
    # These paths are only checked by the CI job of the old asdfghjkl package
    pytest.importorskip("asdfghjkl")
    from laplace.curvature import AsdfghjklEF, AsdfghjklGGN, AsdfghjklHessian

    X, y = soft_data((8,))
    N = len(X)

    ggn = AsdfghjklGGN(mlp(), "classification")

    # The GGN doesn't depend on the targets, the loss comes from the soft ones
    for structure in ["diag", "kron"]:
        loss, H = curvature(ggn, structure, X, y)
        _, H_hard = curvature(ggn, structure, X, y.argmax(-1))

        assert torch.allclose(loss, soft_ce(mlp()(X), y))
        assert torch.allclose(H, H_hard)

    # The empirical Fisher of the diagonal falls back to per-position gradients
    G = position_gradients(mlp(), X, y)
    _, diag_ef = AsdfghjklEF(mlp(), "classification").diag(X, y)
    assert torch.allclose(diag_ef, (G**2).sum(0), atol=1e-8, rtol=1e-6)

    # The other methods have no soft targets, and the error says where to go instead
    with pytest.raises(ValueError, match="CurvlinopsEF"):
        AsdfghjklEF(mlp(), "classification").kron(X, y, N)

    with pytest.raises(ValueError, match="CurvlinopsHessian"):
        AsdfghjklHessian(mlp(), "classification").full(X, y)
