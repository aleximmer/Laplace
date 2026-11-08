"""Contracts from AGENTS.md that every Laplace method must meet.

Each test runs over all methods on the same small multi-output regression problem:

- predictive shapes, with and without `joint`, and the input-then-output order
  of the joint moments;
- predictive moments without observation noise;
- hyperparameter changes after `fit` act like setting them before `fit`;
- `state_dict` round-trips.
"""

from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace

pytestmark = [pytest.mark.default_dtype(torch.float64), pytest.mark.seed(0)]

N_DATA, N_IN, N_OUT = 12, 3, 2

SUBNET_INDICES = torch.LongTensor([0, 4, 9, 17, 21, 25, 30])


def _make(subset, structure, **kwargs):
    return partial(
        Laplace, subset_of_weights=subset, hessian_structure=structure, **kwargs
    )


METHODS = {
    "full": _make("all", "full"),
    "kron": _make("all", "kron"),
    "diag": _make("all", "diag"),
    "lowrank": _make("all", "lowrank"),
    "ll_full": _make("last_layer", "full"),
    "ll_kron": _make("last_layer", "kron"),
    "ll_diag": _make("last_layer", "diag"),
    "subnet_full": _make("subnetwork", "full", subnetwork_indices=SUBNET_INDICES),
    "subnet_diag": _make("subnetwork", "diag", subnetwork_indices=SUBNET_INDICES),
    "functional": _make("all", "gp", n_subset=N_DATA),
    "ll_functional": _make("last_layer", "gp", n_subset=N_DATA),
    "ella": _make(
        "all",
        "gp",
        functional_approximation="nystrom",
        subsample_size=N_DATA,
        n_eigenvalues=6,
    ),
    "valla": None,  # Needs the inputs, see `_build`
}

# VaLLA fits by gradient steps on its posterior and hyperparameters. One step
# keeps the tests fast.
FIT_KWARGS = {"valla": {"iterations": 1}}

# Their objectives aren't a Laplace marginal likelihood, so they raise for it
NO_MARGLIK = {"ella", "valla"}


def _model() -> nn.Module:
    torch.manual_seed(711)
    return nn.Sequential(nn.Linear(N_IN, 5), nn.Tanh(), nn.Linear(5, N_OUT))


def _data() -> tuple[torch.Tensor, DataLoader]:
    torch.manual_seed(712)
    X = torch.randn(N_DATA, N_IN)
    y = torch.randn(N_DATA, N_OUT)
    X_test = torch.randn(4, N_IN)

    # A partial last batch, so per-batch bookkeeping must weight batches right
    return X_test, DataLoader(TensorDataset(X, y), batch_size=5)


def _build(method: str, model: nn.Module, loader: DataLoader, **hyper):
    if method == "valla":
        X = loader.dataset.tensors[0]
        return Laplace(
            model,
            "regression",
            subset_of_weights="all",
            hessian_structure="gp",
            functional_approximation="variational",
            inducing_locations=X[:4].clone(),
            **hyper,
        )

    return METHODS[method](model, "regression", **hyper)


def _fit(la, method: str, loader: DataLoader) -> None:
    la.fit(loader, **FIT_KWARGS.get(method, {}))


def _fitted(method: str, **hyper):
    model = _model()
    X_test, loader = _data()
    la = _build(method, model, loader, **hyper)
    _fit(la, method, loader)
    return la, model, X_test, loader


def _param(method: str):
    marks = [pytest.mark.asdfghjkl] if method == "lowrank" else []
    return pytest.param(method, id=method, marks=marks)


all_methods = pytest.mark.parametrize("method", [_param(m) for m in METHODS])


def _skip_lowrank_without_asdfghjkl(method: str) -> None:
    if method == "lowrank":
        pytest.importorskip("asdfghjkl")


@all_methods
def test_predictive_shapes(method):
    _skip_lowrank_without_asdfghjkl(method)
    la, _, X_test, _ = _fitted(method)
    B = len(X_test)

    mean, cov = la(X_test)
    assert mean.shape == (B, N_OUT)
    assert cov.shape == (B, N_OUT, N_OUT)

    joint_mean, joint_cov = la(X_test, joint=True)
    assert joint_mean.shape == (B * N_OUT,)
    assert joint_cov.shape == (B * N_OUT, B * N_OUT)


@all_methods
def test_joint_is_ordered_input_then_output(method):
    _skip_lowrank_without_asdfghjkl(method)
    la, _, X_test, _ = _fitted(method)
    B = len(X_test)

    mean, cov = la(X_test)
    joint_mean, joint_cov = la(X_test, joint=True)

    # Row `i * N_OUT + k` belongs to input `i` and output `k`
    torch.testing.assert_close(joint_mean, mean.reshape(-1))

    blocks = joint_cov.reshape(B, N_OUT, B, N_OUT)
    diagonal_blocks = blocks[torch.arange(B), :, torch.arange(B), :]
    torch.testing.assert_close(diagonal_blocks, cov)


@all_methods
def test_moments_exclude_observation_noise(method):
    _skip_lowrank_without_asdfghjkl(method)

    # The posterior can't be wider than the prior, whose predictive variance is
    # small for this net. The noise variance would add 1e6 to every variance.
    la, _, X_test, _ = _fitted(method, sigma_noise=1e3)
    _, joint_cov = la(X_test, joint=True)

    assert torch.linalg.eigvalsh(joint_cov).max() < 1e3


HYPERPARAMETERS = [
    pytest.param("prior_precision", 4.0, id="prior_precision"),
    pytest.param("sigma_noise", 0.5, id="sigma_noise"),
    pytest.param("temperature", 0.5, id="temperature"),
]

# VaLLA's fit also optimizes the hyperparameters, so values set before `fit`
# don't survive it. It keeps no cache derived from them either.
refresh_methods = pytest.mark.parametrize(
    "method", [_param(m) for m in METHODS if m != "valla"]
)


@refresh_methods
@pytest.mark.parametrize("name, value", HYPERPARAMETERS)
def test_hyperparameter_change_after_fit_matches_refit(method, name, value):
    _skip_lowrank_without_asdfghjkl(method)

    changed, _, X_test, _ = _fitted(method)
    setattr(changed, name, value)
    reference, _, _, _ = _fitted(method, **{name: value})

    mean, cov = changed(X_test, joint=True)
    ref_mean, ref_cov = reference(X_test, joint=True)
    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(cov, ref_cov)

    if method not in NO_MARGLIK:
        torch.testing.assert_close(
            changed.log_marginal_likelihood(), reference.log_marginal_likelihood()
        )


@all_methods
def test_state_dict_round_trip(method):
    _skip_lowrank_without_asdfghjkl(method)

    la, model, X_test, loader = _fitted(method)
    la.prior_precision = 3.0
    la.sigma_noise = 0.7

    restored = _build(method, model, loader)
    restored.load_state_dict(la.state_dict())

    mean, cov = la(X_test, joint=True)
    restored_mean, restored_cov = restored(X_test, joint=True)
    torch.testing.assert_close(restored_mean, mean)
    torch.testing.assert_close(restored_cov, cov)

    if method not in NO_MARGLIK:
        torch.testing.assert_close(
            restored.log_marginal_likelihood(), la.log_marginal_likelihood()
        )
