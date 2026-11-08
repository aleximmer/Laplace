import inspect
import math
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import marglik_training
from laplace.curvature import CurvlinopsEF, CurvlinopsGGN
from laplace.likelihoods import (
    BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
)
from tests.utils import toy_multilabel_dataset


@pytest.fixture
def model():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )


def _loader(likelihood: str) -> DataLoader:
    X = torch.randn(32, 3)

    if likelihood == "regression":
        y = torch.randn(32, 2)
    else:
        y = torch.randint(0, 2, (32,))

    return DataLoader(TensorDataset(X, y), batch_size=8)


def test_default_backend_is_curvlinops():
    default = inspect.signature(marglik_training).parameters["backend"].default
    assert default is CurvlinopsGGN


@pytest.mark.parametrize("likelihood", ["regression", "classification"])
@pytest.mark.parametrize("hessian_structure", ["full", "kron", "diag"])
def test_marglik_training_default_backend(model, likelihood, hessian_structure):
    la, model, margliks, losses = marglik_training(
        model,
        _loader(likelihood),
        likelihood=likelihood,
        hessian_structure=hessian_structure,
        n_epochs=2,
    )

    assert isinstance(la.backend, CurvlinopsGGN)
    assert all(math.isfinite(m) for m in margliks)


@pytest.mark.parametrize(
    "likelihood,obj",
    [("regression", GaussianLikelihood()), ("classification", CategoricalLikelihood())],
)
def test_marglik_training_object_equivalence(likelihood, obj):
    results = []

    for lik in (likelihood, obj):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
        )
        loader = _loader(likelihood)

        _, _, margliks, _ = marglik_training(
            model, loader, likelihood=lik, hessian_structure="kron", n_epochs=2
        )
        results.append(margliks)

    assert results[0] == results[1]


def test_marglik_training_rejects_reward_modeling(model):
    with pytest.raises(ValueError, match="reward modeling"):
        marglik_training(
            model, _loader("classification"), likelihood="reward_modeling", n_epochs=1
        )


@pytest.mark.parametrize("hessian_structure", ["diag", "full"])
def test_marglik_training_casts_class_weights(model, hessian_structure):
    loader = _loader("classification")
    X, y = loader.dataset.tensors
    loader = DataLoader(TensorDataset(X.double(), y), batch_size=8)

    # Float32 class weights for a float64 model
    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]))

    _, _, margliks, losses = marglik_training(
        model.double(),
        loader,
        likelihood=weighted,
        hessian_structure=hessian_structure,
        n_epochs=2,
        n_epochs_burnin=1,
    )

    assert len(margliks) > 0
    assert all(math.isfinite(m) for m in margliks)
    assert all(math.isfinite(loss) for loss in losses)


def _first_epoch_loss(model, loader, likelihood):
    # Zero learning rate and a negligible prior: the loss is the data term alone
    _, _, _, losses = marglik_training(
        model,
        loader,
        likelihood=likelihood,
        hessian_structure="diag",  # the default, kron, rejects class weights
        n_epochs=1,
        prior_prec_init=1e-12,
        optimizer_kwargs={"lr": 0.0},
    )

    return losses[0]


def test_marglik_training_sums_weighted_categorical_loss(model):
    loader = _loader("classification")
    X, y = loader.dataset.tensors
    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 3.0]))

    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(
            model(X), y, weight=weighted.weight, reduction="sum"
        ).item() / len(X)

    loss = _first_epoch_loss(model, loader, weighted)

    assert math.isclose(loss, expected, rel_tol=1e-5)


def test_marglik_training_unweighted_categorical_loss_is_the_mean(model):
    loader = _loader("classification")
    X, y = loader.dataset.tensors

    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(X), y).item()

    loss = _first_epoch_loss(model, loader, CategoricalLikelihood())

    assert math.isclose(loss, expected, rel_tol=1e-5)


def test_marglik_training_sums_gaussian_loss_over_outputs(model):
    # Two outputs, so a mean over all entries would halve the data term
    loader = _loader("regression")
    X, y = loader.dataset.tensors

    with torch.no_grad():
        expected = torch.nn.functional.mse_loss(
            model(X), y, reduction="sum"
        ).item() / len(X)

    loss = _first_epoch_loss(model, loader, GaussianLikelihood())

    assert math.isclose(loss, expected, rel_tol=1e-5)


def _multilabel_loader(dtype: torch.dtype | None = None) -> DataLoader:
    return toy_multilabel_dataset(n_train=32, batch_size=8, out_dim=2, dtype=dtype)[2]


def _check_bernoulli_marglik_training(model, likelihood, hessian_structure):
    la, model, margliks, losses = marglik_training(
        model,
        _multilabel_loader(),
        likelihood=likelihood,
        hessian_structure=hessian_structure,
        n_epochs=2,
    )

    assert la.likelihood == "bernoulli"
    assert len(margliks) > 0
    assert all(math.isfinite(m) for m in margliks)
    assert all(math.isfinite(loss) for loss in losses)

    # No observation noise to tune, and the predictive is a sigmoid
    assert la.sigma_noise == 1
    probs = la(torch.randn(4, 3))
    assert ((probs > 0) & (probs < 1)).all()


@pytest.mark.parametrize("hessian_structure", ["diag", "full", "kron"])
def test_marglik_training_bernoulli(model, hessian_structure):
    _check_bernoulli_marglik_training(model, "bernoulli", hessian_structure)


def test_marglik_training_bernoulli_from_criterion(model):
    _check_bernoulli_marglik_training(model, torch.nn.BCEWithLogitsLoss(), "kron")


@pytest.mark.parametrize("hessian_structure", ["diag", "full"])
def test_marglik_training_casts_bernoulli_options(model, hessian_structure):
    # Float32 options for a float64 model, in the data loss and in the Laplace fit
    options = dict(weight=torch.tensor([1.0, 2.0]), pos_weight=torch.tensor([3.0, 0.5]))

    la, _, margliks, losses = marglik_training(
        model.double(),
        _multilabel_loader(torch.float64),
        likelihood=BernoulliLikelihood(**options),
        hessian_structure=hessian_structure,
        n_epochs=2,
        n_epochs_burnin=1,
    )

    assert la.H.dtype == torch.float64
    assert len(margliks) > 0
    assert all(math.isfinite(m) for m in margliks)
    assert all(math.isfinite(loss) for loss in losses)


def test_marglik_training_bernoulli_options_with_kron_raise(model):
    weighted = BernoulliLikelihood(pos_weight=torch.tensor([3.0, 0.5]))

    with pytest.raises(ValueError, match="does not support the option"):
        marglik_training(model, _multilabel_loader(), likelihood=weighted, n_epochs=1)

    # The empirical Fisher honours them
    la, _, margliks, _ = marglik_training(
        model,
        _multilabel_loader(),
        likelihood=weighted,
        backend=CurvlinopsEF,
        n_epochs=1,
    )

    assert all(math.isfinite(m) for m in margliks)


def test_marglik_training_sums_bernoulli_loss(model):
    loader = _multilabel_loader()
    X, y = loader.dataset.tensors
    weighted = BernoulliLikelihood(pos_weight=torch.tensor([3.0, 0.5]))

    with torch.no_grad():
        expected = torch.nn.BCEWithLogitsLoss(
            pos_weight=weighted.pos_weight, reduction="sum"
        )(model(X), y).item() / len(X)

    loss = _first_epoch_loss(model, loader, weighted)

    assert math.isclose(loss, expected, rel_tol=1e-5)


def _soft_loader(one_hot: bool, n_classes: int = 2) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(32, 3)
    y = torch.randint(0, n_classes, (32,))

    if one_hot:
        y = torch.nn.functional.one_hot(y, n_classes).float()

    return DataLoader(TensorDataset(X, y), batch_size=8)


@pytest.mark.parametrize("hessian_structure", ["diag", "full", "kron"])
def test_marglik_training_soft_targets(model, hessian_structure, caplog):
    # Float one-hot targets `(B, K)` are class indices written differently, so
    # training, the marginal likelihood, and the logged accuracy are the same. The
    # accuracy compares the logits' argmax with the argmax of the targets.
    runs = {}

    for one_hot in [False, True]:
        torch.manual_seed(1)
        caplog.clear()

        with caplog.at_level("INFO"):
            _, _, margliks, losses = marglik_training(
                deepcopy(model),
                _soft_loader(one_hot),
                likelihood="classification",
                hessian_structure=hessian_structure,
                n_epochs=2,
            )

        perfs = [m for m in caplog.messages if "Perf=" in m]
        assert len(perfs) == 2
        runs[one_hot] = (margliks, losses, [m.split("Perf=")[1] for m in perfs])

    assert all(math.isfinite(m) for m in runs[True][0])
    assert runs[True][2] == runs[False][2]
    assert torch.allclose(torch.tensor(runs[True][0]), torch.tensor(runs[False][0]))
    assert torch.allclose(torch.tensor(runs[True][1]), torch.tensor(runs[False][1]))


@pytest.mark.parametrize("hessian_structure", ["full", "kron", "diag"])
def test_marglik_training_raises_for_noise_per_output(model, hessian_structure):
    with pytest.raises(ValueError, match="#349"):
        marglik_training(
            model,
            _loader("regression"),
            likelihood=GaussianLikelihood(sigma_noise=torch.tensor([0.5, 2.0])),
            hessian_structure=hessian_structure,
            n_epochs=1,
        )
