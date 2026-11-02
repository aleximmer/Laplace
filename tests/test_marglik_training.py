import inspect
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import marglik_training
from laplace.curvature import CurvlinopsGGN
from laplace.likelihoods import CategoricalLikelihood, GaussianLikelihood


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
