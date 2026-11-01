import inspect
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import marglik_training
from laplace.curvature import CurvlinopsGGN

torch.manual_seed(711)


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
