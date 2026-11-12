"""Fixtures shared by the curvature-backend test modules.

They are seeded with 711 and use a `Tanh` network, so `model` here differs from
the one in `tests/conftest.py` and overrides it for this directory.
"""

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector


def _with_attrs(model: nn.Module) -> nn.Module:
    """Set `output_size`, `n_layers` and `n_params` on `model` and return it."""
    model_params = list(model.parameters())

    setattr(model, "output_size", 2)
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))

    return model


@pytest.fixture
def model() -> nn.Module:
    """Return a 3 -> 20 -> 2 Tanh MLP."""
    torch.manual_seed(711)

    return _with_attrs(nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2)))


@pytest.fixture
def class_Xy() -> tuple[torch.Tensor, torch.Tensor]:
    """Return 10 random 3-d inputs with binary class labels."""
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))

    return X, y


@pytest.fixture
def reg_Xy() -> tuple[torch.Tensor, torch.Tensor]:
    """Return 10 random 3-d inputs with 2-d regression targets."""
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)

    return X, y


@pytest.fixture
def complex_model() -> nn.Module:
    """Return a small conv net on 3x5x5 inputs with 2 outputs."""
    torch.manual_seed(711)

    return _with_attrs(
        nn.Sequential(
            nn.Conv2d(3, 4, 2, 2),
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear(16, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )
    )


@pytest.fixture
def complex_class_Xy() -> tuple[torch.Tensor, torch.Tensor]:
    """Return 10 random 3x5x5 inputs with binary class labels."""
    torch.manual_seed(711)
    X = torch.randn(10, 3, 5, 5)
    y = torch.randint(2, (10,))

    return X, y
