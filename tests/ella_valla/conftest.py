"""Fixtures shared by the ELLA and VaLLA test modules."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.ella_valla.helpers import METHODS


@pytest.fixture(params=METHODS)
def method(request: pytest.FixtureRequest) -> str:
    """Run the test once for ELLA and once for VaLLA."""
    return request.param


@pytest.fixture
def data() -> tuple[torch.Tensor, DataLoader, torch.nn.Module]:
    """Return XOR inputs, a loader over them with batch size 2, and a linear model."""
    torch.manual_seed(7)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))

    return (x, DataLoader(TensorDataset(x, y), batch_size=2), model)
