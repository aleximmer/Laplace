import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace.curvature import AugAsdlGGN, AugAsdlEF, AugBackPackGGN, AsdlGGN, AsdlEF, BackPackGGN, BackPackEF


@pytest.fixture
def model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 5, 3, requires_grad=True)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def class_Xy_single():
    torch.manual_seed(711)
    X = torch.randn(1, 1, 3, requires_grad=True)
    y = torch.randint(2, (1,))
    return X, y
