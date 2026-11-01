from collections import UserDict
from collections.abc import MutableMapping
from importlib.util import find_spec
from typing import Any, List

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import laplace.curvature
from laplace.curvature import CurvatureInterface


def toy_regression_dataset_1d(sigma, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    # create simple sinusoid data set
    X_train = (torch.rand(n_train) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.linspace(-5, 13, n_test).unsqueeze(
        -1
    )  # +-5 on top of the training X-range

    return X_train, y_train, train_loader, X_test


def toy_multivariate_regression_dataset(
    sigma, d_input, n_train=150, n_test=500, batch_size=150
):
    torch.manual_seed(711)
    # create simple sinusoid data set
    X_train = torch.rand(n_train, d_input) * 8
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    # y_train = torch.rand(n_train, d_input) * 8 + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.rand(n_test, d_input) * 10
    return X_train, y_train, train_loader, X_test


def toy_classification_dataset(
    n_train=150, n_test=500, batch_size=150, in_dim=3, out_dim=2
):
    torch.manual_seed(711)
    X_train = torch.randn(n_train, in_dim)
    y_train = torch.randint(out_dim, (n_train,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.randn(n_test, in_dim)
    return X_train, y_train, train_loader, X_test


def toy_model(
    train_loader: DataLoader,
    n_epochs=500,
    fit=True,
    in_dim=1,
    out_dim=1,
    regression=True,
):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 50), torch.nn.Tanh(), torch.nn.Linear(50, out_dim)
    )
    if fit:
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
        for i in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    return model


def get_psd_matrix(dim):
    X = torch.randn(dim, dim * 3)
    return X @ X.T / (dim * 3)


def get_diag_psd_matrix(dim):
    return torch.randn(dim) ** 2


def grad(model):
    return torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()


def jacobians_naive(model, data):
    model.zero_grad()
    f = model(data)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = i != (f.shape[0] - 1) or j != (f.shape[1] - 1)
                f[i, j].backward(retain_graph=rg)
                Jij = grad(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = i != (f.shape[0] - 1)
            f[i].backward(retain_graph=rg)
            jacs = grad(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs).transpose(1, 2)
    return Jacs.detach(), f.detach()


class ListDataset(Dataset):
    def __init__(self, data: List[Any]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def dict_data_collator(batch: List[MutableMapping]) -> UserDict:
    ret = UserDict({})

    for k in batch[0].keys():
        vals = torch.stack([v[k] for v in batch]).squeeze(-1)
        ret[k] = vals

    return ret


class ScaledModel(nn.Module):
    """Takes an extra forward kwarg that changes the features."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 2)
        self.output_size = 2

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.fc2(scale * self.fc1(x))


requires_asdl = pytest.mark.skipif(
    find_spec("asdl") is None, reason="asdl not installed"
)
requires_backpack = pytest.mark.skipif(
    find_spec("backpack") is None, reason="backpack not installed"
)


def _optional_backend(name: str) -> type[CurvatureInterface]:
    """Get an optional backend class, or a stand-in if its package is missing.

    The stand-in has the same name and a `_missing_package` attribute. Building
    it skips the running test, and `tests/conftest.py` skips tests that take it
    as a parameter before they run.

    Parameters
    ----------
    name : str
        Name of the backend class in `laplace.curvature`, e.g. `"AsdlGGN"`.

    Returns
    -------
    type[CurvatureInterface]
        The backend class or its stand-in.
    """
    try:
        return getattr(laplace.curvature, name)
    except ImportError:
        module = laplace.curvature._OPTIONAL_BACKENDS[name]
        package = laplace.curvature._OPTIONAL_PACKAGES[module]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pytest.skip(f"{package} not installed")

        return type(
            name,
            (CurvatureInterface,),
            {"__init__": __init__, "_missing_package": package},
        )


AsdlInterface = _optional_backend("AsdlInterface")
AsdlGGN = _optional_backend("AsdlGGN")
AsdlEF = _optional_backend("AsdlEF")
AsdlHessian = _optional_backend("AsdlHessian")
BackPackInterface = _optional_backend("BackPackInterface")
BackPackGGN = _optional_backend("BackPackGGN")
BackPackEF = _optional_backend("BackPackEF")
