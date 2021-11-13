import pytest
import torch
from laplace.baselaplace import FunctionalLaplace
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


def test_sod_data_loader(reg_loader, model):
    M = 5
    func_la = FunctionalLaplace(model, 'regression', M)
    sod_data_loader = func_la._get_SoD_data_loader(reg_loader)

    first_iter = []
    for x, _ in sod_data_loader:
        first_iter.append(x)
    second_iter = []
    for x, y in sod_data_loader:
        second_iter.append(x)

    first_iter = torch.cat(first_iter, dim=0)
    second_iter = torch.cat(second_iter, dim=0)

    assert torch.allclose(first_iter, second_iter)
    assert len(first_iter) == M


def test_store_K_batch_full_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, 'regression', M, diagonal_kernel=False)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    func_la._init_K_MM()
    assert torch.equal(torch.zeros(size=(M*C, M*C)), func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(4, 4)), 0, 0)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.]]),
                       func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(4, 2)), 0, 1)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.]]),
                       func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(2, 2)), 1, 1)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.]]),
                       func_la.K_MM)


def test_store_K_batch_block_diagonal_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, 'regression', M, diagonal_kernel=True)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    def _check(expected_K_MM):
        for c in range(C):
            assert torch.equal(expected_K_MM[c], func_la.K_MM[c])

    func_la._init_K_MM()
    expected = [torch.zeros(size=(M, M)) for _ in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(2, 2, 1)), 2 * torch.ones(size=(2, 2, 1))], dim=2), 0, 0)
    expected = [(c + 1) * torch.tensor([[1., 1., 0.],
                                        [1., 1., 0.],
                                        [0., 0., 0.]]) for c in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(2, 1, 1)), 2 * torch.ones(size=(2, 1, 1))], dim=2), 0, 1)
    expected = [(c + 1) * torch.tensor([[1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 0.]]) for c in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(1, 1, 1)), 2 * torch.ones(size=(1, 1, 1))], dim=2), 1, 1)
    expected = [(c + 1) * torch.tensor([[1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 1.]]) for c in range(C)]
    _check(expected)




