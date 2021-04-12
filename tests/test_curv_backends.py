import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.curvature import BackPackGGN
from laplace.jacobians import Jacobians


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def class_Xy():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def reg_Xy():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return X, y


def test_diag_ggn_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, 'classification', stochastic=False)
    loss, dggn = backend.diag(X, y)
    assert len(dggn) == model.n_params

    
def test_diag_ggn_cls_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, 'classification', stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    Js, f = Jacobians(model, X)
    ps = torch.softmax(f, dim=-1)
    H = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
    full_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H, Js)
    assert torch.allclose(dggn, full_ggn.diagonal())


def test_diag_ggn_reg_backpack(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackGGN(model, 'regression', stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    # compare to naive computation of ggn
    Js, f = Jacobians(model, X)
    # factor 2 since backpack uses MSE, log N has factor 1/2 
    full_ggn = 2 * torch.einsum('mkp,mkq->pq', Js, Js)
    assert torch.allclose(dggn, full_ggn.diagonal())


def test_diag_ggn_stoch_cls_backpack(class_Xy, model):
    X, y = class_Xy
    backend = BackPackGGN(model, 'classification', stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    Js, f = Jacobians(model, X)
    ps = torch.softmax(f, dim=-1)
    H = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
    full_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H, Js)
    assert torch.allclose(dggn, full_ggn.diagonal(), atol=1e-8, rtol=1e1)


def test_diag_ggn_stoch_reg_backpack(reg_Xy, model):
    X, y = reg_Xy
    backend = BackPackGGN(model, 'regression', stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    # compare to naive computation of ggn
    Js, f = Jacobians(model, X)
    # factor 2 since backpack uses MSE, log N has factor 1/2 
    full_ggn = 2 * torch.einsum('mkp,mkq->pq', Js, Js)
    # stochastic so only require same order of magnitude 
    assert torch.allclose(dggn, full_ggn.diagonal(), atol=1e-8, rtol=1e1)