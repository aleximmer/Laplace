from laplace.curvature.asdl import AsdlGGN
import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from asdfghjkl.operations import Bias, Scale

from laplace.curvature import AugAsdlGGN, AugAsdlEF
from laplace.curvature import AugBackPackGGN


@pytest.fixture
def model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20, bias=True), nn.Tanh(), nn.Linear(20, 2, bias=True))
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
def class_Xy_single():
    torch.manual_seed(711)
    X = torch.randn(1, 1, 3, requires_grad=True)
    y = torch.randint(2, (1,))
    return X, y

    
def test_full_ggn_against_backpack(class_Xy, model):
    X, y = class_Xy
    backend_backpack = AugBackPackGGN(model, 'classification')
    loss_cmp, H_cmp = backend_backpack.full(X, y)
    X.grad.data.zero_()
    (loss_cmp + H_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.clone().detach().flatten()
    backend_asdl = AugAsdlGGN(model, 'classification')
    loss, H = backend_asdl.full(X, y)
    X.grad.data.zero_()
    (loss + H.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    assert torch.allclose(H, H_cmp)
    assert torch.allclose(xgrad, xgrad_cmp, atol=1e-7)

    
def test_diag_ggn_against_diagonalized_full(class_Xy, model):
    X, y = class_Xy
    backend = AugAsdlGGN(model, 'classification')
    loss_cmp, H_cmp = backend.full(X, y)
    h_cmp = torch.diag(H_cmp)
    X.grad.data.zero_()
    (loss_cmp + h_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.detach().clone().flatten()
    loss, h = backend.diag(X, y)
    X.grad.data.zero_()
    (loss + h.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    assert torch.allclose(h, h_cmp)
    assert torch.allclose(xgrad, xgrad_cmp)

    
def test_kron_ggn_against_diagonal_ggn(class_Xy_single, model):
    X, y = class_Xy_single
    backend = AugAsdlGGN(model, 'classification')
    loss_cmp, h_cmp = backend.diag(X.repeat(5, 11, 1), y.repeat(5))
    X.grad.data.zero_()
    (loss_cmp + h_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.detach().clone().flatten()
    loss, H_kron = backend.kron(X.repeat(5, 11, 1), y.repeat(5), 5)
    h = H_kron.diag()
    X.grad.data.zero_()
    (loss + h.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    assert torch.allclose(h, h_cmp)
    assert torch.allclose(xgrad, xgrad_cmp)


def test_kron_ggn_against_diagonal_ggn_approx(class_Xy, model):
    X, y = class_Xy
    backend = AugAsdlGGN(model, 'classification')
    loss_cmp, h_cmp = backend.diag(X, y)
    X.grad.data.zero_()
    (loss_cmp + h_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.detach().clone().flatten()
    loss, H_kron = backend.kron(X, y, len(X))
    h = H_kron.diag()
    X.grad.data.zero_()
    (loss + h.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    # these are very rough, should be rather taken as integration test
    # for real test see above `test_kron_ggn_against_diagonal_ggn` with
    # repeated data which makes Kron and diagonal exactly equal and 
    # thus ensures proper scaling.
    assert torch.allclose(h, h_cmp, atol=1e-1, rtol=1e-3)
    assert torch.allclose(xgrad, xgrad_cmp, atol=1e-1, rtol=1e-3)
