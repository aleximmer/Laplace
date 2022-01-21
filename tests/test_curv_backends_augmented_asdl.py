import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from asdfghjkl.operations import Conv2dAug, Bias, Scale

from asdfghjkl.operations import Bias, Scale

from laplace.curvature import AugAsdlGGN, AugAsdlEF, AsdlGGN
from laplace.curvature import AugBackPackGGN


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
def class_Xy_single():
    torch.manual_seed(711)
    X = torch.randn(1, 1, 3, requires_grad=True)
    y = torch.randint(2, (1,))
    return X, y


@pytest.fixture
def complex_model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(Conv2dAug(3, 4, 2, 2), nn.Flatten(start_dim=2), nn.Tanh(),
                                nn.Linear(16, 20), Scale(), Bias(), nn.Tanh(), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model

    
@pytest.fixture
def complex_class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 7, 3, 5, 5, requires_grad=True)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def complex_class_Xy_single():
    torch.manual_seed(711)
    X = torch.randn(1, 1, 3, 5, 5, requires_grad=True)
    y = torch.randint(2, (1,))
    return X, y


@pytest.fixture
def complex_class_Xy_single_aug():
    torch.manual_seed(711)
    X = torch.randn(10, 1, 3, 5, 5, requires_grad=True)
    y = torch.randint(2, (10,))
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


@pytest.mark.parametrize('Backend', [AugAsdlEF, AugAsdlGGN])
def test_kron_against_diagonal(class_Xy_single, model, Backend):
    model.zero_grad()
    torch.manual_seed(723)
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
    

@pytest.mark.parametrize('Backend', [AugAsdlEF, AugAsdlGGN])
def test_diag_against_diagonalized_full_cnn(complex_class_Xy, complex_model, Backend):
    X, y = complex_class_Xy
    backend = Backend(complex_model, 'classification')
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


@pytest.mark.parametrize('Backend', [AugAsdlEF, AugAsdlGGN])
def test_kron_augmentation_cnn(complex_class_Xy_single_aug, complex_model, Backend):
    # augmenting more with simple repetition shouldn't change anything.
    X, y = complex_class_Xy_single_aug
    backend = Backend(complex_model, 'classification')
    # no augmentation dimension
    loss_cmp, H_kron = backend.kron(X, y, len(y))
    X.grad.data.zero_()
    h_cmp = H_kron.diag()
    (loss_cmp + h_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.detach().clone().flatten()
    # use 21 augmentation dimensions but just repeat --> has to be equivalent
    loss, H_kron = backend.kron(X.repeat(1, 21, 1, 1, 1), y, len(y))
    X.grad.data.zero_()
    h = H_kron.diag()
    (loss + h.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    assert torch.allclose(h, h_cmp)
    idx = torch.argmax(torch.abs(xgrad - xgrad_cmp))
    print(xgrad[idx], xgrad_cmp[idx], xgrad_cmp[idx]-xgrad[idx])
    assert torch.allclose(xgrad, xgrad_cmp, atol=1e-7)


@pytest.mark.parametrize('Backend', [AugAsdlEF, AugAsdlGGN])
def test_kron_against_diagonal_approx_cnn(complex_class_Xy_single, complex_model, Backend):
    X, y = complex_class_Xy_single
    backend = Backend(complex_model, 'classification')
    loss_cmp, h_cmp = backend.diag(X, y)
    X.grad.data.zero_()
    (loss_cmp + h_cmp.sum()).backward()
    xgrad_cmp = X.grad.data.detach().clone().flatten()
    loss, H_kron = backend.kron(X, y, 1)
    X.grad.data.zero_()
    h = H_kron.diag()
    (loss + h.sum()).backward()
    xgrad = X.grad.data.clone().flatten()
    assert torch.allclose(loss, loss_cmp)
    # these are very rough, should be rather taken as integration test
    # this is because for CNN,  Kron != diagonal even for a single input
    assert torch.allclose(h, h_cmp, atol=1e-1, rtol=1e-2)
    assert torch.allclose(xgrad, xgrad_cmp, atol=1e-1, rtol=1e-2)


def test_kron_aug_against_kron_fc(model):
    X = torch.randn(11, 1, 3, requires_grad=True)
    y = torch.randint(2, (11,))
    backend = AsdlGGN(model, 'classification')
    loss, H_kron = backend.kron(X.squeeze(), y, len(X))
    (loss + H_kron.diag().sum()).backward()
    grad = X.grad.data.clone()
    X.grad = None
    backend_aug = AugAsdlGGN(model, 'classification')
    loss_aug, H_kron_aug = backend_aug.kron(X, y, len(X))
    (loss_aug + H_kron_aug.diag().sum()).backward()
    grad_aug = X.grad.data.clone()
    assert torch.allclose(loss_aug, loss)
    assert torch.allclose(H_kron.diag(), H_kron_aug.diag())
    assert torch.allclose(grad, grad_aug)


def test_kron_aug_against_kron_cnn(complex_model):
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Conv2d(3, 4, 2, 2), nn.Flatten(), nn.Tanh(),
                                nn.Linear(16, 20), Scale(), Bias(), nn.Tanh(), nn.Linear(20, 2))
    X = torch.randn(11, 1, 3, 5, 5, requires_grad=True)
    y = torch.randint(2, (11,))
    backend = AsdlGGN(model, 'classification')
    loss, H_kron = backend.kron(X.squeeze(), y, len(X))
    (loss + H_kron.diag().sum()).backward()
    grad = X.grad.data.clone()
    X.grad = None
    backend_aug = AugAsdlGGN(complex_model, 'classification')
    loss_aug, H_kron_aug = backend_aug.kron(X, y, len(X))
    (loss_aug + H_kron_aug.diag().sum()).backward()
    grad_aug = X.grad.data.clone()
    assert torch.allclose(loss_aug, loss)
    assert torch.allclose(H_kron.diag(), H_kron_aug.diag())
    assert torch.allclose(grad, grad_aug)
