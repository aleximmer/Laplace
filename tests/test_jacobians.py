import pytest
import torch
from torch import nn

from laplace.curvature import AsdlInterface, BackPackInterface, AugBackPackInterface, AugAsdlInterface, AugAsdlGGN
from laplace.feature_extractor import FeatureExtractor
from tests.utils import jacobians_naive, jacobians_naive_aug, gradients_naive_aug


@pytest.fixture
def multioutput_model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    return model


@pytest.fixture
def singleoutput_model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 1))
    setattr(model, 'output_size', 1)
    return model


@pytest.fixture
def linear_model():
    model = torch.nn.Sequential(nn.Linear(3, 1, bias=False))
    setattr(model, 'output_size', 1)
    return model


@pytest.fixture
def X():
    torch.manual_seed(15)
    return torch.randn(200, 3)

    
@pytest.fixture
def X_aug():
    torch.manual_seed(711)
    return torch.randn(200, 5, 3)

    
@pytest.mark.parametrize('backend', [AsdlInterface, BackPackInterface])
def test_linear_jacobians(linear_model, X, backend):
    # jacobian of linear model is input X.
    Js, f = backend.jacobians(linear_model, X)
    # into Jacs shape (batch_size, output_size, params)
    true_Js = X.reshape(len(X), 1, -1)
    assert true_Js.shape == Js.shape
    assert torch.allclose(true_Js, Js, atol=1e-5)
    assert torch.allclose(f, linear_model(X), atol=1e-5)


@pytest.mark.parametrize('backend', [AsdlInterface, BackPackInterface])
def test_jacobians_singleoutput(singleoutput_model, X, backend):
    model = singleoutput_model
    Js, f = backend.jacobians(model, X)
    Js_naive, f_naive = jacobians_naive(model, X)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(model(X), f_naive)
    assert torch.allclose(f, f_naive)


@pytest.mark.parametrize('backend', [AsdlInterface, BackPackInterface])
def test_jacobians_multioutput(multioutput_model, X, backend):
    model = multioutput_model
    Js, f = backend.jacobians(model, X)
    Js_naive, f_naive = jacobians_naive(model, X)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(model(X), f_naive)
    assert torch.allclose(f, f_naive)


@pytest.mark.parametrize('backend', [AsdlInterface, BackPackInterface])
def test_last_layer_jacobians_singleoutput(singleoutput_model, X, backend):
    model = FeatureExtractor(singleoutput_model)
    Js, f = backend.last_layer_jacobians(model, X)
    _, phi = model.forward_with_features(X)
    Js_naive, f_naive = jacobians_naive(model.last_layer, phi)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(model(X), f_naive)
    assert torch.allclose(f, f_naive)


@pytest.mark.parametrize('backend', [AsdlInterface, BackPackInterface])
def test_last_layer_jacobians_multioutput(multioutput_model, X, backend):
    model = FeatureExtractor(multioutput_model)
    Js, f = backend.last_layer_jacobians(model, X)
    _, phi = model.forward_with_features(X)
    Js_naive, f_naive = jacobians_naive(model.last_layer, phi)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(model(X), f_naive)
    assert torch.allclose(f, f_naive)


@pytest.mark.parametrize('backend', [AugAsdlInterface, AugBackPackInterface])
def test_jacobians_augmented(multioutput_model, X_aug, backend):
    Js, f = backend.jacobians(multioutput_model, X_aug)
    Js_naive, f_naive = jacobians_naive_aug(multioutput_model, X_aug)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(multioutput_model(X_aug).mean(dim=1), f_naive)
    assert torch.allclose(f, f_naive)


@pytest.mark.parametrize('likelihood', ['classification', 'regression'])
def test_gradients_augmented(multioutput_model, X_aug, likelihood):
    if likelihood == 'regression':
        y = torch.randn(len(X_aug), 2)
    else:
        y = torch.empty(len(X_aug), dtype=torch.long).random_(2)
    backend = AugAsdlGGN(multioutput_model, likelihood)
    Gs, loss = backend.gradients(X_aug, y)
    Gs_naive, loss_naive = gradients_naive_aug(multioutput_model, X_aug, y, likelihood)
    assert torch.allclose(loss, loss_naive)
    assert Gs.shape == Gs_naive.shape
    assert torch.allclose(Gs, Gs_naive, atol=1e-7)
