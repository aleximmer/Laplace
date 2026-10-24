from itertools import product

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace import DiagLaplace, FullLaplace, KronLaplace
from laplace.curvature import AsdlEF, AsdlGGN, AsdlHessian, BackPackEF, BackPackGGN
from laplace.curvature.asdfghjkl import AsdfghjklEF, AsdfghjklGGN, AsdfghjklHessian
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN, CurvlinopsHessian

torch.manual_seed(240)
torch.set_default_dtype(torch.double)
flavors = [KronLaplace, DiagLaplace, FullLaplace]
valid_backends = [CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))

    # Subset of params
    for p in model.parameters():
        p.requires_grad = False
    model[0].weight.requires_grad = True

    return model


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.mark.parametrize(
    "laplace,lh", product(flavors, ["classification", "regression"])
)
def test_compatible_backend(laplace, lh, model):
    laplace(model, lh, backend=CurvlinopsEF)
    laplace(model, lh, backend=CurvlinopsGGN)
    laplace(model, lh, backend=CurvlinopsHessian)
    laplace(model, lh, backend=AsdlEF)
    laplace(model, lh, backend=AsdlGGN)
    laplace(model, lh, backend=AsdlHessian)


@pytest.mark.parametrize(
    "laplace,lh", product(flavors, ["classification", "regression"])
)
def test_incompatible_backend(laplace, lh, model):
    with pytest.raises(ValueError):
        laplace(model, lh, backend=BackPackGGN)

    with pytest.raises(ValueError):
        laplace(model, lh, backend=BackPackEF)

    with pytest.raises(ValueError):
        laplace(model, lh, backend=AsdfghjklGGN)

    with pytest.raises(ValueError):
        laplace(model, lh, backend=AsdfghjklEF)

    with pytest.raises(ValueError):
        laplace(model, lh, backend=AsdfghjklHessian)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", valid_backends)
def test_mean_clf(laplace, backend, model, class_loader):
    n_params = model[0].weight.numel()
    lap = laplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    assert lap.mean.shape == (n_params,)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", valid_backends)
def test_mean_reg(laplace, backend, model, reg_loader):
    n_params = model[0].weight.numel()
    lap = laplace(model, "regression", backend=backend)
    lap.fit(reg_loader)
    assert lap.mean.shape == (n_params,)


@pytest.mark.parametrize("backend", valid_backends)
def test_post_precision_diag(model, backend, class_loader):
    n_params = model[0].weight.numel()
    lap = DiagLaplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    assert lap.posterior_precision.shape == (n_params,)


@pytest.mark.parametrize("backend", valid_backends)
def test_post_precision_kron(model, backend, class_loader):
    n_params = model[0].weight.numel()
    lap = KronLaplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    assert lap.posterior_precision.to_matrix().shape == (n_params, n_params)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", valid_backends)
def test_predictive(laplace, backend, model, class_loader):
    lap = laplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    lap(torch.randn(5, 3), pred_type="nn", link_approx="mc")


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", valid_backends)
def test_marglik_glm(laplace, backend, model, class_loader):
    lap = laplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    lap.optimize_prior_precision(method="marglik")


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", valid_backends)
def test_marglik_nn(laplace, backend, model, class_loader):
    lap = laplace(model, "classification", backend=backend)
    lap.fit(class_loader)
    lap.optimize_prior_precision(
        method="gridsearch", val_loader=class_loader, pred_type="nn", link_approx="mc"
    )
