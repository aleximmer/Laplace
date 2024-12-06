from itertools import product

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2

from laplace import DiagSubnetLaplace, FullSubnetLaplace, Laplace, SubnetLaplace
from laplace.baselaplace import DiagLaplace
from laplace.curvature.asdl import AsdlEF, AsdlGGN, AsdlHessian
from laplace.curvature.backpack import BackPackEF, BackPackGGN
from laplace.curvature.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN, CurvlinopsHessian
from laplace.utils import (
    LargestMagnitudeSubnetMask,
    LargestVarianceDiagLaplaceSubnetMask,
    LargestVarianceSWAGSubnetMask,
    LastLayerSubnetMask,
    ModuleNameSubnetMask,
    ParamNameSubnetMask,
    RandomSubnetMask,
    SubnetMask,
)

torch.manual_seed(240)
torch.set_default_dtype(torch.double)
score_based_subnet_masks = [
    RandomSubnetMask,
    LargestMagnitudeSubnetMask,
    LargestVarianceDiagLaplaceSubnetMask,
    LargestVarianceSWAGSubnetMask,
]
layer_subnet_masks = [ParamNameSubnetMask, ModuleNameSubnetMask, LastLayerSubnetMask]
all_subnet_masks = score_based_subnet_masks + layer_subnet_masks
likelihoods = ["classification", "regression"]
hessian_structures = ["full", "diag"]
backends = [
    AsdlEF,
    AsdlGGN,
    AsdlHessian,
    BackPackEF,
    BackPackGGN,
    AsdlHessian,
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
    GGNInterface,
    EFInterface,
    CurvatureInterface,
]


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    model_params = list(model.parameters())
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def large_model():
    model = wide_resnet50_2()
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


@pytest.mark.parametrize("likelihood", likelihoods)
def test_subnet_laplace_init(model, likelihood):
    # use random subnet mask for this test
    subnetwork_mask = RandomSubnetMask
    subnetmask_kwargs = dict(model=model, n_params_subnet=10)
    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select()

    # subnet Laplace with full Hessian should work
    hessian_structure = "full"
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, FullSubnetLaplace)

    # subnet Laplace with diagonal Hessian should work
    hessian_structure = "diag"
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, DiagSubnetLaplace)

    # subnet Laplace without specifying subnetwork indices should raise an error
    with pytest.raises(TypeError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            hessian_structure=hessian_structure,
        )

    # subnet Laplace with kron or lowrank Hessians should raise errors
    hessian_structure = "kron"
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
    hessian_structure = "lowrank"
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )


@pytest.mark.parametrize("likelihood", likelihoods)
@pytest.mark.parametrize("hessian_structure", hessian_structures)
@pytest.mark.parametrize("backend", backends)
def test_subnet_laplace_backend_init(model, likelihood, hessian_structure, backend):
    if issubclass(backend, (GGNInterface, EFInterface)):
        _ = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=torch.LongTensor([1, 2]),
            hessian_structure=hessian_structure,
            backend=backend,
        )
    else:
        with pytest.raises(ValueError):
            _ = Laplace(
                model,
                likelihood=likelihood,
                subset_of_weights="subnetwork",
                subnetwork_indices=torch.LongTensor([1, 2]),
                hessian_structure=hessian_structure,
                backend=backend,
            )


@pytest.mark.parametrize(
    "likelihood,hessian_structure", product(likelihoods, hessian_structures)
)
def test_subnet_laplace_large_init(large_model, likelihood, hessian_structure):
    # use random subnet mask for this test
    subnetwork_mask = RandomSubnetMask
    n_param_subnet = 10
    subnetmask_kwargs = dict(model=large_model, n_params_subnet=n_param_subnet)
    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select()

    lap = Laplace(
        large_model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert lap.n_params_subnet == n_param_subnet
    if hessian_structure == "full":
        assert lap.H.shape == (lap.n_params_subnet, lap.n_params_subnet)
    else:
        assert lap.H.shape == (lap.n_params_subnet,)
    H = lap.H.clone()
    lap._init_H()
    assert torch.allclose(H, lap.H)


@pytest.mark.parametrize(
    "likelihood,hessian_structure", product(likelihoods, hessian_structures)
)
def test_custom_subnetwork_indices(
    model, likelihood, class_loader, reg_loader, hessian_structure
):
    loader = class_loader if likelihood == "classification" else reg_loader

    # subnetwork indices that are None should raise an error
    subnetwork_indices = None
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are not PyTorch tensors should raise an error
    subnetwork_indices = [0, 5, 11, 42]
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are empty tensors should raise an error
    subnetwork_indices = torch.LongTensor([])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are scalar tensors should raise an error
    subnetwork_indices = torch.LongTensor(11)
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are not 1D PyTorch tensors should raise an error
    subnetwork_indices = torch.LongTensor([[0, 5], [11, 42]])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are double tensors should raise an error
    subnetwork_indices = torch.DoubleTensor([0.0, 5.0, 11.0, 42.0])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are float tensors should raise an error
    subnetwork_indices = torch.FloatTensor([0.0, 5.0, 11.0, 42.0])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are half tensors should raise an error
    subnetwork_indices = torch.HalfTensor([0.0, 5.0, 11.0, 42.0])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are int tensors should raise an error
    subnetwork_indices = torch.IntTensor([0, 5, 11, 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are short tensors should raise an error
    subnetwork_indices = torch.ShortTensor([0, 5, 11, 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are char tensors should raise an error
    subnetwork_indices = torch.CharTensor([0, 5, 11, 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that are bool tensors should raise an error
    subnetwork_indices = torch.BoolTensor([0, 5, 11, 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that contain elements smaller than zero should raise an error
    subnetwork_indices = torch.LongTensor([0, -1, -11])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that contain elements larger than n_params should raise an error
    subnetwork_indices = torch.LongTensor([model.n_params + 1, model.n_params + 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # subnetwork indices that contain duplicate entries should raise an error
    subnetwork_indices = torch.LongTensor([0, 0, 5, 11, 11, 42])
    with pytest.raises(ValueError):
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetwork_indices,
            hessian_structure=hessian_structure,
        )

    # Non-empty, 1-dimensional torch.LongTensor with valid entries should work
    subnetwork_indices = torch.LongTensor([0, 5, 11, 42])
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetwork_indices,
        hessian_structure=hessian_structure,
    )
    lap.fit(loader)
    assert isinstance(lap, SubnetLaplace)
    assert lap.n_params_subnet == 4
    if hessian_structure == "full":
        assert lap.H.shape == (4, 4)
    else:
        assert lap.H.shape == (4,)
    assert lap.backend.subnetwork_indices.equal(subnetwork_indices)


@pytest.mark.parametrize(
    "subnetwork_mask,likelihood,hessian_structure",
    product(score_based_subnet_masks, likelihoods, hessian_structures),
)
def test_score_based_subnet_masks(
    model, likelihood, subnetwork_mask, class_loader, reg_loader, hessian_structure
):
    loader = class_loader if likelihood == "classification" else reg_loader
    model_params = parameters_to_vector(model.parameters())

    # set subnetwork mask arguments
    if subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
        diag_laplace_model = DiagLaplace(model, likelihood)
        subnetmask_kwargs = dict(model=model, diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == LargestVarianceSWAGSubnetMask:
        subnetmask_kwargs = dict(model=model, likelihood=likelihood)
    else:
        subnetmask_kwargs = dict(model=model)

    # should raise error if we don't pass number of subnet parameters within the subnetmask_kwargs
    with pytest.raises(TypeError):
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)

    # should raise error if we set number of subnet parameters to None
    subnetmask_kwargs.update(n_params_subnet=None)
    with pytest.raises(ValueError):
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)

    # should raise error if number of subnet parameters is larger than number of model parameters
    subnetmask_kwargs.update(n_params_subnet=99999)
    with pytest.raises(ValueError):
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)

    # define subnetwork mask
    n_params_subnet = 32
    subnetmask_kwargs.update(n_params_subnet=n_params_subnet)
    subnetmask = subnetwork_mask(**subnetmask_kwargs)

    # should raise error if we try to access the subnet indices before the subnet has been selected
    with pytest.raises(AttributeError):
        subnetmask.indices

    # select subnet mask
    subnetmask.select(loader)

    # should raise error if we try to select the subnet again
    with pytest.raises(ValueError):
        subnetmask.select(loader)

    # define valid subnet Laplace model
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, SubnetLaplace)

    # fit Laplace model
    lap.fit(loader)

    # check some parameters
    assert subnetmask.indices.equal(lap.backend.subnetwork_indices)
    assert subnetmask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet
    assert parameters_to_vector(model.parameters()).equal(model_params)

    # check that Hessian and prior precision is of correct shape
    if hessian_structure == "full":
        assert lap.H.shape == (n_params_subnet, n_params_subnet)
    else:
        assert lap.H.shape == (n_params_subnet,)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)


@pytest.mark.parametrize(
    "subnetwork_mask,likelihood,hessian_structure",
    product(layer_subnet_masks, likelihoods, hessian_structures),
)
def test_layer_subnet_masks(
    model, likelihood, subnetwork_mask, class_loader, reg_loader, hessian_structure
):
    loader = class_loader if likelihood == "classification" else reg_loader
    subnetmask_kwargs = dict(model=model)

    # fit last-layer Laplace model
    lllap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="last_layer",
        hessian_structure=hessian_structure,
    )
    lllap.fit(loader)

    # should raise error if we pass number of subnet parameters
    subnetmask_kwargs.update(n_params_subnet=32)
    with pytest.raises(TypeError):
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)

    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask == ParamNameSubnetMask:
        # should raise error if we pass no parameter name list
        subnetmask_kwargs.update()
        with pytest.raises(TypeError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # should raise error if we pass an empty parameter name list
        subnetmask_kwargs.update(parameter_names=[])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # should raise error if we pass a parameter name list with invalid parameter names
        subnetmask_kwargs.update(parameter_names=["123"])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # define last-layer Laplace model by parameter names and check that
        # Hessian is identical to that of a full LLLaplace model
        subnetmask_kwargs.update(parameter_names=["1.weight", "1.bias"])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert torch.allclose(lllap.H, lap.H, rtol=1e-3)

        # define valid parameter name subnet mask
        subnetmask_kwargs.update(parameter_names=["0.weight", "1.bias"])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 62
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

    elif subnetwork_mask == ModuleNameSubnetMask:
        # should raise error if we pass no module name list
        subnetmask_kwargs.update()
        with pytest.raises(TypeError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # should raise error if we pass an empty module name list
        subnetmask_kwargs.update(module_names=[])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # should raise error if we pass a module name list with invalid module names
        subnetmask_kwargs.update(module_names=["123"])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # define last-layer Laplace model by module name and check that
        # Hessian is identical to that of a full LLLaplace model
        subnetmask_kwargs.update(module_names=["1"])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert torch.allclose(lllap.H, lap.H, rtol=1e-3)

        # define valid parameter name subnet mask
        subnetmask_kwargs.update(module_names=["0"])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 80
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

    elif subnetwork_mask == LastLayerSubnetMask:
        # should raise error if we pass invalid last-layer name
        subnetmask_kwargs.update(last_layer_name="123")
        with pytest.raises(KeyError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # define valid last-layer subnet mask (without passing the last-layer name)
        subnetmask_kwargs = dict(model=model)
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

        # check that Hessian is identical to that of a full LLLaplace model
        assert torch.allclose(lllap.H, lap.H, rtol=1e-3)

        # define valid last-layer subnet mask (with passing the last-layer name)
        subnetmask_kwargs.update(last_layer_name="1")
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 42
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(
            model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            subnetwork_indices=subnetmask.indices,
            hessian_structure=hessian_structure,
        )
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

        # check that Hessian is identical to that of a full LLLaplace model
        assert torch.allclose(lllap.H, lap.H, rtol=1e-3)

    # check some parameters
    assert subnetmask.indices.equal(lap.backend.subnetwork_indices)
    assert subnetmask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet

    # check that Hessian and prior precision is of correct shape
    if hessian_structure == "full":
        assert lap.H.shape == (n_params_subnet, n_params_subnet)
    else:
        assert lap.H.shape == (n_params_subnet,)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)


@pytest.mark.parametrize(
    "likelihood,hessian_structure", product(likelihoods, hessian_structures)
)
def test_full_subnet_mask(
    model, likelihood, class_loader, reg_loader, hessian_structure
):
    loader = class_loader if likelihood == "classification" else reg_loader

    # define full model 'subnet' mask class (i.e. where all parameters are part of the subnet)
    class FullSubnetMask(SubnetMask):
        def get_subnet_mask(self, train_loader):
            return torch.ones(model.n_params).byte()

    # define and fit valid subnet Laplace model over all weights
    subnetwork_mask = FullSubnetMask
    subnetmask = subnetwork_mask(model=model)
    subnetmask.select(loader)
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    lap.fit(loader)
    assert isinstance(lap, SubnetLaplace)

    # check some parameters
    assert subnetmask.indices.equal(torch.tensor(list(range(model.n_params))))
    assert subnetmask.n_params_subnet == model.n_params
    assert lap.n_params_subnet == model.n_params

    # check that the Hessian is identical to that of an all-weights Laplace model
    full_lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure=hessian_structure,
    )
    full_lap.fit(loader)
    assert torch.allclose(full_lap.H, lap.H, rtol=1e-3)


@pytest.mark.parametrize(
    "subnetwork_mask,hessian_structure", product(all_subnet_masks, hessian_structures)
)
def test_regression_predictive(model, reg_loader, subnetwork_mask, hessian_structure):
    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask in score_based_subnet_masks:
        subnetmask_kwargs.update(n_params_subnet=32)
        if subnetwork_mask == LargestVarianceSWAGSubnetMask:
            subnetmask_kwargs.update(likelihood="regression")
        elif subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            diag_laplace_model = DiagLaplace(model, "regression")
            subnetmask_kwargs.update(diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == ParamNameSubnetMask:
        subnetmask_kwargs.update(parameter_names=["0.weight", "1.bias"])
    elif subnetwork_mask == ModuleNameSubnetMask:
        subnetmask_kwargs.update(module_names=["0"])

    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select(reg_loader)
    lap = Laplace(
        model,
        likelihood="regression",
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, SubnetLaplace)

    lap.fit(reg_loader)
    X, _ = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    f_mu, f_var = lap(X, pred_type="glm")
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(X, pred_type="nn", link_approx="mc")
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(X)


@pytest.mark.parametrize(
    "subnetwork_mask,hessian_structure", product(all_subnet_masks, hessian_structures)
)
def test_classification_predictive(
    model, class_loader, subnetwork_mask, hessian_structure
):
    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask in score_based_subnet_masks:
        subnetmask_kwargs.update(n_params_subnet=32)
        if subnetwork_mask == LargestVarianceSWAGSubnetMask:
            subnetmask_kwargs.update(likelihood="classification")
        elif subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            diag_laplace_model = DiagLaplace(model, "classification")
            subnetmask_kwargs.update(diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == ParamNameSubnetMask:
        subnetmask_kwargs.update(parameter_names=["0.weight", "1.bias"])
    elif subnetwork_mask == ModuleNameSubnetMask:
        subnetmask_kwargs.update(module_names=["0"])

    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select(class_loader)
    lap = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, SubnetLaplace)

    lap.fit(class_loader)
    X, _ = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    f_pred = lap(X, pred_type="glm", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="probit")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge_norm")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1

    # NN predictive
    f_pred = lap(X, pred_type="nn", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1


@pytest.mark.parametrize(
    "subnetwork_mask,likelihood,hessian_structure",
    product(all_subnet_masks, likelihoods, hessian_structures),
)
def test_subnet_marginal_likelihood(
    model, subnetwork_mask, likelihood, hessian_structure, class_loader, reg_loader
):
    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask in score_based_subnet_masks:
        subnetmask_kwargs.update(n_params_subnet=32)
        if subnetwork_mask == LargestVarianceSWAGSubnetMask:
            subnetmask_kwargs.update(likelihood=likelihood)
        elif subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            diag_laplace_model = DiagLaplace(model, likelihood)
            subnetmask_kwargs.update(diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == ParamNameSubnetMask:
        subnetmask_kwargs.update(parameter_names=["0.weight", "1.bias"])
    elif subnetwork_mask == ModuleNameSubnetMask:
        subnetmask_kwargs.update(module_names=["0"])

    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    loader = class_loader if likelihood == "classification" else reg_loader
    subnetmask.select(loader)
    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    assert isinstance(lap, SubnetLaplace)

    lap.fit(loader)
    lap.log_marginal_likelihood()


@pytest.mark.parametrize(
    "likelihood,hessian_structure", product(likelihoods, hessian_structures)
)
def test_sample(model, likelihood, hessian_structure, class_loader, reg_loader):
    loader = class_loader if likelihood == "classification" else reg_loader

    subnetmask = RandomSubnetMask(model=model, n_params_subnet=10)
    subnetmask.select()

    lap = Laplace(
        model,
        likelihood=likelihood,
        subset_of_weights="subnetwork",
        subnetwork_indices=subnetmask.indices,
        hessian_structure=hessian_structure,
    )
    lap.fit(loader)

    n_samples = 20
    generator = torch.Generator()
    generator.manual_seed(123)
    generator_state = generator.get_state()

    # Generate different samples.
    for gen in [generator, None]:
        samples_1 = lap.sample(n_samples=n_samples, generator=gen)
        samples_2 = lap.sample(n_samples=n_samples, generator=gen)
        assert not (samples_1 == samples_2).all()
    assert samples_1.shape == (n_samples, model.n_params)

    # Test that generator gets used.
    generator.set_state(generator_state)
    samples_1 = lap.sample(n_samples=n_samples, generator=generator)
    generator.set_state(generator_state)
    samples_2 = lap.sample(n_samples=n_samples, generator=generator)
    assert (samples_1 == samples_2).all()

    # Test that only model params indexed by subnetmask.indices vary across samples.
    model_params = parameters_to_vector(model.parameters())
    fixed_mask = torch.ones(model.n_params, dtype=bool)
    fixed_mask[subnetmask.indices] = False
    n_fixed = model.n_params - lap.n_params_subnet
    assert n_fixed == fixed_mask.nonzero().shape[0]
    assert samples_1[:, fixed_mask].shape == (n_samples, n_fixed)
    assert (
        samples_1[:, fixed_mask] == model_params[fixed_mask].repeat(n_samples, 1)
    ).all()
    assert (samples_1[:, ~fixed_mask] == samples_1[:, subnetmask.indices]).all()
    assert not (
        samples_1[:, ~fixed_mask] == model_params[~fixed_mask].repeat(n_samples, 1)
    ).all()


@pytest.mark.parametrize("laplace", [FullSubnetLaplace, DiagSubnetLaplace])
@pytest.mark.parametrize(
    "backend", [AsdlEF, AsdlGGN, BackPackEF, BackPackGGN, CurvlinopsEF, CurvlinopsGGN]
)
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_dtype(laplace, backend, dtype, likelihood):
    X = torch.randn((10, 3), dtype=dtype)
    Y = torch.randn((10, 3), dtype=dtype)

    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=10)

    model = nn.Linear(3, 3, dtype=dtype)

    subnetmask = RandomSubnetMask(model=model, n_params_subnet=10)
    subnetmask.select()

    try:
        la = laplace(
            model, likelihood, subnetwork_indices=subnetmask.indices, backend=backend
        )
        la.fit(dataloader)

        assert la.H is not None

        if isinstance(la.H, torch.Tensor):
            assert la.H.dtype == dtype

        assert la.log_marginal_likelihood().dtype == dtype

        y_pred, y_var = la(X, pred_type="glm")
        assert y_pred.dtype == dtype
        assert y_var.dtype == dtype

        y_pred = la(X, pred_type="nn", num_samples=3)
        assert y_pred.dtype == dtype
    except (ValueError, AttributeError, RuntimeError, SystemExit) as e:
        if "must have the same dtype" in str(e):
            assert False  # Fail the test
        else:
            pass  # Ignore
