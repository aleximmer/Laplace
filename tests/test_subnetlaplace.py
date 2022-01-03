import pytest
from itertools import product 

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace, SubnetLaplace
from laplace.baselaplace import DiagLaplace
from laplace.utils.subnetmask import SubnetMask, RandomSubnetMask, LargestMagnitudeSubnetMask, LargestVarianceDiagLaplaceSubnetMask, LargestVarianceSWAGSubnetMask, ParamNameSubnetMask, ModuleNameSubnetMask, LastLayerSubnetMask


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
score_based_subnet_masks = [RandomSubnetMask, LargestMagnitudeSubnetMask, LargestVarianceDiagLaplaceSubnetMask, LargestVarianceSWAGSubnetMask]
layer_subnet_masks = [ParamNameSubnetMask, ModuleNameSubnetMask, LastLayerSubnetMask]
all_subnet_masks = score_based_subnet_masks + layer_subnet_masks
likelihoods = ['classification', 'regression']


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    model_params = list(model.parameters())
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
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


@pytest.mark.parametrize('likelihood', likelihoods)
def test_subnet_laplace_init(model, likelihood):
    # use random subnet mask for this test
    subnetwork_mask = RandomSubnetMask
    subnetmask_kwargs = dict(model=model, n_params_subnet=10)
    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select()

    # subnet Laplace with full Hessian should work
    hessian_structure = 'full'
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure=hessian_structure)
    assert isinstance(lap, SubnetLaplace)

    # subnet Laplace with diag, kron or lowrank Hessians should raise errors
    hessian_structure = 'diag'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure=hessian_structure)
    hessian_structure = 'kron'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure=hessian_structure)
    hessian_structure = 'lowrank'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure=hessian_structure)


@pytest.mark.parametrize('subnetwork_mask,likelihood', product(score_based_subnet_masks, likelihoods))
def test_score_based_subnet_masks(model, likelihood, subnetwork_mask, class_loader, reg_loader):
    loader = class_loader if likelihood == 'classification' else reg_loader
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

    # should raise error if we set number of subnet parameters to be larger than number of model parameters
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
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
    assert isinstance(lap, SubnetLaplace)

    # fit Laplace model
    lap.fit(loader)

    # check some parameters
    assert subnetmask.indices.equal(lap.backend.subnetwork_indices)
    assert subnetmask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet
    assert parameters_to_vector(model.parameters()).equal(model_params)

    # check that Hessian and prior precision is of correct shape
    assert lap.H.shape == (n_params_subnet, n_params_subnet)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)


@pytest.mark.parametrize('subnetwork_mask,likelihood', product(layer_subnet_masks, likelihoods))
def test_layer_subnet_masks(model, likelihood, subnetwork_mask, class_loader, reg_loader):
    loader = class_loader if likelihood == 'classification' else reg_loader
    subnetmask_kwargs = dict(model=model)

    # fit last-layer Laplace model
    lllap = Laplace(model, likelihood=likelihood, subset_of_weights='last_layer', hessian_structure='full')
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
        subnetmask_kwargs.update(parameter_names=['123'])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # define last-layer Laplace model by parameter names and check that Hessian is identical to that of a full LLLaplace model
        subnetmask_kwargs.update(parameter_names=['1.weight', '1.bias'])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
        lap.fit(loader)
        assert lllap.H.equal(lap.H)

        # define valid parameter name subnet mask
        subnetmask_kwargs.update(parameter_names=['0.weight', '1.bias'])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 62
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
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
        subnetmask_kwargs.update(module_names=['123'])
        with pytest.raises(ValueError):
            subnetmask = subnetwork_mask(**subnetmask_kwargs)
            subnetmask.select(loader)

        # define last-layer Laplace model by module name and check that Hessian is identical to that of a full LLLaplace model
        subnetmask_kwargs.update(module_names=['1'])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)
        subnetmask.select(loader)
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
        lap.fit(loader)
        assert lllap.H.equal(lap.H)

        # define valid parameter name subnet mask
        subnetmask_kwargs.update(module_names=['0'])
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 80
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

    elif subnetwork_mask == LastLayerSubnetMask:
        # should raise error if we pass invalid last-layer name 
        subnetmask_kwargs.update(last_layer_name='123')
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
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

        # check that Hessian is identical to that of a full LLLaplace model
        assert lllap.H.equal(lap.H)

        # define valid last-layer subnet mask (with passing the last-layer name)
        subnetmask_kwargs.update(last_layer_name='1')
        subnetmask = subnetwork_mask(**subnetmask_kwargs)

        # should raise error if we access number of subnet parameters before selecting the subnet
        n_params_subnet = 42
        with pytest.raises(AttributeError):
            n_params_subnet = subnetmask.n_params_subnet

        # select subnet mask and fit Laplace model
        subnetmask.select(loader)
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
        lap.fit(loader)
        assert isinstance(lap, SubnetLaplace)

        # check that Hessian is identical to that of a full LLLaplace model
        assert lllap.H.equal(lap.H)

    # check some parameters
    assert subnetmask.indices.equal(lap.backend.subnetwork_indices)
    assert subnetmask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet

    # check that Hessian and prior precision is of correct shape
    assert lap.H.shape == (n_params_subnet, n_params_subnet)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)


@pytest.mark.parametrize('likelihood', likelihoods)
def test_full_subnet_mask(model, likelihood, class_loader, reg_loader):
    loader = class_loader if likelihood == 'classification' else reg_loader

    # define full model 'subnet' mask class (i.e. where all parameters are part of the subnet)
    class FullSubnetMask(SubnetMask):
        def get_subnet_mask(self, train_loader):
            return torch.ones(model.n_params).byte()

    # define and fit valid full subnet Laplace model
    subnetwork_mask = FullSubnetMask
    subnetmask = subnetwork_mask(model=model)
    subnetmask.select(loader)
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
    lap.fit(loader)
    assert isinstance(lap, SubnetLaplace)

    # check some parameters
    assert subnetmask.indices.equal(torch.tensor(list(range(model.n_params))))
    assert subnetmask.n_params_subnet == model.n_params
    assert lap.n_params_subnet == model.n_params

    # check that the Hessian is identical to that of a all-weights FullLaplace model
    full_lap = Laplace(model, likelihood=likelihood, subset_of_weights='all', hessian_structure='full')
    full_lap.fit(loader)
    assert full_lap.H.equal(lap.H)


@pytest.mark.parametrize('subnetwork_mask', all_subnet_masks)
def test_regression_predictive(model, reg_loader, subnetwork_mask):
    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask in score_based_subnet_masks:
        subnetmask_kwargs.update(n_params_subnet=32)
        if subnetwork_mask == LargestVarianceSWAGSubnetMask:
            subnetmask_kwargs.update(likelihood='regression')
        elif subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            diag_laplace_model = DiagLaplace(model, 'regression')
            subnetmask_kwargs.update(diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == ParamNameSubnetMask:
        subnetmask_kwargs.update(parameter_names=['0.weight', '1.bias'])
    elif subnetwork_mask == ModuleNameSubnetMask:
        subnetmask_kwargs.update(module_names=['0'])

    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select(reg_loader)
    lap = Laplace(model, likelihood='regression', subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
    assert isinstance(lap, SubnetLaplace)

    lap.fit(reg_loader)
    X, _ = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    f_mu, f_var = lap(X, pred_type='glm')
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(X, pred_type='nn')
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(X)


@pytest.mark.parametrize('subnetwork_mask', all_subnet_masks)
def test_classification_predictive(model, class_loader, subnetwork_mask):
    subnetmask_kwargs = dict(model=model)
    if subnetwork_mask in score_based_subnet_masks:
        subnetmask_kwargs.update(n_params_subnet=32)
        if subnetwork_mask == LargestVarianceSWAGSubnetMask:
            subnetmask_kwargs.update(likelihood='classification')
        elif subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            diag_laplace_model = DiagLaplace(model, 'classification')
            subnetmask_kwargs.update(diag_laplace_model=diag_laplace_model)
    elif subnetwork_mask == ParamNameSubnetMask:
        subnetmask_kwargs.update(parameter_names=['0.weight', '1.bias'])
    elif subnetwork_mask == ModuleNameSubnetMask:
        subnetmask_kwargs.update(module_names=['0'])

    subnetmask = subnetwork_mask(**subnetmask_kwargs)
    subnetmask.select(class_loader)
    lap = Laplace(model, likelihood='classification', subset_of_weights='subnetwork', subnetwork_indices=subnetmask.indices, hessian_structure='full')
    assert isinstance(lap, SubnetLaplace)

    lap.fit(class_loader)
    X, _ = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    f_pred = lap(X, pred_type='glm', link_approx='mc', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='probit')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='bridge')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1

    # NN predictive
    f_pred = lap(X, pred_type='nn', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
