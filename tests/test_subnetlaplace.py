import pytest
from itertools import product 

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace, SubnetLaplace
from laplace.subnetmask import SubnetMask, RandomSubnetMask, LargestMagnitudeSubnetMask, LastLayerSubnetMask, LargestVarianceDiagLaplaceSubnetMask


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
score_based_subnet_masks = [RandomSubnetMask, LargestMagnitudeSubnetMask, LargestVarianceDiagLaplaceSubnetMask]
all_subnet_masks = score_based_subnet_masks + [LastLayerSubnetMask] 
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
    # use last-layer subnet mask for this test
    subnetwork_mask = LastLayerSubnetMask

    # subnet Laplace with full Hessian should work
    hessian_structure = 'full'
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure=hessian_structure)
    assert isinstance(lap, SubnetLaplace)

    # subnet Laplace with diag, kron or lowrank Hessians should raise errors
    hessian_structure = 'diag'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure=hessian_structure)
    hessian_structure = 'kron'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure=hessian_structure)
    hessian_structure = 'lowrank'
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure=hessian_structure)


@pytest.mark.parametrize('subnetwork_mask,likelihood', product(score_based_subnet_masks, likelihoods))
def test_score_based_subnet_masks(model, likelihood, subnetwork_mask, class_loader, reg_loader):
    loader = class_loader if likelihood == 'classification' else reg_loader

    # should raise error if we don't pass number of subnet parameters within the subnetmask_kwargs
    subnetmask_kwargs = dict()
    with pytest.raises(TypeError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)

    # should raise error if we set number of subnet parameters to None
    subnetmask_kwargs = dict(n_params_subnet=None)
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)

    # should raise error if we set number of subnet parameters to be larger than number of model parameters
    subnetmask_kwargs = dict(n_params_subnet=99999)
    with pytest.raises(ValueError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)

    # define valid subnet Laplace model
    n_params_subnet = 32
    subnetmask_kwargs = dict(n_params_subnet=n_params_subnet)
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

    # should raise error if we try to access the subnet indices before the subnet has been selected
    with pytest.raises(AttributeError):
        lap._subnetwork_mask.indices

    # select subnet mask
    lap._subnetwork_mask.select(loader)

    # should raise error if we try to select the subnet again
    with pytest.raises(ValueError):
        lap._subnetwork_mask.select(loader)

    # re-define valid subnet Laplace model
    n_params_subnet = 32
    subnetmask_kwargs = dict(n_params_subnet=n_params_subnet)
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

    # fit Laplace model (which internally selects the subnet mask)
    lap.fit(loader)

    # check some parameters
    assert lap._subnetwork_mask.indices.equal(lap.backend.subnetwork_indices)
    assert lap._subnetwork_mask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet

    # check that Hessian and prior precision is of correct shape
    assert lap.H.shape == (n_params_subnet, n_params_subnet)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)

    # should raise error if we try to fit the Laplace mdoel again 
    with pytest.raises(ValueError):
        lap.fit(loader)


@pytest.mark.parametrize('likelihood', likelihoods)
def test_last_layer_subnet_mask(model, likelihood, class_loader, reg_loader):
    subnetwork_mask = LastLayerSubnetMask
    loader = class_loader if likelihood == 'classification' else reg_loader

    # should raise error if we pass number of subnet parameters
    subnetmask_kwargs = dict(n_params_subnet=32)
    with pytest.raises(TypeError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)

    # should raise error if we pass invalid last-layer name 
    subnetmask_kwargs = dict(last_layer_name='123')
    with pytest.raises(KeyError):
        lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)

    # define valid last-layer subnet Laplace model (without passing the last-layer name)
    subnetmask_kwargs = dict()
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

    # define valid last-layer subnet Laplace model (with passing the last-layer name)
    subnetmask_kwargs = dict(last_layer_name='1')
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

    # should raise error if we access number of subnet parameters before selecting the subnet
    with pytest.raises(AttributeError):
        n_params_subnet = lap._subnetwork_mask.n_params_subnet

    # fit Laplace model
    lap.fit(loader)

    # check some parameters
    n_params_subnet = 42
    assert lap._subnetwork_mask.indices.equal(lap.backend.subnetwork_indices)
    assert lap._subnetwork_mask.n_params_subnet == n_params_subnet
    assert lap.n_params_subnet == n_params_subnet

    # check that Hessian and prior precision is of correct shape
    assert lap.H.shape == (n_params_subnet, n_params_subnet)
    assert lap.prior_precision_diag.shape == (n_params_subnet,)

    # check that Hessian is identical to that of a full LLLaplace model
    lllap = Laplace(model, likelihood=likelihood, subset_of_weights='last_layer', hessian_structure='full')
    lllap.fit(loader)
    assert lllap.H.equal(lap.H)


@pytest.mark.parametrize('likelihood', likelihoods)
def test_full_subnet_mask(model, likelihood, class_loader, reg_loader):
    loader = class_loader if likelihood == 'classification' else reg_loader

    # define full model 'subnet' mask class (i.e. where all parameters are part of the subnet)
    class FullSubnetMask(SubnetMask):
        @property
        def n_params_subnet(self):
            if self._n_params_subnet is None:
                self._check_select()
                self._n_params_subnet = len(self._indices)
            return self._n_params_subnet

        def get_subnet_mask(self, train_loader):
            return torch.ones(model.n_params).byte()

    # define and fit valid full subnet Laplace model
    subnetwork_mask = FullSubnetMask
    lap = Laplace(model, likelihood=likelihood, subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full')
    lap.fit(loader)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

    # check some parameters
    assert lap._subnetwork_mask.indices.equal(torch.tensor(list(range(model.n_params))))
    assert lap._subnetwork_mask.n_params_subnet == model.n_params
    assert lap.n_params_subnet == model.n_params

    # check that the Hessian is identical to that of a all-weights FullLaplace model
    full_lap = Laplace(model, likelihood=likelihood, subset_of_weights='all', hessian_structure='full')
    full_lap.fit(loader)
    assert full_lap.H.equal(lap.H)


@pytest.mark.parametrize('subnetwork_mask', all_subnet_masks)
def test_regression_predictive(model, reg_loader, subnetwork_mask):
    subnetmask_kwargs = dict(n_params_subnet=32) if subnetwork_mask in score_based_subnet_masks else dict()
    lap = Laplace(model, likelihood='regression', subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

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
    subnetmask_kwargs = dict(n_params_subnet=32) if subnetwork_mask in score_based_subnet_masks else dict()
    lap = Laplace(model, likelihood='classification', subset_of_weights='subnetwork', subnetwork_mask=subnetwork_mask, hessian_structure='full', subnetmask_kwargs=subnetmask_kwargs)
    assert isinstance(lap, SubnetLaplace)
    assert isinstance(lap._subnetwork_mask, subnetwork_mask)

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