import os
from collections import OrderedDict
from importlib.util import find_spec

import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset

from laplace import (
    DiagLaplace,
    DiagLLLaplace,
    DiagSubnetLaplace,
    FullLaplace,
    FullLLLaplace,
    FullSubnetLaplace,
    FunctionalLaplace,
    FunctionalLLLaplace,
    KronLaplace,
    KronLLLaplace,
    Laplace,
    LLLaplace,
    LowRankLaplace,
    SubnetLaplace,
)

torch.manual_seed(240)
torch.set_default_dtype(torch.double)

lrlaplace_param = pytest.param(
    LowRankLaplace, marks=pytest.mark.xfail(reason="Unimplemented in the new ASDL")
)
flavors = [
    FullLaplace,
    KronLaplace,
    DiagLaplace,
    lrlaplace_param,
    FullLLLaplace,
    KronLLLaplace,
    DiagLLLaplace,
]

flavors_no_llla = [FullLaplace, KronLaplace, DiagLaplace, lrlaplace_param]
flavors_llla = [FullLLLaplace, KronLLLaplace, DiagLLLaplace]
flavors_subnet = [DiagSubnetLaplace, FullSubnetLaplace]
flavors_functional = [FunctionalLaplace, FunctionalLLLaplace]
flavors = flavors_llla


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model2():
    model = torch.nn.Sequential(nn.Linear(3, 25), nn.Linear(25, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model3():
    model = torch.nn.Sequential(
        OrderedDict([("fc1", nn.Linear(3, 20)), ("clf", nn.Linear(20, 2))])
    )
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Run after test
    if os.path.exists("state_dict.bin"):
        os.remove("state_dict.bin")


@pytest.mark.parametrize("laplace", flavors)
def test_serialize(laplace, model, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "regression")
    la2.load_state_dict(torch.load("state_dict.bin"))

    assert la.sigma_noise == la2.sigma_noise

    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X)
    f_mean2, f_var2 = la2(X)
    assert torch.allclose(f_mean, f_mean2)
    assert torch.allclose(f_var, f_var2)


@pytest.mark.parametrize("laplace", flavors_functional)
def test_serialize_functional(laplace, model, reg_loader):
    la = laplace(model, "regression", n_subset=10)
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "regression", n_subset=10)
    la2.load_state_dict(torch.load("state_dict.bin"))

    assert la.sigma_noise == la2.sigma_noise

    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X)
    f_mean2, f_var2 = la2(X)
    assert torch.allclose(f_mean, f_mean2)
    assert torch.allclose(f_var, f_var2)


@pytest.mark.parametrize("laplace", flavors_no_llla[:-1])
def test_serialize_override(laplace, model, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    H_orig = la.H_facs.to_matrix() if laplace == KronLaplace else la.H
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "regression")
    la2.load_state_dict(torch.load("state_dict.bin"))

    # Emulating continual learning
    la2.fit(reg_loader, override=False)

    H_new = la2.H_facs.to_matrix() if laplace == KronLaplace else la2.H
    assert torch.allclose(2 * H_orig, H_new)


@pytest.mark.parametrize("laplace", flavors)
def test_serialize_no_pickle(laplace, model, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")
    state_dict = torch.load("state_dict.bin")

    # Make sure no pickle object
    for val in state_dict.values():
        if val is not None:
            assert isinstance(val, (list, tuple, int, float, str, bool, torch.Tensor))


@pytest.mark.parametrize("laplace", flavors_functional)
def test_serialize_no_pickle_functional(laplace, model, reg_loader):
    la = laplace(model, "regression", n_subset=10)
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")
    state_dict = torch.load("state_dict.bin")

    # Make sure no pickle object
    for val in state_dict.values():
        if val is not None:
            assert isinstance(
                val, (DataLoader, list, tuple, int, float, str, bool, torch.Tensor)
            )


@pytest.mark.parametrize("laplace", flavors_subnet)
def test_serialize_subnetlaplace(laplace, model, reg_loader):
    subnetwork_indices = torch.LongTensor([1, 10, 104, 44])
    la = laplace(model, "regression", subnetwork_indices=subnetwork_indices)
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "regression", subnetwork_indices=subnetwork_indices)
    la2.load_state_dict(torch.load("state_dict.bin"))

    assert la.sigma_noise == la2.sigma_noise

    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X)
    f_mean2, f_var2 = la2(X)
    assert torch.allclose(f_mean, f_mean2)
    assert torch.allclose(f_var, f_var2)


@pytest.mark.parametrize("laplace", flavors_no_llla)
def test_serialize_fail_different_models(laplace, model, model2, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model2, "regression")

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin"))


def test_serialize_fail_different_hess_structures(model, reg_loader):
    la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="kron")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = Laplace(
        model, "regression", subset_of_weights="all", hessian_structure="diag"
    )

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin"))


def test_serialize_fail_different_subset_of_weights(model, reg_loader):
    la = Laplace(
        model, "regression", subset_of_weights="last_layer", hessian_structure="diag"
    )
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = Laplace(
        model, "regression", subset_of_weights="all", hessian_structure="diag"
    )

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin"))


@pytest.mark.parametrize("laplace", flavors)
def test_serialize_fail_different_liks(laplace, model, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "classification")

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin"))


@pytest.mark.parametrize("laplace", flavors_llla)
def test_serialize_fail_llla_different_last_layer_name(
    laplace, model, model3, reg_loader
):
    la = laplace(model, "regression", last_layer_name="1")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model3, "classification", last_layer_name="clf")

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin"))


@pytest.mark.parametrize(
    "model_device_str",
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
)
@pytest.mark.parametrize(
    "map_location",
    ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
)
@pytest.mark.parametrize("laplace", flavors + flavors_subnet)
def test_map_location(
    laplace, model, reg_loader, tmp_path, map_location, model_device_str
):
    # Only for LLLaplace, we need dill.
    #   AttributeError: Can't pickle local object 'FeatureExtractor._get_hook.<locals>.hook'
    if issubclass(laplace, LLLaplace):
        if find_spec("dill") is None:
            pytest.skip(reason="dill package not found but needed for this test")
        else:
            import dill

            def torch_save(obj, fn):
                return torch.save(obj, fn, pickle_module=dill)
    else:
        # Use default pickle_module=pickle, but no need to import pickle here
        # just to set pickle_module=pickle.
        def torch_save(obj, fn):
            return torch.save(obj, fn)

    device = torch.device(model_device_str)
    kwds = dict(model=model.to(device), likelihood="regression")
    if issubclass(laplace, SubnetLaplace):
        kwds.update(subnetwork_indices=torch.LongTensor([1, 10, 104, 44]))
    la = laplace(**kwds)

    # device(type='cuda', index=0) != device(type='cuda')
    ##assert la._device == device
    assert la._device.type == device.type

    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231

    save_fn = tmp_path / "la.pt"
    torch_save(la, save_fn)
    la2 = torch.load(save_fn, map_location=map_location)

    assert la2._device.type == map_location

    # Test tensor-valued properties that use self._device in their definition.
    # This is redundant if *all* will be used in the forward call below, where
    # things would fail if self._device is wrong.
    for name in [
        "prior_precision_diag",
        "prior_mean",
        "prior_precision",
        "sigma_noise",
    ]:
        assert getattr(la, name).device.type == device.type, f"la.{name} failed"
        assert getattr(la2, name).device.type == map_location, f"la2.{name} failed"

    # Test tensor attrs.
    for name, obj in vars(la).items():
        # Only LLLaplace instances have a self.X that comes from the train
        # loader, is assigned in fit() and not explicitly sent to device (it
        # will be dynamically when used, as is done elsewhere in the code, such
        # as in BaseLaplace.fit()). self.X won't be affected by self._device.
        # Since the predict() below passes, we ignore the wrong device here
        # for now.
        if issubclass(laplace, LLLaplace) and name == "X":
            continue
        if isinstance(obj, torch.Tensor):
            assert obj.device.type == device.type, f"la.{name} failed"
            assert getattr(la2, name).device.type == map_location, f"la2.{name} failed"

    assert la.sigma_noise == la2.sigma_noise
    X, _ = next(iter(reg_loader))
    f_mean, f_var = la(X.to(device))
    f_mean2, f_var2 = la2(X.to(map_location))
    assert torch.allclose(f_mean.to("cpu"), f_mean2.to("cpu"))
    assert torch.allclose(f_var.to("cpu"), f_var2.to("cpu"))
