import os
import pickle
from collections import OrderedDict
from copy import deepcopy
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
from laplace.curvature import CurvlinopsEF
from laplace.likelihoods import (
    BernoulliLikelihood,
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
)
from tests.utils import ScaledGaussian, toy_multilabel_dataset

# These tests load only checkpoints written by the same test process.
# Explicit pickle loading preserves their intent on PyTorch 2.6 and newer.
pytestmark = [pytest.mark.default_dtype(torch.float64), pytest.mark.seed(240)]

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
    la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))

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
    la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))

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
    la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))

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
    state_dict = torch.load("state_dict.bin", weights_only=False)

    # Make sure no pickle object
    for val in state_dict.values():
        if val is not None:
            assert isinstance(
                val, (dict, list, tuple, int, float, str, bool, torch.Tensor)
            )


@pytest.mark.parametrize("laplace", flavors_functional)
def test_serialize_no_pickle_functional(laplace, model, reg_loader):
    la = laplace(model, "regression", n_subset=10)
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")
    state_dict = torch.load("state_dict.bin", weights_only=False)

    # Make sure no pickle object
    for val in state_dict.values():
        if val is not None:
            assert isinstance(
                val,
                (DataLoader, dict, list, tuple, int, float, str, bool, torch.Tensor),
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
    la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))

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
        la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))


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
        la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))


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
        la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))


@pytest.mark.parametrize("laplace", flavors)
def test_serialize_fail_different_liks(laplace, model, reg_loader):
    la = laplace(model, "regression")
    la.fit(reg_loader)
    la.optimize_prior_precision()
    la.sigma_noise = 1231
    torch.save(la.state_dict(), "state_dict.bin")

    la2 = laplace(model, "classification")

    with pytest.raises(ValueError):
        la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))


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
        la2.load_state_dict(torch.load("state_dict.bin", weights_only=False))


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
    la2 = torch.load(save_fn, map_location=map_location, weights_only=False)

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


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
def test_load_old_string_checkpoint(laplace, model, class_loader):
    la = laplace(model, "classification")
    la.fit(class_loader)
    state_dict = la.state_dict()
    state_dict["likelihood"] = "classification"

    la2 = laplace(model, CategoricalLikelihood())
    la2.load_state_dict(state_dict)
    assert isinstance(la2.likelihood, CategoricalLikelihood)

    X, _ = next(iter(class_loader))
    assert torch.allclose(la(X), la2(X))

    state_dict["likelihood"] = "regression"
    with pytest.raises(ValueError, match="Different likelihoods"):
        laplace(model, CategoricalLikelihood()).load_state_dict(state_dict)


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize("likelihood", ["regression", GaussianLikelihood()])
def test_state_dict_likelihood_is_str(laplace, likelihood, model, reg_loader):
    la = laplace(model, likelihood)
    la.fit(reg_loader)
    state_dict = la.state_dict()

    assert type(state_dict["likelihood"]) is str
    assert state_dict["likelihood"] == "regression"


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "likelihood", [CategoricalLikelihood(), BradleyTerryLikelihood()]
)
def test_deepcopy_and_pickle_object_likelihood(
    laplace, likelihood, model, class_loader
):
    if likelihood == "reward_modeling":
        X_train = torch.randn(10, 2, 3)
        net = nn.Sequential(nn.Linear(3, 1), nn.Flatten(1))
        loader = DataLoader(
            TensorDataset(X_train, torch.randint(2, (10,))), batch_size=3
        )
        X = X_train[:4]
    else:
        net, loader = model, class_loader
        X = next(iter(class_loader))[0]

    la = laplace(net, likelihood)
    la.fit(loader)

    for la2 in [deepcopy(la), pickle.loads(pickle.dumps(la))]:
        assert la2.likelihood == likelihood
        assert la2.likelihood is not la.likelihood
        assert torch.allclose(la(X, fitting=True), la2(X, fitting=True))


def test_checkpoint_stores_likelihood_hyperparameters(model, reg_loader):
    la = DiagLaplace(model, ScaledGaussian(2.0))
    la.fit(reg_loader)
    state_dict = la.state_dict()

    assert state_dict["likelihood"] == "scaled_gaussian"
    assert state_dict["likelihood_hyperparameters"] == {"scale": 2.0}

    DiagLaplace(model, ScaledGaussian(2.0)).load_state_dict(state_dict)

    with pytest.raises(ValueError, match="Different likelihoods"):
        DiagLaplace(model, ScaledGaussian(1.0)).load_state_dict(state_dict)

    default = DiagLaplace(model, "regression")
    default.fit(reg_loader)
    assert default.state_dict()["likelihood_hyperparameters"] == {}


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
def test_old_checkpoint_without_hyperparameters_loads(laplace, model, class_loader):
    la = laplace(model, "classification")
    la.fit(class_loader)
    state_dict = la.state_dict()
    del state_dict["likelihood_hyperparameters"]

    la2 = laplace(model, "classification")
    la2.load_state_dict(state_dict)

    X, _ = next(iter(class_loader))
    assert torch.allclose(la(X), la2(X))


@pytest.mark.parametrize("laplace", [DiagLaplace, FunctionalLaplace])
def test_checkpoint_copies_tensor_hyperparameters(laplace, model, class_loader):
    weight = torch.tensor([1.0, 2.0])
    kwargs = {"n_subset": 5} if laplace is FunctionalLaplace else {}
    la = laplace(model, CategoricalLikelihood(weight=weight), **kwargs)
    la.fit(class_loader)

    stored = la.state_dict()["likelihood_hyperparameters"]["weight"]
    assert stored is not weight
    assert torch.equal(stored, weight)

    # Later changes to the weight don't reach the checkpoint
    weight.mul_(3.0)
    assert torch.equal(stored, torch.tensor([1.0, 2.0]))


@pytest.fixture
def multilabel_loader():
    return toy_multilabel_dataset(n_train=10, batch_size=3, out_dim=2, soft=True)[2]


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace, FullLLLaplace])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"pos_weight": torch.tensor([3.0, 0.5])},
        {"weight": torch.tensor([1.0, 2.0]), "pos_weight": torch.tensor([3.0, 0.5])},
    ],
)
def test_bernoulli_checkpoint_roundtrip(laplace, options, model, multilabel_loader):
    lik = BernoulliLikelihood(**options)
    la = laplace(model, lik)
    la.fit(multilabel_loader)
    state_dict = la.state_dict()

    assert state_dict["likelihood"] == "bernoulli"
    assert set(state_dict["likelihood_hyperparameters"]) == set(options)

    for name, value in options.items():
        stored = state_dict["likelihood_hyperparameters"][name]

        assert torch.equal(stored, value)
        assert stored is not value

    torch.save(state_dict, "state_dict.bin")
    loaded = torch.load("state_dict.bin", weights_only=False)

    la2 = laplace(model, BernoulliLikelihood(**options))
    la2.load_state_dict(loaded)

    X, _ = next(iter(multilabel_loader))
    assert torch.allclose(la(X), la2(X))
    assert torch.allclose(la.log_marginal_likelihood(), la2.log_marginal_likelihood())

    # A criterion with the same options is the same likelihood
    la3 = laplace(model, nn.BCEWithLogitsLoss(**options))
    la3.load_state_dict(loaded)


def test_bernoulli_checkpoint_rejects_other_options(model, multilabel_loader):
    pos_weight = torch.tensor([3.0, 0.5])
    la = DiagLaplace(model, BernoulliLikelihood(pos_weight=pos_weight))
    la.fit(multilabel_loader)
    state_dict = la.state_dict()

    for other in (
        BernoulliLikelihood(),
        BernoulliLikelihood(pos_weight=2 * pos_weight),
        BernoulliLikelihood(weight=pos_weight),
        BernoulliLikelihood(weight=pos_weight, pos_weight=pos_weight),
        CategoricalLikelihood(),
    ):
        with pytest.raises(ValueError, match="Different likelihoods"):
            DiagLaplace(model, other).load_state_dict(state_dict)

    # And the other way around
    default = DiagLaplace(model, "bernoulli")
    default.fit(multilabel_loader)

    with pytest.raises(ValueError, match="Different likelihoods"):
        DiagLaplace(model, BernoulliLikelihood(pos_weight=pos_weight)).load_state_dict(
            default.state_dict()
        )


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
def test_deepcopy_and_pickle_bernoulli(laplace, model, multilabel_loader):
    lik = BernoulliLikelihood(weight=torch.tensor([1.0, 2.0]))
    kwargs = {"backend": CurvlinopsEF} if laplace is KronLaplace else {}
    la = laplace(model, lik, **kwargs)
    la.fit(multilabel_loader)
    X = next(iter(multilabel_loader))[0]

    for la2 in [deepcopy(la), pickle.loads(pickle.dumps(la))]:
        assert la2.likelihood == lik
        assert la2.likelihood is not la.likelihood
        assert torch.allclose(la(X), la2(X))
