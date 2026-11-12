"""CPU contracts that ELLA and VaLLA share.

Each test runs once per method. Tests whose assertions differ between the two
stay in `test_ella.py` and `test_valla.py`.
"""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    SequentialSampler,
    TensorDataset,
)

from laplace import BradleyTerryLikelihood, GaussianLikelihood
from laplace.curvature import CurvlinopsGGN
from tests.ella_valla.helpers import (
    RewardModel,
    fit_estimator,
    make_estimator,
)
from tests.utils import AsdlGGN, BackPackGGN, requires_backpack


def test_functional_prior_mean_remains_zero(method, data):
    x, _, model = data
    estimator = make_estimator(method, model, x=x)
    with pytest.raises(ValueError, match="prior_mean=0"):
        estimator.prior_mean = 1.0
    torch.testing.assert_close(
        estimator.prior_mean, torch.zeros_like(estimator.prior_mean)
    )


def test_classification_noise_remains_one(method, data):
    x, _, model = data
    estimator = make_estimator(method, model, x=x)
    with pytest.raises(ValueError, match="only available for regression"):
        estimator.sigma_noise = 2.0
    torch.testing.assert_close(
        estimator.sigma_noise, torch.ones_like(estimator.sigma_noise)
    )


@pytest.mark.parametrize("option", ["prior_precision", "sigma_noise", "temperature"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_functional_hyperparameters_must_be_finite(method, option, value, data):
    x, _, _ = data
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match=option):
        make_estimator(method, model, GaussianLikelihood(), x=x, **{option: value})
    estimator = make_estimator(method, model, GaussianLikelihood(), x=x)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, value)


@pytest.mark.parametrize("option", ["prior_precision", "sigma_noise"])
def test_functional_hyperparameters_remain_finite_in_model_dtype(method, option):
    model = torch.nn.Linear(1, 1).float()
    options = {
        "ella": {"subsample_size": 1, "n_eigenvalues": 1},
        "valla": {"inducing_locations": torch.ones(1, 1)},
    }
    overflow = torch.tensor(1e50, dtype=torch.float64)
    with pytest.raises(ValueError, match=option):
        make_estimator(
            method, model, GaussianLikelihood(), **options, **{option: overflow}
        )
    estimator = make_estimator(method, model, GaussianLikelihood(), **options)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, overflow)


def test_checkpoint_rejects_different_mapping_key(method):
    class MappingRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, inputs):
            return self.linear(inputs["features"])

    inputs = torch.randn(4, 2)
    loader = DataLoader(
        [{"features": x, "labels": x.sum().unsqueeze(0)} for x in inputs], batch_size=2
    )
    model = MappingRegressor()
    original = deepcopy(model)
    options = {"dict_key_x": "features"}
    estimator = make_estimator(
        method,
        model,
        GaussianLikelihood(),
        valla={"inducing_locations": {"features": inputs[:2].clone()}},
        **options,
    )
    fit_estimator(method, estimator, loader)
    restored = make_estimator(
        method,
        original,
        GaussianLikelihood(),
        valla={"inducing_locations": "random", "num_inducing": 2},
    )
    with pytest.raises(ValueError, match="dict_key_x"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_rejects_different_parameterless_module(method, data):
    x, loader, _ = data
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    )
    other_model = deepcopy(model)
    other_model[1] = torch.nn.Tanh()
    estimator = make_estimator(method, model, x=x)
    fit_estimator(method, estimator, loader)
    restored = make_estimator(method, other_model, x=x)
    with pytest.raises(ValueError, match="same pretrained model"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_rejects_different_trainable_coordinates(method, data):
    x, loader, _ = data
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    other_model = deepcopy(model)
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    for parameter in other_model[1].parameters():
        parameter.requires_grad_(False)
    estimator = make_estimator(method, model, x=x)
    fit_estimator(method, estimator, loader)
    restored = make_estimator(method, other_model, x=x)
    with pytest.raises(ValueError, match="trainable coordinates"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_requires_same_pretrained_model(method, data):
    x, loader, model = data
    estimator = make_estimator(method, model, x=x)
    fit_estimator(method, estimator, loader, valla={"lr": 0.001})
    changed_model = deepcopy(model)
    with torch.no_grad():
        next(changed_model.parameters()).add_(1.0)
    restored = make_estimator(method, changed_model, x=x)
    with pytest.raises(ValueError, match="same pretrained model"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_restores_evaluation_mode(method, data):
    x, loader, _ = data
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(0.5), torch.nn.Linear(3, 2)
    )
    restored_model = deepcopy(model)
    estimator = make_estimator(method, model, x=x)
    fit_estimator(method, estimator, loader, valla={"iterations": 2})
    restored = make_estimator(method, restored_model, x=x)
    assert restored_model.training
    restored.load_state_dict(estimator.state_dict())
    assert not restored_model.training
    torch.testing.assert_close(
        restored.predictive_moments(x[:2])[0], estimator.predictive_moments(x[:2])[0]
    )


def test_custom_batch_sampler_keeps_subset_batches(method):
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=2, drop_last=False
        ),
    )
    estimator = make_estimator(
        method,
        torch.nn.Linear(2, 2),
        valla={"inducing_locations": "random", "num_inducing": 2},
    )
    fit_estimator(method, estimator, loader)
    assert estimator.predictive_moments(inputs[:2])[0].shape == (2, 2)


def test_backpack_prediction_under_no_grad(method, data):
    x, loader, model = data
    estimator = make_estimator(method, model, x=x, backend=BackPackGGN)
    fit_estimator(
        method,
        estimator,
        loader,
        valla={"lr": 0.001, "val_loader": loader, "val_steps": 1},
    )
    with torch.no_grad():
        mean, covariance = estimator.predictive_moments(x[:2])
    assert mean.shape == (2, 2)
    assert covariance.shape == (2, 2, 2)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_fit_preserves_pretrained_gradient_buffers(method, data, backend):
    x, loader, model = data
    before = []
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = None if index == 0 else torch.full_like(parameter, 0.123)
        before.append(None if parameter.grad is None else parameter.grad.clone())
    estimator = make_estimator(method, model, x=x, backend=backend)
    fit_estimator(method, estimator, loader)
    for parameter, old_gradient in zip(model.parameters(), before):
        if old_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, old_gradient)


def test_functional_methods_reject_empty_queries(method, data):
    x, loader, model = data
    estimator = make_estimator(method, model, x=x)
    fit_estimator(method, estimator, loader)
    for joint in (False, True):
        with pytest.raises(ValueError, match="at least one input"):
            estimator.predictive_moments(x[:0], joint=joint)
        with pytest.raises(ValueError, match="at least one input"):
            estimator.functional_samples(x[:0], n_samples=2, joint=joint)


@requires_backpack
def test_functional_methods_reject_frozen_parameters_with_backpack_subclass(method):
    class AlternateGGN(BackPackGGN):
        pass

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="BackPACK and Asdfghjkl"):
        make_estimator(
            method,
            model,
            valla={"inducing_locations": torch.zeros(2, 2)},
            backend=AlternateGGN,
        )


@pytest.mark.parametrize("val_steps", [0, -1])
def test_functional_methods_reject_invalid_val_steps(method, data, val_steps):
    x, loader, model = data
    estimator = make_estimator(method, model, x=x)
    with pytest.raises(ValueError, match="val_steps must be positive"):
        fit_estimator(method, estimator, loader, val_steps=val_steps)


def test_mapping_backend_limit(method):
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_estimator(
        method,
        RewardModel(),
        BradleyTerryLikelihood(),
        valla={"inducing_locations": {"input_ids": pairs[:, 0][:2].clone()}},
        backend=BackPackGGN,
    )
    with pytest.raises(ValueError, match="mapping-style"):
        fit_estimator(method, estimator, loader, valla={"lr": 0.001})


def test_mapping_reward_asdl_backend(method):
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_estimator(
        method,
        RewardModel(),
        BradleyTerryLikelihood(),
        valla={"inducing_locations": {"input_ids": pairs[:, 0][:2].clone()}},
        backend=AsdlGGN,
    )
    fit_estimator(method, estimator, loader, valla={"lr": 0.001})
    mean, covariance = estimator.predictive_moments({"input_ids": pairs[:2, 0]})
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)


def test_rank_limited_joint_sampling_tolerates_float32_roundoff(method):
    train = torch.ones(2, 1)
    loader = DataLoader(TensorDataset(train, train), batch_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    estimator = make_estimator(
        method,
        model,
        GaussianLikelihood(),
        ella={"subsample_size": 2, "n_eigenvalues": 1},
        valla={"inducing_locations": train[:1].clone()},
    )
    fit_estimator(method, estimator, loader)
    query = torch.tensor([[15409.96], [-2934.29], [-21787.89], [5684.31], [-10845.22]])
    samples = estimator.functional_samples(query, joint=True, n_samples=3)
    assert samples.shape == (3, 5, 1)
    assert torch.isfinite(samples).all()


def test_regression_joint_sampling_and_serialization(method, data):
    x, _, model = data
    regression_model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    restored_model = deepcopy(regression_model)
    loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
    options = {"sigma_noise": 0.5, "temperature": 2.0}
    estimator = make_estimator(
        method, regression_model, GaussianLikelihood(), x=x, **options
    )
    fit_estimator(method, estimator, loader, valla={"iterations": 2, "lr": 0.001})
    mean, covariance = estimator(x[:2])
    joint_mean, joint_covariance = estimator(x[:2], joint=True)
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
    assert joint_mean.shape == (2,)
    assert joint_covariance.shape == (2, 2)
    torch.testing.assert_close(joint_covariance.diag(), covariance[:, 0, 0])
    assert estimator.functional_samples(x[:2], n_samples=3).shape == (3, 2, 1)
    load_options = {"sigma_noise": 0.5}
    restored = make_estimator(
        method, restored_model, GaussianLikelihood(), x=x, **load_options
    )
    restored.load_state_dict(state_dict=estimator.state_dict())
    assert restored.temperature == 2.0
    loaded_mean, loaded_covariance = restored(x[:2])
    torch.testing.assert_close(loaded_mean, mean)
    torch.testing.assert_close(loaded_covariance, covariance)


def test_reward_modeling_recognizes_backpack_subclasses(method):
    class AlternateGGN(BackPackGGN):
        pass

    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_estimator(
        method,
        RewardModel(),
        BradleyTerryLikelihood(),
        valla={"inducing_locations": pairs[:, 0][:2].clone()},
        backend=AlternateGGN,
    )
    fit_estimator(method, estimator, loader)
    assert torch.isfinite(estimator.predictive_moments(pairs[:2, 0])[1]).all()


def test_tensor_reward_backpack_backend(method):
    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_estimator(
        method,
        RewardModel(),
        BradleyTerryLikelihood(),
        valla={"inducing_locations": pairs[:, 0][:2].clone()},
        backend=BackPackGGN,
    )
    fit_estimator(method, estimator, loader, valla={"lr": 0.001})
    mean, covariance = estimator.predictive_moments(pairs[:2, 0])
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
