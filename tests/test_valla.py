"""CPU contracts for the VaLLA estimator."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    SequentialSampler,
    SubsetRandomSampler,
    TensorDataset,
)

from laplace import Laplace
from laplace.curvature import AsdlGGN, BackPackGGN, CurvlinopsGGN

torch.set_num_threads(1)


@pytest.fixture
def data():
    torch.manual_seed(7)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    return (x, DataLoader(TensorDataset(x, y), batch_size=2), model)


def make_valla(model, likelihood="classification", **kwargs):
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        **kwargs,
    )


class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["input_ids"]
        output = self.net(x)
        return output.squeeze(-1) if x.ndim == 3 else output


def test_functional_prior_mean_remains_zero(data):
    x, _, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="prior_mean=0"):
        estimator.prior_mean = 1.0
    torch.testing.assert_close(
        estimator.prior_mean, torch.zeros_like(estimator.prior_mean)
    )


def test_classification_noise_remains_one(data):
    x, _, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="only available for regression"):
        estimator.sigma_noise = 2.0
    torch.testing.assert_close(
        estimator.sigma_noise, torch.ones_like(estimator.sigma_noise)
    )


@pytest.mark.parametrize("option", ["prior_precision", "sigma_noise", "temperature"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_functional_hyperparameters_must_be_finite(option, value, data):
    x, _, _ = data
    model = torch.nn.Linear(2, 1)
    constructor = make_valla
    options = {"inducing_locations": x[:2].clone()}
    with pytest.raises(ValueError, match=option):
        constructor(model, "regression", **options, **{option: value})
    estimator = constructor(model, "regression", **options)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, value)


def test_valla_alpha_requires_mc_for_classification(data):
    x, _, model = data
    with pytest.raises(ValueError, match="requires mc_softmax_samples"):
        make_valla(model, inducing_locations=x[:2].clone(), alpha=0.5)


def test_valla_fit_validates_mutated_options(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.alpha = 2.0
    with pytest.raises(ValueError, match="alpha must be"):
        estimator.fit(loader, iterations=1)
    estimator.alpha = 1.0
    estimator.mc_softmax_samples = -1
    with pytest.raises(ValueError, match="mc_softmax_samples"):
        estimator.fit(loader, iterations=1)
    estimator.mc_softmax_samples = 0
    with pytest.raises(ValueError, match="temperature must be positive"):
        estimator.temperature = -1.0
    estimator.fit(loader, iterations=1)
    estimator.inducing_classes = torch.tensor([0])
    with pytest.raises(ValueError, match="inducing_classes"):
        estimator.fit(loader, iterations=1, override=False)


@pytest.mark.parametrize("strategy", ["fixed", "random"])
def test_valla_refit_override_and_warm_start(data, strategy):
    x, loader, model = data
    estimator = make_valla(
        model,
        inducing_locations=x[:2].clone() if strategy == "fixed" else "random",
        **{} if strategy == "fixed" else {"num_inducing": 2},
    )
    estimator.fit(loader, iterations=1, lr=0.1)
    first_locations = estimator.inducing_locations.detach().clone()
    estimator.fit(loader, iterations=1, lr=0.1, override=False)
    assert len(estimator.fit_history_["objective"]) == 2
    estimator.fit(loader, iterations=1, lr=1e-10, override=True)
    assert len(estimator.fit_history_["objective"]) == 1
    torch.testing.assert_close(
        estimator.L.detach(), torch.tensor([1.0, 0.0, 1.0]), atol=1e-06, rtol=0
    )
    assert not torch.allclose(estimator.inducing_locations.detach(), first_locations)


def test_valla_checkpoint_preserves_refit_initialization(data):
    x, loader, model = data
    pristine_model = deepcopy(model)
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1, lr=0.1)
    restored = make_valla(pristine_model, inducing_locations=torch.zeros_like(x[:2]))
    restored.load_state_dict(state_dict=estimator.state_dict())
    restored.fit(loader, iterations=1, lr=1e-10, override=True)
    torch.testing.assert_close(
        restored.inducing_locations.detach(), x[:2], atol=1e-06, rtol=0
    )


@pytest.mark.parametrize("target_format", ["column", "onehot"])
def test_classification_target_shapes_agree(data, target_format):
    x, _, model = data
    flat = torch.tensor([0, 1, 1, 0])
    alternative = (
        flat[:, None]
        if target_format == "column"
        else torch.nn.functional.one_hot(flat, num_classes=2)
    )
    histories = []
    for targets in (flat, alternative):
        loader = DataLoader(TensorDataset(x, targets), batch_size=2)
        estimator = make_valla(deepcopy(model), inducing_locations=x[:2].clone())
        estimator.fit(loader, iterations=1, val_loader=loader)
        histories.append(
            (estimator.fit_history_["objective"], estimator.fit_history_["val_nll"])
        )
    for first, second in zip(histories[0], histories[1]):
        torch.testing.assert_close(torch.tensor(first), torch.tensor(second))


def test_valla_functional_covariance_requires_fit_and_raw_jacobians(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    jacobians = torch.zeros(2, 2, estimator.n_params)
    with pytest.raises(RuntimeError, match="fit"):
        estimator.functional_variance(jacobians)
    with pytest.raises(RuntimeError, match="fit"):
        estimator.functional_covariance(jacobians)
    estimator.fit(loader, iterations=1)
    with pytest.raises(ValueError, match="Invalid Jacobians shape"):
        estimator.functional_variance(jacobians[..., :-1])


def test_classification_fit_predict_samples_and_model_preservation(data):
    x, loader, model = data
    before = {key: value.clone() for key, value in model.state_dict().items()}
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    with pytest.raises(RuntimeError, match="fit"):
        estimator.predictive_moments(x[:2])
    with pytest.raises(RuntimeError, match="fit"):
        estimator.state_dict()
    result = estimator.fit(loader, iterations=2, lr=0.001)
    assert result is None
    assert estimator.fit_history_
    mean, covariance = estimator.predictive_moments(x[:2])
    probability = estimator(x[:2])
    torch.testing.assert_close(
        estimator.predictive(x[:2], "gp", "probit", 10), probability
    )
    assert mean.shape == (2, 2)
    assert covariance.shape == (2, 2, 2)
    assert probability.shape == (2, 2)
    torch.testing.assert_close(probability.sum(-1), torch.ones(2))
    assert estimator.functional_samples(x[:2], n_samples=3).shape == (3, 2, 2)
    assert estimator.predictive_samples(x[:2], n_samples=3).shape == (3, 2, 2)
    with pytest.raises(ValueError, match="positive"):
        estimator.functional_samples(x[:2], n_samples=0)
    with pytest.raises(NotImplementedError):
        estimator.log_marginal_likelihood()
    with pytest.raises(NotImplementedError):
        _ = estimator.log_likelihood
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    assert list(model.named_parameters())


def test_regression_joint_sampling_and_serialization(data):
    x, _, model = data
    regression_model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    restored_model = deepcopy(regression_model)
    loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
    options = {"sigma_noise": 0.5, "temperature": 2.0}
    estimator = make_valla(
        regression_model, "regression", inducing_locations=x[:2].clone(), **options
    )
    estimator.fit(loader, iterations=2, lr=0.001)
    mean, covariance = estimator(x[:2])
    joint_mean, joint_covariance = estimator(x[:2], joint=True)
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
    assert joint_mean.shape == (2,)
    assert joint_covariance.shape == (2, 2)
    torch.testing.assert_close(joint_covariance.diag(), covariance[:, 0, 0])
    assert estimator.functional_samples(x[:2], n_samples=3).shape == (3, 2, 1)
    load_options = {"sigma_noise": 0.5}
    restored = make_valla(
        restored_model, "regression", inducing_locations=x[:2].clone(), **load_options
    )
    restored.load_state_dict(state_dict=estimator.state_dict())
    assert restored.temperature == 2.0
    loaded_mean, loaded_covariance = restored(x[:2])
    torch.testing.assert_close(loaded_mean, mean)
    torch.testing.assert_close(loaded_covariance, covariance)


def test_checkpoint_restores_evaluation_mode(data):
    x, loader, _ = data
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Dropout(0.5), torch.nn.Linear(3, 2)
    )
    restored_model = deepcopy(model)
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=2)
    restored = make_valla(restored_model, inducing_locations=x[:2].clone())
    assert restored_model.training
    restored.load_state_dict(estimator.state_dict())
    assert not restored_model.training
    torch.testing.assert_close(
        restored.predictive_moments(x[:2])[0], estimator.predictive_moments(x[:2])[0]
    )


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_fit_preserves_pretrained_gradient_buffers(data, backend):
    x, loader, model = data
    before = []
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = None if index == 0 else torch.full_like(parameter, 0.123)
        before.append(None if parameter.grad is None else parameter.grad.clone())
    estimator = make_valla(model, inducing_locations=x[:2].clone(), backend=backend)
    estimator.fit(loader, iterations=1)
    for parameter, old_gradient in zip(model.parameters(), before):
        if old_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, old_gradient)


def test_backpack_prediction_preserves_pretrained_gradient_buffers(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone(), backend=BackPackGGN)
    estimator.fit(loader, iterations=1)
    predictions = (
        lambda: estimator.predictive_moments(x[:1]),
        lambda: estimator(x[:1]),
        lambda: estimator.functional_samples(x[:1], n_samples=2),
        lambda: estimator.predictive_samples(x[:1], n_samples=2),
    )
    jacobians, _ = estimator.backend.jacobians(x[:1])
    predictions += (
        lambda: estimator.functional_variance(jacobians),
        lambda: estimator.functional_covariance(jacobians),
    )
    for predict in predictions:
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 7)
        predict()
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7))


def test_classification_validation_nll_is_stable_for_saturated_probabilities():
    inputs = torch.ones(4, 1, dtype=torch.float32)
    train = DataLoader(
        TensorDataset(inputs, torch.zeros(4, dtype=torch.long)), batch_size=2
    )
    validation_targets = torch.ones(4, dtype=torch.long)
    validation = DataLoader(TensorDataset(inputs, validation_targets), batch_size=2)
    model = torch.nn.Linear(1, 2).float()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[100.0], [-100.0]], dtype=torch.float32))
        model.bias.zero_()
    estimator = make_valla(model, inducing_locations=inputs[:2].clone())
    estimator.fit(train, iterations=1, val_loader=validation)
    assert torch.all(estimator(inputs)[:, 1] == 0)
    mean, covariance = estimator.predictive_moments(inputs)
    scaled = mean / torch.sqrt(1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1))
    expected = torch.nn.functional.cross_entropy(scaled, validation_targets)
    observed = estimator._validation_nll(validation)
    assert torch.isfinite(observed)
    torch.testing.assert_close(observed, expected)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["val_nll"])).all()


def test_periodic_validation_across_training_batches(data):
    inputs, loader, model = data
    estimator = make_valla(model, inducing_locations=inputs[:2].clone())
    estimator.fit(loader, iterations=3, val_loader=loader, val_steps=2)
    assert len(estimator.fit_history_["objective"]) == 3
    assert len(estimator.fit_history_["val_nll"]) == 2
    assert torch.isfinite(torch.tensor(estimator.fit_history_["val_nll"])).all()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_valla_regression_alpha_energy(data, alpha):
    x, _, _ = data
    model = torch.nn.Linear(2, 1)
    loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
    estimator = make_valla(
        model,
        "regression",
        inducing_locations=x[:2].clone(),
        alpha=alpha,
        sigma_noise=0.5,
    )
    estimator.fit(loader, iterations=1)
    inputs, targets = next(iter(loader))
    mean, covariance, features, _ = estimator._latent_distribution(inputs)
    noise_variance = estimator.sigma_noise.square()
    latent_variance = covariance.diagonal(dim1=-2, dim2=-1)
    if alpha == 0:
        energy = ((targets - mean).square() + latent_variance) / noise_variance
    else:
        energy = (targets - mean).square() / (
            noise_variance + alpha * latent_variance
        ) + torch.log1p(alpha * latent_variance / noise_variance) / alpha
    log_term = -0.5 * (energy + torch.log(2 * torch.pi * noise_variance)).sum()
    hessian = torch.eye(features.shape[0], dtype=features.dtype) + features @ features.T
    kl = 0.5 * (
        torch.linalg.slogdet(hessian).logabsdet
        - torch.trace(torch.linalg.solve(hessian, features @ features.T))
    )
    expected = -(estimator.n_data / len(inputs)) * log_term / estimator.temperature + kl
    torch.testing.assert_close(estimator._objective(inputs, targets), expected)
    assert estimator.state_dict()["alpha"] == alpha


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_valla_classification_mc_energy(data, alpha):
    x, loader, model = data
    estimator = make_valla(
        model, inducing_locations=x[:2].clone(), alpha=alpha, mc_softmax_samples=4
    )
    estimator.fit(loader, iterations=1)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["objective"])).all()
    assert estimator.state_dict()["mc_softmax_samples"] == 4


def test_valla_mc_fit_keeps_finite_parameters_with_zero_covariance():
    inputs = torch.zeros(2, 1)
    targets = torch.tensor([0, 1])
    model = torch.nn.Linear(1, 2, bias=False)
    estimator = make_valla(
        model,
        inducing_locations=inputs[:1].clone(),
        mc_softmax_samples=2,
        backend=CurvlinopsGGN,
    )
    estimator.fit(
        DataLoader(TensorDataset(inputs, targets), batch_size=2), iterations=1
    )
    assert torch.isfinite(estimator.L).all()
    assert torch.isfinite(estimator.prior_precision).all()
    assert torch.isfinite(estimator(inputs[:1], link_approx="mc", n_samples=4)).all()


def test_rank_limited_joint_sampling_tolerates_float32_roundoff():
    train = torch.ones(2, 1)
    loader = DataLoader(TensorDataset(train, train), batch_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    estimator = make_valla(model, "regression", inducing_locations=train[:1].clone())
    estimator.fit(loader, iterations=1)
    query = torch.tensor([[15409.96], [-2934.29], [-21787.89], [5684.31], [-10845.22]])
    samples = estimator.functional_samples(query, joint=True, n_samples=3)
    assert samples.shape == (3, 5, 1)
    assert torch.isfinite(samples).all()


def test_valla_random_inducing_and_prior_fit(data):
    x, _, _ = data
    loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
    estimator = make_valla(
        torch.nn.Sequential(torch.nn.Linear(2, 1)),
        "regression",
        inducing_locations="random",
        num_inducing=2,
        sigma_noise=0.5,
    )
    estimator.fit(loader, iterations=2, lr=0.001)
    assert isinstance(estimator.inducing_locations, torch.nn.Parameter)
    assert len(estimator.fit_history_["objective"]) == 2
    with pytest.raises(NotImplementedError):
        estimator.optimize_prior_precision()


def test_valla_kmeans_initializes_from_floating_training_inputs(data):
    inputs, loader, model = data
    estimator = make_valla(model, inducing_locations="kmeans", num_inducing=2, seed=7)
    estimator.fit(loader, iterations=1, lr=1e-10)
    assert estimator.inducing_locations.shape == (2, 2)
    for location in estimator.inducing_locations:
        assert torch.any(torch.all(torch.isclose(inputs, location, atol=1e-06), dim=-1))


def test_valla_kmeans_rejects_mapping_inputs():
    pairs = torch.randn(4, 2, 2)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(
        [{"input_ids": pair, "labels": label} for pair, label in zip(pairs, labels)],
        batch_size=2,
    )
    estimator = make_valla(
        RewardModel(), "reward_modeling", inducing_locations="kmeans", num_inducing=2
    )
    with pytest.raises(ValueError, match="kmeans requires floating-point tensor"):
        estimator.fit(loader, iterations=1)


def test_valla_kmeans_rejects_integer_inputs():

    class CastLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2).float()

        def forward(self, inputs):
            return self.linear(inputs.float())

    inputs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 1, 0])), batch_size=2)
    estimator = make_valla(CastLinear(), inducing_locations="kmeans", num_inducing=2)
    with pytest.raises(ValueError, match="kmeans requires floating-point tensor"):
        estimator.fit(loader, iterations=1)


@pytest.mark.parametrize("mapping", [False, True])
def test_reward_modeling_pair_fit_single_prediction(mapping):
    torch.manual_seed(10)
    pairs = torch.randn(6, 2, 2)
    labels = torch.randint(0, 2, (6,))
    dataset = (
        [
            {"input_ids": pair, "labels": label, "source": "demo"}
            for pair, label in zip(pairs, labels)
        ]
        if mapping
        else TensorDataset(pairs, labels)
    )
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_valla(
        RewardModel(), "reward_modeling", inducing_locations="random", num_inducing=2
    )
    estimator.fit(loader, iterations=1, lr=0.001)
    if mapping:
        assert estimator.inducing_locations["source"] == ["demo", "demo"]
    pair_input = {"input_ids": pairs[:2]} if mapping else pairs[:2]
    single_input = {"input_ids": pairs[:2, 0]} if mapping else pairs[:2, 0]
    assert estimator(pair_input, fitting=True).shape == (2, 2)
    mean, covariance = estimator(single_input)
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
    assert estimator.predictive_samples(single_input, n_samples=2).shape == (2, 2, 1)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_valla_selected_output_backends(data, backend):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone(), backend=backend)
    estimator.fit(loader, iterations=1, lr=0.001)
    assert torch.isfinite(estimator.predictive_moments(x[:2])[1]).all()


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_selected_output_jacobians_match_autograd(backend):
    torch.manual_seed(31)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)
    )
    model.output_size = 3
    inputs = torch.randn(3, 2)
    selectors = torch.tensor([2, 0, 1])
    interface = backend(model, "classification")
    observed, outputs = interface.selected_output_jacobians(inputs, selectors)
    expected_outputs = model(inputs)
    rows = []
    for index, selected in enumerate(selectors):
        gradients = torch.autograd.grad(
            expected_outputs[index, selected],
            tuple(model.parameters()),
            retain_graph=True,
        )
        rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
    torch.testing.assert_close(outputs, expected_outputs)
    torch.testing.assert_close(observed[:, 0, :], torch.stack(rows))


def test_checkpoint_requires_same_pretrained_model(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1, lr=0.001)
    changed_model = deepcopy(model)
    with torch.no_grad():
        next(changed_model.parameters()).add_(1.0)
    restored = make_valla(changed_model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="same pretrained model"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_rejects_different_trainable_coordinates(data):
    x, loader, _ = data
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    other_model = deepcopy(model)
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    for parameter in other_model[1].parameters():
        parameter.requires_grad_(False)
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1)
    restored = make_valla(other_model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="trainable coordinates"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_rejects_different_parameterless_module(data):
    x, loader, _ = data
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    )
    other_model = deepcopy(model)
    other_model[1] = torch.nn.Tanh()
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1)
    restored = make_valla(other_model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="same pretrained model"):
        restored.load_state_dict(estimator.state_dict())


def test_mapping_backend_limit():
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_valla(
        RewardModel(),
        "reward_modeling",
        backend=BackPackGGN,
        inducing_locations={"input_ids": pairs[:, 0][:2].clone()},
    )
    with pytest.raises(ValueError, match="mapping-style"):
        estimator.fit(loader, iterations=1, lr=0.001)


def test_mapping_reward_asdl_backend():
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_valla(
        RewardModel(),
        "reward_modeling",
        backend=AsdlGGN,
        inducing_locations={"input_ids": pairs[:, 0][:2].clone()},
    )
    estimator.fit(loader, iterations=1, lr=0.001)
    mean, covariance = estimator.predictive_moments({"input_ids": pairs[:2, 0]})
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)


def test_tensor_reward_backpack_backend():
    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_valla(
        RewardModel(),
        "reward_modeling",
        backend=BackPackGGN,
        inducing_locations=pairs[:, 0][:2].clone(),
    )
    estimator.fit(loader, iterations=1, lr=0.001)
    mean, covariance = estimator.predictive_moments(pairs[:2, 0])
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)


def test_backpack_prediction_under_no_grad(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone(), backend=BackPackGGN)
    estimator.fit(loader, iterations=1, lr=0.001, val_loader=loader, val_steps=1)
    with torch.no_grad():
        mean, covariance = estimator.predictive_moments(x[:2])
    assert mean.shape == (2, 2)
    assert covariance.shape == (2, 2, 2)


def test_single_output_regression_vector_targets():
    torch.manual_seed(12)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = x.sum(-1)
    vector_loader = DataLoader(TensorDataset(x, y), batch_size=2)
    matrix_loader = DataLoader(TensorDataset(x, y[:, None]), batch_size=2)
    original = torch.nn.Linear(2, 1)
    model_copy = deepcopy(original)
    options = {"inducing_locations": x[:2].clone()}
    vector = Laplace(
        original,
        "regression",
        "all",
        "gp",
        functional_approximation="variational",
        **options,
    )
    matrix = Laplace(
        model_copy,
        "regression",
        "all",
        "gp",
        functional_approximation="variational",
        **options,
    )
    vector.fit(vector_loader, iterations=1, lr=0.001, val_loader=vector_loader)
    matrix.fit(matrix_loader, iterations=1, lr=0.001, val_loader=matrix_loader)
    torch.testing.assert_close(
        torch.tensor(vector.fit_history_["objective"]),
        torch.tensor(matrix.fit_history_["objective"]),
    )
    torch.testing.assert_close(
        torch.tensor(vector.fit_history_["val_nll"]),
        torch.tensor(matrix.fit_history_["val_nll"]),
    )


def test_valla_mapping_state_is_a_snapshot():
    inputs = {"input_ids": torch.randn(2, 2)}
    pairs = torch.randn(4, 2, 2)
    loader = DataLoader(
        [
            {"input_ids": pair, "labels": torch.tensor(index % 2)}
            for index, pair in enumerate(pairs)
        ],
        batch_size=2,
    )
    estimator = make_valla(RewardModel(), "reward_modeling", inducing_locations=inputs)
    estimator.fit(loader, iterations=1, lr=0.001)
    estimator.fit_history_["objective"] = [1.0]
    state = estimator.state_dict()
    estimator.inducing_locations["input_ids"].add_(10)
    estimator.fit_history_["objective"].append(2.0)
    assert not torch.equal(
        state["inducing_locations"]["input_ids"],
        estimator.inducing_locations["input_ids"],
    )
    assert state["fit_history"]["objective"] == [1.0]


def test_valla_fixed_mapping_accepts_nonleaf_inputs():
    inputs = torch.randn(2, 2, requires_grad=True) * 2
    estimator = make_valla(
        RewardModel(), "reward_modeling", inducing_locations={"input_ids": inputs}
    )
    selected = estimator.inducing_locations["input_ids"]
    torch.testing.assert_close(selected, inputs)
    assert selected.grad_fn is None
    assert not selected.requires_grad
    assert selected.data_ptr() != inputs.data_ptr()


def test_valla_fixed_integer_inducing_is_a_snapshot():
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    estimator = make_valla(torch.nn.Linear(2, 2), inducing_locations=inputs)
    selected = estimator.inducing_locations
    torch.testing.assert_close(selected, inputs)
    assert selected.data_ptr() != inputs.data_ptr()
    inputs.add_(10)
    torch.testing.assert_close(selected, torch.tensor([[1, 2], [3, 4]]))


def test_inducing_and_basis_respect_training_sampler():
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]]
    )
    y = x.sum(-1, keepdim=True)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=1, sampler=SubsetRandomSampler([0, 1])
    )
    estimator = make_valla(
        torch.nn.Linear(2, 1), "regression", inducing_locations="random", num_inducing=2
    )
    estimator.fit(loader, iterations=1, lr=0.001)
    selected = estimator.inducing_locations.detach()
    assert all(
        (
            any((torch.allclose(row, expected, atol=0.01) for expected in x[:2]))
            for row in selected
        )
    )
    assert estimator.n_data == 2


def test_custom_batch_sampler_keeps_subset_batches():
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=2, drop_last=False
        ),
    )
    estimator = make_valla(
        torch.nn.Linear(2, 2), inducing_locations="random", num_inducing=2
    )
    estimator.fit(loader, iterations=1)
    assert estimator.predictive_moments(inputs[:2])[0].shape == (2, 2)


@pytest.mark.parametrize("custom_batch_sampler", [False, True])
def test_deterministic_drop_last_excludes_unseen_rows(custom_batch_sampler):
    inputs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [99.0, 99.0]]
    )
    dataset = TensorDataset(inputs, inputs.sum(dim=-1, keepdim=True))
    loader = (
        DataLoader(
            dataset, batch_sampler=BatchSampler(SequentialSampler(dataset), 2, True)
        )
        if custom_batch_sampler
        else DataLoader(dataset, batch_size=2, drop_last=True)
    )
    estimator = make_valla(
        torch.nn.Linear(2, 1), "regression", inducing_locations="random", num_inducing=2
    )
    estimator.fit(loader, iterations=1)
    assert estimator.n_data == 4
    assert all(
        (
            any((torch.allclose(row, candidate, atol=0.05) for candidate in inputs[:4]))
            for row in estimator.inducing_locations.detach()
        )
    )


@pytest.mark.parametrize("scale", [10000.0, 100000.0])
def test_valla_large_float32_features_keep_positive_variance(scale):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    inducing = torch.tensor([[scale]])
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.zeros(2, 1)), batch_size=2
    )
    estimator = make_valla(model, "regression", inducing_locations=inducing)
    estimator.fit(loader, iterations=1, lr=1e-12)
    query = torch.tensor([[0.99 * scale], [0.98 * scale]])
    _, covariance = estimator.predictive_moments(query)
    expected = query.square() / (1 + inducing.square())
    torch.testing.assert_close(
        covariance[:, 0, 0], expected[:, 0], rtol=0.0001, atol=0.0001
    )
    joint = estimator.predictive_moments(query, joint=True)[1]
    assert torch.linalg.eigvalsh(joint).min() >= -1e-06
    assert torch.isfinite(estimator.functional_samples(query, n_samples=2)).all()


def test_valla_rank_deficient_inducing_features_keep_finite_gradients():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0], [2.0]])
    loader = DataLoader(TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=2)
    estimator = make_valla(model, "regression", inducing_locations=torch.zeros(2, 1))
    estimator.fit(loader, iterations=1)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["objective"])).all()
    assert torch.isfinite(estimator.L).all()
    assert torch.isfinite(estimator.predictive_moments(inputs)[1]).all()


def test_valla_large_duplicate_inducing_features_keep_precision_invertible():
    scale = 100000000.0
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    inducing = torch.full((2, 1), scale)
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.zeros(2, 1)), batch_size=2
    )
    estimator = make_valla(model, "regression", inducing_locations=inducing)
    estimator.fit(loader, iterations=1, lr=1e-12)
    query = torch.tensor([[0.99 * scale]])
    variance = estimator.predictive_moments(query)[1][0, 0, 0]
    expected = query.square() / (1 + 2 * inducing[0].square())
    torch.testing.assert_close(variance, expected.squeeze(), rtol=0.0001, atol=0.0001)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["objective"])).all()


def test_functional_methods_reject_empty_queries(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1)
    for joint in (False, True):
        with pytest.raises(ValueError, match="at least one input"):
            estimator.predictive_moments(x[:0], joint=joint)
        with pytest.raises(ValueError, match="at least one input"):
            estimator.functional_samples(x[:0], n_samples=2, joint=joint)


@pytest.mark.parametrize("val_steps", [0, -1])
def test_functional_methods_reject_invalid_val_steps(data, val_steps):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="val_steps must be positive"):
        estimator.fit(loader, iterations=1, val_steps=val_steps)


def test_functional_methods_reject_frozen_parameters_with_backpack_subclass():

    class AlternateGGN(BackPackGGN):
        pass

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="BackPACK and Asdfghjkl"):
        make_valla(model, inducing_locations=torch.zeros(2, 2), backend=AlternateGGN)


def test_reward_modeling_recognizes_backpack_subclasses():

    class AlternateGGN(BackPackGGN):
        pass

    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_valla(
        RewardModel(),
        "reward_modeling",
        backend=AlternateGGN,
        inducing_locations=pairs[:, 0][:2].clone(),
    )
    estimator.fit(loader, iterations=1)
    assert torch.isfinite(estimator.predictive_moments(pairs[:2, 0])[1]).all()


def test_checkpoint_rejects_different_mapping_key():

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
    estimator = make_valla(
        model,
        "regression",
        inducing_locations={"features": inputs[:2].clone()},
        **options,
    )
    estimator.fit(loader, iterations=1)
    restored = make_valla(
        original, "regression", inducing_locations="random", num_inducing=2
    )
    with pytest.raises(ValueError, match="dict_key_x"):
        restored.load_state_dict(estimator.state_dict())


def test_valla_checkpoint_restores_random_inducing_seed(data):
    x, loader, model = data
    original = deepcopy(model)
    estimator = make_valla(model, inducing_locations="random", num_inducing=2, seed=5)
    estimator.fit(loader, iterations=1, lr=1e-10)
    restored = make_valla(original, inducing_locations="random", num_inducing=2, seed=0)
    restored.load_state_dict(estimator.state_dict())
    assert restored.seed == 5
    torch.testing.assert_close(
        restored.generator.get_state(), estimator.generator.get_state()
    )
    estimator.fit(loader, iterations=1, lr=1e-10, override=True)
    restored.fit(loader, iterations=1, lr=1e-10, override=True)
    torch.testing.assert_close(
        restored.inducing_locations, estimator.inducing_locations
    )


@pytest.mark.parametrize("option", ["prior_precision", "sigma_noise"])
def test_functional_hyperparameters_remain_finite_in_model_dtype(option):
    model = torch.nn.Linear(1, 1).float()
    options = {"inducing_locations": torch.ones(1, 1)}
    constructor = make_valla
    overflow = torch.tensor(1e50, dtype=torch.float64)
    with pytest.raises(ValueError, match=option):
        constructor(model, "regression", **options, **{option: overflow})
    estimator = constructor(model, "regression", **options)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, overflow)


def test_valla_random_mapping_inducing_accepts_row_metadata():

    class MappingDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor([float(index), float(index + 1)]),
                "labels": torch.tensor(index % 2),
                "source": f"row-{index}",
            }

    class MappingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, batch):
            return self.linear(batch["input_ids"])

    loader = DataLoader(MappingDataset(), batch_size=2)
    estimator = make_valla(
        MappingModel(), inducing_locations="random", num_inducing=2, backend=AsdlGGN
    )
    estimator.fit(loader, iterations=1)
    assert len(estimator.inducing_locations["source"]) == 2
    for token, row in zip(
        estimator.inducing_locations["source"],
        estimator.inducing_locations["input_ids"],
    ):
        assert token == f"row-{int(row[0])}"


def test_seeded_valla_regression_reference():
    """Check a seeded VaLLA fit against fixed regression moments and objective."""
    torch.set_num_threads(1)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = inputs.sum(dim=-1, keepdim=True)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.4, -0.2]]))
        model[0].bias.copy_(torch.tensor([0.3]))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    estimator = make_valla(
        model,
        "regression",
        inducing_locations=inputs[:2].clone(),
        sigma_noise=0.5,
        prior_precision=1.0,
        seed=11,
        backend=BackPackGGN,
    )
    assert estimator.fit(loader, iterations=1, lr=1e-12) is None
    mean, covariance = estimator.predictive_moments(inputs[:2])
    torch.testing.assert_close(mean, torch.tensor([[0.3], [0.7]]))
    torch.testing.assert_close(
        covariance, torch.tensor([[[0.4]], [[0.6]]]), rtol=0.0001, atol=1e-05
    )
    assert estimator.fit_history_["objective"] == pytest.approx([3.631515], rel=0.0001)
