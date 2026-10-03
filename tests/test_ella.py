"""CPU contracts for the ELLA estimator."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
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


def make_ella(model, likelihood="classification", **kwargs):
    options = {"subsample_size": 2, "n_eigenvalues": 1}
    options.update(kwargs)
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="nystrom",
        **options,
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
    estimator = make_ella(model)
    with pytest.raises(ValueError, match="prior_mean=0"):
        estimator.prior_mean = 1.0
    torch.testing.assert_close(
        estimator.prior_mean, torch.zeros_like(estimator.prior_mean)
    )


def test_classification_noise_remains_one(data):
    x, _, model = data
    estimator = make_ella(model)
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
    constructor = make_ella
    options = {}
    with pytest.raises(ValueError, match=option):
        constructor(model, "regression", **options, **{option: value})
    estimator = constructor(model, "regression", **options)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, value)


def test_ella_temperature_update_refreshes_posterior(data):
    x, _, _ = data
    targets = x.sum(dim=-1, keepdim=True)
    loader = DataLoader(TensorDataset(x, targets), batch_size=2)
    estimator = make_ella(torch.nn.Linear(2, 1), "regression")
    estimator.fit(loader)
    before = estimator.predictive_moments(x[:1])[1]
    estimator.temperature = 2.0
    after = estimator.predictive_moments(x[:1])[1]
    assert torch.all(after > before)
    with pytest.raises(ValueError, match="temperature must be positive"):
        estimator.temperature = 0.0


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
        estimator = make_ella(deepcopy(model))
        estimator.fit(loader, val_loader=loader)
        histories.append((estimator.fit_history_["val_nll"],))
    for first, second in zip(histories[0], histories[1]):
        torch.testing.assert_close(torch.tensor(first), torch.tensor(second))


def test_classification_fit_predict_samples_and_model_preservation(data):
    x, loader, model = data
    before = {key: value.clone() for key, value in model.state_dict().items()}
    estimator = make_ella(model)
    with pytest.raises(RuntimeError, match="fit"):
        estimator.predictive_moments(x[:2])
    with pytest.raises(RuntimeError, match="fit"):
        estimator.state_dict()
    result = estimator.fit(loader)
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
    jacobians, _ = estimator.backend.jacobians(x[:2])
    torch.testing.assert_close(estimator.functional_variance(jacobians), covariance)
    torch.testing.assert_close(
        estimator.functional_covariance(jacobians),
        estimator.predictive_moments(x[:2], joint=True)[1],
    )
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
    estimator = make_ella(regression_model, "regression", **options)
    estimator.fit(loader)
    mean, covariance = estimator(x[:2])
    joint_mean, joint_covariance = estimator(x[:2], joint=True)
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
    assert joint_mean.shape == (2,)
    assert joint_covariance.shape == (2, 2)
    torch.testing.assert_close(joint_covariance.diag(), covariance[:, 0, 0])
    assert estimator.functional_samples(x[:2], n_samples=3).shape == (3, 2, 1)
    load_options = {"sigma_noise": 0.5}
    restored = make_ella(restored_model, "regression", **load_options)
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
    estimator = make_ella(model)
    estimator.fit(loader)
    restored = make_ella(restored_model)
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
    estimator = make_ella(model, backend=backend)
    estimator.fit(loader)
    for parameter, old_gradient in zip(model.parameters(), before):
        if old_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, old_gradient)


def test_backpack_prediction_preserves_pretrained_gradient_buffers(data):
    x, loader, model = data
    estimator = make_ella(model, backend=BackPackGGN)
    estimator.fit(loader)
    predictions = (
        lambda: estimator.predictive_moments(x[:1]),
        lambda: estimator(x[:1]),
        lambda: estimator.functional_samples(x[:1], n_samples=2),
        lambda: estimator.predictive_samples(x[:1], n_samples=2),
    )
    for predict in predictions:
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 7)
        predict()
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7))


def test_ella_asdl_reverse_mode_fallback_retains_input_gradients():

    class Square(torch.autograd.Function):
        @staticmethod
        def forward(value):
            return value.square()

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])

        @staticmethod
        def backward(ctx, gradient):
            return 2 * ctx.saved_tensors[0] * gradient

    class SquareModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2)

        def forward(self, inputs):
            return Square.apply(self.linear(inputs))

    inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    model = SquareModel()
    targets = model(inputs).detach()
    estimator = make_ella(model, "regression", backend=AsdlGGN, enable_backprop=True)
    estimator.fit(DataLoader(TensorDataset(inputs, targets), batch_size=2))
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7)
    query = inputs[:1].clone().requires_grad_()
    _, covariance = estimator.predictive_moments(query)
    assert covariance.requires_grad
    assert torch.isfinite(torch.autograd.grad(covariance.sum(), query)[0]).all()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7))


def test_ella_noise_and_validation_grid(data):
    x, _, model = data
    loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
    estimator = make_ella(
        model=torch.nn.Sequential(torch.nn.Linear(2, 1)), likelihood="regression"
    )
    estimator.fit(loader, val_loader=loader)
    low = estimator.predictive_moments(x)[1]
    estimator.sigma_noise = 2.0
    high = estimator.predictive_moments(x)[1]
    assert torch.all(
        high.diagonal(dim1=-2, dim2=-1) >= low.diagonal(dim1=-2, dim2=-1) - 1e-06
    )
    estimator.optimize_prior_precision(
        "gp", method="gridsearch", val_loader=loader, grid_size=3
    )
    assert len(estimator.fit_history_["tuning"]) == 3
    assert torch.isfinite(estimator.prior_precision).all()
    with pytest.raises(NotImplementedError):
        estimator.optimize_prior_precision(method="marglik", val_loader=loader)


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
    estimator = make_ella(model, subsample_size=2, n_eigenvalues=1)
    estimator.fit(train, val_loader=validation)
    assert torch.all(estimator(inputs)[:, 1] == 0)
    mean, covariance = estimator.predictive_moments(inputs)
    scaled = mean / torch.sqrt(1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1))
    expected = torch.nn.functional.cross_entropy(scaled, validation_targets)
    observed = estimator._validation_nll(validation)
    assert torch.isfinite(observed)
    torch.testing.assert_close(observed, expected)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["val_nll"])).all()
    estimator.optimize_hyperparameters(validation, [1.0, 2.0])
    assert len(estimator.fit_history_["tuning"]) == 2
    assert all(
        (
            torch.isfinite(torch.tensor(item["val_nll"]))
            for item in estimator.fit_history_["tuning"]
        )
    )


def test_ella_prior_grid_distinguishes_empty_and_nonfinite_scores(data, monkeypatch):
    _, loader, model = data
    estimator = make_ella(model)
    estimator.fit(loader)
    initial_prior = estimator.prior_precision.clone()
    initial_noise = estimator.sigma_noise.clone()
    with pytest.raises(ValueError, match="grid is empty"):
        estimator.optimize_hyperparameters(loader, [])
    monkeypatch.setattr(
        estimator, "_validation_nll", lambda _: torch.tensor(float("inf"))
    )
    with pytest.raises(ValueError, match="No finite validation NLL"):
        estimator.optimize_hyperparameters(loader, [2.0, 3.0])
    torch.testing.assert_close(estimator.prior_precision, initial_prior)
    torch.testing.assert_close(estimator.sigma_noise, initial_noise)


def test_periodic_validation_across_training_batches(data):
    inputs, loader, model = data
    estimator = make_ella(model)
    estimator.fit(loader, val_loader=loader, val_steps=1)
    assert len(estimator.fit_history_["val_nll"]) == len(loader)
    assert torch.isfinite(torch.tensor(estimator.fit_history_["val_nll"])).all()


def test_ella_joint_prior_noise_grid_selects_best_candidate(data):
    inputs, _, _ = data
    loader = DataLoader(
        TensorDataset(inputs, inputs.sum(-1, keepdim=True)), batch_size=2
    )
    estimator = make_ella(torch.nn.Linear(2, 1), "regression")
    estimator.fit(loader)
    estimator.optimize_hyperparameters(loader, [(0.1, 0.5), (10.0, 2.0)])
    scores = estimator.fit_history_["tuning"]
    assert len(scores) == 2
    selected = min(scores, key=lambda item: item["val_nll"])
    assert float(estimator.prior_precision) == pytest.approx(
        selected["prior_precision"]
    )
    assert float(estimator.sigma_noise) == pytest.approx(selected["sigma_noise"])


def test_ella_rejects_rank_deficient_nystrom_subset():
    inputs = torch.ones(4, 1)
    loader = DataLoader(TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=2)
    estimator = make_ella(
        torch.nn.Linear(1, 1, bias=False),
        "regression",
        subsample_size=2,
        n_eigenvalues=2,
    )
    with pytest.raises(ValueError, match="insufficient positive rank"):
        estimator.fit(loader)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_ella_accepts_small_full_rank_subset_kernel(backend):

    class ScaledLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)

        def forward(self, inputs):
            return 0.0001 * self.linear(inputs)

    inputs = torch.eye(2)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(2, 1)), batch_size=2)
    estimator = make_ella(
        ScaledLinear(), "regression", subsample_size=2, n_eigenvalues=1, backend=backend
    )
    estimator.fit(loader)
    _, covariance = estimator.predictive_moments(inputs)
    assert torch.isfinite(covariance).all()
    torch.testing.assert_close(
        covariance.diagonal(dim1=-2, dim2=-1).sum(),
        torch.tensor(1e-08),
        rtol=0.0001,
        atol=1e-12,
    )


def test_rank_limited_joint_sampling_tolerates_float32_roundoff():
    train = torch.ones(2, 1)
    loader = DataLoader(TensorDataset(train, train), batch_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    estimator = make_ella(model, "regression", subsample_size=2, n_eigenvalues=1)
    estimator.fit(loader)
    query = torch.tensor([[15409.96], [-2934.29], [-21787.89], [5684.31], [-10845.22]])
    samples = estimator.functional_samples(query, joint=True, n_samples=3)
    assert samples.shape == (3, 5, 1)
    assert torch.isfinite(samples).all()


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
    estimator = make_ella(RewardModel(), "reward_modeling")
    estimator.fit(loader)
    pair_input = {"input_ids": pairs[:2]} if mapping else pairs[:2]
    single_input = {"input_ids": pairs[:2, 0]} if mapping else pairs[:2, 0]
    assert estimator(pair_input, fitting=True).shape == (2, 2)
    mean, covariance = estimator(single_input)
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)
    assert estimator.predictive_samples(single_input, n_samples=2).shape == (2, 2, 1)


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
def test_ella_curvature_backends(data, backend):
    x, loader, model = data
    estimator = make_ella(model, backend=backend)
    estimator.fit(loader)
    assert torch.isfinite(estimator.predictive_moments(x[:2])[1]).all()


def test_checkpoint_requires_same_pretrained_model(data):
    x, loader, model = data
    estimator = make_ella(model)
    estimator.fit(loader)
    changed_model = deepcopy(model)
    with torch.no_grad():
        next(changed_model.parameters()).add_(1.0)
    restored = make_ella(changed_model)
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
    estimator = make_ella(model)
    estimator.fit(loader)
    restored = make_ella(other_model)
    with pytest.raises(ValueError, match="trainable coordinates"):
        restored.load_state_dict(estimator.state_dict())


def test_checkpoint_rejects_different_parameterless_module(data):
    x, loader, _ = data
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    )
    other_model = deepcopy(model)
    other_model[1] = torch.nn.Tanh()
    estimator = make_ella(model)
    estimator.fit(loader)
    restored = make_ella(other_model)
    with pytest.raises(ValueError, match="same pretrained model"):
        restored.load_state_dict(estimator.state_dict())


def test_ella_backpack_checkpoint_into_fresh_model(data):
    x, loader, model = data
    fresh_model = deepcopy(model)
    estimator = make_ella(model, backend=BackPackGGN)
    estimator.fit(loader)
    restored = make_ella(fresh_model, backend=BackPackGGN)
    restored.load_state_dict(estimator.state_dict())
    observed = restored.predictive_moments(x[:2])
    expected = estimator.predictive_moments(x[:2])
    torch.testing.assert_close(observed[0], expected[0])
    torch.testing.assert_close(observed[1], expected[1])


def test_mapping_backend_limit():
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_ella(RewardModel(), "reward_modeling", backend=BackPackGGN)
    with pytest.raises(ValueError, match="mapping-style"):
        estimator.fit(loader)


def test_mapping_reward_asdl_backend():
    pairs = torch.randn(4, 2, 2)
    dataset = [
        {"input_ids": pair, "labels": torch.tensor(index % 2)}
        for index, pair in enumerate(pairs)
    ]
    loader = DataLoader(dataset, batch_size=2)
    estimator = make_ella(RewardModel(), "reward_modeling", backend=AsdlGGN)
    estimator.fit(loader)
    mean, covariance = estimator.predictive_moments({"input_ids": pairs[:2, 0]})
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)


def test_tensor_reward_backpack_backend():
    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_ella(RewardModel(), "reward_modeling", backend=BackPackGGN)
    estimator.fit(loader)
    mean, covariance = estimator.predictive_moments(pairs[:2, 0])
    assert mean.shape == (2, 1)
    assert covariance.shape == (2, 1, 1)


def test_backpack_prediction_under_no_grad(data):
    x, loader, model = data
    estimator = make_ella(model, backend=BackPackGGN)
    estimator.fit(loader)
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
    options = {"subsample_size": 2, "n_eigenvalues": 1}
    vector = Laplace(
        original,
        "regression",
        "all",
        "gp",
        functional_approximation="nystrom",
        **options,
    )
    matrix = Laplace(
        model_copy,
        "regression",
        "all",
        "gp",
        functional_approximation="nystrom",
        **options,
    )
    vector.fit(vector_loader, val_loader=vector_loader)
    matrix.fit(matrix_loader, val_loader=matrix_loader)
    torch.testing.assert_close(
        torch.tensor(vector.fit_history_["val_nll"]),
        torch.tensor(matrix.fit_history_["val_nll"]),
    )


def test_ella_balanced_subset_includes_rare_class():
    torch.manual_seed(32)
    inputs = torch.randn(10, 2)
    labels = torch.tensor([0] * 9 + [1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    estimator = make_ella(
        torch.nn.Linear(2, 2), subsample_size=4, n_eigenvalues=2, seed=32
    )
    selected = estimator._indices(loader, balanced=True)
    assert torch.bincount(labels[selected], minlength=2).tolist() == [3, 1]
    estimator.fit(loader, balanced=True)
    assert estimator.fit_history_["processed_examples"][-1] == 10


def test_inducing_and_basis_respect_training_sampler():
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]]
    )
    y = x.sum(-1, keepdim=True)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=1, sampler=SubsetRandomSampler([0, 1])
    )
    estimator = make_ella(torch.nn.Linear(2, 1), "regression")
    assert set(estimator._indices(loader, balanced=False).tolist()) == {0, 1}
    estimator.fit(loader)
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
    estimator = make_ella(torch.nn.Linear(2, 2))
    estimator.fit(loader)
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
    estimator = make_ella(torch.nn.Linear(2, 1), "regression")
    estimator.fit(loader)
    assert estimator.n_data == 4
    assert torch.all(estimator._indices(loader, balanced=False) < 4)


def test_ella_mc_link_handles_singular_covariance_and_generator(data):
    x, loader, model = data
    estimator = make_ella(model)
    estimator.fit(loader)
    first = estimator(
        x[:2],
        link_approx="mc",
        n_samples=32,
        generator=torch.Generator().manual_seed(4),
    )
    second = estimator(
        x[:2],
        link_approx="mc",
        n_samples=32,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sum(-1), torch.ones(2))


def test_joint_functional_samples_follow_joint_covariance():
    torch.manual_seed(4)
    train_x = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    loader = DataLoader(TensorDataset(train_x, torch.zeros(2, 1)), batch_size=1)
    estimator = Laplace(
        torch.nn.Linear(2, 1, bias=False),
        "regression",
        "all",
        "gp",
        functional_approximation="nystrom",
        subsample_size=2,
        n_eigenvalues=2,
    )
    estimator.fit(loader)
    query = torch.eye(2)
    _, covariance = estimator.predictive_moments(query, joint=True)
    samples = estimator.functional_samples(
        query, n_samples=4000, joint=True, generator=torch.Generator().manual_seed(5)
    )[:, :, 0]
    torch.testing.assert_close(torch.cov(samples.T), covariance, atol=0.04, rtol=0.1)


def test_functional_methods_reject_empty_queries(data):
    x, loader, model = data
    estimator = make_ella(model)
    estimator.fit(loader)
    for joint in (False, True):
        with pytest.raises(ValueError, match="at least one input"):
            estimator.predictive_moments(x[:0], joint=joint)
        with pytest.raises(ValueError, match="at least one input"):
            estimator.functional_samples(x[:0], n_samples=2, joint=joint)


@pytest.mark.parametrize("val_steps", [0, -1])
def test_functional_methods_reject_invalid_val_steps(data, val_steps):
    x, loader, model = data
    estimator = make_ella(model)
    with pytest.raises(ValueError, match="val_steps must be positive"):
        estimator.fit(loader, val_steps=val_steps)


def test_functional_methods_reject_frozen_parameters_with_backpack_subclass():

    class AlternateGGN(BackPackGGN):
        pass

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="BackPACK and Asdfghjkl"):
        make_ella(model, backend=AlternateGGN)


def test_reward_modeling_recognizes_backpack_subclasses():

    class AlternateGGN(BackPackGGN):
        pass

    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    estimator = make_ella(RewardModel(), "reward_modeling", backend=AlternateGGN)
    estimator.fit(loader)
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
    estimator = make_ella(model, "regression", **options)
    estimator.fit(loader)
    restored = make_ella(original, "regression")
    with pytest.raises(ValueError, match="dict_key_x"):
        restored.load_state_dict(estimator.state_dict())


@pytest.mark.parametrize("option", ["prior_precision", "sigma_noise"])
def test_functional_hyperparameters_remain_finite_in_model_dtype(option):
    model = torch.nn.Linear(1, 1).float()
    options = {"subsample_size": 1, "n_eigenvalues": 1}
    constructor = make_ella
    overflow = torch.tensor(1e50, dtype=torch.float64)
    with pytest.raises(ValueError, match=option):
        constructor(model, "regression", **options, **{option: overflow})
    estimator = constructor(model, "regression", **options)
    with pytest.raises(ValueError, match=option):
        setattr(estimator, option, overflow)


@pytest.mark.parametrize("drop_last", [False, True])
def test_ella_replays_partial_random_sampler_epoch(drop_last):

    class CountingRandomSampler(RandomSampler):
        def __iter__(self):
            self.calls += 1
            yield from super().__iter__()

    row_ids = torch.arange(1, 13, dtype=torch.float32)
    inputs = torch.stack([row_ids, row_ids.square()], dim=-1)
    dataset = TensorDataset(inputs, inputs.sum(dim=-1, keepdim=True))
    sampler = CountingRandomSampler(
        dataset,
        replacement=False,
        num_samples=5 if drop_last else 4,
        generator=torch.Generator().manual_seed(12),
    )
    sampler.calls = 0
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=drop_last)
    estimator = make_ella(
        torch.nn.Linear(2, 1), "regression", subsample_size=2, n_eigenvalues=2
    )
    basis_indices = []
    fitted_indices = []
    original_build_basis = estimator._build_basis
    original_features = estimator._features

    def build_basis(source, indices):
        basis_indices.extend(indices.tolist())
        return original_build_basis(source, indices)

    def features(x):
        fitted_indices.extend((x[:, 0] - 1).long().tolist())
        return original_features(x)

    estimator._build_basis = build_basis
    estimator._features = features
    estimator.fit(loader)
    assert sampler.calls == 1
    assert estimator.n_data == len(fitted_indices) == 4
    assert set(basis_indices) <= set(fitted_indices)


def test_seeded_ella_map_mean_reference():
    """Check a seeded ELLA MAP mean against fixed expected logits."""
    torch.set_num_threads(1)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 1, 0])
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.2], [-0.1, 0.3]]))
        model.bias.copy_(torch.tensor([0.2, -0.3]))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    estimator = make_ella(
        model, subsample_size=4, n_eigenvalues=1, seed=11, backend=BackPackGGN
    )
    assert estimator.fit(loader) is None
    mean, _ = estimator.predictive_moments(inputs[:2])
    expected = torch.tensor([[0.2, -0.3], [0.6, -0.4]])
    torch.testing.assert_close(mean, expected)
