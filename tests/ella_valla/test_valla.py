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
from laplace.curvature import CurvlinopsGGN
from laplace.likelihoods import (
    BernoulliLikelihood,
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
)
from laplace.utils.functional_utils import draw_gaussian
from tests.ella_valla.helpers import RewardModel
from tests.utils import AsdlGGN, BackPackGGN


def make_valla(model, likelihood="classification", **kwargs):
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        **kwargs,
    )


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


@pytest.mark.parametrize("likelihood", ["regression", "classification"])
def test_valla_likelihood_object_equivalence(data, likelihood):
    x, loader, model = data
    if likelihood == "regression":
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
        as_object = GaussianLikelihood()
    else:
        as_object = CategoricalLikelihood()

    estimators = []
    for lik in (likelihood, as_object):
        torch.manual_seed(3)
        estimator = make_valla(
            deepcopy(model), lik, inducing_locations=x[:2].clone(), seed=1
        )
        estimator.fit(loader, iterations=2, lr=0.1)
        estimators.append(estimator)

    by_name, by_object = estimators
    mean_name, cov_name = by_name.predictive_moments(x)
    mean_object, cov_object = by_object.predictive_moments(x)

    torch.testing.assert_close(mean_name, mean_object)
    torch.testing.assert_close(cov_name, cov_object)
    torch.testing.assert_close(by_name(x), by_object(x))


def test_valla_reward_likelihood_object_equivalence():
    torch.manual_seed(10)
    pairs = torch.randn(6, 2, 2)
    labels = torch.randint(0, 2, (6,))
    loader = DataLoader(TensorDataset(pairs, labels), batch_size=2)
    model = RewardModel()

    estimators = []
    for lik in ("reward_modeling", BradleyTerryLikelihood()):
        torch.manual_seed(3)
        estimator = make_valla(
            deepcopy(model), lik, inducing_locations=pairs[:2, 0].clone(), seed=1
        )
        estimator.fit(loader, iterations=2, lr=0.1)
        estimators.append(estimator)

    by_name, by_object = estimators
    mean_name, cov_name = by_name(pairs[:2, 0])
    mean_object, cov_object = by_object(pairs[:2, 0])

    torch.testing.assert_close(mean_name, mean_object)
    torch.testing.assert_close(cov_name, cov_object)
    torch.testing.assert_close(
        by_name(pairs[:2], fitting=True), by_object(pairs[:2], fitting=True)
    )


def test_valla_checkpoint_likelihood_is_str(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1, lr=0.1)
    state = estimator.state_dict()

    assert type(state["likelihood"]) is str

    restored = make_valla(
        deepcopy(model), CategoricalLikelihood(), inducing_locations=x[:2].clone()
    )
    restored.load_state_dict(state)

    assert restored.likelihood == "classification"


def test_valla_takes_likelihood_options(data):
    x, _, model = data
    smoothed = CategoricalLikelihood(label_smoothing=0.1)
    estimator = make_valla(model, smoothed, inducing_locations=x[:2].clone())

    assert estimator.likelihood == smoothed

    weighted = CategoricalLikelihood(weight=torch.tensor([1.0, 2.0]), ignore_index=3)
    estimator.likelihood = weighted

    assert estimator.likelihood == weighted


@pytest.mark.parametrize("mc_softmax_samples", [0, 4], ids=["probit", "mc"])
def test_valla_ignored_rows(data, mc_softmax_samples):
    x, _, model = data
    estimator = make_valla(
        model,
        inducing_locations=x[:2].clone(),
        mc_softmax_samples=mc_softmax_samples,
    )
    y = torch.tensor([0, 1, 1, 0])
    estimator.fit(DataLoader(TensorDataset(x, y), batch_size=2), iterations=1)

    def objective(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The same Monte Carlo draws for every call
        estimator.generator = torch.Generator().manual_seed(3)
        return estimator._objective(x, y)

    y_ignored = torch.tensor([-100, 1, -100, 0])
    y_complement = torch.tensor([0, -100, 1, -100])

    # Nothing observed: only the KL term remains
    kl = objective(x, torch.full_like(y, -100))
    assert torch.isfinite(kl)

    # The ignored rows don't count, and whatever their inputs are
    obj_ignored = objective(x, y_ignored)
    obj_complement = objective(x, y_complement)
    x_other = x.clone()
    x_other[[0, 2]] = torch.randn(2, 2)
    torch.testing.assert_close(objective(x_other, y_ignored), obj_ignored)

    # The rows are independent, so the terms of the two halves add up to all rows
    torch.testing.assert_close(
        obj_ignored + obj_complement - kl, objective(x, y), rtol=1e-5, atol=1e-6
    )

    if not mc_softmax_samples:
        # Dropping the ignored rows rescales the minibatch term, with the number of rows
        torch.testing.assert_close(
            (objective(x[[1, 3]], y[[1, 3]]) - kl) * 2, (obj_ignored - kl) * 4
        )


def _ignored_rows(n_data: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and class targets where a third of the rows are ignored."""
    X, y = torch.randn(n_data, 3), torch.randint(2, (n_data,))
    y[::3] = -100

    return X, y


def _valla_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )


def test_valla_validation_nll_leaves_out_ignored_rows():
    X, y = _ignored_rows()
    observed = y != -100

    la = make_valla(_valla_model(), inducing_locations=X[:3])
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4), iterations=1)

    padded = la._validation_nll(DataLoader(TensorDataset(X, y), batch_size=4))
    dropped = la._validation_nll(
        DataLoader(TensorDataset(X[observed], y[observed]), batch_size=4)
    )

    torch.testing.assert_close(padded, dropped)


@pytest.mark.parametrize("strategy", ["random", "kmeans"])
def test_valla_inducing_selection_skips_ignored_rows(strategy):
    X, y = _ignored_rows()

    # Random selection first draws as many rows as it needs, some of them ignored
    la = make_valla(_valla_model(), inducing_locations=strategy, num_inducing=8)
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4), iterations=1)

    assert len(la.inducing_classes) == 8
    assert torch.all(la.inducing_classes >= 0)


@pytest.mark.parametrize("strategy", ["random", "kmeans"])
def test_valla_inducing_selection_needs_enough_observed_rows(strategy):
    X, y = _ignored_rows()

    la = make_valla(_valla_model(), inducing_locations=strategy, num_inducing=9)

    with pytest.raises(ValueError, match="observed"):
        la.fit(DataLoader(TensorDataset(X, y), batch_size=4), iterations=1)


def test_valla_validation_nll_needs_observed_targets():
    X, y = _ignored_rows()

    la = make_valla(_valla_model(), inducing_locations=X[:3])
    la.fit(DataLoader(TensorDataset(X, y), batch_size=4), iterations=1)

    ignored = DataLoader(TensorDataset(X, torch.full_like(y, -100)), batch_size=4)

    with pytest.raises(ValueError, match="observed"):
        la._validation_nll(ignored)


def _fitted_objective(estimator, x, y, seed=3):
    """Objective with the same Monte Carlo draws for every call."""
    estimator.generator = torch.Generator().manual_seed(seed)

    return estimator._objective(x, y)


@pytest.mark.parametrize(
    ("mc_softmax_samples", "alpha"),
    [(0, 1.0), (4, 0.0), (4, 0.5)],
    ids=["probit", "mc", "mc-alpha"],
)
def test_valla_one_hot_probability_targets_match_indices(
    data, mc_softmax_samples, alpha
):
    x, loader, model = data
    estimator = make_valla(
        model,
        inducing_locations=x[:2].clone(),
        mc_softmax_samples=mc_softmax_samples,
        alpha=alpha,
    )
    estimator.fit(loader, iterations=1)

    y = torch.tensor([0, 1, 1, 0])
    one_hot = torch.nn.functional.one_hot(y, num_classes=2).to(x.dtype)

    torch.testing.assert_close(
        _fitted_objective(estimator, x, one_hot),
        _fitted_objective(estimator, x, y),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("mc_softmax_samples", "alpha"), [(0, 1.0), (4, 0.0)], ids=["probit", "mc"]
)
def test_valla_objective_uses_likelihood_options(data, mc_softmax_samples, alpha):
    x, loader, model = data
    weight = torch.tensor([1.0, 3.0])
    likelihood = CategoricalLikelihood(
        weight=weight, label_smoothing=0.1, ignore_index=-1
    )
    estimator = make_valla(
        model,
        likelihood,
        inducing_locations=x[:2].clone(),
        mc_softmax_samples=mc_softmax_samples,
        alpha=alpha,
    )
    estimator.fit(loader, iterations=1)

    # Reference: the data term of the objective from torch, then the shared KL term
    mean, covariance, features, triangular = estimator._latent_distribution(x)
    solved = torch.linalg.solve_triangular(triangular.T, features, upper=False)
    kl = triangular.diagonal().abs().log().sum() - 0.5 * solved.square().sum()

    if mc_softmax_samples:
        generator = torch.Generator().manual_seed(3)
        logits = draw_gaussian(mean, covariance, mc_softmax_samples, generator)
    else:
        scale = torch.sqrt(1 + torch.pi / 8 * covariance.diagonal(dim1=-2, dim2=-1))
        logits = (mean / scale)[None]

    criterion = torch.nn.CrossEntropyLoss(
        weight=weight, ignore_index=-1, label_smoothing=0.1, reduction="none"
    )
    indices = torch.tensor([0, -1, 1, 1])
    probabilities = torch.tensor([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.5, 0.5]])

    # Index targets with an ignored row, and soft class probabilities
    for y in (indices, probabilities):
        log_term = -torch.stack([criterion(f, y) for f in logits]).mean(0).sum()
        expected = (
            -(estimator.n_data / y.shape[0]) * log_term / estimator.temperature + kl
        )

        torch.testing.assert_close(_fitted_objective(estimator, x, y), expected)


@pytest.mark.parametrize("soft", [False, True])
def test_valla_soft_targets_fit(data, soft):
    x, _, model = data
    flat = torch.tensor([0, 1, 1, 0])
    probabilities = torch.nn.functional.one_hot(flat, num_classes=2).to(x.dtype)
    probabilities = 0.8 * probabilities + 0.1 if soft else probabilities
    loader = DataLoader(TensorDataset(x, probabilities), batch_size=2)

    # Random and k-means selection read a class per row from the probabilities
    for strategy in ("random", "kmeans"):
        estimator = make_valla(
            deepcopy(model), inducing_locations=strategy, num_inducing=2
        )
        estimator.fit(loader, iterations=2, val_loader=loader)

        assert len(estimator.inducing_classes) == 2
        assert all(torch.isfinite(torch.tensor(estimator.fit_history_["objective"])))
        assert all(torch.isfinite(torch.tensor(estimator.fit_history_["val_nll"])))


def test_valla_validation_nll_probability_targets_and_custom_ignore_index(data):
    x, loader, model = data
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    estimator.fit(loader, iterations=1)

    y = torch.tensor([0, 1, 1, 0])
    one_hot = torch.nn.functional.one_hot(y, num_classes=2).to(x.dtype)

    torch.testing.assert_close(
        estimator._validation_nll(DataLoader(TensorDataset(x, one_hot), batch_size=2)),
        estimator._validation_nll(DataLoader(TensorDataset(x, y), batch_size=2)),
    )

    estimator.likelihood = CategoricalLikelihood(ignore_index=-1)
    padded = torch.tensor([0, -1, 1, -1])
    observed = padded != -1

    torch.testing.assert_close(
        estimator._validation_nll(DataLoader(TensorDataset(x, padded), batch_size=2)),
        estimator._validation_nll(
            DataLoader(TensorDataset(x[observed], padded[observed]), batch_size=2)
        ),
    )


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("target_format", ["integer-one-hot", "column"])
def test_valla_classification_targets_like_torch(data, split, target_format):
    x, _, model = data
    flat = torch.tensor([0, 1, 1, 0])
    bad = (
        flat[:, None]
        if target_format == "column"
        else torch.nn.functional.one_hot(flat, num_classes=2)
    )

    good_loader = DataLoader(TensorDataset(x, flat), batch_size=2)
    bad_loader = DataLoader(TensorDataset(x, bad), batch_size=2)
    train_loader = bad_loader if split == "train" else good_loader
    val_loader = bad_loader if split == "val" else good_loader
    estimator = make_valla(deepcopy(model), inducing_locations=x[:2].clone())

    with pytest.raises(ValueError, match="targets"):
        estimator.fit(train_loader, iterations=1, val_loader=val_loader)


@pytest.fixture
def bernoulli_data():
    torch.manual_seed(11)
    X = torch.randn(12, 3)
    y = torch.randint(2, (12, 2)).float()
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )

    return X, y, model


def _kl(features: torch.Tensor) -> torch.Tensor:
    hessian = torch.eye(features.shape[0], dtype=features.dtype) + features @ features.T

    return 0.5 * (
        torch.linalg.slogdet(hessian).logabsdet
        - torch.trace(torch.linalg.solve(hessian, features @ features.T))
    )


# Without options, and with the options of `nn.BCEWithLogitsLoss`
BERNOULLI_OPTIONS = [
    {},
    {"weight": torch.tensor([1.0, 2.0]), "pos_weight": torch.tensor([3.0, 0.5])},
]


@pytest.mark.parametrize("options", BERNOULLI_OPTIONS, ids=["plain", "options"])
def test_valla_bernoulli_probit_objective(bernoulli_data, options):
    X, y, model = bernoulli_data
    loader = DataLoader(TensorDataset(X, y), batch_size=6)
    la = make_valla(
        model, BernoulliLikelihood(**options), inducing_locations=X[:3].clone()
    )
    la.fit(loader, iterations=2)

    # Inducing points take output 0, as in regression
    assert la.inducing_classes.tolist() == [0, 0, 0]

    inputs, targets = next(iter(loader))
    mean, cov, features, _ = la._latent_distribution(inputs)
    scaled = mean / torch.sqrt(1 + torch.pi / 8 * cov.diagonal(dim1=-2, dim2=-1))
    log_term = -torch.nn.functional.binary_cross_entropy_with_logits(
        scaled, targets, reduction="sum", **options
    )
    expected = -(la.n_data / len(inputs)) * log_term / la.temperature + _kl(features)

    torch.testing.assert_close(la._objective(inputs, targets), expected)


@pytest.mark.parametrize("options", BERNOULLI_OPTIONS, ids=["plain", "options"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_valla_bernoulli_mc_objective(bernoulli_data, alpha, options):
    X, y, model = bernoulli_data
    loader = DataLoader(TensorDataset(X, y), batch_size=6)
    la = make_valla(
        model,
        BernoulliLikelihood(**options),
        inducing_locations=X[:3].clone(),
        alpha=alpha,
        mc_softmax_samples=5,
    )
    la.fit(loader, iterations=2)
    assert torch.isfinite(torch.tensor(la.fit_history_["objective"])).all()

    # Replay the draws of the objective
    inputs, targets = next(iter(loader))
    state = la.generator.get_state()
    objective = la._objective(inputs, targets)
    la.generator.set_state(state)

    mean, cov, features, _ = la._latent_distribution(inputs)
    draws = draw_gaussian(mean, cov, 5, la.generator)

    # The log likelihood of a row sums over its binary outputs
    log_lik = -torch.nn.functional.binary_cross_entropy_with_logits(
        draws, targets.expand_as(draws), reduction="none", **options
    ).sum(-1)

    if alpha == 0:
        log_term = log_lik.mean(0).sum()
    else:
        log_term = (
            torch.logsumexp(alpha * log_lik, 0) - torch.log(torch.tensor(5.0))
        ).sum() / alpha

    expected = -(la.n_data / len(inputs)) * log_term / la.temperature + _kl(features)

    torch.testing.assert_close(objective, expected)


def test_valla_bernoulli_alpha_requires_mc(bernoulli_data):
    X, _, model = bernoulli_data

    with pytest.raises(ValueError, match="mc_softmax_samples"):
        make_valla(model, "bernoulli", inducing_locations=X[:2].clone(), alpha=0.5)


def test_valla_bernoulli_fit_and_predict(bernoulli_data):
    X, y, model = bernoulli_data
    loader = DataLoader(TensorDataset(X, y), batch_size=6)

    # Soft labels are fine, as in `nn.BCEWithLogitsLoss`
    soft = DataLoader(TensorDataset(X, 0.8 * y + 0.1), batch_size=6)
    la = make_valla(model, "bernoulli", inducing_locations="random", num_inducing=4)
    la.fit(soft, iterations=3, val_loader=loader)

    assert la.inducing_classes.tolist() == [0, 0, 0, 0]
    assert torch.isfinite(torch.tensor(la.fit_history_["objective"])).all()

    mean, cov = la.predictive_moments(X)
    scaled = mean / torch.sqrt(1 + torch.pi / 8 * cov.diagonal(dim1=-2, dim2=-1))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        scaled, y, reduction="sum"
    ) / len(X)

    torch.testing.assert_close(la._validation_nll(loader), expected)

    probs = la(X, link_approx="probit")
    assert probs.shape == (12, 2)
    assert torch.all((probs > 0) & (probs < 1))
    torch.testing.assert_close(
        la(X, link_approx="mc", n_samples=20000), probs, atol=0.02, rtol=0
    )


def test_valla_bernoulli_targets_like_torch(bernoulli_data):
    X, y, model = bernoulli_data
    la = make_valla(model, "bernoulli", inducing_locations=X[:2].clone())

    with pytest.raises(ValueError, match="BernoulliLikelihood expects"):
        la.fit(DataLoader(TensorDataset(X, y.long()), batch_size=6), iterations=1)
