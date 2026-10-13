"""CUDA contracts for VaLLA on supported curvature backends."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.curvature import AsdlGGN, BackPackGGN, CurvlinopsGGN

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch required"),
]

CUDA_DEVICES = [
    torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())
] or [torch.device("cuda:0")]


@pytest.fixture(params=CUDA_DEVICES, ids=str)
def cuda_device(request):
    return request.param


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN, BackPackGGN])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_cuda_fit_moments_sampling_and_gradients(cuda_device, backend, likelihood):
    torch.manual_seed(21)
    inputs = torch.tensor(
        [[0.2, 0.4], [1.0, -0.3], [-0.5, 0.7], [0.8, 0.9], [-0.4, -0.6], [0.3, -0.2]]
    )
    targets = (
        torch.tensor([0, 1, 0, 1, 1, 0])
        if likelihood == "classification"
        else torch.stack([inputs.sum(dim=-1), inputs[:, 0] - inputs[:, 1]], dim=-1)
    )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
    model = torch.nn.Linear(2, 2).to(cuda_device)
    weights = [parameter.detach().clone() for parameter in model.parameters()]
    options = {"inducing_locations": inputs[:2].clone()}
    estimator = Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        backend=backend,
        enable_backprop=True,
        **options,
    )
    assert estimator.fit(loader, iterations=2, lr=0.001) is None
    for parameter, original in zip(model.parameters(), weights):
        torch.testing.assert_close(parameter, original)
    query = inputs[:2].to(cuda_device).requires_grad_()
    mean, covariance = estimator.predictive_moments(query)
    joint_mean, joint_covariance = estimator.predictive_moments(query, joint=True)
    assert mean.shape == (2, 2)
    assert covariance.shape == (2, 2, 2)
    assert joint_mean.shape == (4,)
    assert joint_covariance.shape == (4, 4)
    assert mean.device == covariance.device == joint_covariance.device == cuda_device
    assert torch.isfinite(mean).all() and torch.isfinite(covariance).all()
    blocks = torch.stack(
        [joint_covariance[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2] for i in range(2)]
    )
    torch.testing.assert_close(blocks, covariance, rtol=0.001, atol=0.0001)
    gradient = torch.autograd.grad(mean.sum() + covariance.sum(), query)[0]
    assert torch.isfinite(gradient).all()
    draws = estimator.functional_samples(
        query.detach(),
        n_samples=3,
        joint=True,
        generator=torch.Generator(device=cuda_device).manual_seed(7),
    )
    predictive = estimator.predictive_samples(query.detach(), n_samples=3)
    assert draws.shape == predictive.shape == (3, 2, 2)
    assert draws.device == predictive.device == cuda_device
    assert torch.isfinite(draws).all() and torch.isfinite(predictive).all()
    if likelihood == "classification":
        probabilities = estimator(query.detach())
        assert probabilities.shape == (2, 2)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(2, device=cuda_device),
            atol=1e-05,
            rtol=1e-05,
        )
        torch.testing.assert_close(
            predictive.sum(dim=-1),
            torch.ones(3, 2, device=cuda_device),
            atol=1e-05,
            rtol=1e-05,
        )
    else:
        regression_mean, regression_covariance = estimator(query.detach())
        torch.testing.assert_close(regression_mean, mean.detach())
        torch.testing.assert_close(regression_covariance, covariance.detach())


@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN])
def test_cuda_reward_mapping_and_checkpoint(cuda_device, backend):

    class RewardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, batch):
            inputs = batch["input_ids"]
            rewards = self.linear(inputs)
            return rewards.squeeze(-1) if inputs.ndim == 3 else rewards

    torch.manual_seed(22)
    pairs = torch.randn(4, 2, 2)
    labels = torch.tensor([0, 1, 1, 0])
    dataset = [
        {"input_ids": pair, "labels": label, "source": f"pair-{index}"}
        for index, (pair, label) in enumerate(zip(pairs, labels))
    ]
    loader = DataLoader(dataset, batch_size=2)
    model = RewardModel().to(cuda_device)
    original_model = deepcopy(model)
    options = {"inducing_locations": "random", "num_inducing": 2}
    estimator = Laplace(
        model,
        "reward_modeling",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        backend=backend,
        **options,
    )
    estimator.fit(loader, iterations=1)
    assert len(estimator.inducing_locations["source"]) == 2
    pair_query = {"input_ids": pairs[:2].to(cuda_device)}
    single_query = {"input_ids": pairs[:2, 0].to(cuda_device)}
    preference = estimator(pair_query, fitting=True)
    mean, covariance = estimator.predictive_moments(single_query)
    assert preference.shape == (2, 2)
    assert mean.shape == (2, 1) and covariance.shape == (2, 1, 1)
    assert preference.device == mean.device == covariance.device == cuda_device
    torch.testing.assert_close(
        preference.sum(dim=-1),
        torch.ones(2, device=cuda_device),
        atol=1e-05,
        rtol=1e-05,
    )
    restored = Laplace(
        original_model,
        "reward_modeling",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        backend=backend,
        **options,
    )
    restored.load_state_dict(estimator.state_dict())
    restored_mean, restored_covariance = restored.predictive_moments(single_query)
    torch.testing.assert_close(restored_mean, mean)
    torch.testing.assert_close(restored_covariance, covariance)
