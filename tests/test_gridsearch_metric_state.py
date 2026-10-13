import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError

from laplace import DiagLaplace
from laplace.utils import RunningNLLMetric, get_nll


class ControlledPrediction(DiagLaplace):
    def __call__(self, inputs, **kwargs):
        prior = self.prior_precision.item()
        if self.likelihood == "classification":
            probability = {1.0: 0.2, 10.0: 0.8, 100.0: 0.5}[prior]
            return inputs.new_tensor([probability, 1 - probability]).expand(
                len(inputs), -1
            )
        mean = inputs.new_full(
            (len(inputs), 1), {1.0: 2.0, 10.0: 1.0, 100.0: 1.5}[prior]
        )
        return mean, torch.ones_like(mean)


@pytest.mark.parametrize("likelihood", ["classification", "regression"])
@pytest.mark.parametrize("metric_kind", ["default", "online", "warm_online", "offline"])
def test_gridsearch_evaluates_metrics_independently(likelihood, metric_kind):
    n_outputs = 2 if likelihood == "classification" else 1
    posterior = ControlledPrediction(nn.Linear(1, n_outputs).double(), likelihood)
    posterior.n_outputs = n_outputs
    inputs = torch.ones(3, 1, dtype=torch.float64)
    targets = (
        torch.zeros(3, dtype=torch.long) if n_outputs == 2 else torch.zeros_like(inputs)
    )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    metric = None
    if metric_kind in ("online", "warm_online"):
        metric = RunningNLLMetric() if n_outputs == 2 else MeanSquaredError()
        if metric_kind == "warm_online":
            if n_outputs == 2:
                metric.update(torch.tensor([[0.01, 0.99]]), torch.tensor([0]))
            else:
                metric.update(torch.tensor([10.0]), torch.tensor([0.0]))
    elif metric_kind == "offline":
        metric = (
            get_nll
            if n_outputs == 2
            else lambda means, variance, y: (means - y).square().mean()
        )
    # Separate candidate losses are -log(.2/.8/.5), or 4/1/2.25.
    # Their cumulative averages incorrectly select the third candidate.
    posterior.optimize_prior_precision(
        "glm",
        method="gridsearch",
        val_loader=loader,
        loss=metric,
        log_prior_prec_min=0,
        log_prior_prec_max=2,
        grid_size=3,
    )
    torch.testing.assert_close(
        posterior.prior_precision, torch.tensor([10.0], dtype=torch.float64)
    )
