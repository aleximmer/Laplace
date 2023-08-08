import torch
from torch.nn import functional as F
from laplace.utils import RunningNLLMetric
import math


def test_running_nll_metric():
    metric = RunningNLLMetric()
    all_probs, all_targets = [], []

    for _ in range(10):
        probs = torch.softmax(torch.randn(3, 5, 10), dim=-1)
        targets = torch.randint(10, size=(3, 5))
        metric.update(probs, targets)
        all_probs.append(probs)
        all_targets.append(targets)

    all_probs, all_targets = torch.cat(all_probs, 0), torch.cat(all_targets, 0)

    nll_running = metric.compute().item()
    nll_offline = F.nll_loss(all_probs.log().flatten(end_dim=-2), all_targets.flatten()).item()

    assert math.isclose(nll_running, nll_offline)


def test_running_nll_metric_ignore_idx():
    ignore_idx = -1232
    metric_orig = RunningNLLMetric()
    metric_ignore = RunningNLLMetric(ignore_index=ignore_idx)

    for _ in range(10):
        probs = torch.softmax(torch.randn(3, 5, 10), dim=-1)
        targets_orig = torch.randint(10, size=(3, 5))
        targets_ignore = targets_orig.clone()
        metric_orig.update(probs, targets_orig)

        mask = torch.FloatTensor(*targets_ignore.shape).uniform_() > 0.8  # ~80% zeros
        targets_ignore[mask] = ignore_idx  # ~80% changed to ignore_idx
        metric_ignore.update(probs, targets_ignore)

    nll_orig = metric_orig.compute().item()
    nll_ignore = metric_ignore.compute().item()

    assert nll_orig > nll_ignore
