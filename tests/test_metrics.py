import math

import torch
from torch.nn import functional as F

from laplace.utils import RunningNLLMetric


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
    nll_offline = F.nll_loss(
        all_probs.log().flatten(end_dim=-2), all_targets.flatten()
    ).item()

    assert math.isclose(nll_running, nll_offline, rel_tol=1e-7)


def test_running_nll_metric_ignore_idx():
    ignore_idx = -1232
    metric = RunningNLLMetric(ignore_index=ignore_idx)
    all_probs, all_targets = [], []

    for _ in range(10):
        probs = torch.softmax(torch.randn(3, 5, 10), dim=-1)
        targets = torch.randint(10, size=(3, 5))
        mask = torch.FloatTensor(*targets.shape).uniform_() > 0.5  # ~50% zeros
        targets[mask] = ignore_idx  # ~50% changed to ignore_idx
        all_probs.append(probs)
        all_targets.append(targets)
        metric.update(probs, targets)

    all_probs, all_targets = torch.cat(all_probs, 0), torch.cat(all_targets, 0)

    nll_running = metric.compute().item()
    nll_offline = F.nll_loss(
        all_probs.log().flatten(end_dim=-2),
        all_targets.flatten(),
        ignore_index=ignore_idx,
    ).item()

    assert math.isclose(nll_running, nll_offline, rel_tol=1e-7)
