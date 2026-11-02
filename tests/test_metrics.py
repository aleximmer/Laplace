import math

import pytest
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

    assert math.isclose(nll_running, nll_offline, rel_tol=1e-5)


@pytest.mark.parametrize("transpose_probs", [False, True])
@pytest.mark.parametrize("transpose_targets", [False, True])
@pytest.mark.parametrize("ignore_idx", [-100, -1232])
def test_running_nll_metric_noncontiguous(
    transpose_probs, transpose_targets, ignore_idx
):
    generator = torch.Generator().manual_seed(42)
    metric = RunningNLLMetric(ignore_index=ignore_idx)
    all_probs, all_targets = [], []

    for batch_size in (2, 4):
        probs = torch.randn(batch_size, 3, 5, generator=generator).softmax(-1)
        targets = torch.randint(5, (batch_size, 3), generator=generator)
        probs = probs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        targets[0, 0] = ignore_idx
        if not transpose_probs:
            probs = probs.contiguous()
        if not transpose_targets:
            targets = targets.contiguous()
        assert probs.is_contiguous() != transpose_probs
        assert targets.is_contiguous() != transpose_targets

        metric.update(probs, targets)
        all_probs.append(probs.flatten(end_dim=-2))
        all_targets.append(targets.flatten())

    expected = F.nll_loss(
        torch.cat(all_probs).log(), torch.cat(all_targets), ignore_index=ignore_idx
    )
    torch.testing.assert_close(metric.compute(), expected)
