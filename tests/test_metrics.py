import math

import pytest
import torch
from torch.nn import functional as F

from laplace.utils import RunningBernoulliNLLMetric, RunningNLLMetric


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


@pytest.mark.parametrize("shape", [(6, 4), (3, 5, 4), (2, 3, 3, 4)])
@pytest.mark.parametrize("soft", [False, True])
def test_bernoulli_nll_is_summed_over_outputs_and_averaged_over_positions(shape, soft):
    torch.manual_seed(0)
    probs = torch.rand(*shape)
    targets = torch.rand(*shape)

    if not soft:
        targets = (targets > 0.5).float()

    metric = RunningBernoulliNLLMetric()
    metric.update(probs, targets)

    nll = -(targets * probs.log() + (1 - targets) * (1 - probs).log())

    # Summed over the last dim, averaged over the others
    assert torch.allclose(metric.compute(), nll.sum(-1).mean())


def test_bernoulli_nll_accumulates_over_batches():
    torch.manual_seed(0)
    probs = torch.rand(10, 4)
    targets = (torch.rand(10, 4) > 0.5).float()

    metric = RunningBernoulliNLLMetric()

    # Uneven batches: the average is over positions, not over batches
    for batch in (slice(0, 3), slice(3, 4), slice(4, 10)):
        metric.update(probs[batch], targets[batch])

    full = RunningBernoulliNLLMetric()
    full.update(probs, targets)

    assert torch.allclose(metric.compute(), full.compute())

    metric.reset()
    metric.update(probs[:2], targets[:2])
    assert not torch.allclose(metric.compute(), full.compute())


def test_bernoulli_nll_matches_the_loss_of_the_logits():
    torch.manual_seed(0)
    f = torch.randn(8, 3)
    y = torch.rand(8, 3)

    metric = RunningBernoulliNLLMetric()
    metric.update(torch.sigmoid(f), y)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        f, y, reduction="sum"
    ) / len(f)

    assert torch.allclose(metric.compute(), expected, atol=1e-6)


def test_bernoulli_nll_is_finite_for_saturated_probabilities():
    metric = RunningBernoulliNLLMetric()
    metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))

    assert torch.isfinite(metric.compute())


def test_bernoulli_nll_rejects_mismatched_shapes():
    metric = RunningBernoulliNLLMetric()

    # Class indices, as the softmax metric takes them
    with pytest.raises(ValueError, match="targets have shape"):
        metric.update(torch.rand(4, 3), torch.randint(3, (4,)))
