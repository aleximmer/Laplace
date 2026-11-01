"""Check the decision-making example and the claims its docs page makes."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from examples.expectation_example import (
    ei_cf,
    expected_utility,
    fit_laplace,
    joint_glm_samples,
    thompson_sample,
)
from laplace import BaseLaplace


@pytest.fixture(scope="module")
def fitted() -> BaseLaplace:
    return fit_laplace()[0]


def test_example_runs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "examples.expectation_example"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    for prefix in [
        "EI-CF x_best:",
        "PI-CF x_best:",
        "qEI-CF batch:",
        "Thompson x_next:",
    ]:
        assert prefix in result.stdout


def test_glm_samples_miss_cross_input_covariance(fitted: BaseLaplace) -> None:
    x = torch.tensor([[0.05, 0.05], [0.95, 0.95]])

    # GLM samples get each input's marginal right but reuse the same noise for
    # every input, so the covariance between inputs is not the joint one
    samples = fitted.functional_samples(x, n_samples=20000).detach()
    corr_glm = torch.corrcoef(samples.reshape(20000, 4).T)
    _, cov = fitted(x, joint=True)
    std = cov.detach().diagonal().sqrt()
    corr_joint = cov.detach() / torch.outer(std, std)

    assert (corr_glm[:2, 2:] - corr_joint[:2, 2:]).abs().max() > 0.3
    assert torch.allclose(
        samples.var(0), fitted(x)[1].diagonal(dim1=-2, dim2=-1).detach(), rtol=0.1
    )


def test_thompson_chunking_consistent(fitted: BaseLaplace) -> None:
    candidates = torch.rand(10, 2)

    full = thompson_sample(fitted, candidates, seed=7)
    chunked = thompson_sample(fitted, candidates, seed=7, chunk_size=3)

    assert full.shape == (10, 2)
    assert torch.allclose(full, chunked)


def test_joint_samples_match_marginals(fitted: BaseLaplace) -> None:
    torch.manual_seed(0)
    x = torch.rand(3, 2)

    samples = joint_glm_samples(fitted, x, torch.randn(20000, 6)).detach()
    mean, var = fitted(x)
    var = var.diagonal(dim1=-2, dim2=-1).detach()

    assert samples.shape == (20000, 3, 2)
    assert torch.allclose(samples.mean(0), mean.detach(), atol=0.05 * var.sqrt().max())
    assert torch.allclose(samples.var(0), var, rtol=0.1)


def test_joint_samples_duplicate_points_no_error(fitted: BaseLaplace) -> None:
    x = torch.rand(1, 2).repeat(3, 1)  # Singular joint covariance without jitter
    assert joint_glm_samples(fitted, x, torch.randn(8, 6)).shape == (8, 3, 2)


def test_expected_utility_differentiable_and_deterministic(fitted: BaseLaplace) -> None:
    x = torch.rand(4, 2, requires_grad=True)

    def utility(f: torch.Tensor) -> torch.Tensor:
        return ei_cf(f, best=-1.0)

    u = expected_utility(fitted, x, utility)

    assert u.shape == (4,)
    assert torch.equal(u, expected_utility(fitted, x, utility))
    assert torch.autograd.grad(u.sum(), x)[0].abs().sum() > 0
