"""Edge-case inputs for every Laplace method.

The tests reuse the problem of `tests/test_contracts.py`. Most of them feed the
same data in another form and check that the posterior doesn't change:

- batch size 1, and the whole data set in one batch;
- dict batches with a model that takes a mapping, like a Hugging Face model;
- a single test input.

Two more check a single output, and that the function-space methods match
`FullLaplace` for multi-output regression.
"""

import pytest
import torch

from tests.test_contracts import (
    METHODS,
    N_DATA,
    N_OUT,
    NO_MARGLIK,
    _fitted,
    _param,
    _skip_lowrank_without_asdfghjkl,
)

pytestmark = [pytest.mark.default_dtype(torch.float64), pytest.mark.seed(0)]

all_methods = pytest.mark.parametrize("method", [_param(m) for m in METHODS])

# VaLLA fits by minibatch gradient steps, so its fit depends on the batches
deterministic_methods = pytest.mark.parametrize(
    "method", [_param(m) for m in METHODS if m != "valla"]
)


LIKELIHOODS = [
    pytest.param("regression", N_OUT, id="regression"),
    pytest.param("classification", 3, id="classification"),
    pytest.param("bernoulli", 1, id="bernoulli"),
]
likelihoods = pytest.mark.parametrize("likelihood, n_out", LIKELIHOODS)

FUNCTION_SPACE = {"functional", "ll_functional", "ella", "valla"}


def _skip_unsupported(method: str, likelihood: str) -> None:
    _skip_lowrank_without_asdfghjkl(method)

    # These raise a clear error for the Bernoulli likelihood
    if likelihood == "bernoulli" and (method in FUNCTION_SPACE or method == "lowrank"):
        pytest.skip(f"{method} doesn't support the Bernoulli likelihood")


def _assert_same_posterior(la, reference, X_test, ref_X_test, method) -> None:
    # The asdfghjkl power iteration stops at a loose tolerance, so the LowRank
    # eigenpairs move a little with the batching
    tol = {"rtol": 1e-4, "atol": 1e-4} if method == "lowrank" else {}

    if la.likelihood.is_gaussian:
        mean, cov = la(X_test, joint=True)
        ref_mean, ref_cov = reference(ref_X_test, joint=True)
        torch.testing.assert_close(mean, ref_mean, **tol)
        torch.testing.assert_close(cov, ref_cov, **tol)
    else:
        torch.testing.assert_close(la(X_test), reference(ref_X_test), **tol)

    if method not in NO_MARGLIK:
        torch.testing.assert_close(
            la.log_marginal_likelihood(), reference.log_marginal_likelihood(), **tol
        )


@deterministic_methods
@likelihoods
@pytest.mark.parametrize("batch_size", [1, N_DATA])
def test_batch_size_does_not_change_posterior(method, likelihood, n_out, batch_size):
    _skip_unsupported(method, likelihood)

    options = {"n_out": n_out, "likelihood": likelihood}
    la, _, X_test, _ = _fitted(method, batch_size=batch_size, **options)
    reference, _, _, _ = _fitted(method, **options)

    _assert_same_posterior(la, reference, X_test, X_test, method)


@deterministic_methods
@likelihoods
def test_shuffled_loader_does_not_change_posterior(method, likelihood, n_out):
    _skip_unsupported(method, likelihood)

    # The curvature sums over the data, and the subsets are drawn by index
    options = {"n_out": n_out, "likelihood": likelihood}
    torch.manual_seed(3)
    la, _, X_test, _ = _fitted(method, shuffle=True, **options)
    reference, _, _, _ = _fitted(method, **options)

    _assert_same_posterior(la, reference, X_test, X_test, method)


@deterministic_methods
@likelihoods
def test_dict_batches_match_tensor_batches(method, likelihood, n_out):
    _skip_unsupported(method, likelihood)

    if method == "lowrank":
        pytest.skip(
            "LowRank rejects dict batches, see `test_lowrank_rejects_dict_batches`"
        )

    options = {"n_out": n_out, "likelihood": likelihood}
    la, _, X_test, _ = _fitted(method, dict_inputs=True, **options)
    reference, _, ref_X_test, _ = _fitted(method, **options)

    _assert_same_posterior(la, reference, X_test, ref_X_test, method)


@all_methods
def test_single_test_input(method):
    _skip_lowrank_without_asdfghjkl(method)
    la, _, X_test, _ = _fitted(method)

    mean, cov = la(X_test[:1])
    batch_mean, batch_cov = la(X_test)
    assert mean.shape == (1, N_OUT)
    torch.testing.assert_close(mean, batch_mean[:1])
    torch.testing.assert_close(cov, batch_cov[:1])

    joint_mean, joint_cov = la(X_test[:1], joint=True)
    assert joint_mean.shape == (N_OUT,)
    torch.testing.assert_close(joint_cov, cov[0])


@all_methods
@pytest.mark.parametrize("likelihood, n_out", LIKELIHOODS[1:])
def test_single_test_input_classification(method, likelihood, n_out):
    _skip_unsupported(method, likelihood)
    la, _, X_test, _ = _fitted(method, n_out=n_out, likelihood=likelihood)

    probs = la(X_test[:1])
    assert probs.shape == (1, n_out)
    torch.testing.assert_close(probs, la(X_test)[:1])


@all_methods
def test_single_output(method):
    _skip_lowrank_without_asdfghjkl(method)
    la, _, X_test, _ = _fitted(method, n_out=1)
    B = len(X_test)

    mean, cov = la(X_test)
    assert mean.shape == (B, 1)
    assert cov.shape == (B, 1, 1)

    joint_mean, joint_cov = la(X_test, joint=True)
    torch.testing.assert_close(joint_mean, mean.reshape(-1))
    torch.testing.assert_close(torch.diagonal(joint_cov), cov.reshape(-1))


# On all of the data, the subset-of-data GP is the GLM predictive of the full GGN.
# ELLA isn't here: its Nyström prior keeps one output per subsample input, so
# it's low rank by design and never exact.
FUNCTION_SPACE_REFERENCES = [
    pytest.param("functional", "full", id="functional"),
    pytest.param("ll_functional", "ll_full", id="ll_functional"),
]


@pytest.mark.parametrize("method, reference_method", FUNCTION_SPACE_REFERENCES)
def test_function_space_multi_output_matches_full(method, reference_method):
    la, _, X_test, _ = _fitted(method)
    reference, _, _, _ = _fitted(reference_method)

    mean, cov = la(X_test, joint=True)
    ref_mean, ref_cov = reference(X_test, joint=True)
    torch.testing.assert_close(mean, ref_mean)
    torch.testing.assert_close(cov, ref_cov)


@pytest.mark.asdfghjkl
def test_lowrank_rejects_dict_batches():
    # The legacy asdfghjkl eigensolver unpacks every batch as `(X, y)`
    pytest.importorskip("asdfghjkl")

    with pytest.raises(ValueError, match="dict-like batches"):
        _fitted("lowrank", dict_inputs=True)
