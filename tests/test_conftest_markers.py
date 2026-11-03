import pytest

from laplace import FullLaplace, LowRankLaplace
from laplace.curvature.asdfghjkl import AsdfghjklGGN

# The old-asdfghjkl CI job runs only the tests with the `asdfghjkl` marker,
# which `tests/conftest.py` adds at collection time.


@pytest.mark.parametrize("laplace", [LowRankLaplace])
def test_lowrank_param_is_marked(laplace, request):
    assert request.node.get_closest_marker("asdfghjkl") is not None


@pytest.mark.parametrize("laplace,lh", [(LowRankLaplace, "regression")])
def test_lowrank_in_tuple_param_is_marked(laplace, lh, request):
    assert request.node.get_closest_marker("asdfghjkl") is not None


@pytest.mark.parametrize("backend", [AsdfghjklGGN])
def test_asdfghjkl_backend_param_is_marked(backend, request):
    assert request.node.get_closest_marker("asdfghjkl") is not None


@pytest.mark.parametrize("laplace", [FullLaplace])
def test_other_param_is_not_marked(laplace, request):
    assert request.node.get_closest_marker("asdfghjkl") is None


def test_unparametrized_is_not_marked(request):
    assert request.node.get_closest_marker("asdfghjkl") is None
