import pytest
import torch

# These tests run in file order: the "leaking" tests change global state on
# purpose, and the test right after each one checks that the change is gone.


def test_unmarked_test_gets_float32():
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.default_dtype(torch.float64)
def test_marker_sets_dtype():
    assert torch.get_default_dtype() == torch.float64


def test_leaking_test_sets_double():
    torch.set_default_dtype(torch.float64)


def test_next_test_sees_float32():
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.default_dtype(torch.float64)
class TestParentMarker:
    def test_parent_marker_applies(self):
        assert torch.get_default_dtype() == torch.float64


@pytest.mark.seed(240)
def test_seed_marker_seeds_rng():
    expected = torch.Generator().manual_seed(240)
    assert torch.equal(torch.randn(5), torch.randn(5, generator=expected))


@pytest.mark.seed(240)
def test_seed_does_not_depend_on_earlier_draws():
    expected = torch.Generator().manual_seed(240)
    assert torch.equal(torch.randn(5), torch.randn(5, generator=expected))


def test_unmarked_test_gets_seed_zero():
    expected = torch.Generator().manual_seed(0)
    assert torch.equal(torch.randn(5), torch.randn(5, generator=expected))


def test_leaking_test_reseeds_rng():
    torch.manual_seed(1234)


def test_next_test_still_gets_seed_zero():
    expected = torch.Generator().manual_seed(0)
    assert torch.equal(torch.randn(5), torch.randn(5, generator=expected))
