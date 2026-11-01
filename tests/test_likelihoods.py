from math import log, pi, sqrt

import pytest
import torch
from torch import nn

from laplace import (
    BaseLikelihood,
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
    Likelihood,
    get_likelihood,
)


@pytest.mark.parametrize(
    "key,cls",
    [
        ("regression", GaussianLikelihood),
        ("classification", CategoricalLikelihood),
        ("reward_modeling", BradleyTerryLikelihood),
        (Likelihood.REGRESSION, GaussianLikelihood),
        (Likelihood.CLASSIFICATION, CategoricalLikelihood),
        (Likelihood.REWARD_MODELING, BradleyTerryLikelihood),
    ],
)
def test_get_likelihood_from_str_and_enum(key, cls):
    assert type(get_likelihood(key)) is cls

    inst = cls()
    assert get_likelihood(inst) is inst


@pytest.mark.parametrize("bad", ["foo", GaussianLikelihood, 3, None])
def test_get_likelihood_invalid(bad):
    with pytest.raises(ValueError, match="Invalid likelihood type"):
        get_likelihood(bad)


def test_eq_and_hash():
    lik = GaussianLikelihood()

    assert lik == "regression"
    assert lik == Likelihood.REGRESSION
    assert Likelihood.REGRESSION == lik
    assert lik == GaussianLikelihood()
    assert lik != "classification"
    assert lik != CategoricalLikelihood()
    assert lik != 3
    assert lik in {"regression"}
    assert {lik: 1}["regression"] == 1
    assert str(lik) == "regression"


def test_loss_matches_torch():
    B, L, C = 4, 3, 5
    f = torch.randn(B, C)

    y = torch.randn(B, C)
    assert torch.allclose(
        GaussianLikelihood().loss(f, y), nn.MSELoss(reduction="sum")(f, y)
    )

    cat = CategoricalLikelihood()
    ce = nn.CrossEntropyLoss(reduction="sum")
    yc = torch.randint(C, (B,))
    assert torch.allclose(cat.loss(f, yc), ce(f, yc))

    f = torch.randn(B, L, C)
    yi = torch.randint(C, (B, L))
    assert torch.allclose(cat.loss(f, yi), ce(f.reshape(-1, C), yi.reshape(-1)))

    yp = torch.softmax(torch.randn(B, L, C), dim=-1)
    assert torch.allclose(cat.loss(f, yp), ce(f.reshape(-1, C), yp.reshape(-1, C)))


def test_loss_fn_is_fresh_module():
    lik = GaussianLikelihood()

    assert lik.loss_fn() is not lik.loss_fn()
    assert lik.loss_fn("mean").reduction == "mean"
    assert isinstance(CategoricalLikelihood().loss_fn(), nn.CrossEntropyLoss)
    assert GaussianLikelihood().loss_factor == 0.5
    assert CategoricalLikelihood().loss_factor == 1.0


def test_check_targets_errors():
    with pytest.raises(ValueError):
        GaussianLikelihood().check_targets(torch.randn(4, 3, 2), torch.randn(4, 2))

    with pytest.raises(ValueError):
        CategoricalLikelihood().check_targets(
            torch.randn(4, 3, 5), torch.randint(5, (4, 5, 3))
        )

    CategoricalLikelihood().check_targets(
        torch.randn(4, 3, 5), torch.randint(5, (4, 3))
    )


def test_n_observations():
    assert (
        GaussianLikelihood().n_observations(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        == 6
    )

    cat = CategoricalLikelihood()
    assert cat.n_observations(torch.randn(2, 3, 5), torch.randint(5, (2, 3))) == 6
    assert cat.n_observations(torch.randn(2, 5), torch.rand(2, 5)) == 2


def test_output_hessian():
    f = torch.randn(4, 3, dtype=torch.float64)

    H = GaussianLikelihood().output_hessian(f)
    assert H.dtype == f.dtype
    assert torch.equal(H, torch.eye(3, dtype=f.dtype).expand(4, 3, 3))

    p = torch.softmax(f, dim=-1)
    H = CategoricalLikelihood().output_hessian(f)
    assert H.dtype == f.dtype
    assert torch.allclose(H, torch.diag_embed(p) - p[:, :, None] * p[:, None, :])


def test_log_normalizer():
    sigma = torch.tensor(0.5, dtype=torch.float64)

    out = GaussianLikelihood().log_normalizer(3, 2, sigma)
    assert torch.allclose(out, torch.tensor(6 * log(0.5 * sqrt(2 * pi))).double())

    out = CategoricalLikelihood().log_normalizer(3, 2, sigma)
    assert out == 0
    assert out.dtype == sigma.dtype


@pytest.mark.parametrize("lik", [GaussianLikelihood(), CategoricalLikelihood()])
def test_sample_and_output_gradient(lik):
    f = torch.randn(6, 4)
    gen = torch.Generator().manual_seed(0)

    y = lik.sample(f, generator=gen)
    assert y.shape == f.shape

    y2 = lik.sample(f, generator=torch.Generator().manual_seed(0))
    assert torch.equal(y, y2)

    g = lik.output_gradient(f, y)
    assert g.shape == f.shape
    assert torch.allclose(g, lik.inverse_link(f) - y)


def test_subclass_contract():
    class Incomplete(BaseLikelihood):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()

    class Minimal(BaseLikelihood):
        name = "minimal"
        loss_factor = 1.0

        def loss_fn(self, reduction="sum"):
            return nn.L1Loss(reduction=reduction)

        def check_targets(self, f, y):
            return None

        def output_hessian(self, f):
            return torch.eye(f.shape[-1]).expand(*f.shape, f.shape[-1])

        def inverse_link(self, f):
            return f

        def sample(self, f, generator=None):
            return f

        def output_gradient(self, f, y):
            return f - y

    lik = Minimal()
    assert get_likelihood(lik) is lik
    assert lik.is_gaussian is False
    assert lik.fit_likelihood is lik
    assert lik.predictive_likelihood is lik
    assert lik.log_normalizer(3, 2, torch.tensor(1.0)) == 0

    f, y = torch.randn(3, 2), torch.randn(3, 2)
    assert torch.allclose(lik.loss(f, y), nn.L1Loss(reduction="sum")(f, y))
    assert lik.n_observations(f, y) == 3


def test_reward_modeling_split():
    lik = BradleyTerryLikelihood()

    assert lik.fit_likelihood == "classification"
    assert lik.predictive_likelihood == "regression"
    assert lik.is_gaussian is False
    assert lik.name == "reward_modeling"
