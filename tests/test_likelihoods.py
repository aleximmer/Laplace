import pickle
import warnings
from math import log, pi, sqrt

import pytest
import torch
from torch import nn

from laplace import (
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
    Likelihood,
    get_likelihood,
)
from tests.utils import (
    DistributionOnlyBernoulli,
    DistributionOnlyCategorical,
    DistributionOnlyGaussian,
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


@pytest.mark.parametrize("lik", [CategoricalLikelihood(), BradleyTerryLikelihood()])
@pytest.mark.parametrize(
    "sigma", [torch.tensor([1.0], dtype=torch.float32), torch.tensor(1.0), 1.0]
)
def test_log_normalizer_zero_is_scalar(lik, sigma):
    out = lik.log_normalizer(3, 2, sigma)

    assert out.shape == ()
    assert out == 0

    if isinstance(sigma, torch.Tensor):
        assert out.dtype == sigma.dtype
        assert out.device == sigma.device
    else:
        assert out.dtype == torch.get_default_dtype()


def test_log_normalizer_gaussian_dtype():
    lik = GaussianLikelihood()

    out = lik.log_normalizer(3, 2, 0.5)
    assert out.dtype == torch.get_default_dtype()
    assert torch.allclose(out, torch.tensor(6 * log(0.5 * sqrt(2 * pi))))

    sigma = torch.tensor(0.5, dtype=torch.float32)
    assert lik.log_normalizer(3, 2, sigma).dtype == torch.float32

    sigma = torch.tensor(0.5, requires_grad=True)
    lik.log_normalizer(3, 2, sigma).backward()
    assert torch.allclose(sigma.grad, torch.tensor(6 / 0.5))


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
    class Incomplete(Likelihood):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()

    with pytest.raises(TypeError):
        Likelihood()

    lik = DistributionOnlyGaussian()
    assert get_likelihood(lik) is lik
    assert lik.is_gaussian is False
    assert lik.fit_likelihood is lik
    assert lik.predictive_likelihood is lik
    assert lik.loss_factor == 1.0
    assert lik.log_normalizer(3, 2, torch.tensor(1.0)) == 0

    f, y = torch.randn(3, 2), torch.randn(3, 2)
    assert lik.n_observations(f, y) == 3


def test_concrete_subclass_requires_name():
    with pytest.raises(TypeError, match="must define a `name`"):

        class Nameless(CategoricalLikelihood):
            name = None

    class AbstractNameless(Likelihood):
        def inverse_link(self, f):
            return f

    with pytest.raises(TypeError, match="must define a `name`"):

        class ConcreteNameless(AbstractNameless):
            def distribution(self, f):
                return torch.distributions.Normal(f, 1.0)


def test_default_loss_matches_builtin_gaussian():
    f = torch.randn(6, 3, dtype=torch.float64)
    y = torch.randn(6, 3, dtype=torch.float64)

    g = GaussianLikelihood()
    expected = g.loss_factor * g.loss(f, y) + g.log_normalizer(
        6, 3, torch.tensor(1.0, dtype=torch.float64)
    )

    lik = DistributionOnlyGaussian()
    assert torch.allclose(lik.loss(f, y), expected)
    assert torch.allclose(lik.loss_fn("mean")(f, y), expected / 18)


def test_default_loss_matches_builtin_categorical():
    f = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randint(4, (6,))

    assert torch.allclose(
        DistributionOnlyCategorical().loss(f, y), CategoricalLikelihood().loss(f, y)
    )


def test_default_check_targets():
    with pytest.raises(ValueError, match="expects"):
        DistributionOnlyGaussian().check_targets(torch.randn(6, 3), torch.randn(6))

    cat = DistributionOnlyCategorical()
    f = torch.randn(6, 4)
    cat.check_targets(f, torch.randint(4, (6,)))

    with pytest.raises(ValueError, match="expects"):
        cat.check_targets(f, torch.randint(4, (6, 4)))


def test_default_inverse_link_and_gradient():
    f = torch.randn(6, 4, dtype=torch.float64)

    y = torch.randn(6, 4, dtype=torch.float64)
    gauss, builtin = DistributionOnlyGaussian(), GaussianLikelihood()
    assert torch.allclose(gauss.inverse_link(f), builtin.inverse_link(f))
    assert torch.allclose(gauss.output_gradient(f, y), builtin.output_gradient(f, y))

    yc = torch.randint(4, (6,))
    cat, builtin = DistributionOnlyCategorical(), CategoricalLikelihood()
    onehot = nn.functional.one_hot(yc, 4).to(f.dtype)
    p = cat.inverse_link(f)
    assert not torch.isnan(p).any()
    assert torch.allclose(p, builtin.inverse_link(f))
    assert torch.allclose(
        cat.output_gradient(f, yc), builtin.output_gradient(f, onehot)
    )


@pytest.mark.parametrize("shape", [(50, 3), (4, 5, 3)])
def test_default_output_hessian_enumerates_discrete(shape):
    f = torch.randn(*shape, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H = DistributionOnlyCategorical().output_hessian(f)

    expected = CategoricalLikelihood().output_hessian(f)
    assert H.shape == expected.shape
    assert H.dtype == f.dtype
    assert torch.allclose(H, expected, atol=1e-10)


def test_default_output_hessian_enumerates_independent_outputs():
    f = torch.randn(50, 3, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H = DistributionOnlyBernoulli().output_hessian(f)

    p = torch.sigmoid(f)
    assert H.shape == (50, 3, 3)
    assert H.dtype == torch.float64
    assert torch.allclose(H, torch.diag_embed(p * (1 - p)), atol=1e-10)


@pytest.mark.parametrize("chunk_size", [10, 7])
def test_default_output_hessian_is_mc_fisher(chunk_size):
    class ManySamples(DistributionOnlyGaussian):
        fisher_samples = 20000
        fisher_chunk_size = chunk_size

    torch.manual_seed(0)
    f = torch.randn(50, 3, dtype=torch.float64)

    with pytest.warns(UserWarning, match="Fisher"):
        H = ManySamples().output_hessian(f)

    assert H.shape == (50, 3, 3)
    assert H.dtype == f.dtype
    assert torch.allclose(H.mean(0), torch.eye(3, dtype=torch.float64), atol=0.05)


def test_defaults_work_without_grad_and_on_graph_outputs():
    model = nn.Linear(2, 3).double()
    x = torch.randn(5, 2, dtype=torch.float64)
    yc = torch.randint(3, (5,))
    y = torch.randn(5, 3, dtype=torch.float64)

    cat, gauss = DistributionOnlyCategorical(), DistributionOnlyGaussian()
    expected_H = CategoricalLikelihood().output_hessian(model(x).detach())

    # Inside `no_grad`, as in the fit loops
    with torch.no_grad():
        f = model(x)
        H = cat.output_hessian(f)
        g = cat.output_gradient(f, yc)

        with pytest.warns(UserWarning, match="Fisher"):
            gauss.output_hessian(f)

    assert torch.allclose(H, expected_H, atol=1e-10)
    assert g.shape == f.shape

    # On an output that is part of the model graph
    f = model(x)
    H = cat.output_hessian(f, yc)
    g = gauss.output_gradient(f, y)

    with pytest.warns(UserWarning, match="Fisher"):
        H_mc = gauss.output_hessian(f, y)

    for out in (H, g, H_mc):
        assert not out.requires_grad
        assert out.grad_fn is None

    assert torch.allclose(H, expected_H, atol=1e-10)
    assert torch.allclose(g, f.detach() - y)
    assert all(p.grad is None for p in model.parameters())


def test_default_sample_generator():
    f = torch.randn(6, 3, dtype=torch.float64)
    state = torch.get_rng_state()

    lik = DistributionOnlyGaussian()
    y1 = lik.sample(f, generator=torch.Generator().manual_seed(3))
    y2 = lik.sample(f, generator=torch.Generator().manual_seed(3))

    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(y1, y2)
    assert y1.dtype == torch.float64
    assert y1.shape == f.shape

    yc = DistributionOnlyCategorical().sample(
        f, generator=torch.Generator().manual_seed(3)
    )
    assert yc.shape == f.shape[:-1]
    assert torch.equal(torch.get_rng_state(), state)


def test_builtin_distributions():
    f = torch.randn(6, 3, dtype=torch.float64)

    dist = GaussianLikelihood().distribution(f)
    assert isinstance(dist, torch.distributions.Normal)
    assert torch.equal(dist.mean, f)
    assert torch.all(dist.scale == 1)

    for lik in (CategoricalLikelihood(), BradleyTerryLikelihood()):
        dist = lik.distribution(f)
        assert isinstance(dist, torch.distributions.Categorical)
        assert torch.allclose(dist.probs, torch.softmax(f, dim=-1))


@pytest.mark.parametrize(
    "lik", [GaussianLikelihood(), CategoricalLikelihood(), BradleyTerryLikelihood()]
)
def test_output_hessian_accepts_targets(lik):
    f = torch.randn(6, 3, dtype=torch.float64)
    y = torch.randint(3, (6,))

    assert torch.equal(lik.output_hessian(f, y), lik.output_hessian(f))


@pytest.mark.parametrize("shape", [(7, 3, 4), (50, 10)])
def test_categorical_sample_matches_multinomial(shape):
    f = torch.randn(*shape, dtype=torch.float64)

    torch.manual_seed(42)
    y = CategoricalLikelihood().sample(f)

    torch.manual_seed(42)
    y_ref = torch.distributions.Multinomial(logits=f).sample()

    assert y.dtype == y_ref.dtype
    assert torch.equal(y, y_ref)


def test_reward_modeling_split():
    lik = BradleyTerryLikelihood()

    assert lik.fit_likelihood == "classification"
    assert lik.predictive_likelihood == "regression"
    assert lik.is_gaussian is False
    assert lik.name == "reward_modeling"


def test_builtin_members_replace_the_enum():
    assert type(Likelihood.REGRESSION) is GaussianLikelihood
    assert type(Likelihood.CLASSIFICATION) is CategoricalLikelihood
    assert type(Likelihood.REWARD_MODELING) is BradleyTerryLikelihood

    assert Likelihood.REGRESSION == "regression"
    assert Likelihood.CLASSIFICATION.value == "classification"
    assert hash(Likelihood.REWARD_MODELING) == hash("reward_modeling")


def test_enum_lookup_iteration_and_membership():
    assert Likelihood("regression") is Likelihood.REGRESSION
    assert Likelihood(Likelihood.CLASSIFICATION) is Likelihood.CLASSIFICATION
    assert Likelihood["REWARD_MODELING"] is Likelihood.REWARD_MODELING

    custom = GaussianLikelihood()
    assert Likelihood(custom) is custom

    assert list(Likelihood) == [
        Likelihood.REGRESSION,
        Likelihood.CLASSIFICATION,
        Likelihood.REWARD_MODELING,
    ]
    assert len(Likelihood) == 3
    assert "classification" in Likelihood
    assert "foo" not in Likelihood

    with pytest.raises(ValueError, match="Invalid likelihood type"):
        Likelihood("foo")

    with pytest.raises(KeyError):
        Likelihood["FOO"]

    # The enum API belongs to `Likelihood`, not to its subclasses
    with pytest.raises(TypeError):
        len(GaussianLikelihood)

    with pytest.raises(TypeError):
        list(CategoricalLikelihood)


def test_base_class_stays_abstract():
    with pytest.raises(TypeError):
        Likelihood()

    # Subclasses are constructed normally, not looked up by value
    with pytest.raises(TypeError):
        GaussianLikelihood("regression")


def test_old_enum_import_path_and_pickle():
    from laplace.utils.enums import Likelihood as EnumsLikelihood

    assert EnumsLikelihood is Likelihood

    # How a member of the old `Likelihood(str, Enum)` was pickled
    old_member = b"claplace.utils.enums\nLikelihood\n(Vclassification\ntR."
    assert pickle.loads(old_member) is Likelihood.CLASSIFICATION

    assert pickle.loads(pickle.dumps(Likelihood.REGRESSION)) == Likelihood.REGRESSION


def test_subclass_equals_its_builtin_by_name():
    class Doubled(CategoricalLikelihood):
        def output_hessian(self, f, y=None):
            return 2 * super().output_hessian(f, y)

    assert Doubled() == Likelihood.CLASSIFICATION
    assert Likelihood.CLASSIFICATION == Doubled()
    assert Doubled() != Likelihood.REGRESSION
    assert len({Doubled(), CategoricalLikelihood()}) == 1
