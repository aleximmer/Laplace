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
    DoubledCategorical,
    ScaledGaussian,
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


def test_equality_includes_hyperparameters():
    assert ScaledGaussian(1.0) != ScaledGaussian(2.0)
    assert ScaledGaussian(2.0) == ScaledGaussian(2.0)
    assert ScaledGaussian(1.0) == "scaled_gaussian"
    assert GaussianLikelihood() == Likelihood.REGRESSION
    assert DoubledCategorical() == Likelihood.CLASSIFICATION
    assert GaussianLikelihood().hyperparameters() == {}


def test_equality_compares_tensor_hyperparameters():
    class Weighted(CategoricalLikelihood):
        def __init__(self, weight: torch.Tensor) -> None:
            self.weight = weight

        def hyperparameters(self) -> dict:
            return {"weight": self.weight}

    assert Weighted(torch.ones(2)) == Weighted(torch.ones(2))
    assert Weighted(torch.ones(2)) != Weighted(torch.zeros(2))
    assert Weighted(torch.ones(2)) != Weighted(torch.ones(3))
    assert Weighted(torch.ones(2)) != CategoricalLikelihood()


_CE_WEIGHT = torch.tensor([0.5, 2.0, 1.5], dtype=torch.float64)

_CE_OPTIONS = [
    {},
    {"weight": _CE_WEIGHT},
    {"ignore_index": 1},
    {"label_smoothing": 0.1},
    {"weight": _CE_WEIGHT, "ignore_index": 1, "label_smoothing": 0.1},
]


def _ce_targets(opts: dict, B: int = 4, L: int = 5, C: int = 3) -> list:
    """Index targets, with some ignored tokens if `ignore_index` is set, and
    probability targets if it is not, since torch ignores `ignore_index` for them."""
    y = torch.randint(C, (B, L))

    if "ignore_index" in opts:
        y[0, :2] = opts["ignore_index"]
        return [y]

    q = torch.softmax(torch.randn(B, L, C, dtype=torch.float64), dim=-1)

    return [y, q]


@pytest.mark.parametrize("opts", _CE_OPTIONS)
def test_categorical_options_mirror_cross_entropy(opts):
    torch.manual_seed(0)
    B, L, C = 4, 5, 3
    f = torch.randn(B, L, C, dtype=torch.float64)

    lik = CategoricalLikelihood(**opts)
    ce = nn.CrossEntropyLoss(**opts, reduction="sum")

    for y in _ce_targets(opts):
        y_flat = y.reshape(-1, C) if y.ndim == f.ndim else y.reshape(-1)
        assert torch.allclose(lik.loss(f, y), ce(f.reshape(-1, C), y_flat))


@pytest.mark.parametrize("opts", _CE_OPTIONS)
def test_categorical_option_gradient_and_hessian_match_autograd(opts):
    torch.manual_seed(0)
    B, L, C = 4, 5, 3
    f = torch.randn(B, L, C, dtype=torch.float64)

    lik = CategoricalLikelihood(**opts)
    ce = nn.CrossEntropyLoss(**opts, reduction="sum")
    ignore_index = opts.get("ignore_index", -100)

    for y in _ce_targets(opts):
        probs = y.ndim == f.ndim
        y_flat = y.reshape(-1, C) if probs else y.reshape(-1)

        grad = torch.func.grad(lambda f_: ce(f_.reshape(-1, C), y_flat))(f)
        assert torch.allclose(lik.output_gradient(f, y), grad)

        H_ref = torch.zeros(B * L, C, C, dtype=f.dtype)

        for t, (f_t, y_t) in enumerate(zip(f.reshape(-1, C), y_flat)):
            if not probs and y_t == ignore_index:
                continue

            def token_loss(f_: torch.Tensor, y_t: torch.Tensor = y_t) -> torch.Tensor:
                return ce(f_[None], y_t[None])

            H_ref[t] = torch.func.hessian(token_loss)(f_t)

        assert torch.allclose(lik.output_hessian(f, y), H_ref.reshape(B, L, C, C))


def test_categorical_n_observations_excludes_ignored():
    f = torch.randn(4, 5, 3)
    y = torch.zeros(4, 5, dtype=torch.long)
    y.view(-1)[:7] = 2

    assert CategoricalLikelihood(ignore_index=2).n_observations(f, y) == 13
    assert CategoricalLikelihood().n_observations(f, y) == 20

    # Torch ignores `ignore_index` for probability targets
    q = torch.softmax(f, dim=-1)
    assert CategoricalLikelihood(ignore_index=2).n_observations(f, q) == 20


def test_categorical_hyperparameters_are_non_default():
    w = torch.tensor([1.0, 2.0])

    assert CategoricalLikelihood().hyperparameters() == {}
    assert CategoricalLikelihood(weight=None, ignore_index=-100).hyperparameters() == {}
    assert CategoricalLikelihood(label_smoothing=0.1).hyperparameters() == {
        "label_smoothing": 0.1
    }
    assert CategoricalLikelihood(ignore_index=1).hyperparameters() == {
        "ignore_index": 1
    }
    assert CategoricalLikelihood(weight=w).hyperparameters()["weight"] is w

    assert CategoricalLikelihood(ignore_index=1) != Likelihood.CLASSIFICATION
    assert CategoricalLikelihood(ignore_index=1) == "classification"
    assert CategoricalLikelihood(weight=w) == CategoricalLikelihood(weight=w.clone())
    assert CategoricalLikelihood(weight=w) != CategoricalLikelihood(weight=2 * w)
    assert BradleyTerryLikelihood().fit_likelihood == Likelihood.CLASSIFICATION

    # Tensors on different devices compare by value instead of raising
    assert CategoricalLikelihood(weight=w) == CategoricalLikelihood(
        weight=w.to("cpu", copy=True)
    )
    assert CategoricalLikelihood(weight=w) != CategoricalLikelihood(weight=w.to("meta"))

    if torch.cuda.is_available():
        assert CategoricalLikelihood(weight=w) == CategoricalLikelihood(weight=w.cuda())


def test_weight_dtype_and_device_follow_logits():
    torch.manual_seed(0)
    f = torch.randn(4, 5, 3, dtype=torch.float64)
    y = torch.randint(3, (4, 5))
    w = torch.tensor([0.5, 2.0, 1.5])

    lik = CategoricalLikelihood(weight=w, label_smoothing=0.1)
    ref = CategoricalLikelihood(weight=w.double(), label_smoothing=0.1)

    for method in ("loss", "output_gradient", "output_hessian"):
        out = getattr(lik, method)(f, y)

        assert out.dtype == torch.float64
        assert torch.allclose(out, getattr(ref, method)(f, y))

    # The stored weight stays as given
    assert lik.weight is w


def test_get_likelihood_from_criterion():
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert type(get_likelihood(nn.MSELoss())) is GaussianLikelihood

    criterion = nn.CrossEntropyLoss(
        weight=w, ignore_index=3, label_smoothing=0.2, reduction="mean"
    )
    lik = get_likelihood(criterion)

    assert type(lik) is CategoricalLikelihood
    assert lik == CategoricalLikelihood(weight=w, ignore_index=3, label_smoothing=0.2)
    assert type(get_likelihood(nn.CrossEntropyLoss())) is CategoricalLikelihood
    assert get_likelihood(nn.CrossEntropyLoss()) == Likelihood.CLASSIFICATION

    # The enum-style lookup accepts criteria too
    assert Likelihood(nn.CrossEntropyLoss()) is Likelihood.CLASSIFICATION
    assert Likelihood(criterion) == lik

    with pytest.raises(ValueError, match="Invalid likelihood type"):
        get_likelihood(nn.L1Loss())


def test_categorical_curvature_checks_targets():
    lik = CategoricalLikelihood()
    f, y = torch.randn(4, 3, 2, 5), torch.randint(5, (4, 2, 3))

    with pytest.raises(ValueError, match="targets have shape"):
        lik.output_hessian(f, y)

    with pytest.raises(ValueError, match="targets have shape"):
        lik.output_gradient(f, y)


def test_categorical_subclass_without_super_init_has_default_options():
    class Custom(CategoricalLikelihood):
        def __init__(self, scale: float = 1.0) -> None:
            self.scale = scale

    f, y = torch.randn(4, 3), torch.randint(3, (4,))

    assert Custom().hyperparameters() == {}
    assert torch.allclose(Custom().loss(f, y), CategoricalLikelihood().loss(f, y))


@pytest.mark.parametrize(
    "lik,f_shape,y_shape,n_obs",
    [
        # `Normal(f)`-style targets with the shape of the outputs
        (DistributionOnlyGaussian(), (1, 2), (7, 2), 7),
        (DistributionOnlyGaussian(), (1, 4, 2), (7, 4, 2), 28),
        # `Categorical(logits=f)`-style index targets
        (DistributionOnlyCategorical(), (1, 3), (7,), 7),
        (DistributionOnlyCategorical(), (1, 4, 3), (7, 4), 28),
        # One Bernoulli per logit, with targets of the outputs' shape
        (DistributionOnlyBernoulli(), (1, 3), (7, 3), 7),
    ],
)
def test_default_n_observations_counts_targets(lik, f_shape, y_shape, n_obs):
    # The outputs may come from a single datum, as in `ParametricLaplace.fit`
    assert lik.n_observations(torch.randn(f_shape), torch.zeros(y_shape)) == n_obs


def test_builtin_n_observations_unchanged():
    f = torch.randn(1, 4, 3)

    assert GaussianLikelihood().n_observations(f, torch.randn(7, 4, 3)) == 28
    assert CategoricalLikelihood().n_observations(f, torch.zeros(7, 4).long()) == 28
    assert BradleyTerryLikelihood().n_observations(f, torch.zeros(7).long()) == 7


def test_categorical_validates_options():
    with pytest.raises(ValueError, match="1-D"):
        CategoricalLikelihood(weight=torch.ones(2, 3))

    with pytest.raises(ValueError, match="1-D"):
        CategoricalLikelihood(weight=torch.tensor(1.0))

    for label_smoothing in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CategoricalLikelihood(label_smoothing=label_smoothing)

    # The bounds are fine, as in torch
    CategoricalLikelihood(label_smoothing=0.0)
    CategoricalLikelihood(label_smoothing=1.0)


def test_repr_shows_options():
    assert repr(CategoricalLikelihood()) == "CategoricalLikelihood()"
    assert (
        repr(CategoricalLikelihood(label_smoothing=0.1))
        == "CategoricalLikelihood(label_smoothing=0.1)"
    )
    assert repr(CategoricalLikelihood(weight=torch.ones(3), ignore_index=2)) == (
        "CategoricalLikelihood(weight=tensor(shape=(3,)), ignore_index=2)"
    )
    assert repr(ScaledGaussian(3.0)) == "ScaledGaussian(scale=3.0)"
    assert repr(GaussianLikelihood()) == "GaussianLikelihood()"
