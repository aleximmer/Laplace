import pickle
import warnings
from math import log, pi, sqrt

import pytest
import torch
from torch import nn

from laplace import (
    BernoulliLikelihood,
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
    bce_options,
)


@pytest.mark.parametrize(
    "cls",
    [
        GaussianLikelihood,
        CategoricalLikelihood,
        BradleyTerryLikelihood,
        BernoulliLikelihood,
    ],
)
def test_get_likelihood_returns_instances_as_is(cls):
    inst = cls()
    assert get_likelihood(inst) is inst


@pytest.mark.parametrize(
    "bad",
    [
        "foo",
        "bernoulli",
        "heteroscedastic_gaussian",
        "gaussian",
        "categorical",
        "bradley_terry",
        GaussianLikelihood,
        3,
        None,
    ],
)
def test_get_likelihood_invalid(bad):
    with pytest.raises(ValueError, match="Invalid likelihood type"):
        get_likelihood(bad)


def test_eq_and_hash():
    lik = GaussianLikelihood()

    assert lik == GaussianLikelihood()
    assert lik != CategoricalLikelihood()
    assert lik != 3
    assert hash(lik) == hash(GaussianLikelihood())
    assert str(lik) == "gaussian"


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


@pytest.mark.parametrize("shape", [(4,), (4, 2)])
def test_integer_one_hot_targets_raise(shape):
    # As in `nn.CrossEntropyLoss`, class probabilities must be floats. Integer
    # targets with the shape of the logits would be read as class indices otherwise.
    f = torch.randn(*shape, 3)
    y = nn.functional.one_hot(torch.randint(3, shape), 3)
    lik = CategoricalLikelihood()

    with pytest.raises(ValueError, match=r"\.float\(\)"):
        lik.check_targets(f, y)

    with pytest.raises(ValueError, match=r"\.float\(\)"):
        lik.loss(f, y)

    lik.check_targets(f, y.float())


def test_n_observations():
    assert (
        GaussianLikelihood().n_observations(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        == 6
    )

    cat = CategoricalLikelihood()
    assert cat.n_observations(torch.randn(2, 3, 5), torch.randint(5, (2, 3))) == 6
    assert cat.n_observations(torch.randn(2, 5), torch.rand(2, 5)) == 2

    # One observation per pair, with index or probability targets
    bt = BradleyTerryLikelihood()
    assert bt.n_observations(torch.randn(3, 2), torch.tensor([0, 1, -100])) == 2
    assert bt.n_observations(torch.randn(3, 2), torch.rand(3, 2)) == 3


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

    assert lik.fit_likelihood == CategoricalLikelihood()
    assert lik.predictive_likelihood == GaussianLikelihood()
    assert lik.is_gaussian is False
    assert lik.name == "bradley_terry"


def test_enum_api_is_not_inherited():
    # The deprecated enum API belongs to `Likelihood`, not to its subclasses
    with pytest.raises(TypeError):
        len(GaussianLikelihood)

    with pytest.raises(TypeError):
        list(CategoricalLikelihood)


def test_base_class_stays_abstract():
    with pytest.raises(TypeError):
        Likelihood()

    # Subclasses are constructed normally, not looked up by value
    with pytest.raises(TypeError):
        GaussianLikelihood("gaussian")


def test_old_enum_import_path_and_pickle():
    from laplace.utils.enums import Likelihood as EnumsLikelihood

    assert EnumsLikelihood is Likelihood

    assert pickle.loads(pickle.dumps(GaussianLikelihood())) == GaussianLikelihood()


def test_subclass_equals_its_builtin_by_name():
    class Doubled(CategoricalLikelihood):
        def output_hessian(self, f, y=None):
            return 2 * super().output_hessian(f, y)

    assert Doubled() == CategoricalLikelihood()
    assert CategoricalLikelihood() == Doubled()
    assert Doubled() != GaussianLikelihood()
    assert len({Doubled(), CategoricalLikelihood()}) == 1


def test_equality_includes_hyperparameters():
    assert ScaledGaussian(1.0) != ScaledGaussian(2.0)
    assert ScaledGaussian(2.0) == ScaledGaussian(2.0)
    assert GaussianLikelihood() == GaussianLikelihood()
    assert DoubledCategorical() == CategoricalLikelihood()
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
    """Index targets, with some ignored positions if `ignore_index` is set, and
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

            def position_loss(
                f_: torch.Tensor, y_t: torch.Tensor = y_t
            ) -> torch.Tensor:
                return ce(f_[None], y_t[None])

            H_ref[t] = torch.func.hessian(position_loss)(f_t)

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

    assert CategoricalLikelihood(ignore_index=1) != CategoricalLikelihood()
    assert CategoricalLikelihood(weight=w) == CategoricalLikelihood(weight=w.clone())
    assert CategoricalLikelihood(weight=w) != CategoricalLikelihood(weight=2 * w)
    assert BradleyTerryLikelihood().fit_likelihood == CategoricalLikelihood()

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
    assert get_likelihood(nn.CrossEntropyLoss()) == CategoricalLikelihood()

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


_BCE = bce_options(4)

_BCE_SHAPES = [(6, 4), (3, 5, 4)]


def _bce_targets(shape: tuple[int, ...]) -> list[torch.Tensor]:
    """Hard and soft targets of the given shape."""
    y = torch.rand(*shape, dtype=torch.float64)

    return [(y > 0.5).to(torch.float64), y]


@pytest.mark.parametrize("opts", _BCE.values(), ids=_BCE.keys())
@pytest.mark.parametrize("shape", _BCE_SHAPES)
def test_bernoulli_options_mirror_bce_with_logits(opts, shape):
    f = torch.randn(*shape, dtype=torch.float64)

    lik = BernoulliLikelihood(**opts)
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()

    for y in _bce_targets(shape):
        assert torch.allclose(lik.loss(f, y), bce(f, y))
        assert torch.allclose(lik.loss_fn()(f, y), bce(f, y))

        # The float32 options of the likelihood stay as given
        assert lik.loss_fn().double()(f, y).dtype == torch.float64


@pytest.mark.parametrize("opts", _BCE.values(), ids=_BCE.keys())
@pytest.mark.parametrize("shape", _BCE_SHAPES)
def test_bernoulli_gradient_and_hessian_match_autograd(opts, shape):
    f = torch.randn(*shape, dtype=torch.float64)

    lik = BernoulliLikelihood(**opts)
    bce = nn.BCEWithLogitsLoss(**opts, reduction="sum").double()

    for y in _bce_targets(shape):
        grad = torch.func.grad(lambda f_: bce(f_, y))(f)
        assert torch.allclose(lik.output_gradient(f, y), grad)

        # Every output has its own loss term, so the Hessian is diagonal
        hess = torch.func.hessian(lambda f_: bce(f_, y))(f)
        H = lik.output_hessian(f, y)

        assert H.shape == (*shape, shape[-1])
        assert torch.allclose(
            torch.diagonal(H, dim1=-2, dim2=-1),
            torch.diagonal(hess.reshape(f.numel(), f.numel())).reshape(shape),
        )
        assert torch.allclose(H, torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1)))


@pytest.mark.parametrize("shape", _BCE_SHAPES)
def test_bernoulli_default_gradient_and_hessian(shape):
    f = torch.randn(*shape, dtype=torch.float64)
    lik = BernoulliLikelihood()
    p = torch.sigmoid(f)

    for y in _bce_targets(shape):
        assert torch.allclose(lik.output_gradient(f, y), p - y)

    # Without targets, the Hessian has no weights, and it is the exact Fisher of the
    # Bernoulli distribution that only defines its `distribution`
    H = lik.output_hessian(f)

    assert torch.allclose(H, torch.diag_embed(p * (1 - p)))
    assert torch.allclose(H, DistributionOnlyBernoulli().output_hessian(f))
    assert torch.allclose(lik.output_hessian(f, _bce_targets(shape)[1]), H)

    # Options scale the Hessian only where it is given the targets
    weighted = BernoulliLikelihood(weight=torch.tensor(2.0))
    assert torch.allclose(weighted.output_hessian(f), H)


def test_bernoulli_hessian_is_the_fisher_of_the_weighted_loss():
    f = torch.randn(5, 4, dtype=torch.float64)
    lik = BernoulliLikelihood(
        weight=torch.tensor([0.5, 2.0, 1.5, 1.0]),
        pos_weight=torch.tensor([3.0, 0.5, 2.0, 1.0]),
    )
    p = torch.sigmoid(f)

    # The expected Hessian over y ~ Bernoulli(p) is E[a] w p (1 - p), with
    # a = p_w y + 1 - y, and the Hessian does not depend on y otherwise
    H = sum(
        w * lik.output_hessian(f, torch.full_like(f, value)).diagonal(dim1=-2, dim2=-1)
        for w, value in ((1 - p, 0.0), (p, 1.0))
    )
    pos = lik.pos_weight.double()
    expected = lik.weight.double() * (1 - p + pos * p) * p * (1 - p)

    assert torch.allclose(H, expected)


def test_bernoulli_check_targets():
    lik = BernoulliLikelihood()
    f = torch.randn(4, 3)

    lik.check_targets(f, torch.rand(4, 3))

    with pytest.raises(ValueError, match="targets of shape"):
        lik.check_targets(f, torch.rand(4))

    with pytest.raises(ValueError, match="targets of shape"):
        lik.check_targets(f, torch.rand(4, 1))

    with pytest.raises(ValueError, match="floating-point"):
        lik.check_targets(f, torch.ones(4, 3, dtype=torch.long))

    # Every entry point checks them, like torch
    bad = torch.rand(4, 2)

    for call in (
        lambda: lik.loss(f, bad),
        lambda: lik.output_gradient(f, bad),
        lambda: lik.output_hessian(f, bad),
    ):
        with pytest.raises(ValueError, match="targets of shape"):
            call()


def test_bernoulli_validates_options():
    BernoulliLikelihood(weight=torch.ones(3), pos_weight=torch.tensor(2.0))

    # A weight per example can't follow a shuffled data loader
    with pytest.raises(ValueError, match="weight must be a scalar or 1-D"):
        BernoulliLikelihood(weight=torch.ones(4, 3))

    with pytest.raises(ValueError, match="pos_weight must be a scalar or 1-D"):
        BernoulliLikelihood(pos_weight=torch.ones(1, 1, 3))

    with pytest.raises(ValueError, match="pos_weight must be a scalar or 1-D"):
        get_likelihood(nn.BCEWithLogitsLoss(pos_weight=torch.ones(2, 3)))


def test_bernoulli_n_observations_counts_positions():
    lik = BernoulliLikelihood()

    assert lik.n_observations(torch.randn(1, 4), torch.rand(6, 4)) == 6
    assert lik.n_observations(torch.randn(1, 5, 4), torch.rand(3, 5, 4)) == 15
    assert lik.loss_factor == 1.0
    assert lik.log_normalizer(10, 4, torch.tensor(1.0)) == 0
    assert lik.is_gaussian is False
    assert lik.fit_likelihood is lik and lik.predictive_likelihood is lik


def test_bernoulli_sample_and_inverse_link():
    f = torch.randn(2000, 3, dtype=torch.float64)
    lik = BernoulliLikelihood(pos_weight=torch.tensor(5.0))

    y = lik.sample(f, generator=torch.Generator().manual_seed(0))
    y2 = lik.sample(f, generator=torch.Generator().manual_seed(0))

    assert y.shape == f.shape and y.dtype == f.dtype
    assert torch.equal(y, y2)
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    assert torch.allclose(lik.inverse_link(f), torch.sigmoid(f))

    # The options leave the predictive unchanged
    assert y.mean() == pytest.approx(torch.sigmoid(f).mean().item(), abs=0.03)
    assert isinstance(lik.distribution(f), torch.distributions.Bernoulli)
    assert torch.equal(lik.distribution(f).probs, torch.sigmoid(f))

    # The global generator is used without one
    torch.manual_seed(3)
    y3 = lik.sample(f)
    torch.manual_seed(3)

    assert torch.equal(y3, lik.sample(f))


def test_bernoulli_options_follow_logits_dtype():
    f = torch.randn(4, 5, 3, dtype=torch.float64)
    y = torch.rand(4, 5, 3, dtype=torch.float64)
    w, pw = torch.tensor([0.5, 2.0, 1.5]), torch.tensor([3.0, 0.5, 2.0])

    lik = BernoulliLikelihood(weight=w, pos_weight=pw)
    ref = BernoulliLikelihood(weight=w.double(), pos_weight=pw.double())

    for method in ("loss", "output_gradient", "output_hessian"):
        out = getattr(lik, method)(f, y)

        assert out.dtype == torch.float64
        assert torch.equal(out, getattr(ref, method)(f, y))

    # Targets of another float dtype than the logits
    assert lik.output_gradient(f, y.float()).dtype == torch.float64

    # The stored options stay as given
    assert lik.weight is w and lik.pos_weight is pw


def test_bernoulli_hyperparameters_and_equality():
    w, pw = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])

    assert BernoulliLikelihood().hyperparameters() == {}
    assert BernoulliLikelihood(weight=w).hyperparameters()["weight"] is w
    assert set(BernoulliLikelihood(weight=w, pos_weight=pw).hyperparameters()) == {
        "weight",
        "pos_weight",
    }

    assert BernoulliLikelihood() == BernoulliLikelihood()
    assert BernoulliLikelihood() != CategoricalLikelihood()
    assert BernoulliLikelihood(pos_weight=pw) != BernoulliLikelihood()
    assert BernoulliLikelihood(pos_weight=pw) == BernoulliLikelihood(
        pos_weight=pw.clone()
    )
    assert BernoulliLikelihood(pos_weight=pw) != BernoulliLikelihood(weight=pw)
    assert BernoulliLikelihood(pos_weight=pw) != BernoulliLikelihood(pos_weight=2 * pw)
    assert hash(BernoulliLikelihood()) == hash("bernoulli")
    assert str(BernoulliLikelihood()) == "bernoulli"

    assert "weight=tensor(shape=(2,))" in repr(BernoulliLikelihood(weight=w))
    assert repr(BernoulliLikelihood()) == "BernoulliLikelihood()"


def test_get_bernoulli_likelihood_from_criterion():
    w, pw = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])

    lik = get_likelihood(
        nn.BCEWithLogitsLoss(weight=w, pos_weight=pw, reduction="mean")
    )

    assert type(lik) is BernoulliLikelihood
    assert lik == BernoulliLikelihood(weight=w, pos_weight=pw)
    assert get_likelihood(nn.BCEWithLogitsLoss()) == BernoulliLikelihood()

    # Not `nn.BCELoss`, which takes probabilities, and not subclasses
    with pytest.raises(ValueError, match="Invalid likelihood type"):
        get_likelihood(nn.BCELoss())

    class Subclass(nn.BCEWithLogitsLoss):
        pass

    with pytest.raises(ValueError, match="Invalid likelihood type"):
        get_likelihood(Subclass())


def test_bernoulli_subclass_without_super_init_has_default_options():
    class NoInit(BernoulliLikelihood):
        def __init__(self) -> None:
            pass

    f, y = torch.randn(4, 3), torch.rand(4, 3)
    lik = NoInit()

    assert lik.hyperparameters() == {}
    assert torch.allclose(lik.loss(f, y), BernoulliLikelihood().loss(f, y))


def test_bernoulli_is_pickleable():
    lik = BernoulliLikelihood(pos_weight=torch.tensor([2.0, 3.0]))

    assert pickle.loads(pickle.dumps(lik)) == lik
    assert pickle.loads(pickle.dumps(BernoulliLikelihood())) == BernoulliLikelihood()


SIGMA = torch.tensor([0.5, 2.0, 1.3])


def test_gaussian_sigma_noise_validation():
    assert torch.equal(GaussianLikelihood(sigma_noise=SIGMA).sigma_noise, SIGMA)
    assert GaussianLikelihood().sigma_noise is None

    for bad in (
        torch.tensor(1.0),
        torch.ones(2, 2),
        [0.5, 2.0],
        1.0,
        torch.ones(2).int(),
    ):
        with pytest.raises(ValueError, match="1-D tensor"):
            GaussianLikelihood(sigma_noise=bad)

    for bad in (
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.5, -1.0]),
        torch.tensor([0.5, float("nan")]),
        torch.tensor([0.5, float("inf")]),
    ):
        with pytest.raises(ValueError, match="positive and finite"):
            GaussianLikelihood(sigma_noise=bad)


def test_gaussian_sigma_noise_is_a_hyperparameter():
    lik = GaussianLikelihood(sigma_noise=SIGMA)

    assert GaussianLikelihood().hyperparameters() == {}
    assert torch.equal(lik.hyperparameters()["sigma_noise"], SIGMA)

    assert lik == GaussianLikelihood(sigma_noise=SIGMA.clone())
    assert lik != GaussianLikelihood(sigma_noise=SIGMA * 2)
    assert lik != GaussianLikelihood()
    assert lik != GaussianLikelihood(sigma_noise=SIGMA[:2])

    assert "sigma_noise=tensor(shape=(3,))" in repr(lik)

    restored = pickle.loads(pickle.dumps(lik))
    assert restored == lik
    assert torch.equal(restored.sigma_noise, SIGMA)


def test_gaussian_sigma_noise_without_super_init():
    class NoInit(GaussianLikelihood):
        def __init__(self) -> None:
            pass

    assert NoInit().sigma_noise is None
    assert NoInit().hyperparameters() == {}


@pytest.mark.parametrize("shape", [(6, 3), (4, 5, 3)])
def test_gaussian_sigma_noise_quantities(shape):
    f, y = torch.randn(shape), torch.randn(shape)
    lik = GaussianLikelihood(sigma_noise=SIGMA)
    plain = GaussianLikelihood()

    # The loss times its factor is the negative log likelihood without the normalizer
    weighted = (((f - y) / SIGMA) ** 2).sum()
    assert torch.allclose(lik.loss(f, y), weighted)
    assert torch.allclose(lik.loss_fn("mean")(f, y), weighted / f.numel())
    assert lik.loss_factor == plain.loss_factor == 0.5

    # The plain likelihood is untouched
    assert torch.allclose(plain.loss(f, y), ((f - y) ** 2).sum())

    n_obs = lik.n_observations(f, y)
    nll = -lik.distribution(f).log_prob(y).sum()
    normalizer = lik.log_normalizer(n_obs, 3, torch.tensor(1.0))

    assert torch.allclose(lik.loss_factor * lik.loss(f, y) + normalizer, nll)
    expected = n_obs * sum(log(s * sqrt(2 * pi)) for s in SIGMA.tolist())
    assert torch.allclose(normalizer, torch.tensor(expected))

    H = lik.output_hessian(f)
    assert H.shape == (*shape, 3)
    assert torch.allclose(H, torch.diag(1 / SIGMA**2).expand(*shape[:-1], 3, 3))

    # The gradient and the Hessian are those of the negative log likelihood
    f_ = f.clone().requires_grad_()
    (grad,) = torch.autograd.grad(-lik.distribution(f_).log_prob(y).sum(), f_)
    assert torch.allclose(lik.output_gradient(f, y), grad)

    f0, y0 = f.reshape(-1, 3)[0], y.reshape(-1, 3)[0]
    hessian = torch.autograd.functional.hessian(
        lambda v: -lik.distribution(v).log_prob(y0).sum(), f0
    )
    assert torch.allclose(H.reshape(-1, 3, 3)[0], hessian)


def test_gaussian_sigma_noise_sample_and_dtype():
    f = torch.zeros(20000, 3, dtype=torch.float64)
    lik = GaussianLikelihood(sigma_noise=SIGMA)

    y = lik.sample(f, generator=torch.Generator().manual_seed(0))

    assert y.dtype == torch.float64
    assert torch.allclose(y.std(0), SIGMA.double(), rtol=0.05)

    noise = torch.randn(
        f.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    assert torch.allclose(y, SIGMA.double() * noise)

    # The vector follows the dtype of the outputs
    assert lik.output_hessian(f).dtype == torch.float64
    assert lik.loss(f, y).dtype == torch.float64


def test_gaussian_sigma_noise_loss_module_follows_the_model():
    lik = GaussianLikelihood(sigma_noise=SIGMA)
    loss_fn = lik.loss_fn().to(dtype=torch.float64)

    f, y = torch.randn(5, 3).double(), torch.randn(5, 3).double()

    assert torch.allclose(loss_fn(f, y), lik.loss(f, y))
    assert not loss_fn.state_dict()


def test_gaussian_sigma_noise_wrong_number_of_outputs():
    lik = GaussianLikelihood(sigma_noise=SIGMA)

    with pytest.raises(ValueError, match="2 outputs"):
        lik.check_targets(torch.randn(5, 2), torch.randn(5, 2))

    with pytest.raises(ValueError, match="2 outputs"):
        lik.loss(torch.randn(5, 2), torch.randn(5, 2))

    GaussianLikelihood().check_targets(torch.randn(5, 2), torch.randn(5, 2))


def test_gaussian_equal_sigma_noise_matches_the_scalar_normalizer():
    lik = GaussianLikelihood(sigma_noise=torch.full((3,), 0.7))
    plain = GaussianLikelihood()

    assert torch.allclose(
        lik.log_normalizer(7, 3, torch.tensor(1.0)),
        plain.log_normalizer(7, 3, torch.tensor(0.7)),
    )
