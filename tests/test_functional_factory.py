"""Factory and cross-estimator function-space contracts."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from laplace import ELLA, FunctionalLaplace, Laplace, VaLLA
from laplace.likelihoods import (
    BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
    Likelihood,
)
from tests.utils import lik_id


@pytest.fixture
def data():
    torch.manual_seed(7)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    return (x, DataLoader(TensorDataset(x, y), batch_size=2), model)


def make_ella(model, likelihood=CategoricalLikelihood(), **kwargs):
    options = {"subsample_size": 2, "n_eigenvalues": 1}
    options.update(kwargs)
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="nystrom",
        **options,
    )


def make_valla(model, likelihood=CategoricalLikelihood(), **kwargs):
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        **kwargs,
    )


def test_factory_and_unsupported_combinations(data):
    from laplace import FunctionalApproximation

    assert FunctionalApproximation.NYSTROM.value == "nystrom"
    x, loader, model = data
    assert isinstance(make_ella(model), ELLA)
    assert isinstance(make_valla(model, inducing_locations=x[:2]), VaLLA)
    assert isinstance(
        Laplace(model, CategoricalLikelihood(), "all", "gp", n_subset=2),
        FunctionalLaplace,
    )
    with pytest.raises(ValueError, match="requires hessian_structure"):
        Laplace(
            model,
            CategoricalLikelihood(),
            "all",
            "diag",
            functional_approximation="nystrom",
        )
    with pytest.raises(ValueError, match="subset_of_weights='all'"):
        Laplace(
            model,
            CategoricalLikelihood(),
            "last_layer",
            "gp",
            functional_approximation="variational",
        )
    with pytest.raises(ValueError, match="functional_approximation"):
        Laplace(
            model,
            CategoricalLikelihood(),
            "all",
            "gp",
            functional_approximation="other",
        )


def test_functional_integer_options_and_learning_rate_validate_early(data):
    x, loader, model = data
    with pytest.raises(ValueError, match="n_eigenvalues"):
        make_ella(model, subsample_size=1.5)
    with pytest.raises(ValueError, match="num_inducing"):
        make_valla(model, inducing_locations="random", num_inducing=1.5)
    estimator = make_valla(model, inducing_locations=x[:2].clone())
    with pytest.raises(ValueError, match="iterations"):
        estimator.fit(loader, iterations=1.5)
    with pytest.raises(ValueError, match="lr"):
        estimator.fit(loader, iterations=1, lr=float("inf"))


def test_predictive_alias_preserves_argument_order_for_weight_space(data):
    x, loader, model = data
    estimator = Laplace(model, CategoricalLikelihood(), "all", "diag")
    estimator.fit(loader)
    torch.testing.assert_close(
        estimator.predictive(x[:2], "glm", "probit", 10),
        estimator(x[:2], pred_type="glm", link_approx="probit", n_samples=10),
    )


@pytest.mark.parametrize(
    "likelihood", [GaussianLikelihood(), CategoricalLikelihood()], ids=lik_id
)
def test_functional_factory_likelihood_object_equivalence(data, likelihood):
    x, loader, model = data
    if isinstance(likelihood, GaussianLikelihood):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        loader = DataLoader(TensorDataset(x, x.sum(-1, keepdim=True)), batch_size=2)
        as_object = GaussianLikelihood()
    else:
        as_object = CategoricalLikelihood()

    by_name = make_ella(deepcopy(model), likelihood)
    by_object = make_ella(deepcopy(model), as_object)

    assert type(by_name) is type(by_object)
    assert by_name.likelihood == by_object.likelihood
    assert isinstance(by_object.likelihood, Likelihood)

    by_name.fit(loader)
    by_object.fit(loader)

    torch.testing.assert_close(by_name(x), by_object(x))


@pytest.mark.parametrize(
    "likelihood",
    [
        BernoulliLikelihood(),
        BernoulliLikelihood(),
        BernoulliLikelihood(),
        torch.nn.BCEWithLogitsLoss(),
    ],
    ids=["name", "member", "object", "criterion"],
)
@pytest.mark.parametrize("subset_of_weights", ["all", "last_layer"])
def test_function_space_methods_accept_bernoulli(data, likelihood, subset_of_weights):
    x, loader, model = data

    la = Laplace(model, likelihood, subset_of_weights, "gp", n_subset=2)
    assert isinstance(la, FunctionalLaplace)
    assert la.likelihood.name == "bernoulli"

    if subset_of_weights == "all":
        ella = make_ella(model, likelihood)
        valla = make_valla(model, likelihood, inducing_locations=x[:2])

        assert isinstance(ella, ELLA)
        assert isinstance(valla, VaLLA)
        assert ella.likelihood.name == valla.likelihood.name == "bernoulli"


def test_bernoulli_options_for_function_space_methods(data):
    x, loader, model = data
    options = BernoulliLikelihood(pos_weight=torch.tensor([2.0, 1.0]))

    FunctionalLaplace(model, options, n_subset=2)
    Laplace(model, options, "last_layer", "gp", n_subset=2)

    assert make_ella(model, options).likelihood == options
    assert make_valla(model, options, inducing_locations=x[:2]).likelihood == options
