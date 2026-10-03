"""Factory and cross-estimator function-space contracts."""

import pytest
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from laplace import ELLA, FunctionalLaplace, Laplace, VaLLA

torch.set_num_threads(1)


@pytest.fixture
def data():
    torch.manual_seed(7)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    return (x, DataLoader(TensorDataset(x, y), batch_size=2), model)


def make_ella(model, likelihood="classification", **kwargs):
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


def make_valla(model, likelihood="classification", **kwargs):
    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        **kwargs,
    )


def test_factory_and_unsupported_combinations(data):
    from laplace.utils import FunctionalApproximation

    assert FunctionalApproximation.NYSTROM.value == "nystrom"
    x, loader, model = data
    assert isinstance(make_ella(model), ELLA)
    assert isinstance(make_valla(model, inducing_locations=x[:2]), VaLLA)
    assert isinstance(
        Laplace(model, "classification", "all", "gp", n_subset=2), FunctionalLaplace
    )
    with pytest.raises(ValueError, match="requires hessian_structure"):
        Laplace(
            model, "classification", "all", "diag", functional_approximation="nystrom"
        )
    with pytest.raises(ValueError, match="subset_of_weights='all'"):
        Laplace(
            model,
            "classification",
            "last_layer",
            "gp",
            functional_approximation="variational",
        )
    with pytest.raises(ValueError, match="functional_approximation"):
        Laplace(model, "classification", "all", "gp", functional_approximation="other")


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
    estimator = Laplace(model, "classification", "all", "diag")
    estimator.fit(loader)
    torch.testing.assert_close(
        estimator.predictive(x[:2], "glm", "probit", 10),
        estimator(x[:2], pred_type="glm", link_approx="probit", n_samples=10),
    )
