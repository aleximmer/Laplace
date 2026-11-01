"""
Integration test checking the correctness of the GP implementation in FunctionalLaplace.
"""

from collections import UserDict
from copy import deepcopy
from itertools import product

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace.baselaplace import FullLaplace, FunctionalLaplace
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.lllaplace import FullLLLaplace, FunctionalLLLaplace
from tests.utils import (
    AsdlGGN,
    BackPackGGN,
    ListDataset,
    dict_data_collator,
    requires_backpack,
    toy_classification_dataset,
    toy_model,
    toy_multivariate_regression_dataset,
    toy_regression_dataset_1d,
)

true_sigma_noise = 0.1

torch.manual_seed(711)
torch.set_default_dtype(torch.double)


@pytest.mark.parametrize(
    "laplace,independent_outputs",
    product(
        [(FullLaplace, FunctionalLaplace), (FullLLLaplace, FunctionalLLLaplace)],
        [True, False],
    ),
)
def test_gp_equivalence_regression(laplace, independent_outputs):
    X_train, y_train, train_loader, X_test = toy_regression_dataset_1d(
        sigma=true_sigma_noise, batch_size=60
    )
    M = len(X_train)
    model = toy_model(train_loader)

    parametric_laplace, functional_laplace = laplace
    full_la = parametric_laplace(
        model, "regression", sigma_noise=true_sigma_noise, prior_precision=2.0
    )
    functional_gp_la = functional_laplace(
        model,
        "regression",
        n_subset=M,
        sigma_noise=true_sigma_noise,
        independent_outputs=independent_outputs,
        prior_precision=2.0,
    )
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    assert torch.allclose(f_mu_full, f_mu_gp)
    # if float64 is used instead of float32, one can use atol=1e-10 in assert below
    assert torch.allclose(f_var_full, f_var_gp, atol=1e-2)


@pytest.mark.parametrize(
    "parametric_laplace,functional_laplace",
    [(FullLaplace, FunctionalLaplace), (FullLLLaplace, FunctionalLLLaplace)],
)
def test_gp_equivalence_regression_multivariate(
    parametric_laplace, functional_laplace, c=3
):
    X_train, y_train, train_loader, X_test = toy_multivariate_regression_dataset(
        sigma=true_sigma_noise, d_input=c, batch_size=60
    )
    model = toy_model(train_loader, in_dim=c, out_dim=c)

    full_la = parametric_laplace(
        model, "regression", sigma_noise=true_sigma_noise, prior_precision=2.0
    )
    functional_gp_la = functional_laplace(
        model,
        "regression",
        n_subset=len(X_train),
        sigma_noise=true_sigma_noise,
        independent_outputs=False,
        prior_precision=2.0,
    )
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    assert torch.allclose(f_mu_full, f_mu_gp)
    # if float64 is used instead of float32, one can use atol=1e-10 in assert below
    assert torch.allclose(f_var_full, f_var_gp, atol=1e-2)


@pytest.mark.parametrize(
    "laplace,independent_outputs,gp_backend",
    product(
        [(FullLaplace, FunctionalLaplace), (FullLLLaplace, FunctionalLLLaplace)],
        [True, False],
        [BackPackGGN, AsdlGGN, CurvlinopsGGN],
    ),
)
def test_gp_equivalence_classification(laplace, independent_outputs, gp_backend, c=2):
    X_train, y_train, train_loader, X_test = toy_classification_dataset(
        batch_size=60, in_dim=4, out_dim=c
    )
    model = toy_model(train_loader, in_dim=4, out_dim=c, regression=False)

    parametric_laplace, functional_laplace = laplace
    full_la = parametric_laplace(model, "classification", prior_precision=1.0)
    functional_gp_la = functional_laplace(
        model,
        "classification",
        n_subset=len(X_train),
        independent_outputs=independent_outputs,
        prior_precision=1.0,
        backend=gp_backend,
    )
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    p_full = full_la(X_test)
    p_gp = functional_gp_la(X_test)

    assert p_full.shape == p_gp.shape
    assert torch.allclose(torch.argmax(p_full, -1), torch.argmax(p_gp, -1))


@pytest.mark.parametrize("laplace", [FunctionalLaplace, FunctionalLLLaplace])
@pytest.mark.parametrize("passed", [{}, {"backend": None}])
def test_default_backend_is_curvlinops(laplace, passed):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )
    la = laplace(model, "regression", n_subset=5, **passed)

    assert la._backend_cls is CurvlinopsGGN


def test_accepts_curvlinops_subclass():
    class MyGGN(CurvlinopsGGN):
        pass

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )
    la = FunctionalLaplace(model, "regression", n_subset=5, backend=MyGGN)

    assert la._backend_cls is MyGGN


@requires_backpack
def test_backpack_dict_inputs_suggest_default_backend():
    class DictModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, data):
            return self.linear(data["input_ids"])

    data = [{"input_ids": torch.randn(3), "labels": torch.randn(2)} for _ in range(8)]
    loader = DataLoader(ListDataset(data), batch_size=4, collate_fn=dict_data_collator)
    la = FunctionalLaplace(DictModel(), "regression", n_subset=4, backend=BackPackGGN)

    with pytest.raises(ValueError, match="CurvlinopsGGN"):
        la.fit(loader)


class _DictInputModel(torch.nn.Module):
    """Linear model that reads its input from `data["input_ids"]`."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, data):
        return self.linear(data["input_ids"])


@pytest.mark.parametrize("independent_outputs", [False, True])
def test_dict_inputs_match_tensor_inputs(independent_outputs):
    torch.manual_seed(711)
    X, y = torch.randn(8, 3), torch.randn(8, 2)
    linear = torch.nn.Linear(3, 2)

    tensor_loader = DataLoader(TensorDataset(X, y), batch_size=4)
    dict_loader = DataLoader(
        ListDataset([{"input_ids": x, "labels": t} for x, t in zip(X, y)]),
        batch_size=4,
        collate_fn=dict_data_collator,
    )

    kwargs = dict(n_subset=6, independent_outputs=independent_outputs)
    la_tensor = FunctionalLaplace(deepcopy(linear), "regression", **kwargs)
    la_dict = FunctionalLaplace(
        _DictInputModel(deepcopy(linear)), "regression", **kwargs
    )
    la_tensor.fit(tensor_loader)
    la_dict.fit(dict_loader)

    mean_t, var_t = la_tensor(X[:3])
    mean_d, var_d = la_dict(UserDict({"input_ids": X[:3]}))

    assert torch.allclose(mean_d, mean_t)
    assert torch.allclose(var_d, var_t)

    _, cov_t = la_tensor(X[:3], joint=True)
    _, cov_d = la_dict(UserDict({"input_ids": X[:3]}), joint=True)

    assert torch.allclose(cov_d, cov_t)
    assert torch.allclose(
        la_dict.log_marginal_likelihood(), la_tensor.log_marginal_likelihood()
    )


def test_sod_loader_keeps_collate_fn():
    loader = DataLoader(
        ListDataset([{"input_ids": torch.randn(3)} for _ in range(8)]),
        batch_size=4,
        collate_fn=dict_data_collator,
    )
    la = FunctionalLaplace(torch.nn.Linear(3, 2), "regression", n_subset=4)

    assert la._get_SoD_data_loader(loader).collate_fn is dict_data_collator
