"""
Integration test checking the correctness of the GP implementation in FunctionalLaplace.
"""

from itertools import product

import pytest
import torch

from laplace.baselaplace import FullLaplace, FunctionalLaplace
from laplace.curvature.asdl import AsdlGGN
from laplace.curvature.backpack import BackPackGGN
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.lllaplace import FullLLLaplace, FunctionalLLLaplace
from tests.utils import (
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
