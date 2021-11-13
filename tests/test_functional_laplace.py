"""
Integration test checking the correctness of the full "naive" GP implementation.
"""
import numpy as np
import torch
from laplace.baselaplace import FullLaplace, FunctionalLaplace
from tests.utils import toy_model, toy_regression_dataset_1d, toy_multivariate_regression_dataset, \
    toy_classification_dataset

true_sigma_noise = 0.1
torch.manual_seed(711)

# torch.set_default_dtype(torch.float64)


def test_gp_equivalence_regression():
    X_train, y_train, train_loader, X_test = toy_regression_dataset_1d(sigma=true_sigma_noise,
                                                                       batch_size=60)
    M = len(X_train)
    model = toy_model(train_loader)

    full_la = FullLaplace(model, 'regression', sigma_noise=true_sigma_noise, prior_precision=2.)
    functional_gp_la = FunctionalLaplace(model, 'regression', M=M,
                                         sigma_noise=true_sigma_noise, diagonal_kernel=False, prior_precision=2.)
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    f_mu_full = f_mu_full.squeeze().detach().cpu().numpy()
    f_var_full = f_var_full.squeeze().detach().cpu().numpy()
    f_mu_gp = f_mu_gp.squeeze().detach().cpu().numpy()
    f_var_gp = f_var_gp.squeeze().detach().cpu().numpy()

    assert np.allclose(f_mu_full, f_mu_gp)
    # if float64 is used instead of float32, one can use atol=1e-10 in assert below
    # print(np.max(np.abs(f_var_gp - f_var_full)))
    assert np.allclose(f_var_full, f_var_gp, atol=1e-2)


def test_gp_equivalence_regression_multivariate(c=3):
    X_train, y_train, train_loader, X_test = toy_multivariate_regression_dataset(sigma=true_sigma_noise,
                                                                                 d_input=c,
                                                                                 batch_size=60)
    model = toy_model(train_loader, in_dim=c, out_dim=c)

    full_la = FullLaplace(model, 'regression', sigma_noise=true_sigma_noise, prior_precision=2.0)
    functional_gp_la = FunctionalLaplace(model, 'regression', M=len(X_train),
                                         sigma_noise=true_sigma_noise, diagonal_kernel=False, prior_precision=2.0)
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    f_mu_full = f_mu_full.squeeze().detach().cpu().numpy()
    f_var_full = f_var_full.squeeze().detach().cpu().numpy()
    f_mu_gp = f_mu_gp.squeeze().detach().cpu().numpy()
    f_var_gp = f_var_gp.squeeze().detach().cpu().numpy()

    assert np.allclose(f_mu_full, f_mu_gp)
    # if float64 is used instead of float32, one can use atol=1e-10 in assert below
    assert np.allclose(f_var_full, f_var_gp, atol=1e-2)


def test_gp_equivalence_classification(c=2):
    X_train, y_train, train_loader, X_test = toy_classification_dataset(batch_size=60, in_dim=4, out_dim=c)
    model = toy_model(train_loader, in_dim=4, out_dim=c, regression=False)

    full_la = FullLaplace(model, 'classification', prior_precision=1.0)
    functional_gp_la = FunctionalLaplace(model, 'classification', M=len(X_train),
                                         diagonal_kernel=True, prior_precision=1.0)
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    p_full = full_la(X_test)
    p_gp = functional_gp_la(X_test)

    p_full = p_full.squeeze().detach().cpu().numpy()
    p_gp = p_gp.squeeze().detach().cpu().numpy()

    # difference due to the approximation with diagonal_L=True
    # diffs = np.abs(p_full - p_gp)
    # print(diffs.mean())
    # print(diffs.max())
    # print(f"{(np.argmax(p_full, axis=1) == np.argmax(p_gp, axis=1)).sum()} out of {p_full.shape[0]}")

    assert p_full.shape == p_gp.shape
    assert np.array_equal(np.argmax(p_full, axis=1), np.argmax(p_gp, axis=1))





