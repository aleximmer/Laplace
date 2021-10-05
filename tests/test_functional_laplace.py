"""
Integration test checking the correctness of the full "naive" GP implementation.
"""
import numpy as np
import torch
from laplace.baselaplace import FullLaplace, FunctionalLaplace
from tests.utils import toy_model, toy_regression_dataset_1d, toy_classification_dataset

true_sigma_noise = 0.1
torch.manual_seed(711)

# torch.set_default_dtype(torch.float64)


def test_gp_equivalence_regression():
    X_train, y_train, train_loader, X_test = toy_regression_dataset_1d(sigma=true_sigma_noise,
                                                                       batch_size=60)
    model = toy_model(train_loader)

    full_la = FullLaplace(model, 'regression', sigma_noise=true_sigma_noise)
    functional_gp_la = FunctionalLaplace(model, 'regression', M=len(X_train), sigma_noise=true_sigma_noise)
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


def test_gp_equivalence_classification():
    X_train, y_train, train_loader, X_test = toy_classification_dataset(batch_size=60, in_dim=3, out_dim=2)
    model = toy_model(train_loader, in_dim=3, out_dim=2, regression=False)

    full_la = FullLaplace(model, 'classification')
    functional_gp_la = FunctionalLaplace(model, 'classification', M=len(X_train))
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    f_mu_full = f_mu_full.squeeze().detach().cpu().numpy()
    f_var_full = f_var_full.squeeze().detach().cpu().numpy()
    f_mu_gp = f_mu_gp.squeeze().detach().cpu().numpy()
    f_var_gp = f_var_gp.squeeze().detach().cpu().numpy()

    assert np.allclose(f_mu_full, f_mu_gp)
    assert np.allclose(f_var_full, f_var_gp)



