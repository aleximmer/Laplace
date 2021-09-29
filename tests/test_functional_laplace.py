"""
Integration test checking the correctness of the full "naive" GP implementation.
"""
import numpy as np
import torch
from laplace.baselaplace import FullLaplace, FunctionalLaplace
from tests.utils import toy_model, toy_regression_dataset

true_sigma_noise = 0.3
torch.manual_seed(711)


def test_gp_equivalence():
    X_train, y_train, train_loader, X_test = toy_regression_dataset(sigma=true_sigma_noise)
    model = toy_model(train_loader)

    # TODO: correctly incorporate _H_factor in BackPackGP
    # full_la = FullLaplace(model, 'regression', sigma_noise=true_sigma_noise)
    # functional_gp_la = FunctionalLaplace(model, 'regression', M=len(X_train), sigma_noise=true_sigma_noise)
    full_la = FullLaplace(model, 'regression')
    functional_gp_la = FunctionalLaplace(model, 'regression', M=len(X_train))
    full_la.fit(train_loader)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    f_mu_full = f_mu_full.squeeze().detach().cpu().numpy()
    f_var_full = f_var_full.squeeze().detach().cpu().numpy()
    f_mu_gp = f_mu_gp.squeeze().detach().cpu().numpy()
    f_var_gp = f_var_gp.squeeze().detach().cpu().numpy()

    print(f_var_full / f_var_gp)
    print(np.abs(f_var_full - f_var_gp))
    print(np.max(np.abs(f_var_full - f_var_gp)))
    print(np.mean(np.abs(f_var_full - f_var_gp)))

    assert np.allclose(f_mu_full, f_mu_gp)
    assert np.allclose(f_var_full, f_var_gp, atol=0.02)



