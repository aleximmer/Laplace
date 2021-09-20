import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from laplace.baselaplace import FullLaplace, FunctionalLaplace


n_epochs = 100
batch_size = 150  # full batch
true_sigma_noise = 0.3
torch.manual_seed(711)

# create simple sinusoid data set
X_train = (torch.rand(150) * 8).unsqueeze(-1)
y_train = torch.sin(X_train) + torch.randn_like(X_train) * true_sigma_noise
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)  # +-5 on top of the training X-range


@pytest.fixture
def fitted_model():
    model = torch.nn.Sequential(torch.nn.Linear(1, 50),
                                torch.nn.Tanh(),
                                torch.nn.Linear(50, 1))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
    for i in range(n_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    return model


def test_gp_equivalence(fitted_model):
    full_la = FullLaplace(fitted_model, 'regression', sigma_noise=true_sigma_noise)
    full_la.fit(train_loader)
    functional_gp_la = FunctionalLaplace(fitted_model, 'regression', M=len(X_train), sigma_noise=true_sigma_noise)
    functional_gp_la.fit(train_loader)

    f_mu_full, f_var_full = full_la(X_test)
    f_mu_gp, f_var_gp = functional_gp_la(X_test)

    f_mu_full = f_mu_full.squeeze().detach().cpu().numpy()
    f_var_full = f_var_full.squeeze().detach().cpu().numpy()
    f_mu_gp = f_mu_gp.squeeze().detach().cpu().numpy()
    f_var_gp = f_var_gp.squeeze().detach().cpu().numpy()

    assert np.allclose(f_mu_full, f_mu_gp)
    assert np.allclose(f_var_full, f_var_gp)


