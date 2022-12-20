from laplace.curvature.backpack import BackPackGGN
import numpy as np
import torch

from laplace import Laplace, marglik_training

from helper.dataloaders import get_sinusoid_example
from helper.util import plot_regression

# specify Laplace approximation type: full, kron, gp...
la_type = 'gp'

n_epochs = 1000
torch.manual_seed(711)

# create toy regression data
X_train, y_train, train_loader, X_test = get_sinusoid_example(sigma_noise=0.3)

# construct single layer neural network
def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1)
    )
model = get_model()

# train MAP
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for i in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure=la_type)

la.fit(train_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

x = X_test.flatten().cpu().numpy()
f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

print(f'sigma={la.sigma_noise.item():.3f} | ',
      f'prior precision={la.prior_precision.item():.3f} | ',
      f'MAE: {np.abs(x - f_mu).mean():.3f}')

plot_regression(X_train, y_train, x, f_mu, pred_std,
                file_name='regression_example', plot=True, la_type=la_type)

# alternatively, optimize parameters and hyperparameters of the prior jointly
model = get_model()
la, model, margliks, losses = marglik_training(
    model=model, train_loader=train_loader, likelihood='regression',
    hessian_structure=la_type, backend=BackPackGGN, n_epochs=n_epochs,
    optimizer_kwargs={'lr': 1e-2}, prior_structure='scalar'
)

f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

print(f'sigma={la.sigma_noise.item():.3f} | ',
      f'prior precision={la.prior_precision.numpy()[0]:.3f} | ',
      f'MAE: {np.abs(x - f_mu).mean():.3f}')

plot_regression(X_train, y_train, x, f_mu, pred_std,
                file_name='regression_example_online', plot=True, la_type=la_type + " (online)")
