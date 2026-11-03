# %% [markdown]
# # Regression on Sinusoidal Toy Data
#
# We train a MAP network on a simple sinusoidal regression task, then use the
# marginal likelihood to tune the Laplace approximation (LA) on top of it.
# The tuned LA gives a predictive with uncertainty around the MAP prediction.
# Finally, the `marglik_training` utility trains the MAP and the hyperparameters
# jointly.
#
# First, we set up the training data with observation noise $\sigma=0.3$:

# %%
import numpy as np
import torch
from helper.dataloaders import get_sinusoid_example
from helper.util import plot_regression

from laplace import Laplace, marglik_training
from laplace.curvature.backpack import BackPackGGN

n_epochs = 1000
torch.manual_seed(711)

# create toy regression data
X_train, y_train, train_loader, X_test = get_sinusoid_example(sigma_noise=0.3)

# %% [markdown]
# ## Training a MAP
#
# We use plain PyTorch to train a network with a single hidden layer and Tanh
# activation. The trained network is our MAP estimate. Nothing new here yet:


# %%
def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1)
    )


model = get_model()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for i in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

# %% [markdown]
# ## Fitting and optimizing the LA with empirical Bayes
#
# With the MAP model at hand, we estimate the prior precision and the observation
# noise by empirical Bayes, after training. `Laplace` builds an LA for
# `"regression"` over `"all"` weights. By default `Laplace` uses a Kronecker-factored
# Hessian; this example is small, so we use `"full"` instead. We fit the LA to the
# training data, initialize `log_prior` and `log_sigma`, and minimize the negative
# log marginal likelihood with Adam for `n_epochs`.

# %%
la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="full")
la.fit(train_loader)

log_prior, log_sigma = (
    torch.ones(1, requires_grad=True),
    torch.ones(1, requires_grad=True),
)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

# %% [markdown]
# The fitted quantities can be saved and loaded back into a fresh `Laplace` object:

# %%
state_dict = la.state_dict()
torch.save(state_dict, "state_dict.bin")

la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="full")
la.load_state_dict(torch.load("state_dict.bin"))

print(
    f"sigma={la.sigma_noise.item():.2f}",
    f"prior precision={la.prior_precision.item():.2f}",
)

# %% [markdown]
# The observation noise comes out close to the ground truth, $\sigma \approx 0.28$,
# without any validation data. The prior precision is $\delta \approx 0.10$.
#
# ## Bayesian predictive
#
# Now we compare the MAP prediction with the LA prediction. The LA has a closed-form
# Gaussian predictive on the output $f$,
# $\mathcal{N}(f(x;\theta_{MAP}), \mathbb{V}[f] + \sigma^2)$.
#
# There are two flavors. The marginal predictive treats each test point on its own:
# the mean has shape `(m, k)` and the variance `(m, k, k)`. The joint predictive
# with `joint=True` returns one Gaussian over all test points: the mean has shape
# `(m * k,)` and the covariance `(m * k, m * k)`. Its diagonal matches the marginal
# variances.

# %%
x = X_test.flatten().cpu().numpy()

f_mu, f_var = la(X_test)
f_mu_joint, f_cov = la(X_test, joint=True)

assert torch.allclose(f_mu.flatten(), f_mu_joint)
assert torch.allclose(f_var.flatten(), f_cov.diag())

f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

plot_regression(
    X_train, y_train, x, f_mu, pred_std, file_name="regression_example", plot=False
)

# %% [markdown]
# ![Posthoc Laplace](../assets/regression_example.png)
#
# Unlike the MAP, the predictive shows useful uncertainty. But if the MAP over- or
# underfits, the LA can't fix that after the fact. Jointly optimizing the MAP and
# the marginal likelihood helps in that case.
#
# ## Jointly optimizing the MAP and hyperparameters with online empirical Bayes
#
# The `marglik_training` utility implements the algorithm of
# [Immer et al. (2021)](https://arxiv.org/abs/2104.04975). It alternates between
# optimizing the network and the hyperparameters, and returns an optimally
# regularized LA. We use it below and plot the predictive uncertainty again:

# %%
model = get_model()
la, model, margliks, losses = marglik_training(
    model=model,
    train_loader=train_loader,
    likelihood="regression",
    hessian_structure="full",
    backend=BackPackGGN,
    n_epochs=n_epochs,
    optimizer_kwargs={"lr": 1e-2},
    prior_structure="scalar",
)

print(
    f"sigma={la.sigma_noise.item():.2f}",
    f"prior precision={la.prior_precision.numpy()}",
)

f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

plot_regression(
    X_train,
    y_train,
    x,
    f_mu,
    pred_std,
    file_name="regression_example_online",
    plot=False,
)

# %% [markdown]
# ![Online Laplace](../assets/regression_example_online.png)
