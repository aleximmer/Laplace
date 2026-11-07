# %% [markdown]
# # Known Noise per Output
#
# A regression network with $K$ outputs often has outputs of different scales or
# different measurement quality, e.g. a temperature and a pressure sensor. If you
# know the noise std of each output, $\sigma \in \mathbb{R}^K$, the likelihood is
#
# $$
#     p(y \mid f) = \mathcal{N}\left(y \mid f, \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2)\right).
# $$
#
# This page shows how to give this noise to a Laplace approximation (LA) and how to
# use the predictive. The noise is fixed when you call `fit`. It is not learned.
#
# We create data with two outputs of the same function family but different noise:
# the first has noise std 0.05, the second 0.4.

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.likelihoods import GaussianLikelihood

torch.manual_seed(711)

sigma = torch.tensor([0.05, 0.4])


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty(n, 1).uniform_(-3, 3)
    f = torch.cat([torch.sin(X), 0.5 * X], dim=1)

    return X, f + sigma * torch.randn(n, 2)


X_train, y_train = make_data(200)
X_test = torch.linspace(-3, 3, 50).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100)

# %% [markdown]
# ## Training a MAP
#
# We train the network with plain PyTorch. Weighting the squared error of each output by
# $1 / (2 \sigma_k^2)$ makes the loss the negative log likelihood for this noise.

# %%
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(300):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = ((model(X) - y) ** 2 / (2 * sigma**2)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

# %% [markdown]
# ## Giving the noise to the likelihood
#
# The noise vector belongs to the likelihood. Pass it to `GaussianLikelihood`, and the
# LA uses it for the curvature, the log likelihood, and the marginal likelihood.
#
# The `sigma_noise` argument of `Laplace` stays a scalar. With a vector in the
# likelihood it must stay at 1, and `Laplace` raises an error otherwise. If you pass a
# vector to it, the error points you to `GaussianLikelihood`.

# %%
likelihood = GaussianLikelihood(sigma_noise=sigma)

la = Laplace(model, likelihood, subset_of_weights="all", hessian_structure="kron")
la.fit(train_loader)

print(f"sigma_noise of the LA: {la.sigma_noise}")

# %% [markdown]
# The prior precision is still a free hyperparameter. For example, we can tune it with
# the marginal likelihood.

# %%
la.optimize_prior_precision(method="marglik")

print(f"prior precision: {la.prior_precision.item():.2f}")
print(f"log marginal likelihood: {la.log_marginal_likelihood().item():.2f}")

# %% [markdown]
# ## Predictive
#
# The GLM predictive does not include the observation noise. It returns the mean and
# the covariance of the latent function $f$. To predict a new observation $y$, add the
# noise variance of each output to the diagonal:

# %%
f_mu, f_var = la(X_test)
y_var = f_var.diagonal(dim1=-2, dim2=-1) + sigma**2

print(f"f_mu: {tuple(f_mu.shape)}, f_var: {tuple(f_var.shape)}")
print(f"predictive std of y at the first test point: {y_var[0].sqrt()}")

# %% [markdown]
# The predictive std of the second output is larger than the first, as the noise in
# the data suggests.
#
# ## Other structures and function-space methods
#
# The vector works with the `full`, `diag` and `kron` structures, for all weights and
# the last layer, and with `full` and `diag` for a subnetwork. The function-space
# methods `FunctionalLaplace`, `ELLA` and `VaLLA` take it too. Declare it in the
# likelihood the same way:

# %%
la_diag = Laplace(
    model, likelihood, subset_of_weights="last_layer", hessian_structure="diag"
)
la_diag.fit(train_loader)

print(
    f"log marginal likelihood (last layer, diag): {la_diag.log_marginal_likelihood().item():.2f}"
)

# %% [markdown]
# The default backend, `CurvlinopsGGN`, supports all of these, as does `CurvlinopsEF`.
# BackPACK and ASDL support `full` and `diag` through the generic GGN code, but raise
# an error for `kron`.
#
# `marglik_training` raises an error for a likelihood with a noise per output, since it
# learns a single noise std. Learning one std per output is tracked in
# [issue #349](https://github.com/aleximmer/Laplace/issues/349).
