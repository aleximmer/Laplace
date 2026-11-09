# %% [markdown]
# # Heteroscedastic Regression
#
# In many regression problems the noise changes with the input. A sensor can be
# precise in one range and noisy in another, and the measurements in Bayesian
# optimization for science are often like that. A mean-variance network models this:
# it predicts a mean $\mu(x)$ and a variance $\sigma^2(x)$, and you train it with
# `nn.GaussianNLLLoss` on the likelihood
#
# $$
#     p(y \mid x) = \mathcal{N}\left(y \mid \mu(x), \sigma^2(x)\right).
# $$
#
# The variance is the aleatoric uncertainty, the noise in the data. It doesn't tell you
# where the network itself is unsure, e.g. far from the training data. This page shows
# how to add that epistemic uncertainty with a post-hoc Laplace approximation (LA),
# and then how to train the network with the marginal likelihood, as in
# [Immer et al. (2023)](https://openreview.net/forum?id=A6EquH0enk).
#
# The data have a gap in the middle, and their noise grows with $|x|$.

# %%
import torch
from helper.util import plot_heteroscedastic
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace, marglik_training

torch.manual_seed(711)


def noise_std(x: torch.Tensor) -> torch.Tensor:
    return 0.05 + 0.15 * x.abs()


X_train = torch.cat(
    [torch.empty(150, 1).uniform_(-3, -0.7), torch.empty(150, 1).uniform_(0.7, 3)]
)
y_train = torch.sin(2 * X_train) + noise_std(X_train) * torch.randn_like(X_train)
X_test = torch.linspace(-4, 4, 200).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)

# %% [markdown]
# ## A mean-variance network
#
# The network returns the pair `(mean, var)`, which is what `nn.GaussianNLLLoss` takes.
# A softplus keeps the variance positive. We train it as usual.


# %%
class MeanVarianceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh()
        )
        self.mean = nn.Linear(50, 1)
        self.var = nn.Sequential(nn.Linear(50, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)

        return self.mean(h), self.var(h) + 1e-4


model = MeanVarianceNet()
criterion = nn.GaussianNLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

for epoch in range(500):
    for X, y in train_loader:
        optimizer.zero_grad()
        mean, var = model(X)
        loss = criterion(mean, y, var)
        loss.backward()
        optimizer.step()

# %% [markdown]
# ## A post-hoc Laplace approximation
#
# Pass the criterion as the likelihood. `Laplace` turns `nn.GaussianNLLLoss` into a
# `HeteroscedasticGaussianLikelihood` with the same `eps` and `full`, and you can also
# pass that object or its name, `"heteroscedastic_gaussian"`.
#
# The LA is over the mean of the network, as in Immer et al. (2023): it holds the
# variance at its value under the trained weights. So the parameters of the variance
# head only get curvature from the prior, and the last-layer LA, which assumes a
# single output layer, isn't available. Use all weights or a subnetwork. The `diag`,
# `full`, and `kron` structures work with the default backend.

# %%
la = Laplace(
    model,
    criterion,  # or HeteroscedasticGaussianLikelihood()
    subset_of_weights="all",
    hessian_structure="kron",
)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

print(f"prior precision: {la.prior_precision.item():.2f}")
print(f"log marginal likelihood: {la.log_marginal_likelihood().item():.2f}")

# %% [markdown]
# ```
# prior precision: 0.51
# log marginal likelihood: -298.16
# ```
#
# ## The predictive
#
# The network returns the mean and the aleatoric variance. The LA returns the same two,
# plus the epistemic variance of the mean:

# %%
f_mean, f_alea, f_epis = la(X_test)

print(
    f"f_mean: {tuple(f_mean.shape)}, f_alea: {tuple(f_alea.shape)}, f_epis: {tuple(f_epis.shape)}"
)

# %% [markdown]
# ```
# f_mean: (200, 1), f_alea: (200, 1), f_epis: (200, 1, 1)
# ```
#
# With the default GLM predictive, `f_mean` and `f_alea` are what the network returns
# at the trained weights, and `f_epis` is the covariance of the mean over the outputs,
# `(batch, outputs, outputs)`. `diagonal_output=True` returns its diagonal instead, and
# `joint=True` the covariance across inputs, as for regression. The predictive variance
# of a new observation $y$ is the sum of both:

# %%
epis_var = f_epis.diagonal(dim1=-2, dim2=-1)
y_var = f_alea + epis_var

plot_heteroscedastic(
    X_train,
    y_train,
    X_test.flatten(),
    f_mean.flatten().detach(),
    f_alea.sqrt().flatten().detach(),
    epis_var.sqrt().flatten().detach(),
    file_name="heteroscedastic_example",
    plot=False,
)

# %% [markdown]
# ![Heteroscedastic Laplace](../assets/heteroscedastic_example.png)
#
# The aleatoric std follows the noise in the data, and the network extrapolates it
# outside. The epistemic std is small where there are data and grows in the gap and
# beyond the data.
#
# `pred_type="nn"` samples weights from the posterior instead and returns the sample
# mean and variance of $\mu(x)$ as `f_mean` and `f_epis`, with the same `f_alea`.
# `functional_samples` and `predictive_samples` draw samples of the mean $\mu(x)$,
# without the noise.
#
# `optimize_prior_precision(method="gridsearch")` scores the predictive of $y$, with
# variance `f_alea` plus the diagonal of `f_epis`, by its negative log likelihood on the
# validation data.
#
# ## Training with the marginal likelihood
#
# Immer et al. (2023) train the network and its prior precision together, by
# maximizing the marginal likelihood of the LA during training. This is what
# `marglik_training` does, and it takes the criterion of a mean-variance network too.
# Weight decay then isn't needed, since the learned prior regularizes the network.
#
# They also parameterize the network by the natural parameters of the Gaussian,
# $\eta_1 = \mu / \sigma^2$ and $\eta_2 = -1 / (2 \sigma^2)$. The network only has to
# return `(mean, var)` at the end, so the natural parameters can stay inside it:


# %%
class NaturalHeadNet(MeanVarianceNet):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        eta_1, eta_2 = self.mean(h), -0.5 * self.var(h) - 1e-4

        return -eta_1 / (2 * eta_2), -1 / (2 * eta_2)


la, model, margliks, losses = marglik_training(
    NaturalHeadNet(),
    train_loader,
    likelihood=nn.GaussianNLLLoss(),
    hessian_structure="kron",
    n_epochs=300,
    n_epochs_burnin=50,
    marglik_frequency=50,
    optimizer_kwargs={"lr": 1e-2},
)

print(f"prior precision: {la.prior_precision.mean().item():.2f}")
print(f"log marginal likelihood: {-margliks[-1]:.2f}")

# %% [markdown]
# ```
# prior precision: 25.01
# log marginal likelihood: -216.43
# ```
#
# The returned `la` is fit at the end of the training, and its predictive works as
# above:

# %%
f_mean, f_alea, f_epis = la(X_test)
epis_var = f_epis.diagonal(dim1=-2, dim2=-1)

plot_heteroscedastic(
    X_train,
    y_train,
    X_test.flatten(),
    f_mean.flatten().detach(),
    f_alea.sqrt().flatten().detach(),
    epis_var.sqrt().flatten().detach(),
    file_name="heteroscedastic_example_marglik",
    plot=False,
)

# %% [markdown]
# ![Heteroscedastic Laplace with marginal-likelihood training](../assets/heteroscedastic_example_marglik.png)
