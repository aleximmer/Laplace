# %% [markdown]
# # Bayesian Optimization
#
# Bayesian optimization looks for the minimum of an expensive black-box function with
# as few evaluations as possible. At every step, it fits a surrogate model to the
# evaluations so far and picks the next input by maximizing an acquisition function,
# such as expected improvement, under the surrogate's predictive distribution.
#
# In this example, the surrogate is a neural network with a Laplace approximation. We
# wrap it as a [BoTorch](https://botorch.org/) model, so that BoTorch's acquisition
# functions and optimizers work with it unchanged, and minimize the 2D Branin function.

# %%
import argparse
import warnings

import torch
import torch.utils.data as data_utils
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.test_functions import Branin
from gpytorch.distributions import MultivariateNormal
from torch import nn, optim

from laplace import Laplace

warnings.filterwarnings("ignore")

# Fewer steps are for quick test runs
parser = argparse.ArgumentParser()
parser.add_argument("--n_steps", type=int, default=20)
args, _ = parser.parse_known_args()

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

# %% [markdown]
# ## A Laplace surrogate for BoTorch
#
# BoTorch talks to a surrogate through its `Model` interface. Two methods matter here.
#
# `posterior` gets inputs of shape `(batch, q, d)`: `batch` candidate sets with `q`
# points each. It must return a Gaussian over the outputs of every candidate set,
# including the correlations between the `q` points, which batch acquisition
# functions such as qEI rely on. We get it from the joint GLM predictive
# `la(X, joint=True)` over all `batch * q` points at once, and keep the
# `q * k` by `q * k` block of each candidate set. A small jitter keeps the covariance
# positive definite.
#
# `condition_on_observations` adds new evaluations. Here, it continues training the
# network on all data so far and fits a new Laplace approximation. We use all weights
# with a Kronecker-factored Hessian, and set `enable_backprop=True` so that BoTorch can
# maximize the acquisition function with gradients.


# %%
class LaplaceBNN(Model):
    """BoTorch model whose posterior is the GLM predictive of a Laplace network.

    Parameters
    ----------
    train_X : torch.Tensor
        inputs `(n, d)`
    train_Y : torch.Tensor
        outputs `(n, k)`
    n_epochs : int, default=1000
        number of full-batch training epochs per fit
    """

    def __init__(
        self, train_X: torch.Tensor, train_Y: torch.Tensor, n_epochs: int = 1000
    ) -> None:
        super().__init__()

        self.train_X = train_X
        self.train_Y = train_Y
        self.n_epochs = n_epochs

        self.net = nn.Sequential(
            nn.Linear(train_X.shape[-1], 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, train_Y.shape[-1]),
        )
        self._fit()

    @property
    def num_outputs(self) -> int:
        return self.train_Y.shape[-1]

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs,
    ) -> GPyTorchPosterior:
        B, Q, D = X.shape
        K = self.num_outputs

        # Joint predictive over all B * Q points: mean (B * Q, K), cov (B * Q * K, B * Q * K)
        mean, cov = self.la(X.reshape(B * Q, D), joint=True)
        cov = cov + 1e-4 * torch.eye(B * Q * K)

        # Keep the (Q * K, Q * K) block of each candidate set
        cov = cov.reshape(B, Q, K, B, Q, K)
        cov = torch.einsum("bqkbrl->bqkrl", cov).reshape(B, Q * K, Q * K)

        posterior = GPyTorchPosterior(MultivariateNormal(mean.reshape(B, Q * K), cov))

        if posterior_transform is not None:
            return posterior_transform(posterior)

        return posterior

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs
    ) -> "LaplaceBNN":
        self.train_X = torch.cat([self.train_X, X])
        self.train_Y = torch.cat([self.train_Y, Y])
        self._fit()

        return self

    def _fit(self) -> None:
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(self.train_X, self.train_Y),
            batch_size=len(self.train_X),
            shuffle=True,
        )

        # MAP training, warm-started from the previous fit
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=1e-1, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs)

        for _ in range(self.n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.net(x), y)
                loss.backward()
                optimizer.step()

            scheduler.step()

        self.net.eval()

        self.la = Laplace(
            self.net,
            "regression",
            subset_of_weights="all",
            hessian_structure="kron",
            enable_backprop=True,
        )
        self.la.fit(train_loader)
        self.la.optimize_prior_precision(n_steps=50)


# %% [markdown]
# ## The optimization loop
#
# Branin has three global minima, all with value about 0.398, on
# $[-5, 10] \times [0, 15]$. We start from 5 random evaluations and fit the
# surrogate on them.

# %%
objective = Branin()
bounds = objective.bounds  # (2, d): lower and upper bounds

train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(5, 2)
train_y = objective(train_x).unsqueeze(-1)

model = LaplaceBNN(train_x, train_y)
best_y = train_y.min().item()
print(f"Best initial f(x) = {best_y:.3f}")

# %% [markdown]
# Each step maximizes expected improvement over the best value so far, evaluates
# Branin at the proposed input, and adds the result to the surrogate. BoTorch
# maximizes by default, and we want a minimum, hence `maximize=False`.

# %%
for step in range(args.n_steps):
    acqf = ExpectedImprovement(model, best_f=best_y, maximize=False)
    new_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    new_y = objective(new_x).unsqueeze(-1)
    model = model.condition_on_observations(new_x, new_y)
    best_y = min(best_y, new_y.item())

    print(f"Step {step + 1}: f(x) = {new_y.item():.3f}, best f(x) = {best_y:.3f}")

# %% [markdown]
# Since `posterior` returns the joint predictive, batch acquisition functions work
# too, for example BoTorch's `qExpectedImprovement` with `optimize_acqf(..., q=5)`
# to propose five inputs per step. The
# [decision-making example](../expectation_example/) shows how to build your own
# acquisition functions from Laplace samples, including ones without a closed form.
