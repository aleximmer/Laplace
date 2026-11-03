# %% [markdown]
# # Bayesian Optimization
#
# Bayesian optimization looks for the minimum of an expensive black-box function with
# as few evaluations as possible. It needs a surrogate model with calibrated
# uncertainty: an acquisition function such as expected improvement trades off
# exploring uncertain regions against exploiting regions that already look good.
# Gaussian processes are the usual choice. Here, we use a neural network with a
# Laplace approximation instead, and plug it into [BoTorch](https://botorch.org/)
# so that BoTorch's acquisition functions and optimizers work with it unchanged.
#
# We compare it with a GP and with random search on a 2D test function.

# %%
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from typing import Any, List, Optional

import numpy as np
import torch
import torch.utils.data as data_utils
import tqdm
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.test_functions import Ackley, Branin
from gpytorch import distributions as gdists
from torch import distributions as dists
from torch import nn, optim
from torch.nn import functional as F

from laplace import BaseLaplace, Laplace

# %% [markdown]
# ## A Laplace surrogate for BoTorch
#
# BoTorch talks to a surrogate through its `Model` interface. Two methods matter here.
#
# `posterior` gets inputs of shape `(batch, q, d)`: `batch` candidate sets with `q`
# points each, which acquisition functions such as qEI score jointly. It must return a
# Gaussian over the `q` outputs of every candidate set. We get it from the joint GLM
# predictive `la(X, joint=True)` over all `batch * q` points at once, then keep the
# `q * k` by `q * k` block of each candidate set. A small jitter keeps the covariance
# positive definite.
#
# `condition_on_observations` adds new evaluations to the data set. The wrapper then
# continues training the network on all data and fits a new Laplace approximation
# over all weights, with a Kronecker-factored Hessian and `enable_backprop=True`, so
# that BoTorch can optimize the acquisition function with gradients.


# %%
class LaplaceBNN(Model):
    """
    BoTorch model whose posterior is the GLM predictive of a Laplace-approximated
    neural network.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        bnn: BaseLaplace | None = None,
        likelihood: str = "regression",
        batch_size: int = 1024,
    ):
        super().__init__()

        self.train_X = train_X
        self.train_Y = train_Y
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.nn = nn.Sequential(
            nn.Linear(train_X.shape[-1], 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, train_Y.shape[-1]),
        )
        self.bnn = bnn

        if self.bnn is None:
            self._train_model(self._get_train_loader())

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """
        Note: q is the num. of x's predicted jointly
        """
        # Transform to `(batch_shape*q, d)`
        B, Q, D = X.shape
        X = X.reshape(B * Q, D)

        # Posterior predictive distribution
        # mean_y is (batch_shape*q, k); cov_y is (batch_shape*q*k, batch_shape*q*k)
        mean_y, cov_y = self._get_prediction(X, use_test_loader=False)

        # Mean in `(batch_shape, q*k)`
        K = self.num_outputs
        mean_y = mean_y.reshape(B, Q * K)

        # Cov is `(batch_shape, q*k, q*k)`
        cov_y += 1e-4 * torch.eye(B * Q * K)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum("bqkbrl->bqkrl", cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q * K, Q * K)

        dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
        post_pred = GPyTorchPosterior(dist)

        if posterior_transform is not None:
            return posterior_transform(post_pred)

        return post_pred

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs: Any
    ) -> Model:
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y], dim=0)

        train_loader = self._get_train_loader()
        self._train_model(train_loader)

        return LaplaceBNN(
            # Added dataset & retrained BNN
            self.train_X,
            self.train_Y,
            self.bnn,
            self.likelihood,
            self.batch_size,
        )

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self.train_Y.shape[-1]

    def _train_model(self, train_loader):
        """
        Train BNN with the Laplace approximation
        """
        n_epochs = 1000
        optimizer = optim.Adam(self.nn.parameters(), lr=1e-1, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_epochs * len(train_loader)
        )
        loss_func = nn.MSELoss()

        for i in range(n_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.nn(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        self.nn.eval()

        self.bnn = Laplace(
            self.nn,
            self.likelihood,
            subset_of_weights="all",
            hessian_structure="kron",
            enable_backprop=True,
        )
        self.bnn.fit(train_loader)
        self.bnn.optimize_prior_precision(n_steps=50)

    def _get_prediction(self, test_x: torch.Tensor, joint=True, use_test_loader=False):
        """
        Batched Laplace prediction.

        Args:
            test_x: Tensor of size `(batch_shape, d)`.

        Returns:
            Tensor of size `(batch_shape, k)`
        """
        if self.bnn is None:
            print("Train your model first before making prediction!")
            sys.exit(1)

        if not use_test_loader:
            mean_y, cov_y = self.bnn(test_x, joint=True)
        else:
            test_loader = data_utils.DataLoader(
                data_utils.TensorDataset(test_x, torch.zeros_like(test_x)),
                batch_size=256,
            )

            mean_y, cov_y = [], []

            for x_batch, _ in test_loader:
                _mean_y, _cov_y = self.bnn(x_batch, joint=joint)
                mean_y.append(_mean_y)
                cov_y.append(_cov_y)

            mean_y = torch.cat(mean_y, dim=0).squeeze()
            cov_y = torch.cat(cov_y, dim=0).squeeze()

        return mean_y, cov_y

    def _get_train_loader(self):
        return data_utils.DataLoader(
            data_utils.TensorDataset(self.train_X, self.train_Y),
            batch_size=self.batch_size,
            shuffle=True,
        )


# %% [markdown]
# ## The test problem
#
# The command-line options pick the test function (Branin or Ackley, both from
# BoTorch), the acquisition function (EI, UCB, or qEI), the number of initial
# evaluations, and the number of optimization steps. We start from random
# evaluations inside the bounds and keep 10,000 more random points to measure how
# well each surrogate fits the function.

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--test_func", choices=["ackley", "branin"], default="branin")
parser.add_argument("--acqf", choices=["EI", "UCB", "qEI"], default="EI")
parser.add_argument("--init_data", type=int, default=20)
parser.add_argument("--exp_len", type=int, default=500)
parser.add_argument("--randseed", type=int, default=1)
args, _ = parser.parse_known_args()

np.random.seed(args.randseed)
torch.set_default_dtype(torch.float64)
torch.manual_seed(args.randseed)

if args.test_func == "ackley":
    true_f = Ackley(dim=2)
    bounds = torch.tensor([[-32.768, 32.768], [-32.768, 32.768]]).T.double()
else:
    true_f = Branin()
    bounds = torch.tensor([[-5, 10], [0, 15]]).T.double()

print(f"Test Function: {args.test_func}")

train_x = torch.cat(
    [
        dists.Uniform(*bounds.T[i]).sample((args.init_data, 1))
        for i in range(2)  # for each dimension
    ],
    dim=1,
)
train_y = true_f(train_x).reshape(-1, 1)

test_x = torch.cat(
    [
        dists.Uniform(*bounds.T[i]).sample((10000, 1))
        for i in range(2)  # for each dimension
    ],
    dim=1,
)
test_y = true_f(test_x)

# %% [markdown]
# ## The surrogates
#
# Random search needs no model. The Laplace surrogate trains its network and fits
# Laplace when we create it. The GP is BoTorch's `SingleTaskGP`. The test error of each
# surrogate is its mean squared error on the held-out points; for the Laplace
# surrogate, we predict them in batches with the marginal (not joint) predictive.

# %%
models = {
    "RandomSearch": None,
    "BNN-LA": LaplaceBNN(train_x, train_y),
    "GP": SingleTaskGP(train_x, train_y),
}


def evaluate_model(model_name, model):
    if model_name == "GP":
        pred = model.posterior(test_x).mean.squeeze()
    elif model_name == "BNN-LA":
        pred, _ = model._get_prediction(test_x, use_test_loader=True, joint=False)
    else:
        return -1

    return F.mse_loss(pred, test_y).squeeze().item()


# %% [markdown]
# ## The optimization loop
#
# Each step asks the acquisition function for a new input, evaluates the test
# function there, and conditions the surrogate on the result. BoTorch maximizes
# acquisition functions by default, and we want a minimum, hence `maximize=False`.
# With qEI, every step proposes a batch of 5 inputs. The progress bar shows the
# surrogate's test error and the best value found so far.

# %%
for model_name, model in models.items():
    np.random.seed(args.randseed)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.randseed)

    best_y = train_y.min().item()
    trace_best_y = []
    pbar = tqdm.trange(args.exp_len)
    pbar.set_description(
        f"[{model_name}, MSE = {evaluate_model(model_name, model):.3f}; Best f(x) = {best_y:.3f}]"
    )

    for i in pbar:
        if model_name == "RandomSearch":
            new_x = torch.cat(
                [
                    dists.Uniform(*bounds.T[i]).sample((1, 1))
                    for i in range(len(bounds))  # for each dimension
                ],
                dim=1,
            ).squeeze()
        else:
            if args.acqf == "EI":
                acq_f = ExpectedImprovement(model, best_f=best_y, maximize=False)
            elif args.acqf == "UCB":
                acq_f = UpperConfidenceBound(model, beta=0.2, maximize=False)
            else:
                acq_f = qExpectedImprovement(model, best_f=best_y, maximize=False)

            # Get a proposal for new x
            new_x, val = optimize_acqf(
                acq_f,
                bounds=bounds,
                q=1 if args.acqf not in ["qEI"] else 5,
                num_restarts=10,
                raw_samples=20,
            )

        if len(new_x.shape) == 1:
            new_x = new_x.unsqueeze(0)

        # Evaluate the objective on the proposed x
        new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

        # Update posterior
        if model_name != "RandomSearch":
            model = model.condition_on_observations(new_x, new_y)

        # Update the current best y
        if new_y.min().item() <= best_y:
            best_y = new_y.min().item()

        trace_best_y.append(best_y)
        pbar.set_description(
            f"[{model_name}, MSE = {evaluate_model(model_name, model):.3f}; Best f(x) = {best_y:.3f}, curr f(x) = {new_y.min().item():.3f}]"
        )

# %% [markdown]
# The joint predictive is what makes batch acquisition functions like qEI work: the
# `q` points of a batch are correlated under the posterior, so the batch spreads out
# instead of piling up at one spot. The
# [decision-making example](../expectation_example/) shows how to build your own
# acquisition functions from Laplace samples, including ones without a closed form.
