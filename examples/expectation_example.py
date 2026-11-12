# %% [markdown]
# # Decision-Making
#
# Once you have fitted a Laplace approximation, you have a belief $p(f \mid \mathcal{D})$ about the function your network models.
# One thing to do with a belief is to act on it.
# Decision theory says to pick the action with the highest expected utility under your belief:
#
# $$
# x_\text{best} = \arg\max_x \, \mathbb{E}_{p(f \mid \mathcal{D})} \big[ u(f, x) \big].
# $$
#
# The utility $u$ can be almost anything, as long as it reflects how much you prefer one outcome over another.
# The price of that freedom is that the expectation rarely has a closed form, even when the posterior is Gaussian.
#
# Laplace makes the Monte Carlo route easy.
# Draw samples of the network outputs with `la.functional_samples`, apply $u$, and average.
# With `enable_backprop=True` the estimate is differentiable in $x$, so the $\arg\max$ becomes plain gradient ascent.
# Bayesian optimization is the obvious use: an acquisition function is an expected utility, and this recipe gives you one for any utility you can write down, not only the few that have closed forms [1].

# %%
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils.data as data_utils
from torch import nn, optim

from laplace import (
    BaseLaplace,
    GaussianLikelihood,
    HessianStructure,
    Laplace,
    PredType,
    SubsetOfWeights,
)

# %% [markdown]
# ## Setup: a calibration problem
#
# Suppose an expensive simulator maps an input $x \in [0, 1]^2$ to two outputs $h(x) \in \mathbb{R}^2$, and we want the input whose outputs match a target $y^*$.
# The outer function $g(y) = -\lVert y - y^* \rVert^2$ is cheap and known; only $h$ is expensive.
# Astudillo and Frazier [2] call this a composite objective $g(h(x))$, and modeling $h$ instead of $g \circ h$ directly uses what we know about $g$.


# %%
def true_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x[:, 0] + 0.3 * torch.sin(4 * x[:, 1]),
            x[:, 1] + 0.3 * torch.sin(4 * x[:, 0]),
        ],
        dim=-1,
    )


X_STAR = torch.tensor([0.7, 0.3])
Y_TARGET = true_fn(X_STAR[None])[0]


def g(y: torch.Tensor) -> torch.Tensor:
    return -((y - Y_TARGET) ** 2).sum(-1)


# %% [markdown]
# We query the simulator 20 times, train a small two-output network on the results, and fit a Laplace approximation on top.
# The steps sit in a function so that we can refit with another seed.


# %%
def fit_laplace(seed: int = 123) -> tuple[BaseLaplace, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    X_train = torch.rand(20, 2)
    Y_train = true_fn(X_train)

    model = nn.Sequential(
        nn.Linear(2, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 2)
    )
    opt = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for _ in range(1000):
        opt.zero_grad()
        loss = ((model(X_train) - Y_train) ** 2).mean()
        loss.backward()
        opt.step()

    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, Y_train), batch_size=20
    )

    la = Laplace(
        model,
        GaussianLikelihood(),
        subset_of_weights=SubsetOfWeights.ALL,
        hessian_structure=HessianStructure.KRON,
        sigma_noise=0.05,
        enable_backprop=True,
    )
    la.fit(train_loader)
    la.optimize_prior_precision(PredType.GLM)

    return la, X_train, Y_train


la, X_train, Y_train = fit_laplace()

# %% [markdown]
# The simulator is noise-free, so we set `sigma_noise` small.
# With the default of 1, the posterior is far wider than the data supports.
#
# > **Tip:** Pass `enable_backprop=True` whenever you need gradients of anything computed
# > from Laplace's predictive or its samples. Continuous optimization over $x$
# > depends on it.
#
# To find the best input, we run Adam on the inputs and clamp them to the unit square after each step.
# A small helper prints the results.


# %%
def maximize(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 100,
    lr: float = 0.05,
) -> torch.Tensor:
    x = x0.clone().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        (-objective(x).sum()).backward()
        opt.step()

        with torch.no_grad():
            x.clamp_(0, 1)

    return x.detach()


def fmt(x: torch.Tensor) -> str:
    return str([round(v, 3) for v in x.flatten().tolist()])


# %% [markdown]
# ## Utilities that look at a single input
#
# Most acquisition functions score one input at a time: $u$ depends on $f$ only through $f(x)$.
# Then the expectation over functions reduces to an expectation over the outputs at $x$:
#
# $$
# \mathbb{E}_{p(f \mid \mathcal{D})} \big[ u(f(x)) \big] = \mathbb{E}_{p(f(x) \mid \mathcal{D})} \big[ u(f(x)) \big].
# $$
#
# The integrand ignores $f$ everywhere except at $x$, so only the distribution of $f(x)$ matters.
# That is exactly what `la.functional_samples(x)` draws from, one input at a time.
#
# Here $f(x)$ has two outputs, and $g$ mixes them, so the samples need the full covariance between the outputs at each $x$.
# `functional_samples` uses it by default; don't pass `diagonal_output=True` here.
#
# Our utility is the improvement of $g$ over the best value seen so far, which gives expected improvement for composite functions (EI-CF) [2].
# Unlike plain EI, EI-CF has no closed form, even under a Gaussian posterior.


# %%
def ei_cf(f: torch.Tensor, best: float) -> torch.Tensor:
    return (g(f) - best).clamp(min=0)


def expected_utility(
    la: BaseLaplace,
    x: torch.Tensor,
    utility: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    f_samples = la.functional_samples(
        x, n_samples=n_samples, generator=torch.Generator().manual_seed(seed)
    )  # (n_samples, batch, 2)
    return utility(f_samples).mean(0)  # (batch,)


# %% [markdown]
# The generator gets the same seed on every call.
# The estimate then uses the same random numbers at every step of the optimizer, which makes it a smooth, deterministic function of $x$ that gradient ascent can work with [1].
# We maximize it from five random starting points and keep the best:

# %%
best = g(Y_train).max().item()
print(f"Target input x*: {fmt(X_STAR)}, best g so far: {best:.4f}")


def ei(x: torch.Tensor) -> torch.Tensor:
    return expected_utility(la, x, lambda f: ei_cf(f, best))


x_restarts = maximize(ei, torch.rand(5, 2))
x_best = x_restarts[ei(x_restarts).argmax()]
print(f"EI-CF x_best: {fmt(x_best)}")

# %% [markdown]
# ```
# EI-CF x_best: [0.688, 0.331]
# ```
#
# That is close to the true answer $x^* = (0.7, 0.3)$.
#
# Switching to another utility changes one line.
# Probability of improvement is the expected value of an indicator:


# %%
def pi_cf(f: torch.Tensor, best: float) -> torch.Tensor:
    return (g(f) > best).float()


grid_1d = torch.linspace(0, 1, 50)
grid = torch.cartesian_prod(grid_1d, grid_1d)

pi = expected_utility(la, grid, lambda f: pi_cf(f, best))
x_best = grid[pi.argmax()]
print(f"PI-CF x_best: {fmt(x_best)}")

# %% [markdown]
# An indicator has zero gradient almost everywhere, so here we score a 50 × 50 grid of candidates and take the best one.
#
# ```
# PI-CF x_best: [0.673, 0.327]
# ```
#
# ## Utilities that look at several inputs
#
# Some utilities depend on $f$ at several inputs at once.
# In batch Bayesian optimization, we choose $q$ inputs together and care about the best outcome among them.
# Thompson sampling picks the maximizer of a single function drawn from the posterior [3].
# Neither reduces to per-input marginals; both need the outputs at all inputs jointly.
#
# > **Warning:** The default `la.functional_samples(x)` gets each input's marginal right
# > but not the correlations between inputs. Pass `joint=True` when the
# > utility looks at several inputs.
#
# ### Batch acquisition with joint samples
#
# With `joint=True`, `la.functional_samples` samples from the joint GLM predictive.
# That is the mean and full covariance of all outputs at all inputs, which `la(x, joint=True)` returns with shapes `(q * 2,)` and `(q * 2, q * 2)`, ordered input first, then output.
# The $q$-batch EI-CF scores the best of the $q$ outcomes in each sample:

# %%
q = 3


def qei(x: torch.Tensor) -> torch.Tensor:
    f_samples = la.functional_samples(
        x, n_samples=256, joint=True, generator=torch.Generator().manual_seed(0)
    )  # (256, q, 2)
    return (g(f_samples).max(-1).values - best).clamp(min=0).mean()


x_batch = maximize(qei, torch.rand(q, 2))
print(f"qEI-CF batch: {fmt(x_batch)}")

# %% [markdown]
# The fixed seed keeps the estimate smooth in $x$, as in the previous section.
#
# ```
# qEI-CF batch: [0.152, 0.903, 0.686, 0.332, 0.371, 0.9]
# ```
#
# One point lands near $x^*$.
# The other two go to uncertain regions: once one point covers the likely optimum, a second point there adds little.
#
# ### Thompson sampling
#
# Thompson sampling needs one coherent function, evaluated at every candidate.
# Laplace gives you two ways to get one from a single weight sample $\theta$, and they make different tradeoffs.
#
# The first plugs $\theta$ into the network itself.
# That is what `pred_type="nn"` does, so every input sees the same function:


# %%
def thompson_sample(
    la: BaseLaplace,
    candidates: torch.Tensor,
    seed: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    chunks = [candidates] if chunk_size is None else candidates.split(chunk_size)
    f_sample = [
        la.functional_samples(
            chunk,
            pred_type=PredType.NN,
            n_samples=1,
            generator=torch.Generator().manual_seed(seed),
        )[0]
        for chunk in chunks
    ]

    return torch.cat(f_sample)


f_sample = thompson_sample(la, grid, seed=0, chunk_size=500)
x_next = grid[g(f_sample).argmax()]
print(f"Thompson (nn) x_next: {fmt(x_next)}")

# %% [markdown]
# If the candidates don't fit in one forward pass, evaluate them in chunks and give every chunk a generator with the same seed.
# Each chunk then gets the same weights.
#
# The second plugs $\theta$ into the network linearized around the MAP estimate, $f(x) = f_{\theta_\text{MAP}}(x) + J(x)(\theta - \theta_\text{MAP})$.
# The Laplace backend already computes $J(x)$ for the GLM predictive, so the draw takes a few lines:


# %%
def linearized_thompson_sample(
    la: BaseLaplace,
    candidates: torch.Tensor,
    seed: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    theta = la.sample(1, generator=torch.Generator().manual_seed(seed))[0]
    delta = theta - la.mean

    # Subnetwork Laplace samples full weight vectors, but its Jacobians only
    # cover the subnetwork
    if la.backend.subnetwork_indices is not None:
        delta = delta[la.backend.subnetwork_indices]

    chunks = [candidates] if chunk_size is None else candidates.split(chunk_size)

    f_sample = []
    for chunk in chunks:
        Js, f_map = la.backend.jacobians(chunk)  # (B, K, P), (B, K)
        f_sample.append(f_map + Js @ delta)

    return torch.cat(f_sample).detach()


f_sample = linearized_thompson_sample(la, grid, seed=0, chunk_size=500)
x_next = grid[g(f_sample).argmax()]
print(f"Thompson (linearized) x_next: {fmt(x_next)}")

# %% [markdown]
# Every chunk reuses the same $\theta$.
# Each chunk also holds a `(chunk_size, outputs, n_params)` Jacobian in memory, so use smaller chunks for bigger networks.
#
# When there are only a few candidates, `la.functional_samples(candidates, n_samples=1, joint=True)` also gives a linearized Thompson sample.
# Its covariance grows as `(n_candidates * outputs)²`, so for large grids use the chunked helper above.
#
# Both are proper Thompson samples, but of different posteriors:
#
# ```
# Thompson (nn) x_next: [0.061, 0.0]
# Thompson (linearized) x_next: [0.633, 0.286]
# ```
#
# The linearized draw follows the same posterior as the GLM predictive used in the rest of this page, and here it lands near $x^*$.
# The `nn` draw skips the linearization, but the Gaussian over weights was fitted for the linearized network.
# Pushing those weights through the nonlinear network often spreads the functions out much more than the GLM predictive does [4].
# In this example the `nn` samples are far wider than the GLM predictive, which is why the draw above picks a corner.
#
# The `nn` draw has practical upsides.
# It only needs forward passes, and it works through the public API for every Laplace variant, including last-layer and subnetwork Laplace.
# The linearized draw needs a Jacobian per candidate, which costs more time and memory than a forward pass.
# The helper above works for Laplace over all weights and for subnetwork Laplace.
# With last-layer Laplace the network is linear in the sampled weights, so the two draws coincide and the `nn` draw is the one to use.
#
# ### Function-space Laplace
#
# `joint=True` works the same way for every Laplace variant, including the function-space methods (`hessian_structure="gp"`, ELLA, and VaLLA; see the [ELLA and VaLLA example](../ella_valla_example/)).
# With any of them, the batch utility and Thompson sampling go through the same call as the single-input case.
#
# ## References
#
# [1] Wilson, J. T., Moriconi, R., Hutter, F., and Deisenroth, M. P. [The reparameterization trick for acquisition functions](https://arxiv.org/abs/1712.00424). NIPS Workshop on Bayesian Optimization, 2017.
#
# [2] Astudillo, R. and Frazier, P. I. [Bayesian optimization of composite functions](https://proceedings.mlr.press/v97/astudillo19a.html). ICML, 2019.
#
# [3] Hernández-Lobato, J. M., Requeima, J., Pyzer-Knapp, E. O., and Aspuru-Guzik, A. [Parallel and distributed Thompson sampling for large-scale accelerated exploration of chemical space](https://proceedings.mlr.press/v70/hernandez-lobato17a.html). ICML, 2017.
#
# [4] Immer, A., Korzepa, M., and Bauer, M. [Improving predictions of Bayesian neural nets via local linearization](https://proceedings.mlr.press/v130/immer21a.html). AISTATS, 2021.
