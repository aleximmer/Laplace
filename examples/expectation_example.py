"""Make decisions with a Laplace posterior by maximizing expected utility.

The task is calibration: find the input whose (expensive) two-dimensional
output is closest to a target. A network learns the outputs, and Laplace
turns it into a posterior over functions. We then pick inputs that maximize
the expected utility under that posterior, first with utilities that look at
one input at a time and then with utilities that look at several at once.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils.data as data_utils
from torch import nn, optim

from laplace import BaseLaplace, Laplace
from laplace.utils.enums import HessianStructure, Likelihood, PredType, SubsetOfWeights


def true_fn(x: torch.Tensor) -> torch.Tensor:
    """The expensive function we can only query a few times.

    Parameters
    ----------
    x : torch.Tensor
        inputs in the unit square `(batch, 2)`

    Returns
    -------
    y : torch.Tensor
        outputs `(batch, 2)`
    """
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
    """The cheap, known outer function: negative squared distance to the target.

    Parameters
    ----------
    y : torch.Tensor
        outputs `(..., 2)`

    Returns
    -------
    value : torch.Tensor
        `(...)`, zero at the target and negative elsewhere
    """
    return -((y - Y_TARGET) ** 2).sum(-1)


def fit_laplace(seed: int = 123) -> tuple[BaseLaplace, torch.Tensor, torch.Tensor]:
    """Train a small network on 20 random queries and fit Laplace on top.

    Parameters
    ----------
    seed : int, default=123
        seed for the data, the initialization, and training

    Returns
    -------
    la : BaseLaplace
        the fitted Laplace approximation, with `enable_backprop=True`
    X_train, Y_train : torch.Tensor
        the queried inputs `(20, 2)` and outputs `(20, 2)`
    """
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
        Likelihood.REGRESSION,
        subset_of_weights=SubsetOfWeights.ALL,
        hessian_structure=HessianStructure.KRON,
        # The queries are noise-free, and the default of 1 makes the posterior far
        # too wide
        sigma_noise=0.05,
        enable_backprop=True,
    )
    la.fit(train_loader)
    la.optimize_prior_precision(PredType.GLM)

    return la, X_train, Y_train


def ei_cf(f: torch.Tensor, best: float) -> torch.Tensor:
    """Improvement of `g(f)` over the best value so far (EI for composite functions).

    Parameters
    ----------
    f : torch.Tensor
        samples of the outputs `(n_samples, batch, 2)`
    best : float
        best value of `g` observed so far

    Returns
    -------
    utility : torch.Tensor
        `(n_samples, batch)`
    """
    return (g(f) - best).clamp(min=0)


def pi_cf(f: torch.Tensor, best: float) -> torch.Tensor:
    """Whether `g(f)` beats the best value so far (PI for composite functions).

    Parameters
    ----------
    f : torch.Tensor
        samples of the outputs `(n_samples, batch, 2)`
    best : float
        best value of `g` observed so far

    Returns
    -------
    utility : torch.Tensor
        `(n_samples, batch)`, either 0 or 1
    """
    return (g(f) > best).float()


def expected_utility(
    la: BaseLaplace,
    x: torch.Tensor,
    utility: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Monte Carlo estimate of `E[u(f(x))]` for each input separately.

    The seed is fixed, so repeated calls reuse the same random numbers. This
    keeps the estimate a smooth, deterministic function of `x`, which is what
    a gradient-based optimizer needs.

    Parameters
    ----------
    la : BaseLaplace
        fitted Laplace approximation
    x : torch.Tensor
        inputs `(batch, 2)`
    utility : Callable
        maps output samples `(n_samples, batch, 2)` to `(n_samples, batch)`
    n_samples : int, default=256
        number of Monte Carlo samples
    seed : int, default=0
        seed of the random numbers

    Returns
    -------
    expected_utility : torch.Tensor
        `(batch,)`
    """
    f_samples = la.functional_samples(
        x, n_samples=n_samples, generator=torch.Generator().manual_seed(seed)
    )
    return utility(f_samples).mean(0)


def thompson_sample(
    la: BaseLaplace,
    candidates: torch.Tensor,
    seed: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Evaluate one network drawn from the posterior on all candidates.

    Parameters
    ----------
    la : BaseLaplace
        fitted Laplace approximation
    candidates : torch.Tensor
        inputs `(n_candidates, 2)`
    seed : int
        seed of the weight sample; reused for every chunk so that all chunks
        see the same network
    chunk_size : int, optional
        number of candidates per forward pass, all at once if `None`

    Returns
    -------
    f_sample : torch.Tensor
        `(n_candidates, 2)`
    """
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


def linearized_thompson_sample(
    la: BaseLaplace,
    candidates: torch.Tensor,
    seed: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Evaluate one draw of the linearized network on all candidates.

    The draw is `f(x) = f_MAP(x) + J(x) (theta - theta_MAP)` for a single weight
    sample `theta`, with the same Jacobians the GLM predictive uses. It follows
    the same posterior as the GLM predictive. Works for Laplace over all weights
    and for subnetwork Laplace.

    Parameters
    ----------
    la : BaseLaplace
        fitted Laplace approximation
    candidates : torch.Tensor
        inputs `(n_candidates, 2)`
    seed : int
        seed of the weight sample
    chunk_size : int, optional
        number of candidates per Jacobian computation, all at once if `None`.
        Each chunk holds a `(chunk_size, outputs, n_params)` Jacobian in memory.

    Returns
    -------
    f_sample : torch.Tensor
        `(n_candidates, 2)`
    """
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


def maximize(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 100,
    lr: float = 0.05,
) -> torch.Tensor:
    """Gradient ascent on `objective(x).sum()`, staying inside the unit square.

    Parameters
    ----------
    objective : Callable
        differentiable function of the inputs
    x0 : torch.Tensor
        starting inputs, e.g. one row per restart
    steps : int, default=100
        number of Adam steps
    lr : float, default=0.05
        Adam learning rate

    Returns
    -------
    x : torch.Tensor
        the optimized inputs, same shape as `x0`
    """
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
    """Format a tensor as a flat list rounded to three decimals, for printing."""
    return str([round(v, 3) for v in x.flatten().tolist()])


def main() -> None:
    """Fit Laplace, then pick inputs with EI-CF, PI-CF, qEI-CF, and Thompson sampling."""
    la, _, Y_train = fit_laplace()
    best = g(Y_train).max().item()

    print(f"Target input x*: {fmt(X_STAR)}, best g so far: {best:.4f}")

    # A grid of candidates, used where gradients don't help
    grid_1d = torch.linspace(0, 1, 50)
    grid = torch.cartesian_prod(grid_1d, grid_1d)

    # Utilities that look at one input: EI for composite functions, maximized by
    # gradient ascent from five random starting points
    def ei(x: torch.Tensor) -> torch.Tensor:
        return expected_utility(la, x, lambda f: ei_cf(f, best))

    x_restarts = maximize(ei, torch.rand(5, 2))
    x_best = x_restarts[ei(x_restarts).argmax()]
    print(f"EI-CF x_best: {fmt(x_best)}")

    # Swapping the utility is a one-line change. PI is an indicator, so its
    # gradient is zero almost everywhere; take the best grid point instead.
    pi = expected_utility(la, grid, lambda f: pi_cf(f, best))
    print(f"PI-CF x_best: {fmt(grid[pi.argmax()])}")

    # Utilities that look at several inputs: q-batch EI needs the outputs at
    # all q inputs jointly, so sample from the joint GLM predictive
    q = 3

    def qei(x: torch.Tensor) -> torch.Tensor:
        f_samples = la.functional_samples(
            x, n_samples=256, joint=True, generator=torch.Generator().manual_seed(0)
        )  # (256, q, 2)
        return (g(f_samples).max(-1).values - best).clamp(min=0).mean()

    x_batch = maximize(qei, torch.rand(q, 2))
    print(f"qEI-CF batch: {fmt(x_batch)}")

    # Thompson sampling: one function from the posterior, evaluated on every
    # candidate, then pick the candidate that function likes best. First with the
    # sampled weights plugged into the network itself...
    f_sample = thompson_sample(la, grid, seed=0, chunk_size=500)
    print(f"Thompson (nn) x_next: {fmt(grid[g(f_sample).argmax()])}")

    # ...then with the same weights in the linearized network
    f_sample = linearized_thompson_sample(la, grid, seed=0, chunk_size=500)
    print(f"Thompson (linearized) x_next: {fmt(grid[g(f_sample).argmax()])}")


if __name__ == "__main__":
    main()
