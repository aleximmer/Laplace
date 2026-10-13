"""Demonstrate VaLLA posterior fitting and querying on a small CPU dataset.

Run with ``python -m examples.valla`` from the repository root. The linear
network is randomly initialized to illustrate the API; applications should
supply a pretrained model because posterior fitting keeps its weights fixed.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace


def main() -> None:
    """Fit a regression posterior and print joint predictive tensor shapes."""
    torch.manual_seed(7)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = inputs.sum(dim=-1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    valla = Laplace(
        torch.nn.Linear(2, 1),
        "regression",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="variational",
        inducing_locations="random",
        num_inducing=2,
        sigma_noise=0.2,
    )
    valla.fit(loader, iterations=2, lr=1e-3)
    mean, latent_covariance = valla(inputs[:2], joint=True)
    print("VaLLA", mean.shape, latent_covariance.shape)


if __name__ == "__main__":
    main()
