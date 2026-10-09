"""Fit and query VaLLA through the Laplace factory on a small CPU dataset."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace


def main() -> None:
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
