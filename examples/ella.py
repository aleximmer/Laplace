"""Fit and query ELLA through the Laplace factory on a small CPU dataset."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace


def main() -> None:
    torch.manual_seed(7)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = inputs.sum(dim=-1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    ella = Laplace(
        torch.nn.Linear(2, 1),
        "regression",
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation="nystrom",
        subsample_size=2,
        n_eigenvalues=1,
        sigma_noise=0.2,
    )
    ella.fit(loader)
    mean, latent_covariance = ella(inputs[:2], joint=True)
    print("ELLA", mean.shape, latent_covariance.shape)


if __name__ == "__main__":
    main()
