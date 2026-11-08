"""CUDA contracts for a known noise std per output."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.likelihoods import GaussianLikelihood

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch required"),
]


@pytest.mark.parametrize("structure", ["full", "diag", "kron"])
def test_cpu_noise_with_cuda_model(structure):
    # The noise vector stays on the CPU while the model and the data are on CUDA
    torch.manual_seed(0)
    device = torch.device("cuda")

    X, y = torch.randn(12, 3, device=device), torch.randn(12, 2, device=device)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=6)

    likelihood = GaussianLikelihood(sigma_noise=torch.tensor([0.5, 2.0]))
    la = Laplace(model, likelihood, hessian_structure=structure)
    la.fit(loader)

    assert la.log_marginal_likelihood().isfinite()
