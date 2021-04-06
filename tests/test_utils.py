import torch
from laplace.utils import invsqrt_precision


def test_sqrt_precision():
    X = torch.randn(20, 100)
    M = X @ X.T
    Scale = sqrt_precision(M)
    torch.allclose(Scale @ Scale.T, torch.inverse(M))

