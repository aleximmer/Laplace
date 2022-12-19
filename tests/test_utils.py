import torch
from laplace.utils import invsqrt_precision, diagonal_add_scalar, symeig, normal_samples


def test_sqrt_precision():
    X = torch.randn(20, 100)
    M = X @ X.T
    Scale = invsqrt_precision(M)
    torch.allclose(Scale @ Scale.T, torch.inverse(M))


def test_diagonal_add_scalar():
    M = torch.randn(50, 50)
    M_plus_diag = diagonal_add_scalar(M, 1.98)
    diag = 1.98 * torch.eye(len(M))
    torch.allclose(M_plus_diag, M + diag)


def test_symeig_custom():
    X = torch.randn(20, 100)
    M = X @ X.T
    l1, W1 = torch.linalg.eigh(M, UPLO='U')
    l2, W2 = symeig(M)
    assert torch.allclose(l1, l2)
    assert torch.allclose(W1, W2)
    

def test_symeig_custom_low_rank():
    X = torch.randn(1000, 10)
    M = X @ X.T
    l1, W1 = torch.linalg.eigh(M, UPLO='U')
    l2, W2 = symeig(M)
    # symeig should fail for low-rank
    assert not torch.all(l1 >= 0.0)
    # test clamping to zeros
    assert torch.all(l2 >= 0.0)

    
def test_diagonal_normal_samples():
    mean = torch.randn(10, 2)
    var = torch.exp(torch.randn(10, 2))
    generator = torch.Generator()
    gen_state = generator.get_state()
    samples = normal_samples(mean, var, n_samples=100, generator=generator)
    assert samples.shape == torch.Size([100, 10, 2])
    # reset generator state
    generator.set_state(gen_state)
    same_samples = normal_samples(mean, var, n_samples=100, generator=generator)
    assert torch.allclose(samples, same_samples)

    
def test_multivariate_normal_samples():
    mean = torch.randn(10, 2)
    rndns = torch.randn(10, 2, 10) / 100
    var = torch.matmul(rndns, rndns.transpose(1, 2))
    generator = torch.Generator()
    gen_state = generator.get_state()
    samples = normal_samples(mean, var, n_samples=100, generator=generator)
    assert samples.shape == torch.Size([100, 10, 2])
    # reset generator state
    generator.set_state(gen_state)
    same_samples = normal_samples(mean, var, n_samples=100, generator=generator)
    assert torch.allclose(samples, same_samples)
