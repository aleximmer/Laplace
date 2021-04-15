import pytest
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.matrix import Kron, KronDecomposed
from laplace.utils import kron as kron_prod
from tests.utils import get_psd_matrix


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


def test_init_from_model(model):
    kron = Kron.init_from_model(model, 'cpu')
    expected_sizes = [[20*20, 3*3], [20*20], [2*2, 20*20], [2*2]]
    for facs, exp_facs in zip(kron.kfacs, expected_sizes):
        for fi, exp_fi in zip(facs, exp_facs):
            assert torch.all(fi == 0)
            assert np.prod(fi.shape) == exp_fi


def test_addition(model):
    kron = Kron.init_from_model(model, 'cpu')
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    to_add = Kron([[torch.randn(i, i) for i in sizes] for sizes in expected_sizes])
    kron += to_add
    for facs, exp_facs in zip(kron.kfacs, to_add.kfacs):
        for fi, exp_fi in zip(facs, exp_facs):
            assert torch.allclose(fi, exp_fi)

def test_multiplication():
    # kron * x should be the same as the expanded kronecker product * x
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    kfacs = [[torch.randn(i, i) for i in sizes] for sizes in expected_sizes]
    kron = Kron(kfacs)
    kron *= 1.5
    for facs, exp_facs in zip(kron.kfacs, kfacs):
        if len(facs) == 1:
            assert torch.allclose(facs[0], 1.5 * exp_facs[0])
        else:  # len(facs) == 2
            exp = 1.5 * kron_prod(*exp_facs)
            facs = kron_prod(*facs)
            assert torch.allclose(exp, facs)
    
def test_decompose():
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    torch.manual_seed(7171)
    kfacs = [[get_psd_matrix(i) for i in sizes] for sizes in expected_sizes]
    kron = Kron(kfacs)
    kron_decomp = kron.decompose()
    for facs, Qs, ls in zip(kron.kfacs, kron_decomp.eigenvectors, kron_decomp.eigenvalues):
        if len(facs) == 1:
            H, Q, l = facs[0], Qs[0], ls[0]
            reconstructed = Q @ torch.diag(l) @ Q.T
            assert torch.allclose(H, reconstructed, rtol=1e-3)
        if len(facs) == 2:
            gtruth = kron_prod(facs[0], facs[1])
            rec_1 = Qs[0] @ torch.diag(ls[0]) @ Qs[0].T
            rec_2 = Qs[1] @ torch.diag(ls[1]) @ Qs[1].T
            reconstructed = kron_prod(rec_1, rec_2)
            assert torch.allclose(gtruth, reconstructed, rtol=1e-2)

def test_logdet_consistent():
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    torch.manual_seed(7171)
    kfacs = [[get_psd_matrix(i) for i in sizes] for sizes in expected_sizes]
    kron = Kron(kfacs)
    kron_decomp = kron.decompose()
    assert torch.allclose(kron.logdet(), kron_decomp.logdet())
    

def test_bmm():
    # TODO: test bmm
    assert False