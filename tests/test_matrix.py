import pytest
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from laplace.utils import Kron, block_diag
from laplace.utils import kron as kron_prod
from laplace.curvature import BackPackGGN
from tests.utils import get_psd_matrix, jacobians_naive


torch.set_default_tensor_type(torch.DoubleTensor)


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def small_model():
    model = torch.nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    setattr(model, 'output_size', 2)
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
    P = 20 * 3 + 20 + 2 * 20 + 2
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
    W = torch.randn(P)
    SW_kron = kron.bmm(W)
    SW_kron_decomp = kron_decomp.bmm(W, exponent=1)
    assert torch.allclose(SW_kron, SW_kron_decomp)


def test_logdet_consistent():
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    torch.manual_seed(7171)
    kfacs = [[get_psd_matrix(i) for i in sizes] for sizes in expected_sizes]
    kron = Kron(kfacs)
    kron_decomp = kron.decompose()
    assert torch.allclose(kron.logdet(), kron_decomp.logdet())


def test_bmm(small_model):
    model = small_model
    # model = single_output_model
    X = torch.randn(5, 3)
    y = torch.randn(5, 2)
    backend = BackPackGGN(model, 'regression', stochastic=False)
    loss, kron = backend.kron(X, y, N=5)
    kron_decomp = kron.decompose()
    Js, f = jacobians_naive(model, X)
    blocks = list()
    for F in kron.kfacs:
        if len(F) == 1:
            blocks.append(F[0])
        else:
            blocks.append(kron_prod(*F))
    S = block_diag(blocks)
    assert torch.allclose(S, S.T)
    assert torch.allclose(S.diagonal(), kron.diag())

    # test J @ Kron_decomp @ Jt (square form)
    JS = kron_decomp.bmm(Js, exponent=1)
    JS_true = Js @ S
    JSJ_true = torch.bmm(JS_true, Js.transpose(1,2))
    JSJ = torch.bmm(JS, Js.transpose(1,2))
    assert torch.allclose(JSJ, JSJ_true)
    assert torch.allclose(JS, JS_true)

    # test J @ Kron @ Jt (square form)
    JS_nodecomp = kron.bmm(Js)
    JSJ_nodecomp = torch.bmm(JS_nodecomp, Js.transpose(1,2))
    assert torch.allclose(JSJ_nodecomp, JSJ)
    assert torch.allclose(JS_nodecomp, JS)

    # test J @ S_inv @ J (funcitonal variance)
    JSJ = kron_decomp.inv_square_form(Js)
    S_inv = S.inverse()
    JSJ_true = torch.bmm(Js @ S_inv, Js.transpose(1,2))
    assert torch.allclose(JSJ, JSJ_true)

    # test J @ S^-1/2  (sampling)
    JS = kron_decomp.bmm(Js, exponent=-1/2)
    JSJ = torch.bmm(JS, Js.transpose(1,2))
    l, Q = torch.linalg.eigh(S_inv, UPLO='U')
    JS_true = Js @ Q @ torch.diag(torch.sqrt(l)) @ Q.T
    JSJ_true = torch.bmm(JS_true, Js.transpose(1,2))
    assert torch.allclose(JS, JS_true)
    assert torch.allclose(JSJ, JSJ_true)

    # test different Js shapes:
    # 2 - dimensional
    JS = kron_decomp.bmm(Js[:, 0, :].squeeze(), exponent=1)
    JS_nodecomp = kron.bmm(Js[:, 0, :].squeeze())
    JS_true = Js[:, 0, :].squeeze() @ S
    assert torch.allclose(JS, JS_true)
    assert torch.allclose(JS, JS_nodecomp)
    # 1 - dimensional
    JS = kron_decomp.bmm(Js[0, 0, :].squeeze(), exponent=1)
    JS_nodecomp = kron.bmm(Js[0, 0, :].squeeze())
    JS_true = Js[0, 0, :].squeeze() @ S
    assert torch.allclose(JS, JS_true)
    assert torch.allclose(JS, JS_nodecomp)


def test_matrix_consistent():
    expected_sizes = [[20, 3], [20], [2, 20], [2]]
    torch.manual_seed(7171)
    kfacs = [[get_psd_matrix(i) for i in sizes] for sizes in expected_sizes]
    kron = Kron(kfacs)
    kron_decomp = kron.decompose()
    assert torch.allclose(kron.to_matrix(), kron_decomp.to_matrix(exponent=1))
    assert torch.allclose(kron.to_matrix().inverse(), kron_decomp.to_matrix(exponent=-1))
    M_true = kron.to_matrix()
    M_true.diagonal().add_(3.4)
    kron_decomp += torch.tensor(3.4)
    assert torch.allclose(M_true, kron_decomp.to_matrix(exponent=1))
