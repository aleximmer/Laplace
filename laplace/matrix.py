from math import pow
import torch
import numpy as np
from typing import Union

from laplace.utils import _is_valid_scalar, symeig


class Kron:
    
    def __init__(self, kfacs):
        self.kfacs = kfacs
       
    @classmethod
    def init_from_model(cls, model, device):
        # FIXME: this does not work for BatchNorm!!
        # TODO: rewrite this functionality in terms of module types
        kfacs = list()
        for p in model.parameters():
            if p.ndim == 1:  # bias
                P = p.size(0)
                kfacs.append([torch.zeros(P, P, device=device)])
            elif 4 >= p.ndim >= 2:  # fully connected or conv
                if p.ndim == 2:  # fully connected
                    P_in, P_out = p.size()
                elif p.ndim > 2:
                    P_in, P_out = p.shape[0], np.prod(p.shape[1:])

                kfacs.append([
                    torch.zeros(P_in, P_in, device=device),
                    torch.zeros(P_out, P_out, device=device)
                ])
            else:
                raise ValueError('Invalid parameter shape in network.')
        return cls(kfacs)

    def __add__(self, other):
        if not isinstance(other, Kron):
            raise ValueError('Can only add Kron to Kron.')
        
        self.kfacs = [[Hi.add(Hj) for Hi, Hj in zip(Fi, Fj)]
                      for Fi, Fj in zip(self.kfacs, other.kfacs)]
        return self

    def __mul__(self, scalar: Union[float, torch.Tensor]):
        if not _is_valid_scalar(scalar):
            raise ValueError('Input not valid python or torch scalar.')
        
        # distribute factors evenly so that each group is multiplied by factor
        self.kfacs = [[pow(scalar, 1/len(F)) * Hi for Hi in F] for F in self.kfacs]
        return self

    def __len__(self):
        return len(self.kfacs)

    def decompose(self):
        eigvecs, eigvals = list(), list()
        for F in self.kfacs:
            Qs, ls = list(), list()
            for Hi in F:
                l, Q = symeig(Hi)
                Qs.append(Q)
                ls.append(l)
            eigvecs.append(Qs)
            eigvals.append(ls)
        return KronDecomposed(eigvecs, eigvals)

    def logdet(self) -> torch.Tensor:
        logdet = 0
        for F in self.kfacs:
            if len(F) == 1:
                logdet += F[0].logdet()
            else:  # len(F) == 2
                Hi, Hj = F
                p_in, p_out = len(Hi), len(Hj) 
                logdet += p_out * Hi.logdet() + p_in * Hj.logdet()
        return logdet

    # inplace and permuted operations
    __radd__ = __add__
    __iadd__ = __add__
    __rmul__ = __mul__
    __imul__ = __mul__


class KronDecomposed:

    def __init__(self, eigenvectors, eigenvalues, deltas=None, dampen=False):
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        device = eigenvectors[0][0].device
        if deltas is None:
            self.deltas = torch.zeros(len(self), device=device)
        else:
            self._check_deltas(deltas)
            self.deltas = deltas
        self.dampen = dampen

    def _check_deltas(self, deltas: torch.Tensor):
        if not isinstance(deltas, torch.Tensor):
            raise ValueError('Can only add torch.Tensor to KronDecomposed.')

        if (deltas.ndim == 0  # scalar
            or (deltas.ndim == 1  # vector of length 1 or len(self)
                and (len(deltas) == 1 or len(deltas) == len(self)))):
            return
        else:
            raise ValueError('Invalid shape of delta added to KronDecomposed.')

    def __add__(self, deltas: torch.Tensor):
        self._check_deltas(deltas)
        return KronDecomposed(self.eigenvectors, self.eigenvalues, self.deltas + deltas)

    def __mul__(self, scalar):
        if not _is_valid_scalar(scalar):
            raise ValueError('Invalid argument, can only multiply Kron with scalar.')

        eigenvalues = [[pow(scalar, 1/len(l)) * l for l in ls] for ls in self.eigenvalues]
        return KronDecomposed(self.eigenvectors, eigenvalues, self.deltas)

    def __len__(self) -> int:
        return len(self.eigenvalues)
    
    def logdet(self) -> torch.Tensor:
        # compute \sum_l log det (kron_l + delta I_l)
        logdet = 0
        for ls, delta in zip(self.eigenvalues, self.deltas):
            if len(ls) == 1:  # not KFAC just full
                logdet += torch.log(ls[0] + delta).sum()
            elif len(ls) == 2:
                l1, l2 = ls
                if self.dampen:
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    logdet += torch.log(torch.ger(l1d, l2d)).sum()
                else:
                    logdet += torch.log(torch.ger(l1, l2) + delta).sum()
            else:
                raise ValueError('Too many Kronecker factors. Something went wrong.')
        return logdet

    def _bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        # self @ W[batch, k, params]
        assert len(W.size()) == 3
        B, K, P = W.size()
        W = W.reshape(B * K, P)
        cur_p = 0
        SW = list()
        for ls, Qs, delta in zip(self.eigenvalues, self.eigenvectors, self.deltas):
            if len(ls) == 1:
                # just Q (Lambda + delta) Q^T W_p
                Q, l, p = Qs[0], ls[0], len(ls[0])
                print(Q.mean(), l.mean(), delta)
                ldelta_exp = torch.pow(l + delta, exponent).reshape(-1, 1)
                W_p = W[:, cur_p:cur_p+p].T
                SW.append((Q @ (ldelta_exp * (Q.T @ W_p))).T)
            elif len(ls) == 2:
                # not so easy to explain...
                Q1, Q2 = Qs
                l1, l2 = ls
                print(Q1.mean(), Q2.mean(), l1.mean(), l2.mean(), delta)
                p = len(l1) * len(l2)
                if self.dampen:
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    ldelta_exp = torch.pow(torch.ger(l1d, l2d), exponent).unsqueeze(0)
                else:
                    ldelta_exp = torch.pow(torch.ger(l1, l2) + delta, exponent).unsqueeze(0)
                p_in, p_out = len(l1), len(l2)
                W_p = W[:, cur_p:cur_p+p].reshape(B * K, p_in, p_out)
                W_p = (Q1.T @ W_p @ Q2) * ldelta_exp
                W_p = Q1 @ W_p @ Q2.T
                SW.append(W_p.reshape(B * K, p_in * p_out))
                cur_p += p
            else:
                raise AttributeError('Shape mismatch')
        SW = torch.cat(SW, dim=1).reshape(B, K, P)
        return SW

    def inv_square_form(self, W: torch.Tensor) -> torch.Tensor:
        # W either Batch x K x params or Batch x params
        SW = self._bmm(W, exponent=-1)
        return torch.bmm(W, SW.transpose(1, 2))

    def bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        # self @ W with W[params], W[batch, params], W[batch, classes, params]
        # returns SW[batch, classes, params]
        if len(W) == 1:
            return self._bmm(W.unsqueeze(0).unsqueeze(0)).squeeze()
        elif len(W) == 2:
            return self._bmm(W.unsqueeze(1)).squeeze()
        elif len(W) == 3:
            return self._bmm(W)

    # FIXME: iadd imul should change mutable types in principle.
    __radd__ = __add__
    __iadd__ = __add__
    __rmul__ = __mul__
    __imul__ = __mul__
