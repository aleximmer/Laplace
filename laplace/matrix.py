from math import pow
import torch
import numpy as np

from laplace.utils import _is_valid_scalar, symeig


class Kron:
    # TODO: split up in Kron and KronEig
    # Kron.decompose() -> KronEig

    def __init__(self, kfacs):
        self.kfacs = kfacs
        self.eigvecs = None
        self.eigvals = None
        self._deltas = None

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
        self.eigvecs = eigvecs
        self.eigvals = eigvals

    @staticmethod
    def add_kfacs(kfacs_a, kfacs_b):
        return [[Hi.add(Hj) for Hi, Hj in zip(Fi, Fj)]
                for Fi, Fj in zip(kfacs_a, kfacs_b)]

    def __add__(self, other):
        if self.eigvals is not None:
            raise ValueError('Unsupported operation.')

        self._check_args_add(other)
        if isinstance(other, torch.Tensor):
            if self.deltas is None:
                self.deltas = other
            else:
                self.deltas += other
        elif isinstance(other, Kron):
            self.kfacs = self.add_kfacs(self.kfacs, other.kfacs)
        return self

    def __mul__(self, scalar):
        if self.eigvals is not None:
            raise ValueError('Unsupported operation.')

        if not _is_valid_scalar(scalar):
            raise ValueError('Invalid argument, can only multiply Kron with scalar.')

        self.kfacs = [[pow(scalar, 1/len(F)) * Hi for Hi in F] for F in self.kfacs]
        return self

    def logdet(self):
        logdet = 0
        for F in self.kfacs:
            if len(F) == 1:
                logdet += F[0].logdet()
            else:  # len(F) == 2
                Hi, Hj = F
                p_in, p_out = len(Hi), len(Hj) 
                logdet += p_out * Hi.logdet() + p_in * Hj.logdet()
        return logdet

    def __len__(self):
        return len(self.kfacs)

    def _check_args_add(self, other):
        if isinstance(other, torch.Tensor):
            if self.eigvals is None:
                raise ValueError('Adding tensor only supported on decomposed. Call first.')
            if len(other) == len(self) or len(other) == 1:
                return
            raise ValueError('Invalid tensor shape input.')

        elif not isinstance(other, Kron):
            raise ValueError('Can only add Kron or scalar')

        if len(self.kfacs) != len(other.kfacs):
            raise ValueError('Unmatched number of blocks.')

        for Fi, Fj in zip(self.kfacs, other.kfacs):
            for Hi, Hj in zip(Fi, Fj):
                if Hi.shape != Hj.shape:
                    raise ValueError('Unmatched Kronecker factors.')
                if Hi.device != Hj.device:
                    raise ValueError('Kronecker factors on different devices.')

    @property
    def deltas(self):
        return self._deltas

    @deltas.setter
    def deltas(self, new_deltas):
        if len(new_deltas) == 1:
            self._deltas = new_deltas.repeat(len(self))
        else:
            self._deltas = new_deltas

    __rmul__ = __mul__
    __radd__ = __add__
    __iadd__ = __add__
