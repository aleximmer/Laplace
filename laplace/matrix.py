from abc import ABC, abstractmethod
import torch
import numpy as np

from laplace.utils import invsqrt_precision


class Hessian(ABC):

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    @abstractmethod
    def __mul__(self, scalar):
        pass


class BlockDiag(Hessian):

    def __init__(self, blocks):
        self.blocks = blocks

    @classmethod
    def init_from_model(cls, model, device):
        n_params_per_param = [np.prod(p.shape) for p in model.parameters()]
        blocks = [torch.zeros(P, P, device=device) for P in n_params_per_param]
        return cls(blocks)

    def __add__(self, other):
        self._check_args_add(other)
        return BlockDiag([Hi.add(Hj) for Hi, Hj in zip(self.blocks, other.blocks)])

    def __iadd__(self, other):
        self._check_args_add(other)
        self.blocks = [Hi.add(Hj) for Hi, Hj in zip(self.blocks, other.blocks)]
        return self

    def __mul__(self, scalar):
        self.blocks = [scalar * Hi for Hi in self.blocks]
        return self

    def logdet(self):
        return sum([Hi.logdet() for Hi in self.blocks])

    def invsqrt(self):
        return BlockDiag([invsqrt_precision(Hi) for Hi in self.blocks])

    def square(self):
        return BlockDiag([Hi @ Hi.T for Hi in self.blocks])

    def _check_args_add(self, other):
        if len(self.blocks) != len(other.blocks):
            raise ValueError('Unmatched number of blocks.')

        for Hi, Hj in zip(self.blocks, other.blocks):
            if Hi.shape != Hj.shape:
                raise ValueError('Unmatched individual blocks.')
            if Hi.device != Hj.device:
                raise ValueError('Individual blocks on different devices.')


class Kron(Hessian):
    """[summary]

    Args:
        Hessian ([type]): [description]

    Attributes:
        kfacs : list[lists]
            inner lists are Kronecker factors or a single PxP GGN (e.g. for biases)
    """

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

    @staticmethod
    def add_kfacs(kfacs_a, kfacs_b):
        return [[Hi.add(Hj) for Hi, Hj in zip(Fi, Fj)]
                for Fi, Fj in zip(self.kfacs, other.kfacs)]

    def __add__(self, other):
        self._check_args_add(other)
        return Kron(self.add_kfacs(self.kfacs, other.kfacs))

    def __iadd__(self, other):
        self._check_args_add(other)
        self.kfacs = self.add_kfacs(self.kfacs, other.kfacs)
        return self

    def __mul__(self, scalar):
        self.blocks = [[scalar * Hi for Hi in F] for F in self.blocks]
        return self

    def logdet(self):
        return sum([Hi.logdet() for Hi in self.blocks])

    def invsqrt(self):
        return BlockDiag([invsqrt_precision(Hi) for Hi in self.blocks])

    def square(self):
        return BlockDiag([Hi @ Hi.T for Hi in self.blocks])

    def _check_args_add(self, other):
        if len(self.blocks) != len(other.blocks):
            raise ValueError('Unmatched number of blocks.')

        for Hi, Hj in zip(self.blocks, other.blocks):
            if Hi.shape != Hj.shape:
                raise ValueError('Unmatched individual blocks.')
            if Hi.device != Hj.device:
                raise ValueError('Individual blocks on different devices.')