import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG
from asdfghjkl import fisher_for_cross_entropy

from laplace.curvature import CurvatureInterface
from laplace.matrix import Kron


class KazukiInterface(CurvatureInterface):

    def __init__(self, model, likelihood, last_layer=False):
        if likelihood != 'classification':
            raise ValueError('This backend does only support classification currently.')
        self.last_layer = last_layer
        super().__init__(model, likelihood)

    @property
    def ggn_type(self):
        raise NotImplementedError()

    def _get_kron_factors(self, curv, M):
        kfacs = list()
        for module in curv._model.modules():
            stats = getattr(module, self.ggn_type, None)
            if stats is None:
                continue
            if hasattr(module, 'bias') and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A[:-1, :-1]])
                kfacs.append([stats.kron.B * stats.kron.A[-1, -1] / M])
            elif hasattr(module, 'weight'):
                p, q = np.prod(stats.kron.B.shape), np.prod(stats.kron.A.shape)
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return kfacs

    @staticmethod
    def _rescale_kron_factors(kron, N):
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= 1/N
        return kron

    def diag(self, X, y, **kwargs):
        curv = fisher_for_cross_entropy(self.model, self.ggn_type, SHAPE_DIAG, inputs=X, targets=y)
        diag_ggn = curv.matrices_to_vector(None)
        with torch.no_grad():
            loss = self.lossfunc(self.model(X), y)
        return self.factor * loss, self.factor * diag_ggn

    def kron(self, X, y, N, **wkwargs) -> [torch.Tensor, Kron]:
        M = len(y)
        curv = fisher_for_cross_entropy(self.model, self.ggn_type, SHAPE_KRON, inputs=X, targets=y)
        kron = Kron(self._get_kron_factors(curv, M))
        kron = self._rescale_kron_factors(kron, N)
        with torch.no_grad():
            loss = self.lossfunc(self.model(X), y)
        return self.factor * loss, self.factor * kron

    def full(self, X, y, **kwargs):
        raise NotImplementedError()


class KazukiGGN(KazukiInterface):

    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    @property
    def ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class KazukiEF(KazukiInterface):

    @property
    def ggn_type(self):
        return COV
