import warnings
import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl.gradient import batch_gradient

from laplace.curvature import CurvatureInterface
from laplace.matrix import Kron
from laplace.utils import is_batchnorm


class KazukiInterface(CurvatureInterface):

    def __init__(self, model, likelihood, last_layer=False):
        if likelihood != 'classification':
            raise ValueError('This backend does only support classification currently.')
        super().__init__(model, likelihood, last_layer)

    @staticmethod
    def jacobians(model, X):
        Js = list()
        for i in range(model.output_size):
            def loss_fn(outputs, targets):
                return outputs[:, i].sum()

            f = batch_gradient(model, loss_fn, X, None).detach()
            Js.append(_get_batch_grad(model))
        Js = torch.stack(Js, dim=1)
        return Js, f

    @property
    def ggn_type(self):
        raise NotImplementedError()

    def _get_kron_factors(self, curv, M):
        kfacs = list()
        for module in curv._model.modules():
            if is_batchnorm(module):
                warnings.warn('BatchNorm unsupported for Kron, ignore.')
                continue

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
        with torch.no_grad():
            if self.last_layer:
                f, X = self.model.forward_with_features(X)
            else:
                f = self.model(X)
            loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(self._model, self.ggn_type, SHAPE_DIAG, inputs=X, targets=y)
        diag_ggn = curv.matrices_to_vector(None)
        return self.factor * loss, self.factor * diag_ggn

    def kron(self, X, y, N, **wkwargs) -> [torch.Tensor, Kron]:
        with torch.no_grad():
            if self.last_layer:
                f, X = self.model.forward_with_features(X)
            else:
                f = self.model(X)
            loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(self._model, self.ggn_type, SHAPE_KRON, inputs=X, targets=y)
        M = len(y)
        kron = Kron(self._get_kron_factors(curv, M))
        kron = self._rescale_kron_factors(kron, N)
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


def flatten_after_batch(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)


def _get_batch_grad(model):
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, 'op_results'):
            res = module.op_results['batch_grads']
            if 'weight' in res:
                batch_grads.append(flatten_after_batch(res['weight']))
            if 'bias' in res:
                batch_grads.append(flatten_after_batch(res['bias']))
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return torch.cat(batch_grads, dim=1)
