from abc import abstractproperty
import warnings
import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl.gradient import batch_gradient

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.matrix import Kron
from laplace.utils import _is_batchnorm


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True):
        if likelihood != 'classification':
            raise ValueError('This backend only supports classification currently.')
        super().__init__(model, likelihood, last_layer, differentiable)

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        Js = list()
        for i in range(self.model.output_size):
            def loss_fn(outputs, targets):
                return outputs[:, i].sum()

            f = batch_gradient(self.model, loss_fn, x, None, **self.backward_kwargs)
            Js.append(_get_batch_grad(self.model))
        Js = torch.stack(Js, dim=1)

        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using asdfghjkl's backend.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        """
        f = batch_gradient(self.model, self.lossfunc, x, y, **self.backward_kwargs)
        Gs = _get_batch_grad(self._model)
        loss = self.lossfunc(f, y)
        if self.differentiable:
            return Gs, loss
        return Gs.detach(), loss.detach()

    @abstractproperty
    def _ggn_type(self):
        raise NotImplementedError()

    def _get_kron_factors(self, curv, M):
        kfacs = list()
        for module in curv._model.modules():
            if _is_batchnorm(module):
                warnings.warn('BatchNorm unsupported for Kron, ignore.')
                continue

            stats = getattr(module, self._ggn_type, None)
            if stats is None:
                continue
            if hasattr(module, 'bias') and module.bias is not None:
                # split up bias and weights
                # TODO: clones are inefficient and should depend on necessity to diff wrt Jacs
                kfacs.append([stats.kron.B, stats.kron.A.clone()[:-1, :-1]])
                kfacs.append([stats.kron.B * stats.kron.A.clone()[-1, -1] / M])
            elif hasattr(module, 'weight'):
                p, q = np.prod(stats.kron.B.shape), np.prod(stats.kron.A.shape)
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return Kron(kfacs)

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
        curv = fisher_for_cross_entropy(self._model, self._ggn_type, SHAPE_DIAG,
                                        inputs=X, targets=y, **self.backward_kwargs)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss, self.factor * diag_ggn

    def kron(self, X, y, N, **kwargs):
        with torch.no_grad():
            if self.last_layer:
                f, X = self.model.forward_with_features(X)
            else:
                f = self.model(X)
            loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(self._model, self._ggn_type, SHAPE_KRON,
                                        inputs=X, targets=y, **self.backward_kwargs)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()


class AsdlGGN(AsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, stochastic=False):
        super().__init__(model, likelihood, last_layer, differentiable)
        self.stochastic = stochastic

    @property
    def _ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AsdlEF(AsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl.
    """

    @property
    def _ggn_type(self):
        return COV


def _flatten_after_batch(tensor: torch.Tensor):
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
                batch_grads.append(_flatten_after_batch(res['weight']))
            if 'bias' in res:
                batch_grads.append(_flatten_after_batch(res['bias']))
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return torch.cat(batch_grads, dim=1)
