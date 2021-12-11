import warnings
import numpy as np
import torch

from asdfghjkl.gradient import batch_aug_gradient
from asdfghjkl.fisher import fisher_for_cross_entropy
from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG

from laplace.curvature import CurvatureInterface, GGNInterface
from laplace.curvature.asdl import _get_batch_grad
from laplace.curvature import EFInterface
from laplace.matrix import Kron
from laplace.utils import _is_batchnorm


class AugAsdlInterface(CurvatureInterface):
    """Interface for Backpack backend when using augmented Laplace.
    This ensures that Jacobians, gradients, and the Hessian approximation remain differentiable
    and deals with S-augmented sized inputs (additional to the batch-dimension).
    """

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension, averages over aug dimension.

        Parameters
        ----------
        model : torch.nn.Module
        x : torch.Tensor
            input data `(batch, n_augs, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            averaged Jacobians over `n_augs` of shape `(batch, parameters, outputs)`
        f : torch.Tensor
            averaged output function over `n_augs` of shape `(batch, outputs)`
        """
        Js = list()
        for i in range(self.model.output_size):
            def loss_fn(outputs, _):
                return outputs.mean(dim=1)[:, i].sum()

            f = batch_aug_gradient(self.model, loss_fn, x, None, **self.backward_kwargs).mean(dim=1)
            Js.append(_get_batch_grad(self.model))
        Js = torch.stack(Js, dim=1)

        # set gradients to zero, differentiation here only serves Jacobian computation
        self.model.zero_grad()
        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def gradients(self, x, y):
        def loss_fn(outputs, targets):
            return self.lossfunc(outputs.mean(dim=1), targets)
        f = batch_aug_gradient(self._model, loss_fn, x, y, **self.backward_kwargs).mean(dim=1)
        Gs = _get_batch_grad(self._model)
        loss = self.lossfunc(f, y)
        if self.differentiable:
            return Gs, loss
        return Gs.detach(), loss.detach()

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
        if self.last_layer:
            raise ValueError('Not supported')
            f, X = self.model.forward_with_features(X)
        else:
            f = self.model(X).mean(dim=1)
        loss = self.lossfunc(f, y)

        curv = fisher_for_cross_entropy(self._model, self._ggn_type, SHAPE_DIAG,
                                        inputs=X, targets=y, **self.backward_kwargs)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss.detach(), self.factor * diag_ggn.detach()

    def kron(self, X, y, N, **wkwargs):
        if self.last_layer:
            raise ValueError('Not supported')
            f, X = self.model.forward_with_features(X)
        else:
            f = self.model(X).mean(dim=1)
        loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(self._model, self._ggn_type, SHAPE_KRON,
                                        inputs=X, targets=y, **self.backward_kwargs)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron


class AugAsdlGGN(AugAsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` with Asdl and augmentation support.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, stochastic=False):
        super().__init__(model, likelihood, last_layer, differentiable)
        self.stochastic = stochastic

    def full(self, x, y, **kwargs):
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, n_augs, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ggn : torch.Tensor
            GGN `(parameters, parameters)`
        """
        if self.stochastic:
            raise ValueError('Stochastic approximation not implemented for full GGN.')
        if self.last_layer:
            raise ValueError('Not yet tested/implemented for last layer.')

        Js, f = self.jacobians(x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn

    @property
    def _ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AugAsdlEF(AugAsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using Asdl and augmentation support.
    """

    @property
    def _ggn_type(self):
        return COV
