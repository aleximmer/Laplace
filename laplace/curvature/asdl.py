import warnings
import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG, SHAPE_FULL
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl.hessian import hessian_eigenvalues, hessian_for_loss
from asdfghjkl.gradient import batch_gradient

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.utils import Kron, _is_batchnorm

EPS = 1e-6


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend.
    """

    def jacobians(self, x, enable_backprop=False):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        enable_backprop : bool, default = False
            whether to enable backprop through the Js and f w.r.t. x

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

            f = batch_gradient(self.model, loss_fn, x, None).detach()
            Jk = _get_batch_grad(self.model)
            if self.subnetwork_indices is not None:
                Jk = Jk[:, self.subnetwork_indices]
            Js.append(Jk)
        Js = torch.stack(Js, dim=1)
        return Js, f

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
        f = batch_gradient(self.model, self.lossfunc, x, y).detach()
        Gs = _get_batch_grad(self._model)
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        loss = self.lossfunc(f, y)
        return Gs, loss

    @property
    def _ggn_type(self):
        raise NotImplementedError

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
                                        inputs=X, targets=y)
        diag_ggn = curv.matrices_to_vector(None)
        if self.subnetwork_indices is not None:
            diag_ggn = diag_ggn[self.subnetwork_indices]
        return self.factor * loss, self.factor * diag_ggn

    def kron(self, X, y, N, **wkwargs):
        with torch.no_grad():
            if self.last_layer:
                f, X = self.model.forward_with_features(X)
            else:
                f = self.model(X)
            loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(self._model, self._ggn_type, SHAPE_KRON,
                                        inputs=X, targets=y)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)
        return self.factor * loss, self.factor * kron


class AsdlHessian(AsdlInterface):

    def __init__(self, model, likelihood, last_layer=False, low_rank=10):
        super().__init__(model, likelihood, last_layer)
        self.low_rank = low_rank

    @property
    def _ggn_type(self):
        raise NotImplementedError()

    def full(self, x, y, **kwargs):
        hessian_for_loss(self.model, self.lossfunc, SHAPE_FULL, x, y)
        H = self._model.hessian.data
        loss = self.lossfunc(self.model(x), y).detach()
        return self.factor * loss, self.factor * H

    def eig_lowrank(self, data_loader):
        # compute truncated eigendecomposition of the Hessian, only keep eigvals > EPS
        eigvals, eigvecs = hessian_eigenvalues(self.model, self.lossfunc, data_loader,
                                               top_n=self.low_rank, max_iters=self.low_rank*10)
        eigvals = torch.from_numpy(np.array(eigvals))
        mask = (eigvals > EPS)
        eigvecs = torch.stack([torch.cat([p.flatten() for p in params])
                               for params in eigvecs], dim=1)[:, mask]
        device = eigvecs.device
        eigvals = eigvals[mask].to(eigvecs.dtype).to(device)
        loss = sum([self.lossfunc(self.model(x.to(device)).detach(), y.to(device)) for x, y in data_loader])
        return eigvecs, self.factor * eigvals, self.factor * loss


class AsdlGGN(AsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        if likelihood != 'classification':
            raise ValueError('This backend only supports classification currently.')
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.stochastic = stochastic

    @property
    def _ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AsdlEF(AsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False):
        if likelihood != 'classification':
            raise ValueError('This backend only supports classification currently.')
        super().__init__(model, likelihood, last_layer)

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
