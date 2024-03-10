import torch
import numpy as np

from curvlinops import (
    HessianLinearOperator, GGNLinearOperator, FisherMCLinearOperator, EFLinearOperator,
    KFACLinearOperator
)

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.utils import Kron


class CurvlinopsInterface(CurvatureInterface):
    """Interface for Curvlinops backend. <https://github.com/f-dangel/curvlinops>
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)

    @property
    def _kron_fisher_type(self):
        raise NotImplementedError

    @property
    def _linop_context(self):
        raise NotImplementedError

    @staticmethod
    def _rescale_kron_factors(kron, M, N):
        # Renormalize Kronecker factor to sum up correctly over N data points with batches of M
        # for M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M/N
        return kron

    def _get_kron_factors(self, linop):
        kfacs = list()
        for name, module in self.model.named_modules():
            if name not in linop._mapping.keys():
                continue

            A = linop._input_covariances[name]
            B = linop._gradient_covariances[name]

            if hasattr(module, 'bias') and module.bias is not None:
                kfacs.append([B, A])
                kfacs.append([B])
            elif hasattr(module, 'weight'):
                p, q = B.numel(), A.numel()
                if p == q == 1:
                    kfacs.append([B * A])
                else:
                    kfacs.append([B, A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return Kron(kfacs)

    def kron(self, X, y, N, **kwargs):
        linop = KFACLinearOperator(
            self.model, self.lossfunc, self.params, [(X, y)],
            fisher_type=self._kron_fisher_type,
            loss_average=None,  # Since self.lossfunc is sum
            separate_weight_and_bias=True,
            check_deterministic=False,  # To avoid overhead
            # `kwargs` for `mc_samples` when `stochastic=True` and `kfac_approx` to
            # choose between `'expand'` and `'reduce'`.
            # Defaults to `mc_samples=1` and `kfac_approx='expand'.
            **kwargs,
        )
        linop._compute_kfac()

        kron = self._get_kron_factors(linop)
        kron = self._rescale_kron_factors(kron, len(y), N)
        kron *= self.factor

        loss = self.lossfunc(self.model(X), y)

        return self.factor * loss.detach(), kron

    def full(self, X, y, **kwargs):
        # Fallback to torch.func backend for SubnetLaplace
        if self.subnetwork_indices is not None:
            return super().full(X, y, **kwargs)

        curvlinops_kwargs = {k: v for k, v in kwargs.items() if k != 'N'}
        linop = self._linop_context(self.model, self.lossfunc, self.params, [(X, y)],
                                    check_deterministic=False, **curvlinops_kwargs)
        H = torch.as_tensor(
            linop @ np.eye(linop.shape[0]),
            dtype=X.dtype,
            device=X.device
        )

        f = self.model(X)
        loss = self.lossfunc(f, y)

        return self.factor * loss.detach(), self.factor * H


class CurvlinopsGGN(CurvlinopsInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Curvlinops."""
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.stochastic = stochastic

    @property
    def _kron_fisher_type(self):
        return 'mc' if self.stochastic else 'type-2'

    @property
    def _linop_context(self):
        return FisherMCLinearOperator if self.stochastic else GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops."""

    @property
    def _kron_fisher_type(self):
        return 'empirical'

    @property
    def _linop_context(self):
        return EFLinearOperator


class CurvlinopsHessian(CurvlinopsInterface):
    """Implementation of the full Hessian using Curvlinops."""

    @property
    def _linop_context(self):
        return HessianLinearOperator
