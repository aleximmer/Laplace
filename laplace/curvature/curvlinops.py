import torch
import numpy as np

import curvlinops as cvls

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.utils import Kron

from collections import UserDict

from typing import *


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

    def _get_kron_factors(self, linop, M):
        kfacs = list()
        print([(k, c.shape) for k, c in linop._input_covariances.items()])
        print([(k, c.shape) for k, c in linop._gradient_covariances.items()])
        for mod_name, param_pos in linop._mapping.items():
            # print(param_pos)
            aaT = linop._input_covariances[mod_name]
            ggT = linop._gradient_covariances[mod_name]
            kfacs.append([ggT, aaT])  # Because the weights in PyTorch is (out_dim, in_dim)
        return Kron(kfacs)

    def kron(self, X, y, N, **kwargs):
        linop = cvls.KFACLinearOperator(
            self.model, self.lossfunc, self.params, [(X, y)],
            fisher_type=self._kron_fisher_type,
            loss_average=None,  # Since self.lossfunc is sum
            separate_weight_and_bias=False
        )
        linop._compute_kfac()

        M = len(y)
        kron = self._get_kron_factors(linop, M)
        kron = self._rescale_kron_factors(kron, len(y), N)

        loss = self.lossfunc(self.model(X), y)

        return self.factor * loss.detach(), self.factor * kron


class CurvlinopsGGN(CurvlinopsInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Curvlinops.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.stochastic = stochastic

    @property
    def _kron_fisher_type(self):
        return 'mc' if self.stochastic else 'type-2'

    @property
    def _linop_context(self):
        return cvls.FisherMCLinearOperator if self.stochastic else cvls.GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops.
    """

    @property
    def _kron_fisher_type(self):
        return 'empirical'

    @property
    def _linop_context(self):
        return cvls.EFLinearOperator


class CurvlinopsHessian(CurvlinopsInterface):

    def __init__(self, model, likelihood, last_layer=False, low_rank=10):
        super().__init__(model, likelihood, last_layer)
        self.low_rank = low_rank

    @property
    def _linop_context(self):
        return cvls.HessianLinearOperator

    def full(self, X, y, **kwargs):
        linop = self._linop_context(self.model, self.lossfunc, self.params, [(X, y)])
        H = linop @ np.eye(linop.shape[0])

        f = self.model(X)
        loss = self.lossfunc(f, y)

        return self.factor * loss.detach(), self.factor * H
