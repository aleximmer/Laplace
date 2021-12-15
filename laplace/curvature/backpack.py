import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFAC, KFLR, SumGradSquared, BatchGrad
from backpack.context import CTX

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.matrix import Kron


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True):
        super().__init__(model, likelihood, last_layer, differentiable)
        extend(self._model)
        extend(self.lossfunc)

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

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
        self.model = extend(self.model)
        to_stack = []
        for i in range(self.model.output_size):
            self.model.zero_grad()
            out = self.model(x)
            with backpack(BatchGrad()):
                if self.model.output_size > 1:
                    out[:, i].sum().backward(**self.backward_kwargs)
                else:
                    out.sum().backward(**self.backward_kwargs)
                to_cat = []
                for param in self.model.parameters():
                    to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                    delattr(param, 'grad_batch')
                if self.differentiable:
                    Jk = torch.cat(to_cat, dim=1)
                else:
                    Jk = torch.cat(to_cat, dim=1).detach()
            to_stack.append(Jk)
            if i == 0:
                f = out

        if not self.differentiable:
            f = f.detach()

        self.model.zero_grad()
        CTX.remove_hooks()
        _cleanup(self.model)
        if self.model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2), f
        else:
            return Jk.unsqueeze(-1).transpose(1, 2), f

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using Backpack's BatchGrad.

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
        f = self.model(x)
        loss = self.lossfunc(f, y)
        with backpack(BatchGrad()):
            loss.backward(**self.backward_kwargs)
        Gs = torch.cat([p.grad_batch.data.flatten(start_dim=1)
                        for p in self._model.parameters()], dim=1)

        if self.differentiable:
            return Gs, loss
        return Gs.detach(), loss.detach()


class BackPackGGN(BackPackInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Backpack.
    """
    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    def _get_diag_ggn(self):
        if self.stochastic:
            return torch.cat([p.diag_ggn_mc.data.flatten() for p in self._model.parameters()])
        else:
            return torch.cat([p.diag_ggn_exact.data.flatten() for p in self._model.parameters()])

    def _get_kron_factors(self):
        if self.stochastic:
            return Kron([p.kfac for p in self._model.parameters()])
        else:
            return Kron([p.kflr for p in self._model.parameters()])

    @staticmethod
    def _rescale_kron_factors(kron, M, N):
        # Renormalize Kronecker factor to sum up correctly over N data points with batches of M
        # for M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M/N
        return kron

    def diag(self, X, y, **kwargs):
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward(**self.backward_kwargs)
        dggn = self._get_diag_ggn()

        if self.differentiable:
            return self.factor * loss, self.factor * dggn
        return self.factor * loss.detach(), self.factor * dggn.detach()

    def kron(self, X, y, N, **kwargs):
        context = KFAC if self.stochastic else KFLR
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward(**self.backward_kwargs)
        kron = self._get_kron_factors()
        kron = self._rescale_kron_factors(kron, len(y), N)

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()


class BackPackEF(BackPackInterface, EFInterface):
    """Implementation of `EFInterface` using Backpack.
    """

    def diag(self, X, y, **kwargs):
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(SumGradSquared()):
            loss.backward(**self.backward_kwargs)
        diag_EF = torch.cat([p.sum_grad_squared.data.flatten()
                             for p in self._model.parameters()])

        if self.differentiable:
            return self.factor * loss, self.factor * diag_EF
        return self.factor * loss.detach(), self.factor * diag_EF.detach()

    def kron(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable through Backpack.')


def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
