import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFAC, KFLR, SumGradSquared, BatchGrad
from backpack.context import CTX

from laplace.curvature import CurvatureInterface
from laplace.matrix import Kron


def cleanup(module):
    for child in module.children():
        cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)


class BackPackInterface(CurvatureInterface):

    def __init__(self, model, likelihood, last_layer=False):
        super().__init__(model, likelihood)
        self.last_layer = last_layer
        extend(self._model)
        extend(self.lossfunc)

    @property
    def _model(self):
        return self.model.last_layer if self.last_layer else self.model

    @staticmethod
    def jacobians(model, X):
        # Jacobians are batch x output x params
        model = extend(model)
        to_stack = []
        for i in range(model.output_size):
            model.zero_grad()
            out = model(X)
            with backpack(BatchGrad()):
                if model.output_size > 1:
                    out[:, i].sum().backward()
                else:
                    out.sum().backward()
                to_cat = []
                for param in model.parameters():
                    to_cat.append(param.grad_batch.detach().reshape(X.shape[0], -1))
                    delattr(param, 'grad_batch')
                Jk = torch.cat(to_cat, dim=1)
            to_stack.append(Jk)
            if i == 0:
                f = out.detach()

        # cleanup
        model.zero_grad()
        CTX.remove_hooks()
        cleanup(model)
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2), f
        else:
            return Jk.unsqueeze(-1).transpose(1, 2), f


class BackPackGGN(BackPackInterface):
    """[summary]

    MSELoss = |y-f|_2^2 -> d/df = -2(y-f)
    log N(y|f,1) propto 1/2|y-f|_2^2 -> d/df = -(y-f)
    --> factor for regression is 0.5 for loss and ggn
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
            loss.backward()
        dggn = self._get_diag_ggn()

        return self.factor * loss.detach(), self.factor * dggn

    def kron(self, X, y, N, **wkwargs) -> [torch.Tensor, Kron]:
        context = KFAC if self.stochastic else KFLR
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        kron = self._get_kron_factors()
        kron = self._rescale_kron_factors(kron, len(y), N)

        return self.factor * loss.detach(), self.factor * kron

    def full(self, X, y, **kwargs):
        # TODO: put in shared GGN interaface for both backends
        if self.stochastic:
            raise ValueError('Stochastic approximation not implemented for full GGN.')

        if self.last_layer:
            Js, f = self.last_layer_jacobians(self.model, X)
        else:
            Js, f = self.jacobians(self.model, X)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn


class BackPackEF(BackPackInterface):

    def _get_individual_gradients(self):
        return torch.cat([p.grad_batch.data.flatten(start_dim=1)
                          for p in self._model.parameters()], dim=1)

    def diag(self, X, y, **kwargs):
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(SumGradSquared()):
            loss.backward()
        diag_EF = torch.cat([p.sum_grad_squared.data.flatten()
                             for p in self._model.parameters()])

        return self.factor * loss.detach(), self.factor * diag_EF

    def kron(self, X, y, **kwargs):
        raise NotImplementedError()

    def full(self, X, y, **kwargs):
        # TODO: put in shared EF interface for both kazuki and backpack
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(BatchGrad()):
            loss.backward()
        # TODO: implement similarly to jacobians as staticmethod
        Gs = self._get_individual_gradients()
        H_ef = Gs.T @ Gs
        return self.factor * loss.detach(), self.factor * H_ef
