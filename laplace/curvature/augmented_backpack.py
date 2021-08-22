import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.curvature.backpack import _cleanup


class AugBackPackInterface(CurvatureInterface):
    """Interface for Backpack backend when using augmented Laplace.
    This ensures that Jacobians, gradients, and the Hessian approximation remain differentiable
    and deals with S-augmented sized inputs (additional to the batch-dimension).
    """
    def __init__(self, model, likelihood, last_layer=False):
        super().__init__(model, likelihood, last_layer)
        extend(self._model)
        extend(self.lossfunc)

    @staticmethod
    def jacobians(model, x):
        # TODO: what should this return for x being `(batch, S, input_shape)`?
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        model : torch.nn.Module
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        model = extend(model)
        to_stack = []
        for i in range(model.output_size):
            model.zero_grad()
            out = model(x)
            with backpack(BatchGrad()):
                if model.output_size > 1:
                    out[:, i].sum().backward(retain_graph=True, create_graph=True)
                else:
                    out.sum().backward(retain_graph=True, create_graph=True)
                to_cat = []
                for param in model.parameters():
                    to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                    delattr(param, 'grad_batch')
                Jk = torch.cat(to_cat, dim=1)
            to_stack.append(Jk)
            if i == 0:
                f = out

        model.zero_grad()
        CTX.remove_hooks()
        _cleanup(model)
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2), f
        else:
            return Jk.unsqueeze(-1).transpose(1, 2), f

    def gradients(self, x, y):
        # TODO: what should this return for x being `(batch, S, input_shape)`?
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
            loss.backward(retain_graph=True, create_graph=True)
        Gs = torch.cat([p.grad_batch.data.flatten(start_dim=1)
                        for p in self._model.parameters()], dim=1)
        return Gs, loss


class AugBackPackGGN(AugBackPackInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Backpack.
    """
    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    def full(self, x, y, **kwargs):
        # TODO: what should this return for x being `(batch, S, input_shape)`?
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
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

        Js, f = self.jacobians(self.model, x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn
    
    def diag(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable for DA.')

    def kron(self, X, y, N, **kwargs):
        raise NotImplementedError('Unavailable for DA.')


class AugBackPackEF(AugBackPackInterface, EFInterface):
    """Implementation of `EFInterface` using Backpack.
    """

    def full(self, x, y, **kwargs):
        # TODO: what should this return for x being `(batch, S, input_shape)`?
        """Compute the full EF \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ef}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ef : torch.Tensor
            EF `(parameters, parameters)`
        """
        Gs, loss = self.gradients(x, y)
        H_ef = Gs.T @ Gs
        return self.factor * loss.detach(), self.factor * H_ef

    def kron(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable through Backpack.')

    def diag(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable for DA.')


def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
