from laplace.curvature.curvature import EFInterface
import torch

from asdfghjkl.gradient import batch_aug_gradient

from laplace.curvature import CurvatureInterface, GGNInterface
from laplace.curvature.asdl import _get_batch_grad


class AugAsdlInterface(CurvatureInterface):
    """Interface for Backpack backend when using augmented Laplace.
    This ensures that Jacobians, gradients, and the Hessian approximation remain differentiable
    and deals with S-augmented sized inputs (additional to the batch-dimension).
    """
    def __init__(self, model, likelihood, last_layer=False):
        super().__init__(model, likelihood, last_layer)

    @staticmethod
    def jacobians(model, x):
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
        for i in range(model.output_size):
            def loss_fn(outputs, _):
                return outputs.mean(dim=1)[:, i].sum()

            f = batch_aug_gradient(model, loss_fn, x, None).mean(dim=1)
            Js.append(_get_batch_grad(model))
        Js = torch.stack(Js, dim=1)

        # set gradients to zero, differentiation here only serves Jacobian computation
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        return Js, f

    def gradients(self, x, y):
        def loss_fn(outputs, targets):
            return self.lossfunc(outputs.mean(dim=1), targets)
        f = batch_aug_gradient(self._model, loss_fn, x, y).mean(dim=1)
        Gs = _get_batch_grad(self._model)
        loss = self.lossfunc(f, y)
        return Gs, loss


class AugAsdlGGN(AugAsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` with Asdl and augmentation support.
    """
    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
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

        Js, f = self.jacobians(self.model, x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn

    def diag(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable for DA.')

    def kron(self, X, y, N, **kwargs):
        raise NotImplementedError('Unavailable for DA.')


class AugAsdlEF(AugAsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using Asdl and augmentation support.
    """

    def diag(self, X, y, **kwargs):
        raise NotImplementedError('Unavailable for DA.')

    def kron(self, X, y, N, **kwargs):
        raise NotImplementedError('Unavailable for DA.')