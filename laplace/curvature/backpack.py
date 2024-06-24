from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from backpack import backpack, extend, memory_cleanup
from backpack.context import CTX
from backpack.extensions import (
    KFAC,
    KFLR,
    BatchGrad,
    DiagGGNExact,
    DiagGGNMC,
    SumGradSquared,
)
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.utils import Kron, Likelihood


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ) -> None:
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )

        extend(self._model)
        extend(self.lossfunc)

    def jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension. Note that BackPACK doesn't play well
        with torch.func, so this method has to be overridden.

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
        if isinstance(x, MutableMapping):
            raise ValueError("BackPACK backend does not support dict-like inputs!")

        model = extend(self.model)
        to_stack = []
        for i in range(model.output_size):
            model.zero_grad()
            out = model(x)
            with backpack(BatchGrad()):
                if model.output_size > 1:
                    out[:, i].sum().backward(
                        create_graph=enable_backprop, retain_graph=enable_backprop
                    )
                else:
                    out.sum().backward(
                        create_graph=enable_backprop, retain_graph=enable_backprop
                    )
                to_cat = []
                for param in model.parameters():
                    to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                    delattr(param, "grad_batch")
                Jk = torch.cat(to_cat, dim=1)
                if self.subnetwork_indices is not None:
                    Jk = Jk[:, self.subnetwork_indices]
            to_stack.append(Jk)
            if i == 0:
                f = out

        model.zero_grad()
        CTX.remove_hooks()
        _cleanup(model)
        if model.output_size > 1:
            J = torch.stack(to_stack, dim=2).transpose(1, 2)
        else:
            J = Jk.unsqueeze(-1).transpose(1, 2)

        return (J, f) if enable_backprop else (J.detach(), f.detach())

    def gradients(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any], y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using Backpack's BatchGrad. Note that BackPACK doesn't play well
        with torch.func, so this method has to be overridden.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        loss : torch.Tensor
        """
        f = self.model(x)
        loss = self.lossfunc(f, y)
        with backpack(BatchGrad()):
            loss.backward()
        Gs = torch.cat(
            [p.grad_batch.data.flatten(start_dim=1) for p in self._model.parameters()],
            dim=1,
        )
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        return Gs, loss


class BackPackGGN(BackPackInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Backpack."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
    ):
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    def _get_diag_ggn(self) -> torch.Tensor:
        if self.stochastic:
            return torch.cat(
                [p.diag_ggn_mc.data.flatten() for p in self._model.parameters()]
            )
        else:
            return torch.cat(
                [p.diag_ggn_exact.data.flatten() for p in self._model.parameters()]
            )

    def _get_kron_factors(self) -> Kron:
        if self.stochastic:
            return Kron([p.kfac for p in self._model.parameters()])
        else:
            return Kron([p.kflr for p in self._model.parameters()])

    @staticmethod
    def _rescale_kron_factors(kron: Kron, M: int, N: int) -> Kron:
        # Renormalize Kronecker factor to sum up correctly over N data points with batches of M
        # for M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M / N
        return kron

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(x)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.likelihood == "regression" else f.view(-1, f.size(-1))
        y = y if self.likelihood == "regression" else y.view(-1)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        dggn = self._get_diag_ggn()
        if self.subnetwork_indices is not None:
            dggn = dggn[self.subnetwork_indices]

        return self.factor * loss.detach(), self.factor * dggn

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        context = KFAC if self.stochastic else KFLR
        f = self.model(x)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.likelihood == "regression" else f.view(-1, f.size(-1))
        y = y if self.likelihood == "regression" else y.view(-1)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        kron = self._get_kron_factors()
        kron = self._rescale_kron_factors(kron, len(y), N)

        return self.factor * loss.detach(), self.factor * kron


class BackPackEF(BackPackInterface, EFInterface):
    """Implementation of `EFInterface` using Backpack."""

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.model(x)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.likelihood == "regression" else f.view(-1, f.size(-1))
        y = y if self.likelihood == "regression" else y.view(-1)
        loss = self.lossfunc(f, y)
        with backpack(SumGradSquared()):
            loss.backward()
        diag_EF = torch.cat(
            [p.sum_grad_squared.data.flatten() for p in self._model.parameters()]
        )
        if self.subnetwork_indices is not None:
            diag_EF = diag_EF[self.subnetwork_indices]

        return self.factor * loss.detach(), self.factor * diag_EF

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        raise NotImplementedError("Unavailable through Backpack.")


def _cleanup(module: nn.Module) -> None:
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
