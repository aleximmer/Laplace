from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, TypeAlias

import torch
from curvlinops import (
    EFLinearOperator,
    FisherMCLinearOperator,
    FisherType,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.utils import Kron, Likelihood

_FullLinearOperator: TypeAlias = (
    EFLinearOperator
    | FisherMCLinearOperator
    | GGNLinearOperator
    | HessianLinearOperator
)


class _ClassDimFirst(nn.Module):
    """Move the class axis of the model's logits from last to dim 1, e.g.
    `(B, L, C)` -> `(B, C, L)`, which is the layout Curvlinops expects for
    `CrossEntropyLoss` with extra dims.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any) -> torch.Tensor:
        out = self.model(x)
        return out.movedim(-1, 1) if out.ndim > 2 else out


class CurvlinopsInterface(CurvatureInterface):
    """Interface for Curvlinops backend. <https://github.com/f-dangel/curvlinops>"""

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

    @property
    def _linop_model(self) -> nn.Module:
        """The model that is passed to the Curvlinops linear operators."""
        if self.likelihood == Likelihood.CLASSIFICATION:
            return _ClassDimFirst(self.model)

        return self.model

    @property
    def _kron_fisher_type(self) -> FisherType:
        raise NotImplementedError

    @property
    def _linop_context(self) -> type[_FullLinearOperator]:
        raise NotImplementedError

    @staticmethod
    def _rescale_kron_factors(kron: Kron, M: int, N: int) -> Kron:
        # Renormalize Kronecker factor to sum up correctly over N data points with
        # batches of M. For M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M / N
        return kron

    def _get_kron_factors(self, linop: KFACLinearOperator, model: nn.Module) -> Kron:
        kfacs = list()
        # `model` must be the one passed to `linop`, so that the module names match
        for name, module in model.named_modules():
            if name not in linop._mapping.keys():
                continue

            A = linop._input_covariances[name]
            B = linop._gradient_covariances[name]

            if hasattr(module, "bias") and module.bias is not None:
                kfacs.append([B, A])
                kfacs.append([B])
            elif hasattr(module, "weight"):
                p, q = B.numel(), A.numel()
                if p == q == 1:
                    kfacs.append([B * A])
                else:
                    kfacs.append([B, A])
            else:
                raise ValueError(f"Whats happening with {module}?")
        return Kron(kfacs)

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if isinstance(x, (dict, MutableMapping)):
            kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        model = self._linop_model
        linop = KFACLinearOperator(
            model,
            self.lossfunc,
            self.params,
            [(x, y)],
            fisher_type=self._kron_fisher_type,
            separate_weight_and_bias=True,
            check_deterministic=False,  # To avoid overhead
            # `kwargs` for `mc_samples` when `stochastic=True` and `kfac_approx` to
            # choose between `'expand'` and `'reduce'`.
            # Defaults to `mc_samples=1` and `kfac_approx='expand'.
            **kwargs,
        )
        linop.compute_kronecker_factors()

        kron = self._get_kron_factors(linop, model)
        kron = self._rescale_kron_factors(kron, len(y), N)
        kron *= self.factor

        loss = self._loss(self.model(x), y)

        return self.factor * loss.detach(), kron

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fallback to torch.func backend for SubnetLaplace
        if self.subnetwork_indices is not None:
            return super().full(x, y, **kwargs)

        curvlinops_kwargs = {k: v for k, v in kwargs.items() if k != "N"}
        if isinstance(x, (dict, MutableMapping)):
            curvlinops_kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        linop = self._linop_context(
            self._linop_model,
            self.lossfunc,
            self.params,
            [(x, y)],
            check_deterministic=False,
            **curvlinops_kwargs,
        )

        p = next(self.model.parameters())
        H = torch.as_tensor(
            linop @ torch.eye(linop.shape[0]), device=p.device, dtype=p.dtype
        )

        f = self.model(x)
        loss = self._loss(f, y)

        return self.factor * loss.detach(), self.factor * H


class CurvlinopsGGN(CurvlinopsInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Curvlinops."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
    ) -> None:
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    @property
    def _kron_fisher_type(self) -> FisherType:
        return FisherType.MC if self.stochastic else FisherType.TYPE2

    @property
    def _linop_context(self) -> type[GGNLinearOperator | FisherMCLinearOperator]:
        return FisherMCLinearOperator if self.stochastic else GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops."""

    @property
    def _kron_fisher_type(self) -> FisherType:
        return FisherType.EMPIRICAL

    @property
    def _linop_context(self) -> type[EFLinearOperator]:
        return EFLinearOperator


class CurvlinopsHessian(CurvlinopsInterface):
    """Implementation of the full Hessian using Curvlinops."""

    @property
    def _linop_context(self) -> type[HessianLinearOperator]:
        return HessianLinearOperator
