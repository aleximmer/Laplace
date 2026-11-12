from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from importlib.util import find_spec
from typing import Any, ClassVar

import numpy as np
import torch

if find_spec("asdfghjkl") is not None:
    from asdfghjkl import (
        COV,
        FISHER_EXACT,
        FISHER_MC,
        SHAPE_DIAG,
        SHAPE_FULL,
        SHAPE_KRON,
        fisher_for_cross_entropy,
    )
    from asdfghjkl.gradient import batch_gradient
    from asdfghjkl.hessian import hessian_eigenvalues, hessian_for_loss

from torch import nn
from torch.utils.data import DataLoader

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.curvature.curvature import _is_probability_target, _observed_mask
from laplace.likelihoods import get_likelihood
from laplace.utils import Kron
from laplace.utils.enums import Likelihood
from laplace.utils.utils import _is_batchnorm

EPS = 1e-6


class AsdfghjklInterface(CurvatureInterface):
    """Interface for asdfghjkl backend."""

    # Unmaintained and unchecked, so it honours no options but the ignored positions of
    # KFAC, which are dropped from the batch since an example has a single position
    _honoured: ClassVar[dict[str, frozenset[str]]] = {
        "kron": frozenset({"ignore_index"})
    }

    def _honours(self, method: str, y: torch.Tensor | None = None) -> bool:
        # asdfghjkl is unchecked for class probabilities, so `diag` takes the
        # torch.func code for them. `kron` and `full` handle them in their methods.
        if (
            method == "diag"
            and y is not None
            and _is_probability_target(self.likelihood, y)
        ):
            return False

        return super()._honours(method, y)

    def jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def gradients(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # Per-position gradients by torch.func, since asdfghjkl is unchecked for class
        # probabilities
        if _is_probability_target(self.likelihood, y):
            return CurvatureInterface.gradients(self, x, y)

        f = batch_gradient(self.model, self.lossfunc, x, y).detach()
        Gs = _get_batch_grad(self._model)
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        loss = self.lossfunc(f, y)
        return Gs, loss

    @property
    def _ggn_type(self) -> str:
        raise NotImplementedError

    def _get_kron_factors(self, curv, M: int) -> Kron:
        kfacs = list()
        for module in curv._model.modules():
            if _is_batchnorm(module):
                warnings.warn("BatchNorm unsupported for Kron, ignore.")
                continue

            stats = getattr(module, self._ggn_type, None)
            if stats is None:
                continue
            if hasattr(module, "bias") and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A[:-1, :-1]])
                kfacs.append([stats.kron.B * stats.kron.A[-1, -1] / M])
            elif hasattr(module, "weight"):
                p, q = stats.kron.B.numel(), stats.kron.A.numel()
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f"Whats happening with {module}?")
        return Kron(kfacs)

    @staticmethod
    def _rescale_kron_factors(kron: Kron, N: int) -> Kron:
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= 1 / N
        return kron

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fallback to torch.func for options or likelihoods asdfghjkl can't handle
        if not self._honours("diag", y):
            if isinstance(self, (GGNInterface, EFInterface)):
                return super().diag(x, y, **kwargs)

            raise self._unsupported_error("diag", y=y)

        with torch.no_grad():
            if self.last_layer:
                f, x = self.model.forward_with_features(x)
            else:
                f = self.model(x)
            loss = self.lossfunc(f, y)
        curv = fisher_for_cross_entropy(
            self._model, self._ggn_type, SHAPE_DIAG, inputs=x, targets=y
        )
        diag_ggn = curv.matrices_to_vector(None)
        if self.subnetwork_indices is not None:
            diag_ggn = diag_ggn[self.subnetwork_indices]
        return self.factor * loss, self._curvature_factor * diag_ggn

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if not self._honours("kron"):
            raise self._unsupported_error("kron")

        soft = _is_probability_target(self.likelihood, y)

        if soft and isinstance(self, EFInterface):
            raise ValueError(
                "The asdfghjkl backend does not support class probabilities as "
                "targets for the empirical Fisher with the `kron` structure. Use "
                "CurvlinopsEF or AsdlEF instead."
            )

        mask = _observed_mask(self.likelihood, y)

        if mask is not None and mask.ndim > 1:
            raise NotImplementedError(
                "The asdfghjkl backend does not support sequences with ignored "
                "targets. Use a Curvlinops or ASDL backend instead."
            )

        with torch.no_grad():
            if self.last_layer:
                f, x = self.model.forward_with_features(x)
            else:
                f = self.model(x)
            loss = self.lossfunc(f, y)

        # The loss keeps all targets. The Fisher only sees the observed examples.
        if mask is not None:
            x, y = x[mask], y[mask]

        # The GGN and the MC Fisher don't depend on the targets, and the loss above
        # has already used the class probabilities
        targets = y.argmax(dim=-1) if soft else y

        curv = fisher_for_cross_entropy(
            self._model, self._ggn_type, SHAPE_KRON, inputs=x, targets=targets
        )
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)
        return self.factor * loss, self._curvature_factor * kron


class AsdfghjklHessian(AsdfghjklInterface):
    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        low_rank: int = 10,
    ) -> None:
        super().__init__(
            model,
            likelihood,
            last_layer,
            None,
            dict_key_x="input_ids",
            dict_key_y="labels",
        )
        self.low_rank = low_rank

    @property
    def _ggn_type(self) -> str:
        raise self._unavailable_error("diag")

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """asdfghjkl computes only the full or low-rank Hessian, so this raises.

        Raises
        ------
        NotImplementedError
            always.
        """
        raise self._unavailable_error("diag")

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        """asdfghjkl computes only the full or low-rank Hessian, so this raises.

        Raises
        ------
        NotImplementedError
            always.
        """
        raise self._unavailable_error("kron")

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._honours("full"):
            raise self._unsupported_error("full")

        if _is_probability_target(self.likelihood, y):
            raise ValueError(
                "The asdfghjkl backend does not support class probabilities as "
                "targets for the Hessian. Use CurvlinopsHessian or AsdlHessian "
                "instead."
            )

        hessian_for_loss(self.model, self.lossfunc, SHAPE_FULL, x, y)
        H = self._model.hessian.data
        loss = self.lossfunc(self.model(x), y).detach()
        return self.factor * loss, self.factor * H

    def eig_lowrank(
        self, data_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._honours("eig_lowrank"):
            raise self._unsupported_error("eig_lowrank")

        # compute truncated eigendecomposition of the Hessian, only keep eigvals > EPS
        eigvals, eigvecs = hessian_eigenvalues(
            self.model,
            self.lossfunc,
            data_loader,
            top_n=self.low_rank,
            max_iters=self.low_rank * 10,
        )
        # asdfghjkl averages the Hessian-vector products over the batches, but the
        # Hessian of the loss sums over them
        eigvals = torch.from_numpy(np.array(eigvals)) * len(data_loader)
        eigvecs = torch.stack(
            [torch.cat([p.flatten() for p in params]) for params in eigvecs], dim=1
        )
        mask = eigvals > EPS
        eigvecs = eigvecs[:, mask]
        _warn_if_not_orthonormal(eigvecs)
        device = eigvecs.device
        eigvals = eigvals[mask].to(eigvecs.dtype).to(device)
        loss = sum(
            [
                self.lossfunc(self.model(x.to(device)).detach(), y.to(device))
                for x, y in data_loader
            ]
        )
        return eigvecs, self.factor * eigvals, self.factor * loss


class AsdfghjklGGN(AsdfghjklInterface, GGNInterface):
    """Implementation of the `GGNInterface` using asdfghjkl."""

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
        if get_likelihood(likelihood).name != Likelihood.CLASSIFICATION.name:
            raise ValueError("This backend only supports classification currently.")
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    @property
    def _ggn_type(self) -> str:
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AsdfghjklEF(AsdfghjklInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str | None,
        last_layer: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ) -> None:
        if get_likelihood(likelihood).name != Likelihood.CLASSIFICATION.name:
            raise ValueError("This backend only supports classification currently.")

        super().__init__(model, likelihood, last_layer, None, dict_key_x, dict_key_y)

    @property
    def _ggn_type(self) -> str:
        return COV


def _flatten_after_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)


def _get_batch_grad(model: nn.Module) -> torch.Tensor:
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, "op_results"):
            res = module.op_results["batch_grads"]
            if "weight" in res:
                batch_grads.append(_flatten_after_batch(res["weight"]))
            if "bias" in res:
                batch_grads.append(_flatten_after_batch(res["bias"]))
            if len(set(res.keys()) - {"weight", "bias"}) > 0:
                raise ValueError(f"Invalid parameter keys {res.keys()}")
    return torch.cat(batch_grads, dim=1)


def _warn_if_not_orthonormal(eigvecs: torch.Tensor, atol: float = 1e-4) -> None:
    """Warn if the eigenvectors from asdfghjkl's power iteration aren't orthonormal.

    When the power iteration for an eigenvector stops at its iteration limit,
    asdfghjkl returns the unnormalized last iterate. The later eigenvectors are
    then orthogonalized against a wrong direction, and the low-rank Hessian is off.

    Parameters
    ----------
    eigvecs : torch.Tensor
        eigenvectors as columns, `(parameters, rank)`.
    atol : float, default=1e-4
        tolerance on the entries of \\(V^\\top V - I\\).
    """
    gram = eigvecs.T @ eigvecs
    identity = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)

    if not torch.allclose(gram, identity, atol=atol):
        warnings.warn(
            "asdfghjkl's power iteration didn't converge for all eigenvectors, so "
            "they aren't orthonormal and the low-rank Hessian is inaccurate. A "
            "smaller `low_rank` in `backend_kwargs` converges more often."
        )
