from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from typing import Any

import torch
from asdl.fisher import FisherConfig, get_fisher_maker
from asdl.grad_maker import LOSS_CROSS_ENTROPY, LOSS_MSE
from asdl.gradient import batch_gradient
from asdl.hessian import HessianConfig, HessianMaker
from asdl.matrices import (
    FISHER_EMP,
    FISHER_EXACT,
    FISHER_MC,
    SHAPE_DIAG,
    SHAPE_FULL,
    SHAPE_KRON,
)
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.utils import Kron, Likelihood, _is_batchnorm

EPS = 1e-6


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ):
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )

    @property
    def loss_type(self) -> str:
        return (
            LOSS_MSE if self.likelihood == Likelihood.REGRESSION else LOSS_CROSS_ENTROPY
        )

    def jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping (e.g. dict, UserDict)
            input data `(batch, input_shape)` on compatible device with model if torch.Tensor.
            If MutableMapping, then at least contains `self.dict_key_x`.
            The latter is specific for reward modeling.
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

            def closure():
                self.model.zero_grad()
                f = self.model(x)
                loss = f[:, i].sum()
                loss.backward(
                    create_graph=enable_backprop, retain_graph=enable_backprop
                )
                return f

            Ji, f = batch_gradient(
                self.model,
                closure,
                return_outputs=True,
                batch_size=self._get_batch_size(x),
            )
            if self.subnetwork_indices is not None:
                Ji = Ji[:, self.subnetwork_indices]
            Js.append(Ji)
        Js = torch.stack(Js, dim=1)
        return Js, f

    def gradients(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any], y: torch.Tensor
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

        def closure():
            self.model.zero_grad()
            loss = self.lossfunc(self.model(x), y)
            loss.backward()
            return loss

        Gs, loss = batch_gradient(
            self.model, closure, return_outputs=True, batch_size=self._get_batch_size(x)
        )
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        return Gs, loss

    @property
    def _ggn_type(self) -> str:
        raise NotImplementedError

    def _get_kron_factors(self, M: int) -> Kron:
        kfacs = list()
        for module in self.model.modules():
            if _is_batchnorm(module):
                warnings.warn("BatchNorm unsupported for Kron, ignore.")
                continue

            stats = getattr(module, "fisher", None)
            if stats is None:
                continue

            if hasattr(module, "bias") and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A])
                kfacs.append([stats.kron.B])
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
        if "N" in kwargs:
            del kwargs["N"]

        if self.last_layer:
            _, x = self.model.forward_with_features(x)

        cfg = FisherConfig(
            fisher_type=self._ggn_type,
            loss_type=self.loss_type,
            fisher_shapes=[SHAPE_DIAG],
            data_size=1,
            **kwargs,
        )
        fisher_maker = get_fisher_maker(self.model, cfg)
        y = y if self.loss_type == LOSS_MSE else y.view(-1)
        if "emp" in self._ggn_type:
            dummy = fisher_maker.setup_model_call(self._model, x)
            dummy = (
                dummy if self.loss_type == LOSS_MSE else dummy.view(-1, dummy.size(-1))
            )
            fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
        else:
            fisher_maker.setup_model_call(self._model, x)
        f, _ = fisher_maker.forward_and_backward()
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.view(-1, f.size(-1))
        loss = self.lossfunc(f.detach(), y)
        # ASDL attaches Fisher statistics to modules, but Laplace indexes a single
        # flattened parameter vector. In last-layer mode that vector belongs to
        # `_model`, not to the feature extractor around it.
        model = self._model
        # `self.params` contains trainable parameters only. Frozen parameters must
        # not consume index positions when we map module statistics into this vector.
        n_params = sum(p.numel() for p in self.params)
        selected = self.subnetwork_indices
        if selected is not None:
            # Validate against the full trainable vector before inspecting modules.
            # Copy indices once so each module's `.any()` range check runs on CPU
            # instead of synchronizing a GPU reduction for every module.
            selected = selected.cpu()
            if ((selected < 0) | (selected >= n_params)).any():
                raise IndexError("Subnetwork indices exceed the model parameters.")

        # Hold each module's values and destination until we know the Fisher
        # device and dtype. Model outputs can have a different dtype from Fisher.
        pieces = []
        result_dtype = None
        result_device = None
        offset = 0
        for module_name, module in model.named_modules():
            # Direct parameters appear before child modules in PyTorch's flattened
            # order. `recurse=False` prevents counting child parameters twice.
            module_params = [
                (name, p)
                for name, p in module.named_parameters(recurse=False)
                if p.requires_grad
            ]
            module_n_params = sum(p.numel() for _, p in module_params)
            if not module_n_params:
                continue
            # These are positions in `selected`, not positions in the full vector.
            # The half-open range identifies indices owned by this module.
            in_module = (
                None
                if selected is None
                else (selected >= offset) & (selected < offset + module_n_params)
            )
            if in_module is not None and not in_module.any():
                # No requested entry needs this module's Fisher statistics, even
                # if ASDL never computed them. Its parameters still shift offsets.
                offset += module_n_params
                continue

            stats = getattr(module, "fisher", None)
            if stats is None:
                # Missing stats can mean an unused module or one ASDL does not
                # support. We cannot infer zero curvature for parameters we need.
                raise ValueError(
                    "ASDL produced no diagonal Fisher statistics for "
                    f"parameters in module '{module_name}' "
                    f"({type(module).__name__})."
                )
            module_vec_parts = []
            for name, param in module_params:
                # Read each statistic by parameter name: ASDL's fixed weight/bias
                # order can differ from the module's registration order.
                part = getattr(getattr(stats, "diag", None), name, None)
                if part is None or part.numel() != param.numel():
                    raise ValueError(
                        "ASDL diagonal Fisher statistics do not match the parameters "
                        f"in module '{module_name}' ({type(module).__name__})."
                    )
                module_vec_parts.append(part.reshape(-1))
            module_vec = torch.cat(module_vec_parts)
            if in_module is None:
                # With no subnetwork, the module fills its contiguous full-vector
                # range directly.
                positions = slice(offset, offset + module_n_params)
                values = module_vec
            else:
                # Subtract the module offset to index its local Fisher vector.
                # `in_module` retains the caller's order, including scattered or
                # repeated indices, when these values are written to the result.
                positions = in_module
                local_indices = (selected[in_module] - offset).to(module_vec.device)
                values = module_vec[local_indices]
            # Use the statistics' device and promote their dtypes across modules;
            # allocating from `f` could silently downcast these values.
            if result_device is None:
                result_device = values.device
            elif result_device != values.device:
                raise ValueError("ASDL Fisher statistics span multiple devices.")
            result_dtype = (
                values.dtype
                if result_dtype is None
                else torch.promote_types(result_dtype, values.dtype)
            )
            pieces.append((positions, values))
            offset += module_n_params
        # A mismatch means module traversal did not describe Laplace's vector,
        # for example because parameters are shared in an unsupported layout.
        if offset != n_params:
            raise ValueError("ASDL module parameters do not match model parameters.")
        if pieces:
            diag_ggn = torch.empty(
                n_params if selected is None else selected.numel(),
                device=result_device,
                dtype=result_dtype,
            )
            for positions, values in pieces:
                # A boolean mask locates destinations in subnetwork order. Move it
                # to the Fisher device only for the final assignment.
                if isinstance(positions, torch.Tensor):
                    positions = positions.to(device=result_device)
                diag_ggn[positions] = values
        else:
            # No Fisher values were gathered, as with an empty subnetwork.
            diag_ggn = f.new_empty(0)
        if type(self) is AsdlEF and self.likelihood == "regression":
            curv_factor = 0.5  # correct scaling for diag ef
        else:
            curv_factor = 1.0  # ASDL uses proper 1/2 * MSELoss
        return self.factor * loss, curv_factor * diag_ggn

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if self.last_layer:
            _, x = self.model.forward_with_features(x)
        cfg = FisherConfig(
            fisher_type=self._ggn_type,
            loss_type=self.loss_type,
            fisher_shapes=[SHAPE_KRON],
            data_size=1,
            **kwargs,
        )
        fisher_maker = get_fisher_maker(self.model, cfg)
        y = y if self.loss_type == LOSS_MSE else y.view(-1)
        if "emp" in self._ggn_type:
            dummy = fisher_maker.setup_model_call(self._model, x)
            dummy = (
                dummy if self.loss_type == LOSS_MSE else dummy.view(-1, dummy.size(-1))
            )
            fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
        else:
            fisher_maker.setup_model_call(self._model, x)
        f, _ = fisher_maker.forward_and_backward()
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.view(-1, f.size(-1))
        loss = self.lossfunc(f.detach(), y)
        M = len(y)
        kron = self._get_kron_factors(M)
        kron = self._rescale_kron_factors(kron, N)
        if type(self) is AsdlEF and self.likelihood == "regression":
            curv_factor = 0.5  # correct scaling for diag ef
        else:
            curv_factor = 1.0  # ASDL uses proper 1/2 * MSELoss
        return self.factor * loss, curv_factor * kron

    def _get_batch_size(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
    ) -> int | None:
        """
        ASDL assumes that all leading dimensions are the batch size by default (batch_size = None).
        Here, we want to specify that only the first dimension is the actual batch size.
        This is the case for LLMs.
        """
        if isinstance(x, MutableMapping):
            return x[self.dict_key_x].shape[0]
        else:
            return None  # Use ASDL default behavior


class AsdlHessian(AsdlInterface):
    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ) -> None:
        super().__init__(
            model,
            likelihood,
            last_layer,
            subnetwork_indices=None,
            dict_key_x=dict_key_x,
            dict_key_y=dict_key_y,
        )

    @property
    def _ggn_type(self) -> str:
        raise NotImplementedError()

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.last_layer:
            _, x = self.model.forward_with_features(x)

        cfg = HessianConfig(hessian_shapes=[SHAPE_FULL])
        hess_maker = HessianMaker(self.model, cfg)

        dummy = hess_maker.setup_model_call(self._model, x)
        dummy = dummy if self.loss_type == LOSS_MSE else dummy.view(-1, dummy.size(-1))
        y = y if self.loss_type == LOSS_MSE else y.view(-1)

        hess_maker.setup_loss_call(self.lossfunc, dummy, y)
        hess_maker.forward_and_backward()

        H = self._model.hessian.data
        f = self.model(x).detach()
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.view(-1, f.size(-1))
        loss = self.lossfunc(f, y)

        return self.factor * loss, self.factor * H


class AsdlGGN(AsdlInterface, GGNInterface):
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
    ):
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    @property
    def _ggn_type(self) -> str:
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AsdlEF(AsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ):
        super().__init__(model, likelihood, last_layer, None, dict_key_x, dict_key_y)

    @property
    def _ggn_type(self) -> str:
        return FISHER_EMP
