from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from contextlib import nullcontext
from importlib.util import find_spec
from typing import Any, Callable

import torch
from torch import nn

try:
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
except ModuleNotFoundError as err:
    if find_spec("asdl") is not None:
        raise

    raise ImportError(
        "The ASDL backends need the asdl package. Install it with: "
        'pip install "laplace-torch[asdl]"'
    ) from err

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.curvature.curvature import (
    _ALL_OPTIONS,
    _KRON_OPTIONS,
    _TARGET_FREE_OPTIONS,
    _observed_input_factors,
    _observed_mask,
)
from laplace.likelihoods import GaussianLikelihood, Likelihood
from laplace.utils import Kron
from laplace.utils.utils import _is_batchnorm

EPS = 1e-6


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend."""

    # The exact and MC Fishers build their own loss from enumerated or sampled
    # classes, so they can't see class weights or ignored targets in `diag`. KFAC
    # drops the ignored positions from the logits and the input factors.
    _honoured = {"diag": _TARGET_FREE_OPTIONS, "kron": _KRON_OPTIONS}

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
            LOSS_MSE
            if self.likelihood.name == GaussianLikelihood.name
            else LOSS_CROSS_ENTROPY
        )

    def _flatten_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Flatten the targets to match the logits that ASDL flattens to `(M, K)`.

        ASDL builds the model output lazily, so its shape is only known after the
        forward pass. As in `nn.CrossEntropyLoss`, the dtype tells class probabilities
        from class indices instead. `_check_targets` verifies the shape afterwards.

        Parameters
        ----------
        y : torch.Tensor
            targets of the batch. For classification, class indices
            `(batch, ...)` or class probabilities `(batch, ..., K)`.

        Returns
        -------
        y : torch.Tensor
            `y` itself for regression, else class probabilities `(M, K)` or class
            indices `(M,)`, where `M` is the number of positions.
        """
        if self.loss_type == LOSS_MSE:
            return y

        if y.is_floating_point():
            return y.reshape(-1, y.shape[-1])

        return y.reshape(-1)

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
            Jacobians `([batch_dims], outputs, parameters)`
        f : torch.Tensor
            output function `([batch_dims], outputs)`
        """
        Js = list()
        # `self.model.output_size` can be stale, e.g. in reward modeling where the
        # output size differs between fitting and prediction. So, get the number of
        # outputs from the first forward pass instead.
        n_outputs = 1
        i = 0
        while i < n_outputs:

            def closure():
                self.model.zero_grad()
                f = self.model(x)
                loss = f[..., i].sum()
                loss.backward(
                    create_graph=enable_backprop, retain_graph=enable_backprop
                )
                return f

            Ji, f = self._batch_gradient(closure)
            n_outputs = f.shape[-1]

            if self.subnetwork_indices is not None:
                Ji = Ji[:, self.subnetwork_indices]

            Js.append(Ji)
            i += 1

        Js = torch.stack(Js, dim=1)  # (n_rows, n_outputs, n_params)

        # ASDL returns one row per row of the layers' inputs, e.g. per position. Sum those
        # belonging to the same observation, e.g. when the outputs are pooled over positions.
        n_obs = f.shape[:-1].numel()
        Js = self._sum_rows_per_observation(Js, n_obs)

        if Js.shape[0] == n_obs:
            # Restore the individual batch dims of the outputs
            Js = Js.reshape(*f.shape[:-1], *Js.shape[-2:])

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
            gradients `(n_observations, parameters)`
        """

        # With `last_layer`, `self._model` is the last layer. Map the inputs to the
        # features, so that the gradients are those of the last layer's parameters only.
        if self.last_layer:
            with torch.no_grad():
                _, x = self.model.forward_with_features(x)

        f = None

        def closure():
            nonlocal f
            self._model.zero_grad()
            f = self._model(x)
            loss = self._loss(f, y)
            loss.backward()
            return loss

        Gs, loss = self._batch_gradient(closure, self._model)

        # The last dim of the outputs is the output dim
        n_obs = f.shape[:-1].numel()
        Gs = self._sum_rows_per_observation(Gs, n_obs)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        return Gs, loss

    def _batch_gradient(
        self, closure: Callable[[], Any], model: nn.Module | None = None
    ) -> tuple[torch.Tensor, Any]:
        """ASDL's `batch_gradient` with a clearer error for models whose layers have
        different numbers of gradient rows, e.g., position-wise layers followed by a
        head applied after pooling over positions.

        Parameters
        ----------
        closure : Callable
            Runs the forward and backward pass of `model`.
        model : nn.Module, optional
            The module whose parameters get gradients. Defaults to `self.model`.
        """
        model = self.model if model is None else model
        self._check_backward_hooks(full=False)

        try:
            return batch_gradient(model, closure, return_outputs=True)
        except RuntimeError as e:
            if "Sizes of tensors must match" not in str(e):
                raise
            raise RuntimeError(
                "ASDL returned different numbers of gradient rows for different layers, "
                "e.g., for outputs pooled over positions. Use a Curvlinops backend "
                "(`CurvlinopsGGN` or `CurvlinopsEF`) for such models."
            ) from e

    @staticmethod
    def _sum_rows_per_observation(A: torch.Tensor, n_obs: int) -> torch.Tensor:
        """ASDL returns one row per row of the layers' inputs (e.g. per position), but
        an observation can span several of them (e.g. pooled outputs, or pairs in
        reward modeling). Sum the rows that belong to the same observation.

        Parameters
        ----------
        A : torch.Tensor
            `(n_rows, ...)`, where rows of the same observation are consecutive.
        n_obs : int
            number of observations, i.e., product of the leading dims of the outputs.

        Returns
        -------
        A : torch.Tensor
            `(n_obs, ...)`

        Raises
        ------
        ValueError
            if `n_rows` is not a multiple of `n_obs`.
        """
        n_rows = A.shape[0]
        if n_rows == n_obs:
            return A
        if n_obs == 0 or n_rows % n_obs != 0:
            raise ValueError(
                f"ASDL returned {n_rows} per-example rows, which cannot be grouped "
                f"into {n_obs} observations (the leading dims of the outputs)."
            )
        return A.reshape(n_obs, n_rows // n_obs, *A.shape[1:]).sum(1)

    @property
    def _ggn_type(self) -> str:
        raise NotImplementedError

    def _get_kron_factors(
        self,
        M: int,
        input_factors: dict[nn.Module, torch.Tensor] | None = None,
    ) -> Kron:
        """Collect the Kronecker factors that ASDL stored on the modules.

        Parameters
        ----------
        M : int
            number of positions in the batch that count as observations.
        input_factors : dict[nn.Module, torch.Tensor], optional
            input factors that replace those of ASDL, from
            `_observed_input_factors`.

        Returns
        -------
        kron : Kron
        """
        kfacs = list()
        for module in self.model.modules():
            if _is_batchnorm(module):
                warnings.warn("BatchNorm unsupported for Kron, ignore.")
                continue

            stats = getattr(module, "fisher", None)
            if stats is None:
                continue

            B = stats.kron.B

            if input_factors is None:
                A = stats.kron.A
            else:
                # ASDL's factors are sums over the positions, and
                # `_rescale_kron_factors` divides them by the number of data. So scale
                # the mean by the number `M` of observed positions.
                A = input_factors[module] * M

            if hasattr(module, "bias") and module.bias is not None:
                # split up bias and weights
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

        # Fallback to torch.func for options or likelihoods ASDL can't handle
        if not self._honours("diag", y):
            if isinstance(self, (GGNInterface, EFInterface)):
                return super().diag(x, y, **kwargs)

            raise self._unsupported_error("diag", y=y)

        if self.loss_type == LOSS_MSE and self._ggn_type == FISHER_EXACT and y.ndim > 2:
            # ASDL's exact MSE Fisher loops over `logits[:, i]`, which assumes 2D
            # outputs. With multidim outputs, use the Jacobian-based GGN instead.
            return super(AsdlInterface, self).diag(x, y, **kwargs)

        if self.last_layer:
            _, x = self.model.forward_with_features(x)

        cfg = FisherConfig(
            fisher_type=self._ggn_type,
            loss_type=self.loss_type,
            fisher_shapes=[SHAPE_DIAG],
            data_size=1,
            **kwargs,
        )
        self._check_backward_hooks(full=False)
        fisher_maker = get_fisher_maker(self.model, cfg)
        y_orig = y
        y = self._flatten_targets(y)
        if "emp" in self._ggn_type:
            dummy = fisher_maker.setup_model_call(self._model, x)
            dummy = (
                dummy
                if self.loss_type == LOSS_MSE
                else dummy.reshape(-1, dummy.size(-1))
            )
            fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
        else:
            dummy = fisher_maker.setup_model_call(self._model, x)
            if self.loss_type != LOSS_MSE:
                # ASDL's exact and MC cross-entropy Fishers assume 2D logits
                fisher_maker.setup_logits_repr(dummy.reshape(-1, dummy.size(-1)))
        f, _ = fisher_maker.forward_and_backward()
        self._check_targets(f, y_orig)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.reshape(-1, f.size(-1))
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

        # ASDL's GGN uses the proper 1/2 * MSELoss, but its EF squares the gradients
        # of the MSELoss
        curv_factor = self._curvature_factor if isinstance(self, AsdlEF) else 1.0
        return self.factor * loss, curv_factor * diag_ggn

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if not self._honours("kron"):
            raise self._unsupported_error("kron")

        if self.likelihood.name == GaussianLikelihood.name and y.ndim > 2:
            raise NotImplementedError(
                "Kronecker-factored curvature with ASDL does not support regression "
                "targets with more than one batch dimension (e.g. sequences). "
                "Use a Curvlinops backend instead."
            )

        if self.last_layer:
            _, x = self.model.forward_with_features(x)
        cfg = FisherConfig(
            fisher_type=self._ggn_type,
            loss_type=self.loss_type,
            fisher_shapes=[SHAPE_KRON],
            data_size=1,
            **kwargs,
        )
        self._check_backward_hooks(full=False)
        fisher_maker = get_fisher_maker(self.model, cfg)
        y_orig = y
        y = self._flatten_targets(y)

        mask = _observed_mask(self.likelihood, y_orig)
        observed = (
            nullcontext()
            if mask is None
            else _observed_input_factors(self._model, mask)
        )

        # The forward pass of `setup_model_call` fills the input factors of the mask
        with observed as input_factors:
            if "emp" in self._ggn_type:
                dummy = fisher_maker.setup_model_call(self._model, x)
                dummy = (
                    dummy
                    if self.loss_type == LOSS_MSE
                    else dummy.reshape(-1, dummy.size(-1))
                )

                if mask is None:
                    fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
                else:
                    # The ignored positions are gone, so no class is ignored anymore
                    m = mask.reshape(-1)
                    plain = nn.CrossEntropyLoss(
                        reduction="sum", label_smoothing=self.likelihood.label_smoothing
                    )
                    fisher_maker.setup_loss_call(plain, dummy[m], y[m])
            else:
                dummy = fisher_maker.setup_model_call(self._model, x)
                if self.loss_type != LOSS_MSE:
                    # ASDL's exact and MC cross-entropy Fishers assume 2D logits
                    logits = dummy.reshape(-1, dummy.size(-1))

                    if mask is not None:
                        # Only the observed positions are enumerated or sampled,
                        # whatever their classes are
                        logits = logits[mask.reshape(-1)]

                    fisher_maker.setup_logits_repr(logits)
            f, _ = fisher_maker.forward_and_backward()
        self._check_targets(f, y_orig)
        self._check_conv_with_multidim_outputs(f)
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.reshape(-1, f.size(-1))
        loss = self.lossfunc(f.detach(), y)
        M = len(y) if mask is None else int(mask.sum())
        kron = self._get_kron_factors(M, None if mask is None else input_factors)
        # ASDL sums the input factors over the observed positions, e.g. `B * L` rows for
        # logits `(B, L, C)`. `KronLaplace.fit` normalizes by the number of observations.
        kron = self._rescale_kron_factors(kron, N)
        # ASDL's GGN uses the proper 1/2 * MSELoss, but its EF squares the gradients
        # of the MSELoss
        curv_factor = self._curvature_factor if isinstance(self, AsdlEF) else 1.0
        return self.factor * loss, curv_factor * kron

    def _check_conv_with_multidim_outputs(self, f: torch.Tensor) -> None:
        """Raise if the model has a conv layer and the outputs have more than one
        batch dim, e.g. `(B, H, W, K)` with one prediction per pixel.

        For such outputs, the KFAC input factor that ASDL computes for a conv layer
        is \\( H W \\) times smaller than the one from Curvlinops, or from the same
        net written with `nn.Linear` layers applied to every pixel. Outputs
        `(B, K)`, e.g. after global pooling, are not affected. ASDL is no longer
        maintained, so we raise instead of patching it.

        Parameters
        ----------
        f : torch.Tensor
            The model outputs, with the output dim last.

        Raises
        ------
        NotImplementedError
            If `f` has more than two dims and the model has a trainable conv layer.
        """
        if f.ndim <= 2:
            return

        convs = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        has_conv = any(
            isinstance(module, convs)
            and any(p.requires_grad for p in module.parameters())
            for module in self._model.modules()
        )

        if has_conv:
            raise NotImplementedError(
                "ASDL computes wrong Kronecker factors for conv layers when the "
                "outputs have more than one batch dim, here outputs of shape "
                f"{tuple(f.shape)}. Use a "
                "Curvlinops backend (`CurvlinopsGGN` or `CurvlinopsEF`) instead."
            )


class AsdlHessian(AsdlInterface):
    # ASDL differentiates `lossfunc` twice at the observed targets
    _honoured = {"full": _ALL_OPTIONS}

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
        raise self._unavailable_error("diag")

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ASDL computes only the full Hessian here, so this raises.

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
        """ASDL computes only the full Hessian here, so this raises.

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

        if self.last_layer:
            _, x = self.model.forward_with_features(x)

        cfg = HessianConfig(hessian_shapes=[SHAPE_FULL])
        # With `last_layer`, `x` are the features and `self._model` the last layer
        hess_maker = HessianMaker(self._model, cfg)

        dummy = hess_maker.setup_model_call(self._model, x)
        dummy = (
            dummy if self.loss_type == LOSS_MSE else dummy.reshape(-1, dummy.size(-1))
        )
        y = self._flatten_targets(y)

        hess_maker.setup_loss_call(self.lossfunc, dummy, y)
        hess_maker.forward_and_backward()

        H = self._model.hessian.data
        f = self._model(x).detach()
        # Assumes that the last dimension of f is of size outputs.
        f = f if self.loss_type == LOSS_MSE else f.reshape(-1, f.size(-1))
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

    # The empirical Fisher differentiates `lossfunc` at the observed targets. KFAC
    # drops the ignored positions from the logits and the input factors.
    _honoured = {"diag": _ALL_OPTIONS, "kron": _KRON_OPTIONS}

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
