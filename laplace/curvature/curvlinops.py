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
from curvlinops.kfac import KFACType
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.curvature.curvature import (
    _ALL_OPTIONS,
    _KRON_OPTIONS,
    _TARGET_FREE_OPTIONS,
    _is_probability_target,
    _n_observed_examples,
    _observed_input_factors,
    _observed_mask,
)
from laplace.likelihoods import (
    BernoulliLikelihood,
    CategoricalLikelihood,
    HeteroscedasticGaussianLikelihood,
    Likelihood,
    _is_heteroscedastic,
    _noise_per_output,
)
from laplace.utils import Kron
from laplace.utils.heteroscedastic_utils import WhitenHeteroscedastic

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


class _ProbabilityCrossEntropy(nn.CrossEntropyLoss):
    """`nn.CrossEntropyLoss` for class probabilities, for the empirical Fisher of
    Curvlinops' KFAC.

    KFAC flattens the targets of a batch before it calls the loss: class indices
    `(B, ...)` become `(B * ...,)`, and class probabilities `(B, ..., K)` become
    `(B * ... * K,)`. This loss restores the class dim from the shape of the logits,
    which KFAC has flattened to `(B * ..., K)`.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy of flattened class probabilities.

        Parameters
        ----------
        input : torch.Tensor
            logits `(M, K)`
        target : torch.Tensor
            class probabilities with `M * K` entries, in the order of the logits.

        Returns
        -------
        loss : torch.Tensor
        """
        return super().forward(input, target.reshape(input.shape))


class _KeepObserved(nn.Module):
    """Keep the logits of the observed positions only: `(B, ..., K)` -> `(M, K)`.

    Curvlinops then sees every observed position as an example, and the ignored ones
    never reach its gradient factors.

    Parameters
    ----------
    model : nn.Module
        the model, with logits `(B, ..., K)`.
    mask : torch.Tensor
        boolean mask of the observed positions `(B, ...)`.
    """

    def __init__(self, model: nn.Module, mask: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.mask = mask

    def forward(self, x: Any) -> torch.Tensor:
        """Compute the logits of the observed positions.

        Parameters
        ----------
        x : Any
            model inputs.

        Returns
        -------
        logits : torch.Tensor
            `(M, K)`, for the `M` observed positions.
        """
        return self.model(x)[self.mask]


# Curvlinops whitens the outputs and the targets by the noise std per output, see
# `_Whiten`, which is exact for every curvature. A heteroscedastic likelihood is
# whitened by its predicted noise std, clamped at `eps`, and `full` only adds a
# constant to the loss, see `WhitenHeteroscedastic`.
_WHITENED_OPTIONS = frozenset({"sigma_noise", "eps", "full"})


class _Whiten(nn.Module):
    # Divide the outputs by the noise std per output, so that the plain squared error
    # of Curvlinops is the weighted one of the likelihood

    def __init__(self, model: nn.Module, sigma_noise: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.sigma_noise = sigma_noise

    def forward(self, x: Any) -> torch.Tensor:
        out = self.model(x)

        return out / self.sigma_noise.to(out)


class CurvlinopsInterface(CurvatureInterface):
    """Interface for Curvlinops backend. <https://github.com/f-dangel/curvlinops>"""

    # Curvlinops knows the loss of a Bernoulli likelihood, `nn.BCEWithLogitsLoss`, and
    # the mean of a heteroscedastic Gaussian is a whitened `nn.MSELoss`
    _library_types = (
        *CurvatureInterface._library_types,
        BernoulliLikelihood,
        HeteroscedasticGaussianLikelihood,
    )

    # The dense operators differentiate `lossfunc` at the observed targets. KFAC drops
    # the ignored positions from both Kronecker factors and has no class weights.
    # Curvlinops' KFAC also ignores the `weight` and `pos_weight` of
    # `nn.BCEWithLogitsLoss` without a warning, except for the empirical Fisher, see
    # `CurvlinopsEF`. So `weight` and `pos_weight` are absent from `_KRON_OPTIONS`,
    # which makes the other `kron` raise for them until Curvlinops handles them:
    # https://github.com/f-dangel/curvlinops/issues/311
    _honoured = {"full": _ALL_OPTIONS, "kron": _KRON_OPTIONS}

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
        if self.likelihood.name == CategoricalLikelihood.name:
            return _ClassDimFirst(self.model)

        # Whitening makes the curvature of the plain squared error exact
        sigma = _noise_per_output(self.likelihood)
        if sigma is not None:
            return _Whiten(self.model, sigma)

        if _is_heteroscedastic(self.likelihood):
            return WhitenHeteroscedastic(self.model, self.likelihood.eps)

        return self.model

    @property
    def _linop_lossfunc(self) -> nn.Module:
        # The loss passed to the Curvlinops operators. The whitened model needs the
        # plain squared error.
        if _noise_per_output(self.likelihood) is not None or _is_heteroscedastic(
            self.likelihood
        ):
            return nn.MSELoss(reduction="sum")

        return self.lossfunc

    @property
    def _linop_curvature_factor(self) -> float:
        """Factor that turns the curvature of `_linop_lossfunc` into that of the
        negative log likelihood. The whitened squared error of a heteroscedastic
        likelihood is twice its NLL, which gives \\(\\tfrac{1}{2}\\), or
        \\(\\tfrac{1}{4}\\) for the EF, which squares it."""
        if not _is_heteroscedastic(self.likelihood):
            return self._curvature_factor

        return 0.25 if isinstance(self, EFInterface) else 0.5

    def _honoured_options(self, method: str) -> frozenset[str]:
        return super()._honoured_options(method) | _WHITENED_OPTIONS

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

    def _get_kron_factors(
        self,
        linop: KFACLinearOperator,
        model: nn.Module,
        input_factors: dict[nn.Module, torch.Tensor] | None = None,
    ) -> Kron:
        """Collect the Kronecker factors of a computed KFAC operator.

        Parameters
        ----------
        linop : KFACLinearOperator
            operator whose factors are computed.
        model : nn.Module
            the model passed to `linop`.
        input_factors : dict[nn.Module, torch.Tensor], optional
            input factors that replace those of `linop`, from
            `_observed_input_factors`.

        Returns
        -------
        kron : Kron
        """
        kfacs = list()
        # `model` must be the one passed to `linop`, so that the module names match
        for name, module in model.named_modules():
            if name not in linop._mapping.keys():
                continue

            if input_factors is None:
                A = linop._input_covariances[name]
            else:
                A = input_factors[module]

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

    def _linop_targets(
        self, y: torch.Tensor, method: str, f: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Targets in the form that the Curvlinops operator of `method` takes.

        - A target-free Fisher, i.e. the GGN and the MC Fisher, doesn't depend on the
          targets, but Curvlinops needs valid ones. Class probabilities become their
          class indices, which `CrossEntropyLoss` accepts for any layout, and
          Bernoulli soft labels are rounded to 0 or 1. The loss uses the real targets.
        - The dense GGN and Hessian differentiate the loss at the class probabilities.
          Their model puts the class dim first, see `_ClassDimFirst`, so the targets
          do too.
        - The KFAC of the empirical Fisher takes class probabilities as they are, see
          `_ProbabilityCrossEntropy`.

        Parameters
        ----------
        y : torch.Tensor
            targets of a batch.
        method : str
            curvature method, `"kron"` or `"full"`.
        f : torch.Tensor, optional
            joined outputs of a heteroscedastic likelihood, whose predicted noise std
            whitens the targets, see `WhitenHeteroscedastic`.

        Returns
        -------
        y : torch.Tensor
        """
        target_free = (
            self._kron_fisher_type != FisherType.EMPIRICAL
            if method == "kron"
            else self._linop_context is FisherMCLinearOperator
        )

        if _is_probability_target(self.likelihood, y):
            if target_free:
                return y.argmax(dim=-1)

            if method == "full" and y.ndim > 2:
                return y.movedim(-1, 1)

            return y

        if target_free and isinstance(self.likelihood, BernoulliLikelihood):
            return (y >= 0.5).to(y.dtype)

        sigma = _noise_per_output(self.likelihood)
        if sigma is not None:
            return y / sigma.to(y)

        if _is_heteroscedastic(self.likelihood) and f is not None:
            var = self.likelihood.split(f.detach())[1]

            return y / var.clamp(min=self.likelihood.eps).sqrt().to(y)

        return y

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if not self._honours("kron"):
            raise self._unsupported_error("kron")

        if isinstance(x, (dict, MutableMapping)):
            kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        # Computed first, so that the targets are checked before the curvature
        f = self.model(x)
        loss = self._loss(f, y)

        mask = _observed_mask(self.likelihood, y)

        if mask is None:
            model = self._linop_model
            data = [(x, self._linop_targets(y, "kron", f))]
            kron_kwargs = {}
            M = f.shape[:-1].numel()

            soft = _is_probability_target(self.likelihood, y)

            if soft:
                # Curvlinops would count the class dim of the targets as loss terms
                kron_kwargs["num_per_example_loss_terms"] = M // f.shape[0]

            if soft and self._kron_fisher_type == FisherType.EMPIRICAL:
                lossfunc = _ProbabilityCrossEntropy(
                    reduction="sum", label_smoothing=self.likelihood.label_smoothing
                )
            elif self._has_custom_ignore_index:
                # A label sampled for the MC Fisher may equal a custom `ignore_index`
                lossfunc = nn.CrossEntropyLoss(
                    reduction="sum", label_smoothing=self.likelihood.label_smoothing
                )
            else:
                lossfunc = self._linop_lossfunc
        else:
            # Every observed position is an example with one loss term. The losses
            # don't ignore any class, since the ignored positions are gone, and a
            # sampled label may equal `ignore_index`.
            model = _KeepObserved(self.model, mask)
            lossfunc = nn.CrossEntropyLoss(
                reduction="sum", label_smoothing=self.likelihood.label_smoothing
            )
            data = [(x, y[mask])]
            kron_kwargs = dict(num_per_example_loss_terms=1, num_data=mask.shape[0])
            M = int(mask.sum())

        linop = KFACLinearOperator(
            model,
            lossfunc,
            self.params,
            data,
            fisher_type=self._kron_fisher_type,
            separate_weight_and_bias=True,
            check_deterministic=False,  # To avoid overhead
            # `kwargs` for `mc_samples` when `stochastic=True` and `kfac_approx` to
            # choose between `'expand'` and `'reduce'`.
            # Defaults to `mc_samples=1` and `kfac_approx='expand'.
            **kron_kwargs,
            **kwargs,
        )

        reduce = kwargs.get("kfac_approx", KFACType.EXPAND) == KFACType.REDUCE

        if mask is None:
            linop.compute_kronecker_factors()
            kron = self._get_kron_factors(linop, model)
        else:
            # Curvlinops averages the input factors over all positions, so the observed
            # ones are collected separately
            with _observed_input_factors(self._model, mask, reduce) as input_factors:
                linop.compute_kronecker_factors()

            kron = self._get_kron_factors(linop, model, input_factors)

        # Curvlinops averages the input factors over the positions of the batch, e.g.
        # `B * L` for outputs `(B, L, K)`. Undo it, so that they sum over the
        # positions like with ASDL. KFAC-reduce averages over the examples instead,
        # so it sums over those. `KronLaplace.fit` normalizes by the matching count.
        if reduce:
            M = _n_observed_examples(y, mask)

        kron = self._rescale_kron_factors(kron, M, N)
        kron *= self._linop_curvature_factor

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

        # Fallback to torch.func for options or likelihoods Curvlinops can't handle
        if not self._honours("full", y):
            if isinstance(self, (GGNInterface, EFInterface)):
                return super().full(x, y, **curvlinops_kwargs)

            raise self._unsupported_error("full", y=y)
        if isinstance(x, (dict, MutableMapping)):
            curvlinops_kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        # Computed first, so that the targets are checked before the curvature
        f = self.model(x)
        loss = self._loss(f, y)

        linop = self._linop_context(
            self._linop_model,
            self._linop_lossfunc,
            self.params,
            [(x, self._linop_targets(y, "full", f))],
            check_deterministic=False,
            **curvlinops_kwargs,
        )

        p = next(self.model.parameters())
        H = torch.as_tensor(
            linop @ torch.eye(linop.shape[0]), device=p.device, dtype=p.dtype
        )

        return self.factor * loss.detach(), self._linop_curvature_factor * H


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

    def _honoured_options(self, method: str) -> frozenset[str]:
        if not self.stochastic:
            return super()._honoured_options(method)

        # The MC Fisher of the dense operator samples gradients that ignore the
        # loss. MC KFAC differentiates `lossfunc` at sampled labels, so label
        # smoothing changes it. Ignored positions are dropped before sampling.
        if method == "full":
            options = _TARGET_FREE_OPTIONS
        elif method == "kron":
            options = frozenset({"ignore_index"})
        else:
            options = frozenset()

        return options | _WHITENED_OPTIONS

    @property
    def _kron_fisher_type(self) -> FisherType:
        return FisherType.MC if self.stochastic else FisherType.TYPE2

    @property
    def _linop_context(self) -> type[GGNLinearOperator | FisherMCLinearOperator]:
        return FisherMCLinearOperator if self.stochastic else GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops."""

    def _honoured_options(self, method: str) -> frozenset[str]:
        # Curvlinops' empirical Fisher differentiates `lossfunc` at the observed
        # targets, so KFAC honours the `weight` and `pos_weight` of a Bernoulli
        # likelihood, unlike the GGN and MC Fishers, see `CurvlinopsInterface`.
        if method == "kron" and isinstance(self.likelihood, BernoulliLikelihood):
            return super()._honoured_options(method) | {"weight", "pos_weight"}

        return super()._honoured_options(method)

    def _honours(self, method: str, y: torch.Tensor | None = None) -> bool:
        # `EFLinearOperator` can't differentiate the loss at class probabilities, so
        # the dense empirical Fisher falls back to the per-position gradients
        if (
            method == "full"
            and y is not None
            and _is_probability_target(self.likelihood, y)
        ):
            return False

        return super()._honours(method, y)

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

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        """The Hessian has no Kronecker factors in Curvlinops, so this raises.

        Raises
        ------
        NotImplementedError
            always.
        """
        raise self._unavailable_error("kron")
