from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, MutableMapping

import torch
from curvlinops.kfac_utils import extract_patches
from torch import nn
from torch.utils._pytree import tree_map

from laplace.likelihoods import (
    BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
    Likelihood,
    _noise_per_output,
    get_likelihood,
)
from laplace.utils import Kron

_BUILTIN_TYPES = (GaussianLikelihood, CategoricalLikelihood, BernoulliLikelihood)

# Options of the built-in likelihoods, see `CurvatureInterface._honours`
_ALL_OPTIONS = frozenset({"weight", "ignore_index", "label_smoothing", "pos_weight"})

# Options that leave a target-independent curvature unchanged, e.g. the GGN, since
# label smoothing does not change the Hessian of the cross-entropy w.r.t. the logits
_TARGET_FREE_OPTIONS = frozenset({"label_smoothing"})

# Options that KFAC honours: label smoothing as above, and ignored positions, which
# are dropped from both Kronecker factors, see `_observed_mask`
_KRON_OPTIONS = _TARGET_FREE_OPTIONS | {"ignore_index"}


def _observed_mask(likelihood: Likelihood, y: torch.Tensor) -> torch.Tensor | None:
    """Mask of the positions of the targets `y` that a categorical likelihood observes.

    Parameters
    ----------
    likelihood : Likelihood
    y : torch.Tensor
        targets of a batch, e.g. class indices `(B, L)`.

    Returns
    -------
    mask : torch.Tensor or None
        boolean, with the shape of `y`. `None` if nothing is ignored, so that the
        callers keep the code path for batches without ignored positions.
    """
    if _ignored_positions(likelihood, y) == 0:
        return None

    return y != likelihood.ignore_index


@contextmanager
def _observed_input_factors(
    model: nn.Module, mask: torch.Tensor, reduce: bool = False
) -> Iterator[dict[nn.Module, torch.Tensor]]:
    """Collect the input factors of the `nn.Linear` and `nn.Conv2d` layers of `model`
    over the observed positions only.

    Forward pre-hooks record the inputs of the layers during the forward passes inside
    the `with` block. The factor of a layer is the mean
    \\( \\frac{1}{|S|} \\sum_{i \\in S} x_i x_i^\\top \\) of the outer products of its
    inputs (or, for a convolution, its patches) over a set \\( S \\) of positions.
    \\( S \\) is

    - the observed positions, if the layer has as many positions per example as
      `mask`, e.g. a layer applied to every token of a sequence. The positions are
      assumed to be in the same order as those of the logits, which can't be checked
      here;
    - otherwise all positions of the examples that have at least one observation,
      e.g. a convolution before pooling, whose positions don't line up with the
      logits.

    With `reduce`, as in KFAC-reduce, the inputs are first averaged over the positions
    of \\( S \\) in each example, and the mean runs over the examples.

    The factors are complete once the `with` block exits.

    Parameters
    ----------
    model : nn.Module
        the module whose layers are hooked.
    mask : torch.Tensor
        boolean mask of the observed positions `(B, ...)`, from `_observed_mask`.
    reduce : bool, default=False
        whether to compute the input factors of KFAC-reduce instead of KFAC-expand.

    Yields
    ------
    factors : dict[nn.Module, torch.Tensor]
        input factor of every layer that ran. A layer without a position in \\( S \\)
        has a zero factor.
    """
    n_examples = mask.shape[0]
    mask = mask.reshape(n_examples, -1)
    has_observation = mask.any(dim=1)

    sums: dict[nn.Module, torch.Tensor] = {}
    counts: dict[nn.Module, int] = {}

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach()

        if isinstance(module, nn.Conv2d):
            x = extract_patches(
                x,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )

        x = x.reshape(n_examples, -1, x.shape[-1])

        if x.shape[1] == mask.shape[1]:
            keep = mask
        else:
            keep = has_observation[:, None].expand(-1, x.shape[1])

        if reduce:
            totals = (x * keep[..., None]).sum(dim=1)[has_observation]
            x = totals / keep[has_observation].sum(dim=1, keepdim=True)
        else:
            x = x[keep]

        sums[module] = sums.get(module, 0) + x.T @ x
        counts[module] = counts.get(module, 0) + x.shape[0]

    handles = [
        m.register_forward_pre_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    factors: dict[nn.Module, torch.Tensor] = {}

    try:
        yield factors
    finally:
        for handle in handles:
            handle.remove()

        for module, total in sums.items():
            factors[module] = total / max(counts[module], 1)


def _is_probability_target(likelihood: Likelihood, y: torch.Tensor) -> bool:
    """Whether the targets `y` are class probabilities for a categorical likelihood.

    Class probabilities are floats with the shape of the logits `(B, ..., K)`, while
    class indices are integers `(B, ...)`. This is how `nn.CrossEntropyLoss` tells
    them apart, and the likelihood rejects integer targets with the shape of the
    logits.

    Parameters
    ----------
    likelihood : Likelihood
    y : torch.Tensor
        targets of a batch.

    Returns
    -------
    is_probability : bool
        `False` for other likelihoods and for class indices.
    """
    return isinstance(likelihood, CategoricalLikelihood) and y.is_floating_point()


def _ignored_positions(likelihood: Likelihood, y: torch.Tensor) -> int:
    """Number of positions that a categorical likelihood ignores in the targets `y`.

    For outputs `(B, ..., K)`, a position is an entry of the dims between the batch
    dim `B` and the output dim `K`, e.g. a sequence position or a pixel.

    Only index targets can be ignored. They are integers, while probability targets
    are floats, as in `nn.CrossEntropyLoss`. This holds for the default
    `ignore_index=-100` too, e.g. for padded sequences.

    Parameters
    ----------
    likelihood : Likelihood
    y : torch.Tensor
        targets of a batch.

    Returns
    -------
    n_ignored : int
        zero for other likelihoods and for probability targets.
    """
    if not isinstance(likelihood, CategoricalLikelihood) or _is_probability_target(
        likelihood, y
    ):
        return 0

    return int((y == likelihood.ignore_index).sum())


class CurvatureInterface:
    """Interface to access curvature for a model and corresponding likelihood.
    A `CurvatureInterface` must inherit from this baseclass and implement the
    necessary functions `jacobians`, `full`, `kron`, and `diag`.
    The interface might be extended in the future to account for other curvature
    structures, for example, a block-diagonal one.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : Likelihood or str or torch.nn.Module
        a likelihood object, its name, or a torch criterion, see
        [`get_likelihood`][laplace.likelihoods.get_likelihood]. A composite
        likelihood like reward modeling is not supported, pass its
        `fit_likelihood` instead.
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.LongTensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.

    Attributes
    ----------
    likelihood : Likelihood
    lossfunc : torch.nn.Module
        the loss module built by the likelihood, e.g. `torch.nn.MSELoss` or
        `torch.nn.CrossEntropyLoss`, on the device and with the dtype of the model.
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.

    Notes
    -----
    Backends that compute the curvature from their own loss honour the options of a
    built-in likelihood only where their library implements them, see `_honours`.
    Otherwise, a GGN or EF backend falls back to the torch.func code of
    `GGNInterface` or `EFInterface` for `diag` and `full`, which supports any
    likelihood. Other structures raise a `ValueError`.
    """

    _honoured: ClassVar[dict[str, frozenset[str]]] = {}
    """Options of the built-in likelihoods that the library of the backend
    implements, per curvature method. Missing methods honour none."""

    _library_types: ClassVar[tuple[type[Likelihood], ...]] = (
        GaussianLikelihood,
        CategoricalLikelihood,
    )
    """Built-in likelihoods whose losses the library of the backend knows. Any other
    likelihood, built-in or not, is outside the library, see `_honours`."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ):
        # The model comes first, since the likelihood setter casts the loss to it
        self.model: nn.Module = model
        self.likelihood = likelihood

        self.last_layer: bool = last_layer
        self.subnetwork_indices: torch.LongTensor | None = subnetwork_indices
        self.dict_key_x = dict_key_x
        self.dict_key_y = dict_key_y

        self.params: list[nn.Parameter] = [
            p for p in self._model.parameters() if p.requires_grad
        ]
        self.params_dict: dict[str, nn.Parameter] = {
            k: v for k, v in self._model.named_parameters() if v.requires_grad
        }
        self.buffers_dict: dict[str, torch.Tensor] = {
            k: v for k, v in self.model.named_buffers()
        }

    @property
    def likelihood(self) -> Likelihood:
        """The likelihood object."""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood: Likelihood | str | nn.Module) -> None:
        """Set the likelihood from an object, a `Likelihood` member, a name, or a
        torch criterion, see `get_likelihood`. It also rebuilds `lossfunc` with the
        device and dtype of the model's first parameter, so that the class weights of
        a criterion match the model, and sets `factor`.

        Raises
        ------
        ValueError
            if the likelihood fits its curvature with another likelihood, e.g.
            reward modeling.
        """
        likelihood = get_likelihood(likelihood)

        if likelihood.fit_likelihood is not likelihood:
            raise ValueError(
                f"Curvature backends fit the curvature of a single likelihood, but "
                f"{likelihood!r} fits it with {likelihood.fit_likelihood!r}. Pass "
                f"`fit_likelihood` instead."
            )

        lossfunc = likelihood.loss_fn()
        param = next(self.model.parameters(), None)

        if isinstance(lossfunc, nn.Module) and param is not None:
            lossfunc = lossfunc.to(device=param.device, dtype=param.dtype)

        self._likelihood = likelihood
        self.lossfunc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lossfunc
        self.factor: float = likelihood.loss_factor

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled backend.

        Pickles made before likelihood objects existed store the likelihood as a
        string under `likelihood`. It is turned into a likelihood object here.

        Parameters
        ----------
        state : dict[str, Any]
            the pickled `__dict__`.
        """
        state = dict(state)

        if "likelihood" in state:
            state["_likelihood"] = get_likelihood(state.pop("likelihood"))

        self.__dict__.pop("likelihood", None)
        self.__dict__.update(state)

    @property
    def _model(self) -> nn.Module:
        return self.model.last_layer if self.last_layer else self.model

    def _honoured_options(self, method: str) -> frozenset[str]:
        """Options of the built-in likelihoods that this backend honours natively.

        Parameters
        ----------
        method : str
            curvature method, e.g. `"diag"`, `"kron"`, or `"full"`.

        Returns
        -------
        options : frozenset[str]
            names of the honoured options, as in `Likelihood.hyperparameters`.
        """
        return self._honoured.get(method, frozenset())

    def _honours(self, method: str, y: torch.Tensor | None = None) -> bool:
        """Whether the library of this backend computes the curvature of `method`
        for the likelihood exactly.

        That is the case if the likelihood is exactly one of the `_library_types`,
        e.g. a `GaussianLikelihood` or a `CategoricalLikelihood`, whose losses the
        libraries know, and the library implements all of its non-default options. A
        subclass might override any member, e.g. `output_hessian`, which the library
        cannot see.

        With the targets `y`, the option `ignore_index` also counts as set if `y`
        holds ignored positions, even with the default `-100`. Libraries that don't
        honour it would count those positions, e.g. padding.

        Parameters
        ----------
        method : str
            curvature method, e.g. `"diag"`, `"kron"`, or `"full"`.
        y : torch.Tensor, optional
            targets of the batch.

        Returns
        -------
        honours : bool
        """
        if type(self.likelihood) not in self._library_types:
            return False

        options = set(self.likelihood.hyperparameters())

        if y is not None and _ignored_positions(self.likelihood, y) > 0:
            options.add("ignore_index")

        return options <= self._honoured_options(method)

    @property
    def _has_custom_ignore_index(self) -> bool:
        """Whether the likelihood ignores another class than the default `-100`."""
        return (
            isinstance(self.likelihood, CategoricalLikelihood)
            and self.likelihood.ignore_index != -100
        )

    def _unsupported_error(
        self,
        method: str,
        honoured: frozenset[str] | None = None,
        stochastic: bool = False,
        y: torch.Tensor | None = None,
    ) -> ValueError:
        """Error for a curvature method that cannot handle the likelihood.

        Parameters
        ----------
        method : str
            curvature method, e.g. `"kron"`.
        honoured : frozenset[str], optional
            options that the method honours, `_honoured_options(method)` by
            default.
        stochastic : bool, default=False
            whether the method computes a Monte Carlo Fisher.
        y : torch.Tensor, optional
            targets of the batch. If they hold ignored positions, `ignore_index`
            counts as an option, as in `_honours`.

        Returns
        -------
        error : ValueError
        """
        likelihood = self.likelihood

        if type(likelihood) in self._library_types:
            if honoured is None:
                honoured = self._honoured_options(method)

            options = set(likelihood.hyperparameters())

            if y is not None and _ignored_positions(likelihood, y) > 0:
                options.add("ignore_index")

            options = sorted(options - honoured)
            noun = "options" if len(options) > 1 else "option"
            what = f"the {noun} {', '.join(options)} of {likelihood!r}"
        else:
            what = f"the likelihood {likelihood!r}"

        where = f"{type(self).__name__}.{method}"

        if stochastic:
            where += " with stochastic=True"

        return ValueError(
            f"{where} does not support {what}. Use a deterministic GGN or EF "
            f"backend with the `diag` or `full` structure, which support any "
            f"likelihood."
        )

    def _check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        """Check that the targets `y` match the outputs `f` with the likelihood.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`
        y : torch.Tensor
            targets
        """
        self.likelihood.check_targets(f, y)

    def _loss(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the summed loss of `f` w.r.t. `y`, where every leading dim of `f`
        but the last is a batch dim.

        Parameters
        ----------
        f : torch.Tensor
            outputs `(batch, ..., outputs)`
        y : torch.Tensor
            targets

        Returns
        -------
        loss : torch.Tensor
        """
        return self.likelihood.loss(f, y)

    def jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\),
        via torch.func.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        enable_backprop : bool, default = False
            whether to enable backprop through the Js and f w.r.t. x

        Returns
        -------
        Js : torch.Tensor
            Jacobians `([batch_dims], outputs, parameters)`
        f : torch.Tensor
            output function `([batch_dims], outputs)`
        """

        def model_fn_params_only(params_dict, buffers_dict):
            out = torch.func.functional_call(self.model, (params_dict, buffers_dict), x)
            return out, out

        Js, f = torch.func.jacrev(model_fn_params_only, has_aux=True)(
            self.params_dict, self.buffers_dict
        )

        # Concatenate over flattened parameters
        Js = [
            j.flatten(start_dim=-p.dim())
            for j, p in zip(Js.values(), self.params_dict.values())
        ]
        Js = torch.cat(Js, dim=-1)

        if self.subnetwork_indices is not None:
            Js = Js[..., self.subnetwork_indices]

        return (Js, f) if enable_backprop else (Js.detach(), f.detach())

    def last_layer_jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\)
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).

        Parameters
        ----------
        x : torch.Tensor
        enable_backprop : bool, default=False

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, outputs, last-layer-parameters)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        f, phi = self.model.forward_with_features(x)
        if phi.ndim > 2:
            raise NotImplementedError(
                "Last-layer Laplace does not support inputs/outputs with more than "
                "one batch dimension (e.g. sequences) yet."
            )
        bsize = phi.shape[0]
        output_size = int(f.numel() / bsize)

        # calculate Jacobians using the feature vector 'phi'
        p = next(self.model.parameters())
        identity = (
            torch.eye(output_size, device=p.device, dtype=p.dtype)
            .unsqueeze(0)
            .tile(bsize, 1, 1)
        )
        # Jacobians are batch x output x params
        Js = torch.einsum("kp,kij->kijp", phi, identity).reshape(bsize, output_size, -1)
        if self.model.last_layer.bias is not None:
            Js = torch.cat([Js, identity], dim=2)

        return (Js, f) if enable_backprop else (Js.detach(), f.detach())

    def selected_output_jacobians(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        outputs: torch.Tensor,
        enable_backprop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute parameter Jacobians for selected output coordinates per input.

        The default computes all output Jacobians before selecting coordinates.
        Backends may override it to avoid materializing the full Jacobian.

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            Input data on the model's device, with a leading batch dimension.
        outputs : torch.Tensor
            Output indices for each input, shaped `(batch,)` or
            `(batch, selected_outputs)`.
        enable_backprop : bool, default=False
            Whether to retain gradients through the Jacobians and model outputs.

        Returns
        -------
        Js : torch.Tensor
            Selected Jacobians of shape `(batch, selected_outputs, parameters)`.
        f : torch.Tensor
            Model outputs of shape `(batch, outputs)`.
        """
        # Use the output shape observed at this input. Reward models return two
        # training logits but one output for an individual inference input.
        jacobians, values = CurvatureInterface.jacobians(
            self, x, enable_backprop=enable_backprop
        )

        if outputs.ndim == 1:
            outputs = outputs[:, None]

        if outputs.shape[0] != jacobians.shape[0]:
            raise ValueError("Output selectors must match the input batch size.")

        selected = torch.gather(
            jacobians,
            dim=1,
            index=outputs.to(jacobians.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(-1, -1, jacobians.shape[-1]),
        )

        return selected, values

    def gradients(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any], y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batch gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at
        current parameter \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor or MutableMapping
            input data `(batch, input_shape)` on compatible device with model. If
            MutableMapping, every tensor in it must have the batch as its first dim.
        y : torch.Tensor

        Returns
        -------
        Gs : torch.Tensor
            gradients `(batch * positions, parameters)`, where `positions` is the number
            of leading output dims per datum (1 if there are none)
        loss : torch.Tensor
        """
        # `params_dict` holds the parameters of `_model`, so with a last layer the
        # gradients are those of the last layer at the features. Its buffers are its
        # own, since `buffers_dict` is named after the full model.
        buffers_dict = self.buffers_dict

        if self.last_layer:
            buffers_dict = dict(self._model.named_buffers())

            with torch.no_grad():
                _, x = self.model.forward_with_features(x)

        if isinstance(x, MutableMapping):
            # vmap only maps over tensors in plain containers, e.g. not `UserDict`.
            # The model then gets plain dicts, also for nested mappings; non-tensor
            # entries are passed as is. Tensors in nested containers are mapped over.
            def to_dict(x):
                if isinstance(x, MutableMapping):
                    return {k: to_dict(v) for k, v in x.items()}

                return x

            x = to_dict(x)
            in_dims_x = tree_map(lambda v: 0 if torch.is_tensor(v) else None, x)
        else:
            in_dims_x = 0

        def unsqueeze(x):
            return tree_map(lambda v: v.unsqueeze(0) if torch.is_tensor(v) else v, x)

        def loss_single(params_dict, buffers_dict, x, y):
            """Compute the loss of each position of a single sample."""
            x, y = unsqueeze(x), y.unsqueeze(0)  # vmap removes the batch dimension
            output = torch.func.functional_call(
                self._model, (params_dict, buffers_dict), x
            )
            self._check_targets(output, y)

            # Every leading dim of the output is a "position" with its own loss
            n_lead = output.ndim - 1
            output = output.flatten(end_dim=n_lead - 1)
            y = y.reshape(output.shape[0], *y.shape[n_lead:])

            def loss_position(output_t, y_t):
                return torch.func.functional_call(
                    self.lossfunc, {}, (output_t.unsqueeze(0), y_t.unsqueeze(0))
                )

            losses = torch.func.vmap(loss_position)(output, y)
            return losses, losses.sum()

        jac_fn = torch.func.jacrev(loss_single, argnums=0, has_aux=True)
        batch_jac_fn = torch.func.vmap(jac_fn, in_dims=(None, None, in_dims_x, 0))

        try:
            batch_grad, batch_loss = batch_jac_fn(self.params_dict, buffers_dict, x, y)
        except RuntimeError as e:
            if "data-dependent control flow" not in str(e):
                raise

            raise ValueError(
                "Per-sample gradients for the empirical Fisher are computed with "
                + "`torch.func.vmap`, which does not support models with "
                + "data-dependent control flow (e.g. Huggingface transformers). "
                + "Consider using a GGN backend, e.g. `CurvlinopsGGN`, instead."
            ) from e

        # (batch * positions, parameters)
        Gs = torch.cat([bg.flatten(start_dim=2) for bg in batch_grad.values()], dim=2)
        Gs = Gs.flatten(end_dim=1)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]

        loss = batch_loss.sum(0)

        return Gs, loss

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ):
        """Compute a dense curvature (approximation) in the form of a \\(P \\times P\\) matrix
        \\(H\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            Hessian approximation `(parameters, parameters)`
        """
        raise NotImplementedError

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        """Compute a Kronecker factored curvature approximation (such as KFAC).
        The approximation to \\(H\\) takes the form of two Kronecker factors \\(Q, H\\),
        i.e., \\(H \\approx Q \\otimes H\\) for each Module in the neural network permitting
        such curvature.
        \\(Q\\) is quadratic in the input-dimension of a module \\(p_{in} \\times p_{in}\\)
        and \\(H\\) in the output-dimension \\(p_{out} \\times p_{out}\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`
        N : int
            total number of data points

        Returns
        -------
        loss : torch.Tensor
        H : `laplace.utils.matrix.Kron`
            Kronecker factored Hessian approximation.
        """
        raise NotImplementedError

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ):
        """Compute a diagonal Hessian approximation to \\(H\\) and is represented as a
        vector of the dimensionality of parameters \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            vector representing the diagonal of H
        """
        raise NotImplementedError

    functorch_jacobians = jacobians


class GGNInterface(CurvatureInterface):
    """Generalized Gauss-Newton or Fisher Curvature Interface.
    The GGN is equal to the Fisher information for the available likelihoods.
    In addition to `CurvatureInterface`, methods for Jacobians are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : Likelihood or str or torch.nn.Module
        a likelihood object, its name, or a torch criterion, see
        [`get_likelihood`][laplace.likelihoods.get_likelihood].
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    stochastic : bool, default=False
        Fisher if stochastic else GGN
    num_samples: int, default=1
        Number of samples used to approximate the stochastic Fisher
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
        num_samples: int = 1,
    ) -> None:
        self.stochastic: bool = stochastic
        self.num_samples: int = num_samples

        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )

    def _check_mc_fisher(self, method: str) -> None:
        """Make sure the Monte Carlo Fisher supports the likelihood.

        It draws targets from the likelihood and uses `output_gradient` at them. So
        it cannot see the observed targets, which class weights and ignored positions
        refer to. `output_gradient` would also smooth the sampled targets, which
        changes the expected outer product. Custom likelihoods are fine, since the
        sampled gradients estimate their Fisher. Positions with the default ignore
        index, e.g. padding, are masked with the observed targets.


        Parameters
        ----------
        method : str
            curvature method, `"diag"` or `"full"`.

        Raises
        ------
        ValueError
            if the likelihood is a built-in one with non-default options.
        """
        # The samples and the output gradient of a Gaussian likelihood use its noise
        # std per output, so the sampled Fisher stays exact
        honoured = frozenset({"sigma_noise"})

        if (
            type(self.likelihood) in _BUILTIN_TYPES
            and set(self.likelihood.hyperparameters()) - honoured
        ):
            raise self._unsupported_error(method, honoured, stochastic=True)

    def _get_mc_functional_fisher(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Approximate the Fisher's middle matrix (expected outer product of the functional gradient)
        using MC integral with `self.num_samples` many samples.

        Parameters
        ----------
        f : torch.Tensor
            model outputs `(batch, ..., outputs)`
        y : torch.Tensor, optional
            observed targets. The positions that a categorical likelihood ignores in
            them, e.g. padding, get a zero Fisher.

        Returns
        -------
        F : torch.Tensor
            `(batch, ..., outputs, outputs)`
        """
        F = 0

        for _ in range(self.num_samples):
            y_sample = self.likelihood.sample(f)
            grad_sample = self.likelihood.output_gradient(f, y_sample)

            F += (
                1
                / self.num_samples
                * torch.einsum("...c,...k->...ck", grad_sample, grad_sample)
            )

        if y is not None and _ignored_positions(self.likelihood, y) > 0:
            mask = (y != self.likelihood.ignore_index).reshape(f.shape[:-1])
            F = mask[..., None, None] * F

        return F

    def _get_functional_hessian(
        self, f: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Hessian of the negative log likelihood w.r.t. the outputs `f`.

        Parameters
        ----------
        f : torch.Tensor
            model outputs `(batch, ..., outputs)`
        y : torch.Tensor, optional
            targets, passed on to the likelihood's `output_hessian`.

        Returns
        -------
        H : torch.Tensor or None
            `(batch, ..., outputs, outputs)`, or `None` for a Gaussian likelihood,
            whose output Hessian is the identity and is skipped in the GGN. With a
            noise std per output, the output Hessian is not the identity.
        """
        if self.likelihood.is_gaussian and _noise_per_output(self.likelihood) is None:
            return None

        return self.likelihood.output_hessian(f, y)

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        H : torch.Tensor
            GGN `(parameters, parameters)`
        """
        if self.stochastic:
            self._check_mc_fisher("full")

        Js, f = self.last_layer_jacobians(x) if self.last_layer else self.jacobians(x)
        H_lik = (
            self._get_mc_functional_fisher(f, y)
            if self.stochastic
            else self._get_functional_hessian(f, y)
        )

        if H_lik is not None:
            H = torch.einsum("...cp,...ck,...kq->pq", Js, H_lik, Js)
        else:  # The case of exact GGN for regression
            H = torch.einsum("...cp,...cq->pq", Js, Js)
        loss = self.factor * self._loss(f, y)

        return loss.detach(), H.detach()

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stochastic:
            self._check_mc_fisher("diag")

        Js, f = self.last_layer_jacobians(x) if self.last_layer else self.jacobians(x)
        loss = self.factor * self._loss(f, y)

        H_lik = (
            self._get_mc_functional_fisher(f, y)
            if self.stochastic
            else self._get_functional_hessian(f, y)
        )

        if H_lik is not None:
            H = torch.einsum("...cp,...ck,...kp->p", Js, H_lik, Js)

        else:  # The case of exact GGN for regression
            H = torch.einsum("...cp,...cp->p", Js, Js)

        return loss.detach(), H.detach()


class EFInterface(CurvatureInterface):
    """Interface for Empirical Fisher as Hessian approximation.
    In addition to `CurvatureInterface`, methods for gradients are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : Likelihood or str or torch.nn.Module
        a likelihood object, its name, or a torch criterion, see
        [`get_likelihood`][laplace.likelihoods.get_likelihood].
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    dict_key_x: str, default='input_ids'
        The dictionary key under which the input tensor `x` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.
    dict_key_y: str, default='labels'
        The dictionary key under which the target tensor `y` is stored. Only has effect
        when the model takes a `MutableMapping` as the input. Useful for Huggingface
        LLM models.

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        Gs, loss = Gs.detach(), loss.detach()
        H_ef = torch.einsum("bp,bq->pq", Gs, Gs)
        return self.factor * loss.detach(), self.factor * H_ef

    def diag(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gs is (..., n_params)
        Gs, loss = self.gradients(x, y)
        Gs, loss = Gs.detach(), loss.detach()
        diag_ef = torch.einsum("...p,...p->p", Gs, Gs)
        return self.factor * loss, self.factor * diag_ef
