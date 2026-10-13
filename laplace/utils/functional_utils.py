"""Reusable input and sampling utilities for function-space estimators.

Public helpers are available through ``laplace.utils``. Private helpers support
the estimators' reward-input preparation and checkpoint validation.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from functools import wraps
from typing import Any, Callable

import torch
from torch.linalg import LinAlgError
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
)

__all__ = [
    "preserve_model_gradients",
    "same_configuration",
    "draw_gaussian",
    "to_device",
    "split_batch",
    "regression_targets",
    "classification_targets",
    "training_indices",
    "subset_loader",
    "batch_size",
    "concatenate_rows",
    "select_rows",
]


def preserve_model_gradients(fit: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve model parameter gradients around an estimator method.

    Existing gradient buffers, including `None` values, are restored whether
    the method returns or raises an exception. This applies to fitting and
    prediction methods that compute parameter Jacobians.
    The wrapped method's first argument must have a ``model`` attribute
    containing a ``torch.nn.Module``.

    Parameters
    ----------
    fit : Callable[..., Any]
        Estimator method to wrap.

    Returns
    -------
    Callable[..., Any]
        Wrapped method with the original return value.
    """

    @wraps(fit)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        parameters = list(self.model.parameters())
        gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ]
        try:
            return fit(self, *args, **kwargs)
        finally:
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient

    return wrapped


def same_configuration(left: Any, right: Any) -> bool:
    """Compare nested checkpoint settings, including tensor contents.

    Parameters
    ----------
    left : Any
        Saved setting or nested collection of settings.
    right : Any
        Setting to compare with ``left``.

    Returns
    -------
    bool
        Whether both settings have matching nested structure and values.
    """
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, MutableMapping) and isinstance(right, MutableMapping):
        return left.keys() == right.keys() and all(
            same_configuration(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            same_configuration(a, b) for a, b in zip(left, right)
        )
    return left == right


def draw_gaussian(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw from a possibly rank-deficient Gaussian output distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean shaped ``[B, C]`` or flattened joint mean ``[B*C]``.
    covariance : torch.Tensor
        Covariance shaped ``[B, C, C]`` or joint covariance ``[B*C, B*C]``.
    n_samples : int
        Number of output draws.
    generator : torch.Generator or None
        Random generator on the same device as ``mean``.

    Returns
    -------
    torch.Tensor
        Samples shaped ``[n_samples, B, C]`` or ``[n_samples, B*C]``.

    Raises
    ------
    torch.linalg.LinAlgError
        If the covariance has negative eigenvalues beyond rounding tolerance.
    RuntimeError
        If a covariance requiring gradients cannot be factored after jitter.

    Notes
    -----
    Singular positive-semidefinite covariances use an eigendecomposition.
    When covariance gradients are required, a small diagonal jitter enables
    differentiable Cholesky sampling instead.
    """
    covariance = (covariance + covariance.transpose(-1, -2)) / 2
    try:
        root = torch.linalg.cholesky(covariance)
    except LinAlgError:
        # A rank-limited function-space posterior can be singular.
        values, vectors = torch.linalg.eigh(
            covariance.detach() if covariance.requires_grad else covariance
        )
        # Low-rank PSD matrices can acquire negative eigenvalues from
        # float32 rounding; judge them relative to the covariance scale.
        tolerance = (
            10
            * torch.finfo(covariance.dtype).eps
            * covariance.shape[-1]
            * values.abs().amax(dim=-1, keepdim=True)
        )
        if torch.any(values < -tolerance):
            raise
        if covariance.requires_grad:
            # Eigenvectors and sqrt have undefined derivatives at repeated
            # zero eigenvalues. Keep MC objectives differentiable there.
            scale = covariance.diagonal(dim1=-2, dim2=-1).detach().abs().amax(dim=-1)
            scale = torch.where(scale > 0, scale, 1)
            identity = torch.eye(
                covariance.shape[-1],
                device=covariance.device,
                dtype=covariance.dtype,
            )
            negative_shift = -values.amin(dim=-1).clamp_max(0)
            for multiplier in (1, 10, 100):
                jitter = (
                    negative_shift
                    + multiplier * torch.finfo(covariance.dtype).eps * scale
                )
                try:
                    root = torch.linalg.cholesky(
                        covariance + jitter[..., None, None] * identity
                    )
                    break
                except LinAlgError:
                    continue
            else:
                raise RuntimeError(
                    "Could not factor a differentiable functional covariance."
                )
        else:
            root = vectors @ torch.diag_embed(values.clamp_min(0).sqrt())
    noise = torch.randn(
        (n_samples, *mean.shape),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    if mean.ndim == 1:
        draws = noise @ root.T
    else:
        draws = torch.einsum("bij,sbj->sbi", root, noise)
    return mean.unsqueeze(0) + draws


def to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move tensors in a nested input to a device and floating-point dtype.

    Parameters
    ----------
    value : Any
        Tensor, mapping, sequence, or non-tensor metadata.
    device : torch.device
        Destination device for tensors.
    dtype : torch.dtype
        Destination dtype for floating-point tensors. Integer tensors retain
        their dtype.

    Returns
    -------
    Any
        Input structure with tensors moved to ``device``.
    """
    if isinstance(value, torch.Tensor):
        return value.to(
            device=device, dtype=dtype if value.is_floating_point() else None
        )
    if isinstance(value, MutableMapping):
        return {key: to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device, dtype) for item in value)
    return value


def split_batch(batch: Any, dict_key_y: str) -> tuple[Any, torch.Tensor]:
    """Separate model inputs and targets from a loader batch.

    Parameters
    ----------
    batch : Any
        Mapping containing a target key, or an ``(inputs, targets)`` batch.
    dict_key_y : str
        Target key when ``batch`` is a mapping.

    Returns
    -------
    tuple[Any, torch.Tensor]
        Model inputs and targets. A mapping is kept intact as model input.
    """
    if isinstance(batch, MutableMapping):
        return batch, batch[dict_key_y]
    return batch[0], batch[1]


def regression_targets(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Match regression target shape to model output without broadcasting.

    Parameters
    ----------
    targets : torch.Tensor
        Targets shaped like ``outputs``, or ``[B]`` for a single output.
    outputs : torch.Tensor
        Model outputs used to determine the required target shape.

    Returns
    -------
    torch.Tensor
        Targets with the same shape as ``outputs``.

    Raises
    ------
    ValueError
        If target and output shapes cannot be matched exactly.
    """
    if targets.ndim == 1 and outputs.ndim == 2 and outputs.shape[-1] == 1:
        targets = targets.unsqueeze(-1)
    if targets.shape != outputs.shape:
        raise ValueError(
            f"Regression targets have shape {tuple(targets.shape)}; "
            f"expected {tuple(outputs.shape)}."
        )
    return targets


def classification_targets(targets: torch.Tensor) -> torch.Tensor:
    """Convert class targets to a flat integer index tensor.

    Parameters
    ----------
    targets : torch.Tensor
        Flat indices ``[B]``, column indices ``[B, 1]``, or one-hot targets
        ``[B, C]``.

    Returns
    -------
    torch.Tensor
        Class indices shaped ``[B]`` with integer dtype.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as one class per row.
    """
    if targets.ndim == 2:
        targets = targets[:, 0] if targets.shape[1] == 1 else targets.argmax(-1)
    if targets.ndim != 1:
        raise ValueError("Classification targets must have shape [B] or [B, C].")
    return targets.long()


def training_indices(loader: DataLoader) -> torch.Tensor:
    """Find dataset indices reachable through a training loader's sampler.

    Parameters
    ----------
    loader : DataLoader
        Loader over an indexable, sized dataset whose sampler or batch
        sampler defines the training rows.

    Returns
    -------
    torch.Tensor
        One-dimensional indices of eligible dataset rows. Full-dataset random
        sampling and subset-random sampling use their index sets without
        consuming the loader's generator. Other samplers are iterated once.

    Raises
    ------
    ValueError
        If the sampler is empty, emits invalid indices, or has an invalid
        ``drop_last`` batch size.
    """
    batch_sampler = loader.batch_sampler
    sampler = (
        getattr(batch_sampler, "sampler", None)
        if batch_sampler is not None
        else loader.sampler
    )
    if sampler is None:
        indices = [index for batch_indices in batch_sampler for index in batch_indices]
    elif isinstance(sampler, SequentialSampler):
        indices = list(range(len(loader.dataset)))
        if getattr(batch_sampler, "drop_last", False):
            batch_size = getattr(batch_sampler, "batch_size", None)
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(
                    "A drop-last batch sampler requires a positive batch size."
                )
            indices = indices[: len(batch_sampler) * batch_size]
    elif (
        isinstance(sampler, RandomSampler)
        and not sampler.replacement
        and sampler.num_samples == len(loader.dataset)
    ):
        # All dataset rows are eligible; avoid consuming the loader's RNG.
        indices = list(range(len(loader.dataset)))
    elif isinstance(sampler, SubsetRandomSampler):
        indices = list(sampler.indices)
    else:
        indices = list(sampler)
    result = torch.as_tensor(indices, dtype=torch.long)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("A function-space fit requires a nonempty index sampler.")
    if torch.any(result < 0) or torch.any(result >= len(loader.dataset)):
        raise ValueError("The training sampler yielded an invalid dataset index.")
    return result


def subset_loader(loader: DataLoader, indices: torch.Tensor) -> DataLoader:
    """Create a sequential loader over selected rows of a dataset.

    Parameters
    ----------
    loader : DataLoader
        Source loader providing the dataset, batch size, and collate function.
    indices : torch.Tensor
        Dataset indices to load in the supplied order.

    Returns
    -------
    DataLoader
        Non-shuffled loader over the selected rows.
    """
    batch_size = loader.batch_size or getattr(loader.batch_sampler, "batch_size", 1)
    return DataLoader(
        Subset(loader.dataset, indices.cpu().tolist()),
        batch_size=batch_size,
        collate_fn=loader.collate_fn,
        shuffle=False,
    )


def batch_size(inputs: Any, dict_key_x: str) -> int:
    """Read the leading batch dimension from tensor or mapping inputs.

    Parameters
    ----------
    inputs : Any
        Batched tensor or mapping containing the model input tensor.
    dict_key_x : str
        Input key when ``inputs`` is a mapping.

    Returns
    -------
    int
        Number of input rows.
    """
    if isinstance(inputs, MutableMapping):
        return inputs[dict_key_x].shape[0]
    return inputs.shape[0]


def concatenate_rows(chunks: list[Any]) -> Any:
    """Concatenate batched inputs while retaining per-row metadata.

    Parameters
    ----------
    chunks : list[Any]
        Nonempty list of compatible input batches. Tensors carry a batch axis
        along dimension 0, mutable mappings are combined recursively, and lists
        or tuples contain one metadata entry per row. Constant metadata must
        match across chunks.

    Returns
    -------
    Any
        Combined rows in input order, preserving mapping keys and list or tuple
        types. Matching constant metadata is returned unchanged.

    Raises
    ------
    ValueError
        If mapping keys differ or non-batched metadata conflicts.
    """
    first = chunks[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(chunks, dim=0)
    if isinstance(first, MutableMapping):
        if any(chunk.keys() != first.keys() for chunk in chunks):
            raise ValueError("Mapping batches must have matching input keys.")
        return {
            key: concatenate_rows([chunk[key] for chunk in chunks]) for key in first
        }
    if isinstance(first, (list, tuple)):
        return type(first)(item for chunk in chunks for item in chunk)
    if all(chunk == first for chunk in chunks):
        return first
    raise ValueError("Cannot combine non-batched mapping metadata across batches.")


def select_rows(inputs: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select batch rows recursively without indexing constant metadata.

    Parameters
    ----------
    inputs : Any
        Tensor, mapping, sequence, or non-batched metadata.
    indices : torch.Tensor
        Indices of rows to retain.
    batch_size : int
        Leading size identifying values that carry a batch axis.

    Returns
    -------
    Any
        Selected rows for batched values and unchanged constant metadata.
    """
    if isinstance(inputs, MutableMapping):
        return {
            key: select_rows(value, indices, batch_size)
            for key, value in inputs.items()
        }
    if isinstance(inputs, torch.Tensor):
        return (
            inputs[indices] if inputs.ndim and inputs.shape[0] == batch_size else inputs
        )
    if isinstance(inputs, (list, tuple)) and len(inputs) == batch_size:
        row_indices = indices.detach().cpu().tolist()
        return type(inputs)(inputs[index] for index in row_indices)
    return inputs


def _individual_reward_inputs(inputs: Any, dict_key_x: str) -> Any:
    """Flatten preference pairs into individual reward-model inputs.

    Parameters
    ----------
    inputs : Any
        Tensor or mapping with input shape ``[B, 2, ...]`` and optional
        per-pair metadata.
    dict_key_x : str
        Pair-input key when ``inputs`` is a mapping.

    Returns
    -------
    Any
        Inputs with ``2*B`` individual rows; per-pair metadata is repeated.

    Raises
    ------
    ValueError
        If the model input lacks a pair axis of length two.
    """
    if isinstance(inputs, MutableMapping):
        pair_tensor = inputs[dict_key_x]
        if pair_tensor.ndim < 3 or pair_tensor.shape[1] != 2:
            raise ValueError(
                "Reward training inputs must have a pair axis of length 2."
            )
        batch = pair_tensor.shape[0]
        return {
            key: (
                value.reshape(batch * 2, *value.shape[2:])
                if isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[:2] == (batch, 2)
                else value.repeat_interleave(2, dim=0)
                if isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] == batch
                else type(value)(item for entry in value for item in (entry, entry))
                if isinstance(value, (list, tuple)) and len(value) == batch
                else value
            )
            for key, value in inputs.items()
        }
    if inputs.ndim < 3 or inputs.shape[1] != 2:
        raise ValueError("Reward training inputs must have a pair axis of length 2.")
    return inputs.reshape(inputs.shape[0] * 2, *inputs.shape[2:])


def _model_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """Capture the pretrained model configuration for checkpoint validation.

    Parameters
    ----------
    model : torch.nn.Module
        Fixed model used to build a function-space posterior.

    Returns
    -------
    dict[str, Any]
        Copied parameter and buffer values, active parameter shapes and dtypes,
        and module structure.
    """
    return {
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "active_parameters": [
            (name, tuple(parameter.shape), parameter.dtype)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "module_structure": [
            (
                name,
                type(module).__module__,
                type(module).__qualname__,
                module.extra_repr(),
            )
            for name, module in model.named_modules()
        ],
    }


def _check_model_fingerprint(
    model: torch.nn.Module, fingerprint: dict[str, Any]
) -> None:
    """Require a model to match the one saved in a functional checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to compare with the saved configuration.
    fingerprint : dict[str, Any]
        Configuration returned by :func:`_model_fingerprint`.

    Raises
    ------
    ValueError
        If parameters, buffers, active coordinates, or modules differ.
    """
    current = model.state_dict()
    expected_state = fingerprint["state_dict"]
    active_parameters = [
        (name, tuple(parameter.shape), parameter.dtype)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    module_structure = [
        (name, type(module).__module__, type(module).__qualname__, module.extra_repr())
        for name, module in model.named_modules()
    ]
    if (
        current.keys() != expected_state.keys()
        or active_parameters != fingerprint["active_parameters"]
        or module_structure != fingerprint["module_structure"]
        or any(
            not torch.equal(value.detach().cpu(), expected_state[name])
            for name, value in current.items()
        )
    ):
        raise ValueError(
            "Checkpoint requires the same pretrained model parameters, trainable coordinates, and buffers."
        )
