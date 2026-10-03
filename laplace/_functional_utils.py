"""Input helpers shared by function-space Laplace estimators."""

from __future__ import annotations

from collections.abc import MutableMapping
from functools import wraps
from typing import Any, Callable

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
)


def preserve_model_gradients(fit: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve model parameter gradients around a fitting method.

    Existing gradient buffers, including `None` values, are restored whether
    the fitting method returns or raises an exception.

    Parameters
    ----------
    fit : Callable[..., Any]
        Estimator fitting method to wrap.

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


def to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
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
    if isinstance(batch, MutableMapping):
        return batch, batch[dict_key_y]
    return batch[0], batch[1]


def regression_targets(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Match single-output vector targets without allowing accidental broadcasting."""
    if targets.ndim == 1 and outputs.ndim == 2 and outputs.shape[-1] == 1:
        targets = targets.unsqueeze(-1)
    if targets.shape != outputs.shape:
        raise ValueError(
            f"Regression targets have shape {tuple(targets.shape)}; "
            f"expected {tuple(outputs.shape)}."
        )
    return targets


def classification_targets(targets: torch.Tensor) -> torch.Tensor:
    """Convert flat, column, or one-hot class targets to flat class indices."""
    if targets.ndim == 2:
        targets = targets[:, 0] if targets.shape[1] == 1 else targets.argmax(-1)
    if targets.ndim != 1:
        raise ValueError("Classification targets must have shape [B] or [B, C].")
    return targets.long()


def training_indices(loader: DataLoader) -> torch.Tensor:
    """Dataset indices reachable through this loader's sampler."""
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


def training_epoch(loader: DataLoader) -> tuple[DataLoader, torch.Tensor]:
    """Replay one sampled epoch when the loader's row set may change per pass."""
    batch_sampler = loader.batch_sampler
    sampler = getattr(batch_sampler, "sampler", None)
    drop_last = getattr(batch_sampler, "drop_last", False)
    fixed_rows = type(batch_sampler) is BatchSampler and (
        isinstance(sampler, SequentialSampler)
        or (
            isinstance(sampler, RandomSampler)
            and not sampler.replacement
            and sampler.num_samples == len(loader.dataset)
            and not drop_last
        )
        or (isinstance(sampler, SubsetRandomSampler) and not drop_last)
    )
    if fixed_rows:
        return loader, training_indices(loader)

    batches = [list(batch) for batch in batch_sampler]
    indices = torch.as_tensor(
        [index for batch in batches for index in batch], dtype=torch.long
    )
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("A function-space fit requires a nonempty index sampler.")
    if torch.any(indices < 0) or torch.any(indices >= len(loader.dataset)):
        raise ValueError("The training sampler yielded an invalid dataset index.")
    replay = DataLoader(
        loader.dataset,
        batch_sampler=batches,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=getattr(loader, "pin_memory_device", ""),
    )
    return replay, indices


def subset_loader(loader: DataLoader, indices: torch.Tensor) -> DataLoader:
    batch_size = loader.batch_size or getattr(loader.batch_sampler, "batch_size", 1)
    return DataLoader(
        Subset(loader.dataset, indices.cpu().tolist()),
        batch_size=batch_size,
        collate_fn=loader.collate_fn,
        shuffle=False,
    )


def batch_size(inputs: Any, dict_key_x: str) -> int:
    if isinstance(inputs, MutableMapping):
        return inputs[dict_key_x].shape[0]
    return inputs.shape[0]


def concatenate_rows(chunks: list[Any]) -> Any:
    """Join batched tensor or mapping inputs, including per-row metadata."""
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


def individual_reward_inputs(inputs: Any, dict_key_x: str) -> Any:
    """Flatten the pair axis of preference-training inputs."""
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


def model_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """Capture the fixed MAP model needed to interpret a functional posterior."""
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


def check_model_fingerprint(
    model: torch.nn.Module, fingerprint: dict[str, Any]
) -> None:
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
