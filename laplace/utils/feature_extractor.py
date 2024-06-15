from __future__ import annotations

from collections.abc import MutableMapping
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn

__all__ = ["FeatureReduction", "FeatureExtractor"]


class FeatureReduction(str, Enum):
    """Possible choices of feature reduction before applying last-layer Laplace."""

    PICK_FIRST = "pick_first"
    PICK_LAST = "pick_last"
    AVERAGE = "average"


class FeatureExtractor(nn.Module):
    """Feature extractor for a PyTorch neural network.
    A wrapper which can return the output of the penultimate layer in addition to
    the output of the last layer for each forward pass. If the name of the last
    layer is not known, it can determine it automatically. It assumes that the
    last layer is linear and that for every forward pass the last layer is the same.
    If the name of the last layer is known, it can be passed as a parameter at
    initilization; this is the safest way to use this class.
    Based on https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    last_layer_name : str, default=None
        if the name of the last layer is already known, otherwise it will
        be determined automatically.
    enable_backprop: bool, default=False
        whether to enable backprop through the feature extactor to get the gradients of
        the inputs. Useful for e.g. Bayesian optimization.
    feature_reduction: FeatureReduction or str, default=None
        when the last-layer `features` is a tensor of dim >= 3, this tells how to reduce
        it into a dim-2 tensor. E.g. in LLMs for non-language modeling problems,
        the penultultimate output is a tensor of shape `(batch_size, seq_len, embd_dim)`.
        But the last layer maps `(batch_size, embd_dim)` to `(batch_size, n_classes)`.
        Note: Make sure that this option faithfully reflects the reduction in the model
        definition. When inputting a string, available options are
        `{'pick_first', 'pick_last', 'average'}`.
    """

    def __init__(
        self,
        model: nn.Module,
        last_layer_name: str | None = None,
        enable_backprop: bool = False,
        feature_reduction: FeatureReduction | str | None = None,
    ) -> None:
        if feature_reduction is not None and feature_reduction not in [
            fr.value for fr in FeatureReduction
        ]:
            raise ValueError(
                "`feature_reduction` must take value in the `FeatureReduction enum` or "
                "one of `{'pick_first', 'pick_last', 'average'}`!"
            )

        super().__init__()
        self.model: nn.Module = model
        self._features: dict[str, torch.Tensor] = dict()
        self.enable_backprop: bool = enable_backprop
        self.feature_reduction: FeatureReduction | None = feature_reduction

        self.last_layer: nn.Module | None
        if last_layer_name is None:
            self.last_layer = None
        else:
            self.set_last_layer(last_layer_name)

    def forward(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any]
    ) -> torch.Tensor:
        """Forward pass. If the last layer is not known yet, it will be
        determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor or a dict-like object containing the input tensors
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is None:
            # if this is the first forward pass and last layer is unknown
            out = self.find_last_layer(x)
        else:
            # if last and penultimate layers are already known
            out = self.model(x)
        return out

    def forward_with_features(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass which returns the output of the penultimate layer along
        with the output of the last layer. If the last layer is not known yet,
        it will be determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor or a dict-like object containing the input tensors
            one batch of data to use as input for the forward pass
        """
        out = self.forward(x)
        features = self._features[self._last_layer_name]

        if features.dim() > 2 and self.feature_reduction is not None:
            n_intermediate_dims = len(features.shape) - 2

            if self.feature_reduction == FeatureReduction.PICK_FIRST:
                features = features[
                    (slice(None), *([0] * n_intermediate_dims), slice(None))
                ].squeeze()
            elif self.feature_reduction == FeatureReduction.PICK_LAST:
                features = features[
                    (slice(None), *([-1] * n_intermediate_dims), slice(None))
                ].squeeze()
            else:
                ndim = features.ndim
                features = features.mean(
                    dim=tuple(d for d in range(ndim) if d not in [0, ndim - 1])
                ).squeeze()

        return out, features

    def set_last_layer(self, last_layer_name: str) -> None:
        """Set the last layer of the model by its name. This sets the forward
        hook to get the output of the penultimate layer.

        Parameters
        ----------
        last_layer_name : str
            the name of the last layer (fixed in `model.named_modules()`).
        """
        # set last_layer attributes and check if it is linear
        self._last_layer_name = last_layer_name
        self.last_layer = dict(self.model.named_modules())[last_layer_name]
        if not isinstance(self.last_layer, nn.Linear):
            raise ValueError("Use model with a linear last layer.")

        # set forward hook to extract features in future forward passes
        self.last_layer.register_forward_hook(self._get_hook(last_layer_name))

    def _get_hook(self, name: str) -> Callable:
        def hook(_, input, __):
            # only accepts one input (expects linear layer)
            self._features[name] = input[0]

            if not self.enable_backprop:
                self._features[name] = self._features[name].detach()

        return hook

    def find_last_layer(
        self, x: torch.Tensor | MutableMapping[str, torch.Tensor | Any]
    ) -> torch.Tensor:
        """Automatically determines the last layer of the model with one
        forward pass. It assumes that the last layer is the same for every
        forward pass and that it is an instance of `torch.nn.Linear`.
        Might not work with every architecture, but is tested with all PyTorch
        torchvision classification models (besides SqueezeNet, which has no
        linear last layer).

        Parameters
        ----------
        x : torch.Tensor or dict-like object containing the input tensors
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is not None:
            raise ValueError("Last layer is already known.")

        act_out = dict()

        def get_act_hook(name):
            def act_hook(_, input, __):
                # only accepts one input (expects linear layer)
                try:
                    act_out[name] = input[0].detach()
                except (IndexError, AttributeError):
                    act_out[name] = None
                # remove hook
                handles[name].remove()

            return act_hook

        # set hooks for all modules
        handles = dict()
        for name, module in self.model.named_modules():
            handles[name] = module.register_forward_hook(get_act_hook(name))

        # check if model has more than one module
        # (there might be pathological exceptions)
        if len(handles) <= 2:
            raise ValueError("The model only has one module.")

        # forward pass to find execution order
        out = self.model(x)

        # find the last layer, store features, return output of forward pass
        keys = list(act_out.keys())
        for key in reversed(keys):
            layer = dict(self.model.named_modules())[key]
            if len(list(layer.children())) == 0:
                self.set_last_layer(key)

                # save features from first forward pass
                self._features[key] = act_out[key]

                return out

        raise ValueError("Something went wrong (all modules have children).")
