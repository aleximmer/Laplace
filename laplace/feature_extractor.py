import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional


__all__ = ['FeatureExtractor']


class FeatureExtractor(nn.Module):
    """Feature extractor for a PyTorch neural network.
    A wrapper which returns the output of the penultimate layer in addition to
    the output of the last layer for each forward pass. It assumes that the
    last layer is linear.
    Based on https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76.

    Arguments
    ----------
    model : torch.nn.Module
        PyTorch model

    last_layer_name (optional) : str, default=None
        If the user already knows the name of the last layer, otherwise it will
        be determined automatically.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying PyTorch model.

    last_layer : torch.nn.module
        The torch module corresponding to the last layer (has to be instance
        of torch.nn.Linear).

    Examples
    --------

    Notes
    -----
    Limitations:
        - Assumes that the last layer is always the same for any forward pass
        - Assumes that the last layer is an instance of torch.nn.Linear
    """
    def __init__(self, model: nn.Module, last_layer_name: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self._features = dict()
        if last_layer_name is None:
            self._found = False
        else:
            self.set_last_layer(last_layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._found:
            # if last and penultimate layers are already known
            out = self.model(x)
        else:
            # if this is the first forward pass
            out = self.find_last_layer(x)
        return out

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x)
        features = self._features[self._last_layer_name]
        return out, features

    def set_last_layer(self, last_layer_name: str) -> None:
        # set last_layer attributes and check if it is linear
        self._last_layer_name = last_layer_name
        self.last_layer = dict(self.model.named_modules())[last_layer_name]
        if not isinstance(self.last_layer, nn.Linear):
            raise ValueError('Use model with a linear last layer.')

        # set forward hook to extract features in future forward passes
        self.last_layer.register_forward_hook(self._get_hook(last_layer_name))

        # last layer is now identified and hook is set
        self._found = True

    def _get_hook(self, name: str) -> Callable:
        def hook(_, input, __):
            # only accepts one input (expects linear layer)
            self._features[name] = input[0].detach()
        return hook

    def find_last_layer(self, x: torch.Tensor) -> torch.Tensor:
        if self._found:
            raise ValueError('Last layer is already known.')

        act_out = dict()
        def get_act_hook(name):
            def act_hook(_, input, __):
                # only accepts one input (expects linear layer)
                if isinstance(input[0], torch.Tensor):
                    act_out[name] = input[0].detach()
                else:
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
            raise ValueError('The model only has one module.')

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

        raise ValueError('Something went wrong (all modules have children).')
