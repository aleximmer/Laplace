from typing import List
from contextlib import contextmanager

import torch.nn as nn
from .utils import im2col_2d, record_original_requires_grad
from .operations import OP_ACCUMULATE_GRADS, get_op_class


@contextmanager
def extend(model, op_names):
    if not isinstance(op_names, (list, tuple)):
        op_names = [op_names]
    accumulate_grads = False
    if OP_ACCUMULATE_GRADS in op_names:
        accumulate_grads = True
        op_names = [name for name in op_names if name != OP_ACCUMULATE_GRADS]
    handles = []

    def forward_hook(module, in_data, out_data):
        in_data = in_data[0].clone().detach()
        in_data = _preprocess_in_data(module, in_data, out_data)
        _call_operations_in_forward(module, in_data)

        def backward_hook(out_grads):
            out_grads = out_grads.clone().detach()
            out_grads = _preprocess_out_grads(module, out_grads)
            _call_operations_in_backward(module, in_data, out_grads)

        if out_data.requires_grad:
            handles.append(out_data.register_hook(backward_hook))

    for module in model.modules():
        requires_grad = False
        for attr in ['weight', 'bias']:
            param = getattr(module, attr, None)
            if param is not None:
                requires_grad = requires_grad or param.requires_grad
                record_original_requires_grad(param)
        if not requires_grad:
            continue
        # register hooks and operations in modules
        handles.append(module.register_forward_hook(forward_hook))
        _register_operations(model, module, op_names)

    yield

    # remove hooks and operations from modules
    for handle in handles:
        handle.remove()
    for module in model.modules():
        _remove_operations(module)

    # accumulate param.grad to param.acc_grad
    if accumulate_grads:
        attr = OP_ACCUMULATE_GRADS
        for param in model.parameters():
            if param.grad is None:
                continue
            if not hasattr(param, attr):
                setattr(param, attr, param.grad)
            else:
                acc_grad = getattr(param, attr)
                acc_grad.add_(param.grad)


def _preprocess_in_data(module, in_data, out_data):
    if isinstance(module, nn.Conv2d):
        in_data = im2col_2d(in_data, module)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        bnorm = module
        f = bnorm.num_features
        if isinstance(module, nn.BatchNorm1d):
            shape = (1, f)
        elif isinstance(module, nn.BatchNorm2d):
            shape = (1, f, 1, 1)
        else:
            shape = (1, f, 1, 1, 1)
        # restore normalized input
        in_data_norm = (out_data -
                        bnorm.bias.view(shape)).div(bnorm.weight.view(shape))
        in_data = in_data_norm

    if isinstance(module, nn.LayerNorm):
        layernorm = module
        # restore normalized input
        in_data_norm = (out_data - layernorm.bias).div(layernorm.weight)
        in_data = in_data_norm

    return in_data


def _preprocess_out_grads(module, out_grads):
    if isinstance(module, nn.Conv2d):
        out_grads = out_grads.flatten(start_dim=2)

    return out_grads


def _register_operations(model: nn.Module, module: nn.Module, op_names: List):
    op_class = get_op_class(module)
    if op_class is not None:
        setattr(module, 'operation', op_class(module, model, op_names))


def _call_operations_in_forward(module, in_data):
    if hasattr(module, 'operation'):
        module.operation.forward_post_process(in_data)


def _call_operations_in_backward(module, in_data, out_grads):
    if hasattr(module, 'operation'):
        module.operation.backward_pre_process(in_data, out_grads)


def _remove_operations(module):
    if hasattr(module, 'operation'):
        delattr(module, 'operation')
