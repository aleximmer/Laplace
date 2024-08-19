import warnings
from torch import nn
from .operation import *
from .linear import Linear
from .conv import Conv2d
from .batchnorm import BatchNorm1d, BatchNorm2d
from .bias import Bias, BiasExt
from .scale import Scale, ScaleExt

__all__ = [
    'Linear',
    'Conv2d',
    'BatchNorm1d',
    'BatchNorm2d',
    'Bias',
    'Scale',
    'BiasExt',
    'ScaleExt',
    'get_op_class',
    'Operation',
    'OP_COV_KRON',
    'OP_COV_DIAG',
    'OP_COV_UNIT_WISE',
    'OP_GRAM_DIRECT',
    'OP_GRAM_HADAMARD',
    'OP_BATCH_GRADS',
    'OP_ACCUMULATE_GRADS'
]


def get_op_class(module):
    if isinstance(module, nn.Linear):
        return Linear
    elif isinstance(module, nn.Conv2d):
        return Conv2d
    elif isinstance(module, nn.BatchNorm1d):
        return BatchNorm1d
    elif isinstance(module, nn.BatchNorm2d):
        return BatchNorm2d
    elif isinstance(module, Bias):
        return BiasExt
    elif isinstance(module, Scale):
        return ScaleExt
    else:
        warnings.warn(f'Failed to lookup operations for Module {module}.')
        return None
