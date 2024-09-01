from .core import extend
from .operations import OP_BATCH_GRADS

__all__ = ['batch_gradient']


def batch_gradient(model, loss_fn, inputs, targets):
    with extend(model, OP_BATCH_GRADS):
        model.zero_grad()
        f = model(inputs)
        loss = loss_fn(f, targets)
        loss.backward()
    return f

