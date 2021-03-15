import numpy as np


def parameters_per_layer(model):
    return [np.prod(p.shape) for p in model.parameters()]

