from laplace.baselaplace import BaseLaplace
from laplace import *


def Laplace(model, likelihood, subset_of_weights='last_layer', hessian_structure='kron',
            *args, **kwargs):
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : {'classification', 'regression'}
    subset_of_weights : {'last_layer', 'all'}, default='last_layer'
        subset of weights to consider for inference
    hessian_structure : {'diag', 'kron', 'full'}, default='kron'
        structure of the Hessian approximation

    Returns
    -------
    laplace : BaseLaplace
        chosen subclass of BaseLaplace instantiated with additional arguments
    """
    laplace_map = {subclass._key: subclass for subclass in _all_subclasses(BaseLaplace)
                   if hasattr(subclass, '_key')}
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
