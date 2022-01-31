from laplace.baselaplace import ParametricLaplace
from laplace import *


def Laplace(model, likelihood, subset_of_weights='last_layer', hessian_structure='kron',
            *args, **kwargs):
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : {'classification', 'regression'}
    subset_of_weights : {'last_layer', 'subnetwork', 'all'}, default='last_layer'
        subset of weights to consider for inference
    hessian_structure : {'diag', 'kron', 'full', 'lowrank'}, default='kron'
        structure of the Hessian approximation

    Returns
    -------
    laplace : ParametricLaplace
        chosen subclass of ParametricLaplace instantiated with additional arguments
    """
    if subset_of_weights == 'subnetwork' and hessian_structure not in ['full', 'diag']:
        raise ValueError('Subnetwork Laplace requires a full or diagonal Hessian approximation!')

    laplace_map = {subclass._key: subclass for subclass in _all_subclasses(ParametricLaplace)
                   if hasattr(subclass, '_key')}
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
