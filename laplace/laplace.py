from laplace.baselaplace import BaseLaplace
from laplace import *


def Laplace(model, likelihood, subset_of_weights='last_layer', hessian_structure='kron', *args, **kwargs):
    laplace_map = {subclass.key: subclass for subclass in _all_subclasses(BaseLaplace)
                   if hasattr(subclass, 'key')}
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
