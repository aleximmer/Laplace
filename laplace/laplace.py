from laplace.baselaplace import BaseLaplace
from laplace import *


def Laplace(model, likelihood, weights='last-layer', cov_structure='kron', *args, **kwargs):
    laplace_map = {subclass.id: subclass for subclass in _all_subclasses(BaseLaplace)
                   if hasattr(subclass, 'id')}
    laplace_class = laplace_map[(weights, cov_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
