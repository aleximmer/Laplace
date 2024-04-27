from laplace.baselaplace import (
    SubsetOfWeights,
    HessianStructure,
    Likelihood,
    ParametricLaplace,
)
import torch


def Laplace(
    model: torch.nn.Module,
    likelihood: Likelihood | str,
    subset_of_weights: SubsetOfWeights | str = SubsetOfWeights.LAST_LAYER,
    hessian_structure: HessianStructure | str = HessianStructure.KRON,
    *args,
    **kwargs,
) -> ParametricLaplace:
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression'}
    subset_of_weights : SubsetofWeights or {'last_layer', 'subnetwork', 'all'}, default=SubsetOfWeights.LAST_LAYER
        subset of weights to consider for inference
    hessian_structure : HessianStructure or str in {'diag', 'kron', 'full', 'lowrank'}, default=HessianStructure.KRON
        structure of the Hessian approximation

    Returns
    -------
    laplace : ParametricLaplace
        chosen subclass of ParametricLaplace instantiated with additional arguments
    """
    if subset_of_weights == 'subnetwork' and hessian_structure not in ['full', 'diag']:
        raise ValueError(
            'Subnetwork Laplace requires a full or diagonal Hessian approximation!'
        )

    laplace_map = {
        subclass._key: subclass
        for subclass in _all_subclasses(ParametricLaplace)
        if hasattr(subclass, '_key')
    }
    laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )
