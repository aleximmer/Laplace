from __future__ import annotations

import torch

from laplace.baselaplace import ELLA, BaseLaplace, VaLLA
from laplace.utils.enums import (
    FunctionalApproximation,
    HessianStructure,
    Likelihood,
    SubsetOfWeights,
)


def Laplace(
    model: torch.nn.Module,
    likelihood: Likelihood | str,
    subset_of_weights: SubsetOfWeights | str = SubsetOfWeights.LAST_LAYER,
    hessian_structure: HessianStructure | str = HessianStructure.KRON,
    *args,
    functional_approximation: FunctionalApproximation
    | str = FunctionalApproximation.SOD,
    **kwargs,
) -> BaseLaplace:
    """Simplified Laplace access using strings instead of different classes.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : Likelihood or str in {'classification', 'regression'}
    subset_of_weights : SubsetofWeights or {'last_layer', 'subnetwork', 'all'}, default=SubsetOfWeights.LAST_LAYER
        subset of weights to consider for inference
    hessian_structure : HessianStructure or str in {'diag', 'kron', 'full', 'lowrank', 'gp'}, default=HessianStructure.KRON
        structure of the Hessian approximation (note that in case of 'gp',
        we are not actually doing any Hessian approximation, the inference is instead done in the functional space)
    functional_approximation : {'sod', 'nystrom', 'variational'}, default='sod'
        Function-space method for `hessian_structure='gp'` and
        `subset_of_weights='all'`. The default uses `FunctionalLaplace` and
        requires `n_subset`. Nyström selects `ELLA` and requires
        `subsample_size` and `n_eigenvalues`. Variational selects `VaLLA` and
        requires `inducing_locations`; a strategy also needs `num_inducing`.
    Returns
    -------
    laplace : BaseLaplace
        chosen subclass of BaseLaplace instantiated with additional arguments
    """
    if subset_of_weights == "subnetwork" and hessian_structure not in ["full", "diag"]:
        raise ValueError(
            "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
        )
    try:
        approximation = FunctionalApproximation(functional_approximation)
    except ValueError as exc:
        raise ValueError(
            "functional_approximation must be 'sod', 'nystrom', or 'variational'."
        ) from exc
    if (
        hessian_structure != HessianStructure.GP
        and approximation != FunctionalApproximation.SOD
    ):
        raise ValueError("functional_approximation requires hessian_structure='gp'.")
    if (
        hessian_structure == HessianStructure.GP
        and approximation != FunctionalApproximation.SOD
    ):
        if subset_of_weights != SubsetOfWeights.ALL:
            raise ValueError(
                "Nystrom and variational function-space methods require subset_of_weights='all'."
            )
        cls = ELLA if approximation == FunctionalApproximation.NYSTROM else VaLLA
        return cls(model, likelihood, *args, **kwargs)
    laplace_map = {
        subclass._key: subclass
        for subclass in _all_subclasses(BaseLaplace)
        if hasattr(subclass, "_key")
    }
    try:
        laplace_class = laplace_map[(subset_of_weights, hessian_structure)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Laplace combination: {subset_of_weights}, {hessian_structure}."
        ) from exc
    return laplace_class(model, likelihood, *args, **kwargs)


def _all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )
