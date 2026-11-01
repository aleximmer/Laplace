from __future__ import annotations

import torch

from laplace.baselaplace import ELLA, BaseLaplace, VaLLA
from laplace.likelihoods import BaseLikelihood
from laplace.utils.enums import (
    FunctionalApproximation,
    HessianStructure,
    Likelihood,
    SubsetOfWeights,
)


def Laplace(
    model: torch.nn.Module,
    likelihood: BaseLikelihood | Likelihood | str,
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
        Pretrained network around which to approximate the posterior.
    likelihood : BaseLikelihood or Likelihood or str in {'classification', 'regression', 'reward_modeling'}
        Likelihood used for fitting, either a likelihood object such as
        `CategoricalLikelihood()` or its name. Reward modeling requires an
        estimator that supports pairwise preferences, such as ELLA or VaLLA.
    subset_of_weights : SubsetOfWeights or {'last_layer', 'subnetwork', 'all'}, default=SubsetOfWeights.LAST_LAYER
        subset of weights to consider for inference
    hessian_structure : HessianStructure or str in {'diag', 'kron', 'full', 'lowrank', 'gp'}, default=HessianStructure.KRON
        structure of the Hessian approximation (note that in case of 'gp',
        we are not actually doing any Hessian approximation, the inference is instead done in the functional space)
    *args : Any
        Additional positional arguments passed to the selected estimator.
    functional_approximation : FunctionalApproximation or {'sod', 'nystrom', 'variational'}, default=FunctionalApproximation.SOD
        Function-space method for `hessian_structure='gp'`. The default uses
        `FunctionalLaplace` for all weights or `FunctionalLLLaplace` for the
        last layer, and requires `n_subset`. Nyström selects `ELLA` and requires
        `subsample_size` and `n_eigenvalues`. Variational selects `VaLLA` and
        requires `inducing_locations`; a strategy also needs `num_inducing`.
        Nyström and variational methods require `subset_of_weights='all'`.
    **kwargs : Any
        Additional keyword arguments passed to the selected estimator,
        including its method-specific configuration.

    Returns
    -------
    laplace : BaseLaplace
        chosen subclass of BaseLaplace instantiated with additional arguments

    Raises
    ------
    ValueError
        If an approximation name or combination of weight subset, Hessian
        structure, and function-space method is unsupported.
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
