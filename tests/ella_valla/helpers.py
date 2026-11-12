"""Builders shared by the ELLA and VaLLA test modules."""

from typing import Any

import torch

from laplace import Laplace

METHODS = ["ella", "valla"]


class RewardModel(torch.nn.Module):
    """Linear reward model that also accepts a dict with an `input_ids` entry."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["input_ids"]

        output = self.net(x)

        return output.squeeze(-1) if x.ndim == 3 else output


def make_estimator(
    method: str,
    model: torch.nn.Module,
    likelihood: str = "classification",
    x: torch.Tensor | None = None,
    *,
    ella: dict[str, Any] | None = None,
    valla: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Build an ELLA or VaLLA estimator over all weights.

    Parameters
    ----------
    method : {"ella", "valla"}
        Which estimator to build.
    model : torch.nn.Module
        The pretrained model.
    likelihood : str, default="classification"
        The likelihood name.
    x : torch.Tensor or None, default=None
        If given, VaLLA uses `x[:2].clone()` as its inducing locations.
    ella, valla : dict or None
        Extra options for one method only. They override the defaults
        (`subsample_size=2, n_eigenvalues=1` for ELLA, `x[:2]` for VaLLA).
    **kwargs
        Options passed to both methods.

    Returns
    -------
    ELLA or VaLLA
        The unfitted estimator.
    """
    if method == "ella":
        options = {"subsample_size": 2, "n_eigenvalues": 1, **(ella or {})}
        approximation = "nystrom"
    else:
        options = {} if x is None else {"inducing_locations": x[:2].clone()}
        options.update(valla or {})
        approximation = "variational"

    return Laplace(
        model,
        likelihood,
        subset_of_weights="all",
        hessian_structure="gp",
        functional_approximation=approximation,
        **options,
        **kwargs,
    )


def fit_estimator(
    method: str,
    estimator,
    loader,
    *,
    valla: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Fit `estimator`, giving VaLLA `iterations=1` unless `valla` overrides it.

    `valla` holds fit options for VaLLA only; `kwargs` go to both methods.
    """
    if method == "ella":
        return estimator.fit(loader, **kwargs)

    return estimator.fit(loader, **{"iterations": 1, **(valla or {}), **kwargs})
