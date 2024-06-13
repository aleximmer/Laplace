from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

import laplace
from laplace.utils.enums import Likelihood
from laplace.utils.feature_extractor import FeatureExtractor
from laplace.utils.swag import fit_diagonal_swag_var

__all__ = [
    "SubnetMask",
    "RandomSubnetMask",
    "LargestMagnitudeSubnetMask",
    "LargestVarianceDiagLaplaceSubnetMask",
    "LargestVarianceSWAGSubnetMask",
    "ParamNameSubnetMask",
    "ModuleNameSubnetMask",
    "LastLayerSubnetMask",
]


class SubnetMask:
    """Baseclass for all subnetwork masks in this library (for subnetwork Laplace).

    Parameters
    ----------
    model : torch.nn.Module
    """

    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model
        self.parameter_vector: torch.Tensor = parameters_to_vector(
            self.model.parameters()
        ).detach()
        self._n_params: int = len(self.parameter_vector)
        self._indices: torch.LongTensor | None = None
        self._n_params_subnet: int | None = None

    def _check_select(self) -> None:
        if self._indices is None:
            raise AttributeError("Subnetwork mask not selected. Run select() first.")

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def indices(self) -> torch.LongTensor:
        self._check_select()
        return self._indices

    @property
    def n_params_subnet(self) -> int:
        if self._n_params_subnet is None:
            self._check_select()
            self._n_params_subnet = len(self._indices)
        return self._n_params_subnet

    def convert_subnet_mask_to_indices(
        self, subnet_mask: torch.Tensor
    ) -> torch.LongTensor:
        """Converts a subnetwork mask into subnetwork indices.

        Parameters
        ----------
        subnet_mask : torch.Tensor
            a binary vector of size (n_params) where 1s locate the subnetwork parameters
            within the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)

        Returns
        -------
        subnet_mask_indices : torch.LongTensor
            a vector of indices of the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
            that define the subnetwork
        """
        if not isinstance(subnet_mask, torch.Tensor):
            raise ValueError("Subnetwork mask needs to be torch.Tensor!")
        elif (
            subnet_mask.dtype
            not in [
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.bool,
            ]
            or len(subnet_mask.shape) != 1
        ):
            raise ValueError(
                "Subnetwork mask needs to be 1-dimensional integral or boolean tensor!"
            )
        elif (
            len(subnet_mask) != self._n_params
            or len(subnet_mask[subnet_mask == 0]) + len(subnet_mask[subnet_mask == 1])
            != self._n_params
        ):
            raise ValueError(
                "Subnetwork mask needs to be a binary vector of"
                "size (n_params) where 1s locate the subnetwork"
                "parameters within the vectorized model parameters"
                "(i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)!"
            )

        subnet_mask_indices = subnet_mask.nonzero(as_tuple=True)[0]
        return subnet_mask_indices

    def select(self, train_loader: DataLoader | None = None) -> torch.LongTensor:
        """Select the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader, default=None
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set

        Returns
        -------
        subnet_mask_indices : torch.LongTensor
            a vector of indices of the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
            that define the subnetwork
        """
        if self._indices is not None:
            raise ValueError("Subnetwork mask already selected.")

        subnet_mask = self.get_subnet_mask(train_loader)
        self._indices = self.convert_subnet_mask_to_indices(subnet_mask)
        return self._indices

    def get_subnet_mask(self, train_loader: DataLoader) -> torch.Tensor:
        """Get the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set

        Returns
        -------
        subnet_mask: torch.Tensor
            a binary vector of size (n_params) where 1s locate the subnetwork parameters
            within the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
        """
        raise NotImplementedError


class ScoreBasedSubnetMask(SubnetMask):
    """Baseclass for subnetwork masks defined by selecting
    the top-scoring parameters according to some criterion.

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    """

    def __init__(self, model: nn.Module, n_params_subnet: int) -> None:
        super().__init__(model)

        if n_params_subnet is None:
            raise ValueError(
                "Need to pass number of subnetwork parameters when using subnetwork Laplace."
            )
        if n_params_subnet > self._n_params:
            raise ValueError(
                f"Subnetwork ({n_params_subnet}) cannot be larger than model ({self._n_params})."
            )
        self._n_params_subnet = n_params_subnet
        self._param_scores: torch.Tensor | None = None

    def compute_param_scores(self, train_loader: DataLoader) -> torch.Tensor:
        raise NotImplementedError

    def _check_param_scores(self) -> None:
        assert self._param_scores is not None
        if self._param_scores.shape != self.parameter_vector.shape:
            raise ValueError(
                "Parameter scores need to be of same shape as parameter vector."
            )

    def get_subnet_mask(self, train_loader):
        """Get the subnetwork mask by (descendingly) ranking parameters based on their scores."""
        if self._param_scores is None:
            self._param_scores = self.compute_param_scores(train_loader)
        self._check_param_scores()

        idx = torch.argsort(self._param_scores, descending=True)[
            : self._n_params_subnet
        ]
        idx = idx.sort()[0]
        subnet_mask = torch.zeros_like(self.parameter_vector).bool()
        subnet_mask[idx] = 1
        return subnet_mask


class RandomSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask of parameters sampled uniformly at random."""

    def compute_param_scores(self, train_loader: DataLoader) -> torch.Tensor:
        return torch.rand_like(self.parameter_vector)


class LargestMagnitudeSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest magnitude."""

    def compute_param_scores(self, train_loader: DataLoader) -> torch.Tensor:
        return self.parameter_vector.abs()


class LargestVarianceDiagLaplaceSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest marginal variances
    (estimated using a diagonal Laplace approximation over all model parameters).

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for variance estimation
    """

    def __init__(
        self,
        model: nn.Module,
        n_params_subnet: int,
        diag_laplace_model: laplace.baselaplace.DiagLaplace,
    ):
        super().__init__(model, n_params_subnet)
        self.diag_laplace_model: laplace.baselaplace.DiagLaplace = diag_laplace_model

    def compute_param_scores(self, train_loader: DataLoader) -> torch.Tensor:
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")

        self.diag_laplace_model.fit(train_loader)
        return self.diag_laplace_model.posterior_variance


class LargestVarianceSWAGSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest marginal variances
    (estimated using diagonal SWAG over all model parameters).

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    likelihood : str
        'classification' or 'regression'
    swag_n_snapshots : int
        number of model snapshots to collect for SWAG
    swag_snapshot_freq : int
        SWAG snapshot collection frequency (in epochs)
    swag_lr : float
        learning rate for SWAG snapshot collection
    """

    def __init__(
        self,
        model: nn.Module,
        n_params_subnet: int,
        likelihood: Likelihood | str = Likelihood.CLASSIFICATION,
        swag_n_snapshots: int = 40,
        swag_snapshot_freq: int = 1,
        swag_lr: float = 0.01,
    ):
        if likelihood not in [Likelihood.CLASSIFICATION, Likelihood.REGRESSION]:
            raise ValueError("Only available for classification and regression!")

        super().__init__(model, n_params_subnet)

        self.likelihood: Likelihood | str = likelihood
        self.swag_n_snapshots: int = swag_n_snapshots
        self.swag_snapshot_freq: int = swag_snapshot_freq
        self.swag_lr: float = swag_lr

    def compute_param_scores(self, train_loader: DataLoader) -> torch.Tensor:
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")

        if self.likelihood == Likelihood.CLASSIFICATION:
            criterion = CrossEntropyLoss(reduction="mean")
        else:
            criterion = MSELoss(reduction="mean")

        param_variances = fit_diagonal_swag_var(
            self.model,
            train_loader,
            criterion,
            n_snapshots_total=self.swag_n_snapshots,
            snapshot_freq=self.swag_snapshot_freq,
            lr=self.swag_lr,
        )
        return param_variances


class ParamNameSubnetMask(SubnetMask):
    """Subnetwork mask corresponding to the specified parameters of the neural network.

    Parameters
    ----------
    model : torch.nn.Module
    parameter_names: List[str]
        list of names of the parameters (as in `model.named_parameters()`)
        that define the subnetwork
    """

    def __init__(self, model: nn.Module, parameter_names: list[str]) -> None:
        super().__init__(model)
        self._parameter_names: list[str] = parameter_names
        self._n_params_subnet: int | None = None

    def _check_param_names(self) -> None:
        param_names = deepcopy(self._parameter_names)
        if len(param_names) == 0:
            raise ValueError("Parameter name list cannot be empty.")

        for name, _ in self.model.named_parameters():
            if name in param_names:
                param_names.remove(name)
        if len(param_names) > 0:
            raise ValueError(f"Parameters {param_names} do not exist in model.")

    def get_subnet_mask(self, train_loader: DataLoader) -> torch.Tensor:
        """Get the subnetwork mask identifying the specified parameters."""

        self._check_param_names()

        subnet_mask_list = []
        for name, param in self.model.named_parameters():
            if name in self._parameter_names:
                mask_method = torch.ones_like
            else:
                mask_method = torch.zeros_like
            subnet_mask_list.append(mask_method(parameters_to_vector(param)))
        subnet_mask = torch.cat(subnet_mask_list).bool()
        return subnet_mask


class ModuleNameSubnetMask(SubnetMask):
    """Subnetwork mask corresponding to the specified modules of the neural network.

    Parameters
    ----------
    model : torch.nn.Module
    parameter_names: List[str]
        list of names of the modules (as in `model.named_modules()`) that define the subnetwork;
        the modules cannot have children, i.e. need to be leaf modules
    """

    def __init__(self, model: nn.Module, module_names: list[str]):
        super().__init__(model)
        self._module_names: list[str] = module_names
        self._n_params_subnet: int | None = None

    def _check_module_names(self) -> None:
        module_names = deepcopy(self._module_names)
        if len(module_names) == 0:
            raise ValueError("Module name list cannot be empty.")

        for name, module in self.model.named_modules():
            if name in module_names:
                if len(list(module.children())) > 0:
                    raise ValueError(
                        f'Module "{name}" has children, which is not supported.'
                    )
                elif len(list(module.parameters())) == 0:
                    raise ValueError(f'Module "{name}" does not have any parameters.')
                else:
                    module_names.remove(name)
        if len(module_names) > 0:
            raise ValueError(f"Modules {module_names} do not exist in model.")

    def get_subnet_mask(self, train_loader: DataLoader) -> torch.Tensor:
        """Get the subnetwork mask identifying the specified modules."""

        self._check_module_names()

        subnet_mask_list = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0 or len(list(module.parameters())) == 0:
                continue
            if name in self._module_names:
                mask_method = torch.ones_like
            else:
                mask_method = torch.zeros_like
            subnet_mask_list.append(
                mask_method(parameters_to_vector(module.parameters()))
            )
        subnet_mask = torch.cat(subnet_mask_list).bool()
        return subnet_mask


class LastLayerSubnetMask(ModuleNameSubnetMask):
    """Subnetwork mask corresponding to the last layer of the neural network.

    Parameters
    ----------
    model : torch.nn.Module
    last_layer_name: str, default=None
        name of the model's last layer, if None it will be determined automatically
    """

    def __init__(self, model: nn.Module, last_layer_name: str | None = None):
        super().__init__(model, [])
        self._feature_extractor: FeatureExtractor = FeatureExtractor(
            self.model, last_layer_name=last_layer_name
        )
        self._n_params_subnet: int | None = None

    def get_subnet_mask(self, train_loader: DataLoader) -> torch.Tensor:
        """Get the subnetwork mask identifying the last layer."""
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")

        self._feature_extractor.eval()
        if self._feature_extractor.last_layer is None:
            X = next(iter(train_loader))[0]
            with torch.no_grad():
                self._feature_extractor.find_last_layer(X[:1].to(self._device))
        self._module_names = [self._feature_extractor._last_layer_name]

        return super().get_subnet_mask(train_loader)
