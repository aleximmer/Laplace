from abc import ABC, abstractmethod
from math import sqrt, pi
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import MultivariateNormal

from laplace.laplace import Laplace, FullLaplace, KronLaplace, DiagLaplace
from laplace.feature_extractor import FeatureExtractor
from laplace.utils import parameters_per_layer, invsqrt_precision
from laplace.matrix import Kron
from laplace.curvature import BackPackGGN
from laplace.jacobians import last_layer_jacobians


__all__ = ['FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace']


class LLLaplace(Laplace):
    """Last-Layer Laplace approximation for a pytorch neural network.
    The Laplace approximation is a Gaussian distribution but can have different
    sparsity structures. Further, it provides an approximation to the marginal
    likelihood.

    Parameters
    ----------
    model : torch.nn.Module
        torch model

    likelihood : str
        'classification' or 'regression' are supported

    sigma_noise : float
        observation noise for likelihood = 'regression'

    prior_precision : one-dimensional torch.Tensor, str, default='auto'
        prior precision of a Gaussian prior corresponding to weight decay
        'auto' determines the prior automatically during fitting

    temperature : float, default=1
        posterior temperature scaling affects the posterior covariance as
        `Sigma' = temperature * Sigma`, so low temperatures lead to a more
        concentrated posterior.

    backend : CurvatureInterface
        provides access to curvature/second-order quantities.

    Attributes
    ----------

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None,
                 **backend_kwargs):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
        self.model = FeatureExtractor(model, last_layer_name=last_layer_name)
        self.n_layers = 1
        if self.model._found:
            self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(self.mean)
            self._init_H()
            self.backend = backend(self.model, self.likelihood, last_layer=True,
                                   **backend_kwargs)
        else:
            self._backend = backend
            self._backend_kwargs = backend_kwargs

    def fit(self, train_loader, compute_scale=True):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : iterator
            each iterate is a training batch (X, y)
        """
        if self._fit:
            raise ValueError('Already fit.')

        self.model.eval()

        if not self.model._found:
            self.model.find_last_layer(train_loader.dataset[0][0])
            self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(self.mean)
            self._init_H()
            self.backend = self._backend(self.model, self.likelihood, last_layer=True,
                                         **self._backend_kwargs)

        N = len(train_loader.dataset)
        for X, y in train_loader:
            self.model.zero_grad()
            X, y = X.to(self._device), y.to(self._device)
            loss_batch, H_batch = self._curv_closure(X, y, N)
            self.loss += loss_batch
            self.H += H_batch

        with torch.no_grad():
            self.n_outputs = self.model(X[:1]).shape[-1]
        self.n_data = N

        self._fit = True
        # compute optimal representation of posterior Cov/Prec.
        if compute_scale:
            self.compute_scale()

    @abstractmethod
    def _init_H(self):
        pass

    def glm_predictive_distribution(self, X):
        Js, f_mu = last_layer_jacobians(self.model, X)
        f_var = self.functional_variance(Js)
        return f_mu.detach(), f_var.detach()

    def nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        for sample in self.sample(n_samples):
            vector_to_parameters(sample, self.model.last_layer.parameters())
            fs.append(self.model(X.to(self._device)).detach())
        vector_to_parameters(self.mean, self.model.last_layer.parameters())
        fs = torch.stack(fs)
        if self.likelihood == 'classification':
            fs = torch.softmax(fs, dim=-1)
        return fs

    @property
    def prior_precision_diag(self):
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones_like(self.mean)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision

        else:
            raise ValueError('Mismatch of prior and model. Diagonal or scalar prior.')


class FullLLLaplace(LLLaplace, FullLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, self.n_params, device=self._device)


class KronLLLaplace(LLLaplace, KronLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = Kron.init_from_model(self.model.last_layer, self._device)


class DiagLLLaplace(LLLaplace, DiagLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, device=self._device)
