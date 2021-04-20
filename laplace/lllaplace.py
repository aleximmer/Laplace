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
from laplace.curvature import LastLayer, BackPackGGN
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
            self.backend = LastLayer(self.model, self.likelihood, backend, **backend_kwargs)
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
            self.backend = LastLayer(self.model, self.likelihood, self._backend,
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


class FullLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.full(X, y, N=N)

    def compute_scale(self):
        self.posterior_scale = invsqrt_precision(self.posterior_precision)

    @property
    def posterior_covariance(self):
        return self.posterior_scale @ self.posterior_scale.T

    @property
    def posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        # TODO: could make more efficient for scalar prior precision.
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)

    def sample(self, n_samples=100):
        dist = MultivariateNormal(loc=self.mean, scale_tril=self.posterior_scale)
        return dist.sample((n_samples,))


class KronLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = Kron.init_from_model(self.model.last_layer, self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.kron(X, y, N=N)

    def fit(self, train_loader):
        # Kron requires postprocessing as all quantities depend on the decomposition.
        super().fit(train_loader, compute_scale=True)

    def compute_scale(self):
        self.H = self.H.decompose()

    @property
    def posterior_precision(self):
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H * self.H_factor + self.prior_precision

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        return self.posterior_precision.inv_square_form(Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.mean.reshape(1, self.n_params) + samples.reshape(n_samples, self.n_params)

    @Laplace.prior_precision.setter
    def prior_precision(self, prior_precision):
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLLLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers]:
            raise ValueError('Prior precision for Kron either scalar or per-layer.')


class DiagLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.diag(X, y, N=N)

    @property
    def posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H_factor * self.H + self.prior_precision_diag

    def compute_scale(self):
        # For diagonal this is implemented lazily since computing is for free.
        pass

    @property
    def posterior_scale(self):
        return 1 / self.posterior_precision.sqrt()

    @property
    def posterior_variance(self):
        return 1 / self.posterior_precision

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        return self.posterior_precision.log().sum()

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_variance, Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples
