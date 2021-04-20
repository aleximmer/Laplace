from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.Laplace import Laplace, FullLaplace, KronLaplace, DiagLaplace
from laplace.feature_extractor import FeatureExtractor
from laplace.utils import parameters_per_layer, invsqrt_precision
from laplace.matrix import Kron
from laplace.curvature import LastLayer, BackPackGGN


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
            self.llbackend = LastLayer(self.model, self.likelihood, backend, **backend_kwargs)
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
            self.llbackend = LastLayer(self.model, self.likelihood, self._backend,
                                       **self._backend_kwargs)

        N = len(train_loader.dataset)
        for X, y in train_loader:
            self.model.zero_grad()
            X, y = X.to(self._device), y.to(self._device)
            loss_batch, H_batch = self._curv_closure(X, y, N)
            self.loss += loss_batch
            self.H += H_batch

        self._fit = True
        # compute optimal representation of posterior Cov/Prec.
        if compute_scale:
            self.compute_scale()

    @abstractmethod
    def _init_H(self):
        pass

    @abstractmethod
    def samples(self, n_samples=100):
        """Sample from the Laplace posterior torch.Tensor (n_samples, P)"""
        pass

    def predictive(self, X, n_samples=100, pred_type='lin'):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        X : torch.Tensor (batch_size, *input_size)

        n_samples : int
            number of samples in case necessary

        pred_type : str 'lin' or 'nn'
            type of posterior predictive, linearized GLM predictive or
            neural network sampling predictive.
        """
        pass

    def predictive_samples(self, X, n_samples, pred_type='lin'):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        X : torch.Tensor (batch_size, *input_size)

        n_samples : int
            number of samples

        pred_type : str 'lin' or 'nn'
            type of posterior predictive, linearized GLM predictive or
            neural network sampling predictive.
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        if pred_type not in ['lin', 'nn']:
            raise ValueError('Invalid pred_type parameter.')

        if pred_type == 'nn':
            prev_param = parameters_to_vector(self.model.last_layer.parameters())
            predictions = list()
            for sample in self.posterior_samples(n_samples):
                vector_to_parameters(sample, self.model.last_layer.parameters())
                predictions.append(self.model(X.to(self._device)).detach())
            vector_to_parameters(prev_param, self.model.last_layer.parameters())
            return torch.stack(predictions)

        elif pred_type == 'lin':
            raise NotImplementedError()

    @abstractmethod
    def functional_variance(self, Jacs):
        """Compute functional variance for the predictive:
        `f_var[i] = Jacs[i] @ Sigma @ Jacs[i].T`, which is a output x output
        predictive covariance matrix.

        Parameters
        ----------
        Jacs : torch.Tensor batch x outputs x parameters
            Jacobians of model output wrt parameters.
        """
        pass

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        raise NotImplementedError()


class FullLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.llbackend.full(X, y, N=N)

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

    def samples(self, n_samples=100):
        samples = torch.randn(self.P, n_samples, device=self._device)
        return self.mean + (self.posterior_scale @ samples)


class KronLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = Kron.init_from_model(self.model.last_layer, self._device)

    def _curv_closure(self, X, y, N):
        return self.llbackend.kron(X, y, N=N)

    def fit(self, train_loader):
        # Kron requires postprocessing as all quantities depend on the decomposition.
        super().fit(train_loader, compute_scale=True)

    def compute_scale(self):
        pass
        # self.posterior_scale = self.posterior_precision.invsqrt()

    @property
    def posterior_covariance(self):
        return self.posterior_scale.square()

    @property
    def posterior_precision(self):
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H * self.H_factor + self.prior_precision_kron

    @Laplace.prior_precision.setter
    def prior_precision(self, prior_precision):
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLLLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers]:
            raise ValueError('Prior precision for Kron either scalar or per-layer.')

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        n, c, p = Js.shape
        fvar = torch.zeros(n, c, c)
        ix = 0
        for Si in self.posterior_covariance.blocks:
            P = Si.size(0)
            Jsi = Js[:, :, ix:ix+P]
            fvar += torch.einsum('ncp,pq,nkq->nck', Jsi, Si, Jsi)
            ix += P
        return fvar

    def samples(self, n_samples=100):
        samples = torch.randn(self.P, n_samples, device=self._device)
        samples_list = list()
        ix = 0
        for Si in self.posterior_scale.blocks:
            P = Si.size(0)
            samples_i = samples[ix:ix+P]
            samples_list.append(Si @ samples_i)
        return self.mean + torch.cat(samples_list, dim=0)

    @property
    def prior_precision_kron(self):
        return Kron.init_from_model(self.model.last_layer, self._device)


class DiagLLLaplace(LLLaplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN, last_layer_name=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend, last_layer_name)

    def _init_H(self):
        self.H = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.llbackend.diag(X, y, N=N)

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
        return self.posterior_scale.square()

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        return self.posterior_precision.log().sum()

    def functional_variance(self, Js):
        return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_variance, Js)

    def samples(self, n_samples=100):
        samples = torch.randn(n_samples, self.P, device=self._device)
        return self.mean + samples * self.posterior_scale
