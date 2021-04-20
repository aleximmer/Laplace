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
from laplace.jacobians import LLJacobians


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

    def marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        """Compute the Laplace approximation to the marginal likelihood.
        The resulting value is differentiable in differentiable likelihood
        and prior parameters.
        """
        # make sure we can differentiate wrt prior and sigma_noise for regression
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run fit() first.')

        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            if self.likelihood != 'regression':
                raise ValueError('Can only change sigma_noise for regression.')
            self.sigma_noise = sigma_noise

        return self.log_lik - 0.5 * (self.log_det_ratio + self.scatter)

    @property
    def log_lik(self):
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run fit() first.')

        factor = - self.H_factor
        if self.likelihood == 'regression':
            # loss used is just MSE, need to add normalizer for gaussian likelihood
            c = self.n_data * self.n_outputs * torch.log(self.sigma_noise * sqrt(2 * pi))
            return factor * self.loss - c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

    def __call__(self, X, pred_type='glm', link_approx='mc', n_samples=100):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        X : torch.Tensor (batch_size, *input_size)

        pred_type : str 'lin' or 'nn'
            type of posterior predictive, linearized GLM predictive or
            neural network sampling predictive.

        link_approx : str 'mc' or 'probit'
            how to approximate the classification link function for GLM.
            For NN, only 'mc' is possible.

        n_samples : int
            number of samples in case necessary
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if pred_type == 'glm':
            f_mu, f_var = self.glm_predictive_distribution(X)
            # regression
            if self.likelihood == 'regression':
                return f_mu, f_var
            # classification
            if link_approx == 'mc':
                dist = MultivariateNormal(f_mu, f_var)
                return torch.softmax(dist.sample((n_samples,)), dim=-1).mean(dim=0)
            elif link_approx == 'probit':
                kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
                return torch.softmax(kappa * f_mu, dim=-1)
        else:
            samples = self.nn_predictive_samples(X, n_samples)
            if self.likelihood == 'regression':
                return samples.mean(dim=0), samples.var(dim=0)
            return samples.mean(dim=0)

    def predictive(self, X, pred_type='glm', link_approx='mc', n_samples=100):
        return self(X, pred_type, link_approx, n_samples)

    def predictive_samples(self, X, pred_type='glm', n_samples=100):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        X : torch.Tensor (batch_size, *input_size)

        pred_type : str 'lin' or 'nn'
            type of posterior predictive, linearized GLM predictive or
            neural network sampling predictive.

        n_samples : int
            number of samples
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if pred_type == 'glm':
            f_mu, f_var = self.glm_predictive_distribution(X)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            dist = MultivariateNormal(f_mu, f_var)
            samples = dist.sample((n_samples,))
            if self.likelihood == 'regression':
                return samples
            return torch.softmax(samples, dim=-1)

        else:  # 'nn'
            return self.nn_predictive_samples(X, n_samples)

    def glm_predictive_distribution(self, X):
        Js, f_mu = LLJacobians(self.model, X)
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

    @abstractmethod
    def sample(self, n_samples=100):
        """Sample from the Laplace posterior torch.Tensor (n_samples, P)"""
        pass

    @property
    def scatter(self):
        """Computes the scatter used for the marginal likelihood `m^T P_0 m`
        """
        return (self.mean * self.prior_precision_diag) @ self.mean

    @property
    def log_det_prior_precision(self):
        """Computes log determinant of the prior precision `log det P_0`
        """
        return self.prior_precision_diag.log().sum()

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

    def _check_jacobians(self, Js):
        if not isinstance(Js, torch.Tensor):
            raise ValueError('Jacobians have to be torch.Tensor.')
        if not Js.device == self._device:
            raise ValueError('Jacobians need to be on the same device as Laplace.')
        m, k, p = Js.size()
        if p != self.n_params:
            raise ValueError('Invalid Jacobians shape for Laplace posterior approx.')

    @property
    def log_det_ratio(self):
        """Computes the log of the determinant ratios for the marginal likelihood
        `log (det P / det P_0) = log det P - log det P_0`
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        raise NotImplementedError()

    @property
    def prior_precision_diag(self):
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones_like(self.mean)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision

        else:
            raise ValueError('Mismatch of prior and model. Diagonal or scalar prior.')

    # TODO: protect prior precision and sigma updates when covariance computed/update covariance?
    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision])
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1)
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError('Length of prior precision does not align with architecture.')
                self._prior_precision = prior_precision
            else:
                raise ValueError('Prior precision needs to be at most one-dimensional tensor.')
        else:
            raise ValueError('Prior precision either scalar or torch.Tensor up to 1-dim.')

    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.tensor(sigma_noise)
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim == 0:
                self._sigma_noise = sigma_noise
            elif sigma_noise.ndim == 1:
                if len(sigma_noise) > 1:
                    raise ValueError('Only homoscedastic output noise supported.')
                self._sigma_noise = sigma_noise
            else:
                raise ValueError('Sigma noise needs to be scalar or 1-dimensional.')
        else:
            raise ValueError('Invalid type: sigma noise needs to be torch.Tensor or scalar.')

    @property
    def H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 * self.temperature


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
        return self.llbackend.kron(X, y, N=N)

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