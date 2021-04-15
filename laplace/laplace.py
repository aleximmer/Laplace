from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.utils import parameters_per_layer, invsqrt_precision
from laplace.matrix import Kron
from laplace.curvature import BackPackGGN


class Laplace(ABC):
    """Laplace approximation for a pytorch neural network.
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
                 temperature=1., backend=BackPackGGN, **backend_kwargs):
        if likelihood not in ['classification', 'regression']:
            raise ValueError(f'Invalid likelihood type {likelihood}')

        self.model = model
        # initialize state #
        # posterior mean/mode
        self.mean = parameters_to_vector(self.model.parameters()).detach()
        self.n_params = len(self.mean)
        self.n_layers = len(list(self.model.parameters()))
        self.prior_precision = prior_precision
        if sigma_noise != 1 and likelihood != 'regression':
            raise ValueError('Sigma noise != 1 only available for regression.')
        self.likelihood = likelihood
        self.sigma_noise = sigma_noise
        self.temperature = temperature
        self.backend = backend(self.model, self.likelihood, **backend_kwargs)
        self._fit = False
        self._device = next(model.parameters()).device

        # log likelihood = g(loss)
        self.loss = 0.

    @abstractmethod
    def _curv_closure(self, X, y, N):
        pass

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
    def compute_scale(self):
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

        factor = self.H_factor
        if self.likelihood == 'regression':
            # Hessian factor for Gaussian likelihood is 2x, so halve loglik
            # TODO: compute offset constant c
            c = 0
            return factor * 0.5 * self.loss + c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

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
            prev_param = parameters_to_vector(self.model.parameters())
            predictions = list()
            for sample in self.posterior_samples(n_samples):
                vector_to_parameters(sample, self.model.parameters())
                predictions.append(self.model(X.to(self._device)).detach())
            vector_to_parameters(prev_param, self.model.parameters())
            return torch.stack(predictions)

        elif pred_type == 'lin':
            raise NotImplementedError()

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

        elif len(self.prior_precision) == self.n_layers:  # per layer
            n_params_per_layer = parameters_per_layer(self.model)
            return torch.cat([prior * torch.ones(n_params, device=self._device) for prior, n_params
                              in zip(self.prior_precision, n_params_per_layer)])

        else:
            raise ValueError('Mismatch of prior and model. Diagonal, scalar, or per-layer prior.')

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


class FullLaplace(Laplace):
    # TODO: list additional attributes
    # TODO: recompute scale once prior precision or sigma noise change?
    #       do in lazy way with a flag probably.

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
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

    def samples(self, n_samples=100):
        samples = torch.randn(self.P, n_samples, device=self._device)
        return self.mean + (self.posterior_scale @ samples)


class KronLaplace(Laplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
        self.H = Kron.init_from_model(self.model, self._device)

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

    def samples(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        return self.mean + self.posterior_precision.bmm(samples, exponent=-1/2)

    @Laplace.prior_precision.setter
    def prior_precision(self, prior_precision):
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers]:
            raise ValueError('Prior precision for Kron either scalar or per-layer.')


class DiagLaplace(Laplace):
    # TODO: list additional attributes
    # TODO: caching prior_precision_diag for fast lazy computation?

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=BackPackGGN):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
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
        return self.posterior_scale.square()

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        return self.posterior_precision.log().sum()

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_variance, Js)

    def samples(self, n_samples=100):
        samples = torch.randn(n_samples, self.P, device=self._device)
        return self.mean + samples * self.posterior_scale
