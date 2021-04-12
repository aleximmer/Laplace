from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.utils import parameters_per_layer, invsqrt_precision
from laplace.matrix import BlockDiag, Kron


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
                 temperature=1., backend=None):
        # TODO: add automatic determination of prior precision
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
            raise ValueError('Sigma noise only available for regression.')
        self.likelihood = likelihood
        self.sigma_noise = sigma_noise
        self.temperature = temperature
        # self.backend = backend(self.model)
        self._fit = False
        self._device = next(model.parameters()).device

        # log likelihood = g(loss)
        self.loss = 0.

    @abstractmethod
    def _curv_closure(self, X, y, N):
        pass

    def fit(self, train_loader, compute_covariance=True):
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

        # invert posterior precision (not always necessary for marglik e.g.)
        if compute_covariance:
            self.compute_covariance()

        self._fit = True

    @abstractmethod
    def compute_covariance(self):
        raise NotImplementedError()

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

    def samples(self, n_samples=100):
        """Sample from the Laplace posterior torch.Tensor (n_samples, P)"""
        raise NotImplementedError()
    
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
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        raise NotImplementedError()

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

    def functional_variance(self, Jacs):
        """Compute functional variance for the predictive:
        `f_var[i] = Jacs[i] @ Sigma @ Jacs[i].T`, which is a output x output
        predictive covariance matrix.

        Parameters
        ----------
        Jacs : torch.Tensor batch x outputs x parameters
            Jacobians of model output wrt parameters.
        """
        raise NotImplementedError()

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
            self._sigma_noise = sigma_noise
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
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
        self.H = torch.zeros(self.n_params, self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.full(X, y, N)

    def compute_covariance(self):
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
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)

    def samples(self, n_samples=100):
        samples = torch.randn(self.P, n_samples, device=self._device)
        return self.mean + (self.posterior_scale @ samples)


class BlockLaplace(Laplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
        self.H = BlockDiag.init_from_model(self.model, self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.block(X, y, N)

    def compute_covariance(self):
        self.posterior_scale = self.posterior_precision.invsqrt()

    @property
    def posterior_covariance(self):
        return self.posterior_scale.square() 

    @property
    def posterior_precision(self):
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H * self.H_factor + self.prior_precision_block_diag

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
    def prior_precision_block_diag(self):
        prior_prec_diag = self.prior_precision_diag
        n_params_per_param = [np.prod(p.shape) for p in self.model.parameters()]
        ix = 0
        blocks = list()
        for p in n_params_per_param:
            blocks.append(torch.diag(prior_prec_diag[ix:ix+p]))
            ix += p
        return BlockDiag(blocks)


class KronLaplace(Laplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)

    def _curv_closure(self, X, y, N):
        return self.backend.kron(X, y, N)

    def compute_covariance(self):
        pass

    @property
    def posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')
        
        raise NotImplementedError

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        raise NotImplementedError()

    def samples(self, n_samples=100):
        raise NotImplementedError()


class DiagLaplace(Laplace):
    # TODO list additional attributes

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 temperature=1., backend=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         temperature, backend)
        self.H = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.diag(X, y, N)

    @property
    def posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run Laplace.fit() first')

        return self.H_factor * self.H + self.prior_precision_diag

    def compute_covariance(self):
        self.posterior_scale = 1 / self.posterior_precision.sqrt()

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

