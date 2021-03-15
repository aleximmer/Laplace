import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.utils import parameters_per_layer


class Laplace:
    """Laplace approximation for a pytorch neural network

    Parameters
    ----------
    model : torch.nn.Module
        torch model

    likelihood : str
        'classification' or 'regression' are supported

    prior_precision : one-dimensional torch.Tensor, str, default='auto'
        prior precision of a Gaussian prior corresponding to weight decay
        'auto' determines the prior automatically during fitting

    cov_type : str
        type of covariance/precision approximation
        choices = ['full', 'kron', 'diag']

    cov_closure : method
        closure to call after the forward-pass which can modify the
        backward path and returns a Hessian approximation matching the
        cov_type.
        cov_closure: model, X, y -> log_lik, batch_hessian

    temperature : float, default=1
        posterior temperature scaling affects the posterior covariance as
        `Sigma' = temperature * Sigma`, so low temperatures lead to a more
        concentrated posterior.

    Attributes
    ----------

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, model, likelihood, prior_precision='auto', cov_type='full',
                 cov_closure=None, temperature=1.):
        # TODO: add other options for priors?
        # TODO: add automatic determination of prior precision
        # TODO: not sure about the case where we don't have the cov_closure...
        if likelihood not in ['classification', 'regression']:
            raise ValueError(f'Invalid likelihood type {likelihood}')

        self.model = model
        self.prior_precision = prior_precision
        self.likelihood = likelihood
        self.temperature = temperature
        self.cov_type = cov_type
        self.cov_closure = cov_closure
        self._fit = False
        self._device = next(model.parameters()).device

        # initialize state #
        # posterior mean/mode
        self.mean = parameters_to_vector(self.model.parameters()).detach()
        self.n_params = len(self.mean)
        self.n_layers = len(list(self.model.parameters()))
        # log likelihood
        self.log_lik = 0.
        # Hessian
        # NOTE: I think having classes Laplace, DiagLaplace, KFACLaplace
        # would be better actually. Especially, looking into the posterior_det methods etc.
        if cov_type == 'full':
            self.H = torch.zeros(self.n_params, self.n_params, device=self._device)
        elif cov_type == 'diag':
            self.H = torch.zeros(self.n_params, device=self._device)
        elif cov_type == 'kron':
            # TODO: Kron class for this?
            # @aleximmer could provide it
            raise NotImplementedError()
        else:
            raise ValueError(f'Invalid cov_type {cov_type}!')

    def fit(self, train_loader):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : iterator
            each iterate is a training batch (X, y)
        """
        if self._fit:
            raise ValueError('Already fit.')

        self.model.eval()

        # TODO: make sure we can parallelize this in the future
        # TODO: make sure we can differentiate wrt noise in Gaussian lh!
        for X, y in train_loader:
            self.model.zero_grad()
            X, y = X.to(self._device), y.to(self._device)
            log_lik_batch, H_batch = self.cov_closure(self.model, X, y)
            self.log_lik += log_lik_batch
            self.H += H_batch

        if self.cov_type == 'diag':
            self.cov_sqrt = self.posterior_precision.sqrt()
        elif self.cov_type == 'full':
            self.cov_sqrt = torch.cholesky(self.posterior_precision)
        elif self.cov_type == 'kron':
            raise NotImplementedError

        self._fit = True

    @property
    def marginal_likelihood(self):
        """Compute the Laplace approximation to the marginal likelihood.
        The resulting value is differentiable in differentiable likelihood
        and prior parameters.
        """
        if not self._fit:
            raise AttributeError('Laplace not fitted. Run fit() first.')

        return self.log_lik - 0.5 * (self.log_det_ratio + self.scatter)

    def posterior_samples(self, n_samples=100):
        """Sample from the Laplace posterior torch.Tensor (n_samples, P)"""
        samples = torch.randn(n_samples, self.P, device=self._device)
        if self.cov_type == 'diag':
            return self.mean + (samples * self.cov_sqrt)
        elif self.cov_type == 'full':
            return self.mean + (samples @ self.cov_sqrt)
        elif self.cov_type == 'kron':
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

    @property
    def posterior_precision(self):
        if self.cov_type == 'diag':
            return self.H + self.prior_precision_diag
        elif self.cov_type == 'full':
            return self.H + torch.diag(self.prior_precision_diag)
        elif self.cov_type == 'kron':
            raise NotImplementedError()

    @property
    def log_det_posterior_precision(self):
        """Computes log determinant of the posterior precision `log det P`
        """
        if self.cov_type == 'diag':
            return self.posterior_precision.log().sum()
        elif self.cov_type == 'full':
            return self.posterior_precision.logdet()
        elif self.cov_type == 'kron':
            raise NotImplementedError()

    @property
    def log_det_ratio(self):
        """Computes the log of the determinant ratios for the marginal likelihood
        `log (det P / det P_0) = log det P - log det P_0`
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

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

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        self._prior_precision = prior_precision

