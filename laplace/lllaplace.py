import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.baselaplace import BaseLaplace, FullLaplace, KronLaplace, DiagLaplace
from laplace.feature_extractor import FeatureExtractor

from laplace.matrix import Kron
from laplace.curvature import BackPackGGN


__all__ = ['FullLLLaplace', 'KronLLLaplace', 'DiagLLLaplace']


class LLLaplace(BaseLaplace):
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
                 prior_mean=0., temperature=1., backend=BackPackGGN, last_layer_name=None,
                 **backend_kwargs):
        super().__init__(model, likelihood, sigma_noise=sigma_noise, prior_precision=1.,
                         prior_mean=0., temperature=temperature, backend=backend, **backend_kwargs)
        self.model = FeatureExtractor(model, last_layer_name=last_layer_name)
        if self.model._found:
            self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(self.mean)
            self.n_layers = len(list(self.model.last_layer.parameters()))
            self.prior_precision = prior_precision
            self.prior_mean = prior_mean
        else:
            self.mean = None
            self.n_params = None
            self.n_layers = None
            # ignore checks of prior mean setter temporarily, check on .fit()
            self._prior_precision = prior_precision
            self._prior_mean = prior_mean
        self._backend_kwargs['last_layer'] = True

    def fit(self, train_loader):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : iterator
            each iterate is a training batch (X, y)
        """
        if self.H is not None:
            raise ValueError('Already fit.')

        self.model.eval()

        if not self.model._found:
            X, _ = next(iter(train_loader))
            with torch.no_grad():
                self.model.find_last_layer(X[:1].to(self._device))
            self.mean = parameters_to_vector(self.model.last_layer.parameters()).detach()
            self.n_params = len(self.mean)
            self.n_layers = len(list(self.model.last_layer.parameters()))
            # here, check the already set prior precision again
            self.prior_precision = self._prior_precision
            self.prior_mean = self._prior_mean

        super().fit(train_loader)

    def glm_predictive_distribution(self, X):
        Js, f_mu = self.backend.last_layer_jacobians(self.model, X)
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

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    key = ('last_layer', 'full')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, last_layer_name=None,
                 **backend_kwargs):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, last_layer_name, **backend_kwargs)


class KronLLLaplace(LLLaplace, KronLaplace):
    # TODO list additional attributes

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    key = ('last_layer', 'kron')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, last_layer_name=None,
                 **backend_kwargs):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, last_layer_name, **backend_kwargs)

    def _init_H(self):
        self.H = Kron.init_from_model(self.model.last_layer, self._device)


class DiagLLLaplace(LLLaplace, DiagLaplace):
    # TODO list additional attributes

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    key = ('last_layer', 'diag')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, last_layer_name=None,
                 **backend_kwargs):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, last_layer_name, **backend_kwargs)
