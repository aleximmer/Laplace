import torch
from torch.distributions import MultivariateNormal

from laplace.baselaplace import ParametricLaplace, FullLaplace, DiagLaplace


__all__ = ['SubnetLaplace', 'FullSubnetLaplace', 'DiagSubnetLaplace']


class SubnetLaplace(ParametricLaplace):
    """Class for subnetwork Laplace, which computes the Laplace approximation over just a subset
    of the model parameters (i.e. a subnetwork within the neural network), as proposed in [1].
    Subnetwork Laplace can only be used with either a full or a diagonal Hessian approximation.

    A Laplace approximation is represented by a MAP which is given by the
    `model` parameter and a posterior precision or covariance specifying
    a Gaussian distribution \\(\\mathcal{N}(\\theta_{MAP}, P^{-1})\\).
    Here, only a subset of the model parameters (i.e. a subnetwork of the
    neural network) are treated probabilistically.
    The goal of this class is to compute the posterior precision \\(P\\)
    which sums as
    \\[
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    \\]
    The prior is assumed to be Gaussian and therefore we have a simple form for
    \\(\\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}} = P_0 \\).
    In particular, we assume a scalar or diagonal prior precision so that in
    all cases \\(P_0 = \\textrm{diag}(p_0)\\) and the structure of \\(p_0\\) can be varied.

    The subnetwork Laplace approximation only supports a full, i.e., dense, log likelihood
    Hessian approximation and hence posterior precision.  Based on the chosen `backend`
    parameter, the full approximation can be, for example, a generalized Gauss-Newton
    matrix.  Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `FullLaplace` and `BaseLaplace` for the full interface.

    References
    ----------
    [1] Daxberger, E., Nalisnick, E., Allingham, JU., Antorán, J., Hernández-Lobato, JM.
    [*Bayesian Deep Learning via Subnetwork Inference*](https://arxiv.org/abs/2010.14689). 
    ICML 2021.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
    likelihood : {'classification', 'regression'}
        determines the log likelihood Hessian approximation
    subnetwork_indices : torch.LongTensor
        indices of the vectorized model parameters
        (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
        that define the subnetwork to apply the Laplace approximation over
    sigma_noise : torch.Tensor or float, default=1
        observation noise for the regression setting; must be 1 for classification
    prior_precision : torch.Tensor or float, default=1
        prior precision of a Gaussian prior (= weight decay);
        can be scalar, per-layer, or diagonal in the most general case
    prior_mean : torch.Tensor or float, default=0
        prior mean of a Gaussian prior, useful for continual learning
    temperature : float, default=1
        temperature of the likelihood; lower temperature leads to more
        concentrated posterior and vice versa.
    backend : subclasses of `laplace.curvature.CurvatureInterface`
        backend for access to curvature/Hessian approximations
    backend_kwargs : dict, default=None
        arguments passed to the backend on initialization, for example to
        set the number of MC samples for stochastic approximations.
    """
    def __init__(self, model, likelihood, subnetwork_indices, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=None, backend_kwargs=None):
        self.H = None
        super().__init__(model, likelihood, sigma_noise=sigma_noise,
                         prior_precision=prior_precision, prior_mean=prior_mean,
                         temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
        # check validity of subnetwork indices and pass them to backend
        self._check_subnetwork_indices(subnetwork_indices)
        self.backend.subnetwork_indices = subnetwork_indices
        self.n_params_subnet = len(subnetwork_indices)
        self._init_H()

    def _check_subnetwork_indices(self, subnetwork_indices):
        """Check that subnetwork indices are valid indices of the vectorized model parameters
           (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`).
        """
        if subnetwork_indices is None:
            raise ValueError('Subnetwork indices cannot be None.')
        elif not ((isinstance(subnetwork_indices, torch.LongTensor) or
                   isinstance(subnetwork_indices, torch.cuda.LongTensor)) and
                   subnetwork_indices.numel() > 0 and len(subnetwork_indices.shape) == 1):
            raise ValueError('Subnetwork indices must be non-empty 1-dimensional torch.LongTensor.')
        elif not (len(subnetwork_indices[subnetwork_indices < 0]) == 0 and
            len(subnetwork_indices[subnetwork_indices >= self.n_params]) == 0):
            raise ValueError(f'Subnetwork indices must lie between 0 and n_params={self.n_params}.')
        elif not (len(subnetwork_indices.unique()) == len(subnetwork_indices)):
            raise ValueError('Subnetwork indices must not contain duplicate entries.')

    @property
    def prior_precision_diag(self):
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params_subnet, device=self._device)

        elif len(self.prior_precision) == self.n_params_subnet:  # diagonal
            return self.prior_precision

        else:
            raise ValueError('Mismatch of prior and model. Diagonal or scalar prior.')
    
    @property
    def mean_subnet(self):
        return self.mean[self.backend.subnetwork_indices]

    @property
    def scatter(self):
        delta = (self.mean_subnet - self.prior_mean)
        return (delta * self.prior_precision_diag) @ delta

    def assemble_full_samples(self, subnet_samples):
        full_samples = self.mean.repeat(subnet_samples.shape[0], 1)
        full_samples[:, self.backend.subnetwork_indices] = subnet_samples
        return full_samples


class FullSubnetLaplace(SubnetLaplace, FullLaplace):
    """Subnetwork Laplace approximation with full, i.e., dense, log likelihood Hessian
    approximation and hence posterior precision. Based on the chosen `backend` parameter,
    the full approximation can be, for example, a generalized Gauss-Newton matrix.
    Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `FullLaplace`, `SubnetLaplace`, and `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('subnetwork', 'full')

    def _init_H(self):
        self.H = torch.zeros(self.n_params_subnet, self.n_params_subnet, device=self._device)

    def sample(self, n_samples=100):
        # sample only subnetwork parameters and set all other parameters to their MAP estimates
        dist = MultivariateNormal(loc=self.mean_subnet, scale_tril=self.posterior_scale)
        subnet_samples = dist.sample((n_samples,))
        return self.assemble_full_samples(subnet_samples)


class DiagSubnetLaplace(SubnetLaplace, DiagLaplace):
    """Subnetwork Laplace approximation with diagonal log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have \\(P \\approx \\textrm{diag}(P)\\).
    See `DiagLaplace`, `SubnetLaplace`, and `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('subnetwork', 'diag')

    def _init_H(self):
        self.H = torch.zeros(self.n_params_subnet, device=self._device)

    def _check_jacobians(self, Js):
        if not isinstance(Js, torch.Tensor):
            raise ValueError('Jacobians have to be torch.Tensor.')
        if not Js.device == self._device:
            raise ValueError('Jacobians need to be on the same device as Laplace.')
        m, k, p = Js.size()
        if p != self.n_params_subnet:
            raise ValueError('Invalid Jacobians shape for Laplace posterior approx.')

    def sample(self, n_samples=100):
        # sample only subnetwork parameters and set all other parameters to their MAP estimates
        samples = torch.randn(n_samples, self.n_params_subnet, device=self._device)
        samples = samples * self.posterior_scale.reshape(1, self.n_params_subnet)
        subnet_samples = self.mean_subnet.reshape(1, self.n_params_subnet) + samples
        return self.assemble_full_samples(subnet_samples)
