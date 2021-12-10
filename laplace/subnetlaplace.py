import torch

from laplace.baselaplace import FullLaplace, DiagLaplace

from laplace.curvature import BackPackGGN
from laplace.subnetmask import LargestVarianceDiagLaplaceSubnetMask


__all__ = ['SubnetLaplace']


class SubnetLaplace(FullLaplace):
    """Class for subnetwork Laplace, which computes the Laplace approximation over
    just a subset of the model parameters (i.e. a subnetwork within the neural network).
    Subnetwork Laplace only supports a full Hessian approximation; other Hessian
    approximations could be used in theory, but would not make as much sense conceptually.

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

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
    likelihood : {'classification', 'regression'}
        determines the log likelihood Hessian approximation
    subnetwork_mask : subclasses of `laplace.subnetmask.SubnetMask`, default=None
        mask defining the subnetwork to apply the Laplace approximation over
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
    subnetmask_kwargs : dict, default=None
        arguments passed to the subnetwork mask on initialization.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('subnetwork', 'full')

    def __init__(self, model, likelihood, subnetwork_mask=None, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None, subnetmask_kwargs=None):
        super().__init__(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_precision,
                         prior_mean=prior_mean, temperature=temperature, backend=backend,
                         backend_kwargs=backend_kwargs)
        self._subnetmask_kwargs = dict() if subnetmask_kwargs is None else subnetmask_kwargs
        if subnetwork_mask == LargestVarianceDiagLaplaceSubnetMask:
            # instantiate and pass diagonal Laplace model for largest variance subnetwork selection
            self._subnetmask_kwargs.update(diag_laplace_model=DiagLaplace(self.model, likelihood, sigma_noise,
                prior_precision, prior_mean, temperature, backend, backend_kwargs))
        self.subnetwork_mask = subnetwork_mask(self.model, **self._subnetmask_kwargs)
        self.n_params_subnet = None

    def _init_H(self):
        self.H = torch.zeros(self.n_params_subnet, self.n_params_subnet, device=self._device)

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

    def fit(self, train_loader):
        """Fit the local Laplace approximation at the parameters of the subnetwork.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        """

        # select subnetwork and pass it to backend
        self.subnetwork_mask.select(train_loader)
        self.backend.subnetwork_indices = self.subnetwork_mask.indices
        self.n_params_subnet = self.subnetwork_mask.n_params_subnet

        # fit Laplace approximation over subnetwork
        super().fit(train_loader)
