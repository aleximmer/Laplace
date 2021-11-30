import torch

from laplace.baselaplace import ParametricLaplace, FullLaplace

from laplace.curvature import BackPackGGN


__all__ = ['FullSubnetLaplace']


class SubnetLaplace(ParametricLaplace):
    """Baseclass for all subnetwork Laplace approximations in this library.
    Subclasses specify the structure of the Hessian approximation.
    See `BaseLaplace` for the full interface.

    A Laplace approximation is represented by a MAP which is given by the
    `model` parameter and a posterior precision or covariance specifying
    a Gaussian distribution \\(\\mathcal{N}(\\theta_{MAP}, P^{-1})\\).
    Here, only the parameters of a subnetwork of the neural network
    are treated probabilistically.
    The goal of this class is to compute the posterior precision \\(P\\)
    which sums as
    \\[
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    \\]
    There is one subclass, which implements the only supported option of a full
    approximation to the log likelihood Hessian. The prior is assumed to be Gaussian and
    therefore we have a simple form for
    \\(\\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}} = P_0 \\).
    In particular, we assume a scalar or diagonal prior precision so that in
    all cases \\(P_0 = \\textrm{diag}(p_0)\\) and the structure of \\(p_0\\) can be varied.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
    likelihood : {'classification', 'regression'}
        determines the log likelihood Hessian approximation
    subnetwork_mask : torch.Tensor, default=None
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
    """
    def __init__(self, model, likelihood, subnetwork_mask=None, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        super().__init__(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_precision,
                         prior_mean=prior_mean, temperature=temperature, backend=backend,
                         backend_kwargs=backend_kwargs)
        self.subnetwork_mask = subnetwork_mask
        self.n_params_subnet = len(self.subnetwork_mask)

    @property
    def subnetwork_mask(self):
        return self._subnetwork_mask

    @subnetwork_mask.setter
    def subnetwork_mask(self, subnetwork_mask):
        """Check validity of subnetwork mask and convert it to a vector of indices of the vectorized
        model parameters that define the subnetwork to apply the Laplace approximation over.
        """
        if isinstance(subnetwork_mask, torch.Tensor):
            if subnetwork_mask.type() not in ['torch.ByteTensor', 'torch.IntTensor', 'torch.LongTensor'] or\
                len(subnetwork_mask.shape) != 1:
                raise ValueError('Subnetwork mask needs to be 1-dimensional torch.{Byte,Int,Long}Tensor!')

            elif len(subnetwork_mask) == self.n_params and\
                len(subnetwork_mask[subnetwork_mask == 0]) +\
                    len(subnetwork_mask[subnetwork_mask == 1]) == self.n_params:
                self._subnetwork_mask = subnetwork_mask.nonzero(as_tuple=True)[0]

            elif len(subnetwork_mask) <= self.n_params and\
                len(subnetwork_mask[subnetwork_mask >= self.n_params]) == 0:
                self._subnetwork_mask = subnetwork_mask

            else:
                raise ValueError('Subnetwork mask needs to identify the subnetwork parameters '\
                    'from the vectorized model parameters as:\n'\
                    '1) a vector of indices of the subnetwork parameters, or\n'\
                    '2) a binary vector of size (parameters) where 1s locate the subnetwork parameters.')

        elif subnetwork_mask is None:
            raise ValueError('Subnetwork Laplace requires passing a subnetwork mask!')

        else:
            raise ValueError('Subnetwork mask needs to be torch.Tensor!')

        self.backend.subnetwork_indices = self._subnetwork_mask

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


class FullSubnetLaplace(SubnetLaplace, FullLaplace):
    """Subnetwork Laplace approximation with full, i.e., dense, log likelihood Hessian approximation
    and hence posterior precision. Based on the chosen `backend` parameter, the full
    approximation can be, for example, a generalized Gauss-Newton matrix.
    Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `FullLaplace`, `LLLaplace`, and `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('subnetwork', 'full')

    def __init__(self, model, likelihood, subnetwork_mask=None, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        super().__init__(model, likelihood, subnetwork_mask, sigma_noise, prior_precision,
                         prior_mean, temperature, backend,  backend_kwargs)

    def _init_H(self):
        self.H = torch.zeros(self.n_params_subnet, self.n_params_subnet, device=self._device)
