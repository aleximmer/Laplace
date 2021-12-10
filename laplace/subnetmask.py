import torch
from torch.nn.utils import parameters_to_vector

from laplace.feature_extractor import FeatureExtractor

__all__ = ['SubnetMask', 'RandomSubnetMask', 'LargestMagnitudeSubnetMask', 'LastLayerSubnetMask', 'LargestVarianceDiagLaplaceSubnetMask']


class SubnetMask:
	"""Baseclass for all subnetwork masks in this library (for subnetwork Laplace).

	Parameters
	----------
	model : torch.nn.Module
	"""
	def __init__(self, model):
		self.model = model
		self.parameter_vector = parameters_to_vector(self.model.parameters()).detach()
		self._n_params = len(self.parameter_vector)
		self._device = next(self.model.parameters()).device
		self._indices = None
		self._n_params_subnet = None

	@property
	def n_params_subnet(self):
		raise NotImplementedError

	def _check_select(self):
		if self._indices is None:
			raise AttributeError('Subnetwork mask not selected. Run select() first.')

	@property
	def indices(self):
		self._check_select()
		return self._indices

	def convert_subnet_mask_to_indices(self, subnet_mask):
		"""Converts a subnetwork mask into subnetwork indices.

		Parameters
		----------
		subnet_mask : torch.Tensor
			a binary vector of size (n_params) where 1s locate the subnetwork parameters
			within the vectorized model parameters

		Returns
		-------
		subnet_mask_indices : torch.Tensor
			a vector of indices of the vectorized model parameters that define the subnetwork
		"""
		if not isinstance(subnet_mask, torch.Tensor):
			raise ValueError('Subnetwork mask needs to be torch.Tensor!')
		elif subnet_mask.type() not in ['torch.ByteTensor', 'torch.IntTensor', 'torch.LongTensor'] or\
                len(subnet_mask.shape) != 1:
				raise ValueError('Subnetwork mask needs to be 1-dimensional torch.{Byte,Int,Long}Tensor!')
		elif len(subnet_mask) != self._n_params or\
                len(subnet_mask[subnet_mask == 0]) + len(subnet_mask[subnet_mask == 1]) != self._n_params:
			raise ValueError('Subnetwork mask needs to be a binary vector of size (n_params) where 1s'\
							 'locate the subnetwork parameters within the vectorized model parameters!')

		subnet_mask_indices = subnet_mask.nonzero(as_tuple=True)[0]
		return subnet_mask_indices

	def select(self, train_loader):
		""" Select the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
		"""
		if self._indices is not None:
			raise ValueError('Subnetwork mask already selected.')

		subnet_mask = self.get_subnet_mask(train_loader)
		self._indices = self.convert_subnet_mask_to_indices(subnet_mask)

	def get_subnet_mask(self, train_loader):
		""" Get the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set

		Returns
		-------
		subnet_mask: torch.Tensor
			a binary vector of size (n_params) where 1s locate the subnetwork parameters
			within the vectorized model parameters
		"""
		raise NotImplementedError


class ScoreBasedSubnetMask(SubnetMask):
	"""Baseclass for subnetwork masks defined by selecting the top-scoring parameters according to some criterion.

	Parameters
	----------
	model : torch.nn.Module
	n_params_subnet : int
		the number of parameters in the subnetwork (i.e. the number of top-scoring parameters to select)
	"""
	def __init__(self, model, n_params_subnet):
		super().__init__(model)

		if n_params_subnet is None:
			raise ValueError(f'Need to pass number of subnetwork parameters when using subnetwork Laplace.')
		if n_params_subnet > self._n_params:
			raise ValueError(f'Subnetwork ({n_params_subnet}) cannot be larger than model ({self._n_params}).')
		self._n_params_subnet = n_params_subnet
		self._param_scores = None

	@property
	def n_params_subnet(self):
		return self._n_params_subnet

	def compute_param_scores(self, train_loader):
		raise NotImplementedError

	def _check_param_scores(self):
		if self._param_scores.shape != self.parameter_vector.shape:
			raise ValueError('Parameter scores need to be of same shape as parameter vector.')

	def get_subnet_mask(self, train_loader):
		""" Get the subnetwork mask by (descendingly) ranking parameters based on their scores."""

		if self._param_scores is None:
			self._param_scores = self.compute_param_scores(train_loader)
		self._check_param_scores()

		idx = torch.argsort(self._param_scores, descending=True)[:self._n_params_subnet]
		idx = idx.sort()[0]
		subnet_mask = torch.zeros_like(self.parameter_vector).byte()
		subnet_mask[idx] = 1
		return subnet_mask


class RandomSubnetMask(ScoreBasedSubnetMask):
	"""Subnetwork mask of parameters sampled uniformly at random."""
	def compute_param_scores(self, train_loader):
		return torch.rand_like(self.parameter_vector)


class LargestMagnitudeSubnetMask(ScoreBasedSubnetMask):
	"""Subnetwork mask identifying the parameters with the largest magnitude. """
	def compute_param_scores(self, train_loader):
		return self.parameter_vector


class LargestVarianceDiagLaplaceSubnetMask(ScoreBasedSubnetMask):
	"""Subnetwork mask identifying the parameters with the largest marginal variances
	(estimated using a diagional Laplace approximation over all model parameters).

	Parameters
	----------
	model : torch.nn.Module
	n_params_subnet : int
		the number of parameters in the subnetwork (i.e. the number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for variance estimation
	"""
	def __init__(self, model, n_params_subnet, diag_laplace_model):
		super().__init__(model, n_params_subnet)
		self.diag_laplace_model = diag_laplace_model

	def compute_param_scores(self, train_loader):
		self.diag_laplace_model.fit(train_loader)
		return self.diag_laplace_model.posterior_variance


class LastLayerSubnetMask(SubnetMask):
	"""Subnetwork mask corresponding to the last layer of the neural network.

	Parameters
	----------
	model : torch.nn.Module
    last_layer_name: str, default=None
        name of the model's last layer, if None it will be determined automatically
	"""
	def __init__(self, model, last_layer_name=None):
		super().__init__(model)
		self.model = FeatureExtractor(self.model, last_layer_name=last_layer_name)
		self._n_params_subnet = None

	@property
	def n_params_subnet(self):
		if self._n_params_subnet is None:
			self._check_select()
			self._n_params_subnet = torch.count_nonzero(self._indices).item()
		return self._n_params_subnet

	def get_subnet_mask(self, train_loader):
		""" Get the subnetwork mask identifying the last layer."""

		self.model.eval()
		if self.model.last_layer is None:
			X, _ = next(iter(train_loader))
			with torch.no_grad():
				self.model.find_last_layer(X[:1].to(self._device))

		subnet_mask_list = []
		for name, layer in self.model.model.named_modules():
			if len(list(layer.children())) > 0:
				continue
			if name == self.model._last_layer_name:
				mask_method = torch.ones_like
			else:
				mask_method = torch.zeros_like
			subnet_mask_list.append(mask_method(parameters_to_vector(layer.parameters())))
		subnet_mask = torch.cat(subnet_mask_list).byte()
		return subnet_mask
