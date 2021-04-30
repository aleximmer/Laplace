import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector


class ClassificationLogJoint(CrossEntropyLoss):

    def __init__(self, model, n_datapoints):
        self.model = model
        self.N = n_datapoints
        self.n_params = len(parameters_to_vector(model.parameters()))
        self.n_layers = len(list(self.model.parameters()))
        super().__init__(reduction='sum')

    def _compute_log_prior(self, prior_precision):
        # Not exact log N, ignore irrelevant parameters for objective on model
        prior_precision = prior_precision.detach()
        assert isinstance(prior_precision, torch.Tensor)
        if prior_precision.ndim == 0:
            prior_precision = prior_precision.reshape(-1)
        if len(prior_precision) == 1:
            theta = parameters_to_vector(self.model.parameters())
            return 0.5 * (prior_precision * theta) @ theta
        elif len(prior_precision) == self.n_layers:
            log_prior = 0
            for p, prior_prec_p in zip(self.model.parameters(), prior_precision):
                log_prior += 0.5 * (prior_prec_p * p) @ p
            return log_prior
        elif len(prior_precision) == self.n_params:
            theta = parameters_to_vector(self.model.parameters())
            return 0.5 * (prior_precision * theta) @ theta

    def forward(self, input, target, prior_precision):
        M = len(target)
        f = self.model(input)
        log_lik = self.N / M * super().forward(f, target)
        log_prior = self._compute_log_prior(prior_precision)
        return log_lik + log_prior
