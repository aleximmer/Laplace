from abc import ABC, abstractmethod
import torch
from torch.nn import MSELoss, CrossEntropyLoss


class CurvatureInterface(ABC):

    def __init__(self, model, likelihood):
        assert likelihood in ['regression', 'classification']
        self.likelihood = likelihood
        self.model = model
        if likelihood == 'regression':
            self.lossfunc = MSELoss(reduction='sum')
            self.factor = 0.5  # convert to standard Gauss. log N(y|f,1)
        else:
            self.lossfunc = CrossEntropyLoss(reduction='sum')
            self.factor = 1.

    @abstractmethod
    def full(self, X, y, **kwargs):
        pass

    @abstractmethod
    def kron(self, X, y, **kwargs):
        pass

    @abstractmethod
    def diag(self, X, y, **kwargs):
        pass

    def _get_full_ggn(self, Js, f, y):
        loss = self.factor * self.lossfunc(f, y)
        if self.likelihood == 'regression':
            H_ggn = torch.einsum('mkp,mkq->pq', Js, Js)
        else:
            # second derivative of log lik is diag(p) - pp^T
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)
        return loss.detach(), H_ggn
