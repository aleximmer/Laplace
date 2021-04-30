from abc import ABC, abstractmethod, abstractstaticmethod
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

    @abstractstaticmethod
    def jacobians(model, X):
        raise NotImplementedError()

    @staticmethod
    def last_layer_jacobians(model, X):
        f, phi = model.forward_with_features(X)
        bsize = len(X)
        output_size = f.shape[-1]

        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=X.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        if model.last_layer.bias is not None:
            Js = torch.cat([Js, identity], dim=2)

        return Js, f.detach()

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
