from abc import ABC, abstractmethod
import torch
from torch.nn import MSELoss, CrossEntropyLoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC


class CurvatureInterface(ABC):

    @abstractmethod
    def full():
        pass

    @abstractmethod
    def block():
        pass

    @abstractmethod
    def kron():
        pass

    @abstractmethod
    def diag():
        pass


class BackPackInterface(CurvatureInterface):

    def __init__(self, model, likelihood):
        assert likelihood in ['regression', 'classification']
        self.model = extend(model)
        if likelihood == 'regression':
            self.lossfunc = extend(MSELoss(reduction='sum'))
        else:
            self.lossfunc = extend(CrossEntropyLoss(reduction='sum'))


class BackPackGGN(BackPackInterface):

    def __init__(self, model, likelihood, stochastic=False):
        super().__init__(model, likelihood)
        self.stochastic = stochastic

    def _get_diag_ggn(self):
        if self.stochastic:
            dggn = torch.cat([p.diag_ggn_mc.data.flatten() for p in self.model.parameters()])
        else:
            dggn = torch.cat([p.diag_ggn_exact.data.flatten() for p in self.model.parameters()])
        return dggn.detach()

    def diag(self, X, y, **kwargs):
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        return loss, self._get_diag_ggn()

    def block(**kwargs):
        pass

    def kron(**wkwargs):
        pass

    def full(**kwargs):
        pass
