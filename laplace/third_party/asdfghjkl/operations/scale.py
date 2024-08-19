import torch
from torch import nn

from .operation import Operation


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        
    def forward(self, input):
        return self.weight * input


class ScaleExt(Operation):
    """
    module.weight: 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out = f_in
    """
    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return (out_grads * in_data).view(N, -1).sum(dim=1)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return (out_grads * in_data).view(N, -1).sum(dim=1).square().sum()

    @staticmethod
    def cov_kron_A(module, in_data):
        N = in_data.size(0)
        setattr(module, 'n_in_data', in_data.reshape(N, -1))
        return torch.ones(1, 1, device=in_data.device) 

    @staticmethod
    def cov_kron_B(module, out_grads):
        N = out_grads.size(0)
        in_data = module.n_in_data
        cov = (in_data * out_grads.view(N, -1)).sum(dim=1).square().sum()
        return cov.reshape(1, 1)
