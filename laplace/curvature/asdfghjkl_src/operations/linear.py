import torch
from torch import nn

from .operation import Operation


class Linear(Operation):
    """
    module.weight: f_out x f_in
    module.bias: f_out x 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out
    """
    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        return torch.bmm(
            out_grads.unsqueeze(2), in_data.unsqueeze(1)
        )  # n x f_out x f_in

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in)  # f_out x f_in

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return grad_grad.sum(dim=0)  # f_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        return torch.matmul(in_data.T, in_data)  # f_in x f_in

    @staticmethod
    def cov_kron_B(module, out_grads):
        return torch.matmul(out_grads.T, out_grads)  # f_out x f_out

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        return torch.matmul(in_data1, in_data2.T)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        return torch.matmul(out_grads1, out_grads2.T)  # n x n
