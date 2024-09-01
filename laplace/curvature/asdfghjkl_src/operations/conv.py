import torch
from torch import nn

from .operation import Operation


class Conv2d(Operation):
    """
    module.weight: c_out x c_in x k_h x k_w
    module.bias: c_out x 1

    Argument shapes
    in_data: n x (c_in)(kernel_size) x out_size
    out_grads: n x c_out x out_size

    kernel_size = (k_h)(k_w)
    out_size = output feature map size
    """
    @staticmethod
    def batch_grads_weight(
        module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor
    ):
        grads = torch.bmm(
            out_grads, in_data.transpose(2, 1)
        )  # n x c_out x (c_in)(kernel_size)
        return grads.view(
            -1, *module.weight.size()
        )  # n x c_out x c_in x k_h x k_w

    @staticmethod
    def batch_grads_bias(module: nn.Module, out_grads: torch.tensor):
        return out_grads.sum(axis=2)  # n x c_out

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        grads = torch.bmm(
            out_grads, in_data.transpose(2, 1)
        )  # n x c_out x (c_in)(kernel_size)
        rst = grads.mul(grads).sum(dim=0)  # c_out x (c_in)(kernel_size)
        return rst.view_as(module.weight)  # c_out x c_in x k_h x k_w

    @staticmethod
    def cov_diag_bias(module, out_grads):
        grads = out_grads.sum(axis=2)  # n x c_out
        return grads.mul(grads).sum(axis=0)  # c_out x 1

    @staticmethod
    def cov_kron_A(module, in_data):
        m = in_data.transpose(0, 1).flatten(
            start_dim=1
        )  # (c_in)(kernel_size) x n(out_size)
        return torch.matmul(
            m, m.T
        )  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @staticmethod
    def cov_kron_B(module, out_grads):
        out_size = out_grads.shape[-1]
        m = out_grads.transpose(0,
                                1).flatten(start_dim=1)  # c_out x n(out_size)
        return torch.matmul(m, m.T).div(out_size)  # c_out x c_out

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        # n x (c_in)(kernel_size)(out_size)
        m1 = in_data1.flatten(start_dim=1)
        m2 = in_data2.flatten(start_dim=1)
        return torch.matmul(m1, m2.T)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        out_size = out_grads1.shape[-1]
        # n x (c_out)(out_size)
        m1 = out_grads1.flatten(start_dim=1)
        m2 = out_grads2.flatten(start_dim=1)
        return torch.matmul(m1, m2.T).div(out_size)  # n x n
