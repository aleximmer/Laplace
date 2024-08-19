import torch
from torch import nn

from .operation import Operation, OP_COV_KRON, OP_COV_UNIT_WISE, OP_GRAM_HADAMARD, OP_GRAM_DIRECT  # NOQA


class _BatchNormNd(Operation):
    def __init__(self, module, model, op_names, save_attr='op_results'):
        if OP_COV_KRON in op_names:
            op_names = op_names.copy()
            # kron operation is not supported. unit_wise will be used instead.
            op_names.remove(OP_COV_KRON)
            op_names.append(OP_COV_UNIT_WISE)

        if OP_GRAM_HADAMARD in op_names:
            op_names = op_names.copy()
            # gram hadamard operation is not supported. gram direct will be used instead.
            op_names.remove(OP_GRAM_HADAMARD)
            op_names.append(OP_GRAM_DIRECT)

        super().__init__(module, model, op_names, save_attr)

    @staticmethod
    def _reduce(tensor: torch.Tensor):
        raise NotImplementedError

    def batch_grads_weight(
        self,
        module: nn.Module,
        in_data: torch.Tensor,
        out_grads: torch.Tensor
    ):
        return self._reduce(in_data.mul(out_grads))  # n x f

    def batch_grads_bias(self, module, out_grads):
        return self._reduce(out_grads)  # n x f

    def cov_diag_weight(self, module, in_data, out_grads):
        grads = self._reduce(in_data.mul(out_grads))
        return grads.mul(grads).sum(dim=0)  # f x 1

    def cov_diag_bias(self, module, out_grads):
        grads = self._reduce(out_grads)  # n x f
        return grads.mul(grads).sum(dim=0)  # f x 1

    def cov_unit_wise(self, module, in_data, out_grads):
        n_features = in_data.shape[1]  # f
        grads_w = self.batch_grads_weight(module, in_data, out_grads)  # n x f
        grads_b = self.batch_grads_bias(module, out_grads)  # n x f
        cov_ww = (grads_w ** 2).sum(0)  # f
        cov_bb = (grads_b ** 2).sum(0)  # f
        cov_wb = (grads_w * grads_b).sum(0)  # f
        blocks = torch.zeros(n_features, 2, 2).to(in_data.device)
        for i in range(n_features):
            blocks[i][0][0] = cov_ww[i]
            blocks[i][1][1] = cov_bb[i]
            blocks[i][0][1] = blocks[i][1][0] = cov_wb[i]
        return blocks  # f x 2 x 2

    @staticmethod
    def cov_kron_A(module, in_data):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in BatchNormNd.'
        )

    @staticmethod
    def cov_kron_B(module, out_grads):
        raise ValueError(
            f'{OP_COV_KRON} operation is not supported in BatchNormNd.'
        )

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in BatchNormNd.'
        )

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise ValueError(
            f'{OP_GRAM_HADAMARD} operation is not supported in BatchNormNd.'
        )


class BatchNorm1d(_BatchNormNd):
    """
    module.weight: f x 1
    module.bias: f x 1

    Argument shapes
    in_data: n x f
    out_grads: n x f
    """
    def _reduce(self, tensor):
        return tensor


class BatchNorm2d(_BatchNormNd):
    """
    module.weight: c x 1
    module.bias: c x 1

    Argument shapes
    in_data: n x c x h x w
    out_grads: n x c x h x w
    """
    def _reduce(self, tensor):
        return tensor.sum(dim=(2, 3))
