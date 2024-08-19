import torch
from ..utils import original_requires_grad

# compute no-centered covariance
OP_COV_KRON = 'cov_kron'  # Kronecker-factored
OP_COV_DIAG = 'cov_diag'  # diagonal
OP_COV_UNIT_WISE = 'cov_unit_wise'  # unit-wise

# compute Gram matrix
OP_GRAM_DIRECT = 'gram_direct'  # direct
OP_GRAM_HADAMARD = 'gram_hada'  # Hadamard-factored

OP_BATCH_GRADS = 'batch_grads'  # compute batched gradients (per-example gradients)
OP_ACCUMULATE_GRADS = 'acc_grad'  # accumulate gradients


class Operation:
    def __init__(self, module, model, op_names, save_field='op_results'):
        self._module = module
        self._model = model
        if isinstance(op_names, str):
            op_names = [op_names]
        # remove duplicates
        op_names = set(op_names)
        self._op_names = op_names
        self._save_field = save_field
        self._grads_scale = None
        self._kron_A = None

    def get_op_results(self):
        return getattr(self._module, self._save_field, {})

    def set_op_results(self, op_results):
        setattr(self._module, self._save_field, op_results)

    def clear_op_results(self):
        if hasattr(self._module, self._save_field):
            delattr(self._module, self._save_field)

    @property
    def grads_scale(self):
        return self._grads_scale

    @grads_scale.setter
    def grads_scale(self, value):
        self._grads_scale = value

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_COV_KRON in self._op_names or OP_GRAM_HADAMARD in self._op_names:
            if original_requires_grad(module, 'bias'):
                # Extend in_data with ones.
                # linear: n x f_in
                #      -> n x (f_in + 1)
                # conv2d: n x (c_in)(kernel_size) x out_size
                #      -> n x {(c_in)(kernel_size) + 1} x out_size
                shape = list(in_data.shape)
                shape[1] = 1
                ones = in_data.new_ones(shape)
                in_data = torch.cat((in_data, ones), dim=1)

            op_results = self.get_op_results()

            if OP_COV_KRON in self._op_names:
                A = self.cov_kron_A(module, in_data)
                self._kron_A = A
                op_results[OP_COV_KRON] = {'A': A.clone()}

            if OP_GRAM_HADAMARD in self._op_names:
                n_data = in_data.shape[0]
                n1 = self._model.kernel.shape[0]
                if n_data == n1:
                    A = self.gram_A(module, in_data, in_data)
                else:
                    A = self.gram_A(module, in_data[:n1], in_data[n1:])
                op_results[OP_GRAM_HADAMARD] = {'A': A.clone()}

            self.set_op_results(op_results)

    def backward_pre_process(self, in_data, out_grads):
        if self._grads_scale is not None:
            shape = (-1, ) + (1, ) * (out_grads.ndim - 1)
            out_grads = torch.mul(out_grads, self._grads_scale.reshape(shape))

        module = self._module
        op_results = self.get_op_results()
        for op_name in self._op_names:
            if op_name == OP_COV_KRON:
                rst = self.cov_kron_B(module, out_grads)
                if op_name in op_results:
                    op_results[op_name]['B'] = rst
                else:
                    assert self._kron_A is not None
                    op_results[op_name] = {'A': self._kron_A.clone(), 'B': rst}

            elif op_name == OP_COV_UNIT_WISE:
                assert original_requires_grad(module, 'weight')
                assert original_requires_grad(module, 'bias')
                op_results[op_name] = self.cov_unit_wise(module, in_data, out_grads)

            elif op_name == OP_GRAM_HADAMARD:
                n_data = in_data.shape[0]
                n1 = self._model.kernel.shape[0]
                if n_data == n1:
                    B = self.gram_B(module, out_grads, out_grads)
                else:
                    B = self.gram_B(module, out_grads[:n1], out_grads[n1:])
                A = op_results[OP_GRAM_HADAMARD]['A']
                self._model.kernel += B.mul(A)

            elif op_name == OP_GRAM_DIRECT:
                n_data = in_data.shape[0]
                n1 = self._model.kernel.shape[0]

                grads = self.batch_grads_weight(module, in_data, out_grads)
                v = [grads]
                if original_requires_grad(module, 'bias'):
                    grads_b = self.batch_grads_bias(module, out_grads)
                    v.append(grads_b)
                g = torch.cat([_v.flatten(start_dim=1) for _v in v], axis=1)

                precond = getattr(module, 'gram_precond', None)
                if precond is not None:
                    precond.precondition_vector_module(v, module)
                    g2 = torch.cat([_v.flatten(start_dim=1) for _v in v], axis=1)
                else:
                    g2 = g

                if n_data == n1:
                    self._model.kernel += torch.matmul(g, g2.T)
                else:
                    self._model.kernel += torch.matmul(g[:n1], g2[n1:].T)
            else:
                rst = getattr(self,
                              f'{op_name}_weight')(module, in_data, out_grads)
                op_results[op_name] = {'weight': rst}
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    op_results[op_name]['bias'] = rst

        self.set_op_results(op_results)

    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_kron_A(module, in_data):
        raise NotImplementedError

    @staticmethod
    def cov_kron_B(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise NotImplementedError

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise NotImplementedError
