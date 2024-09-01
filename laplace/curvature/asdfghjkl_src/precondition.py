import torch
from torch import nn

from .matrices import FISHER_EXACT, SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG  # NOQA
from .fisher import fisher_for_cross_entropy
from .utils import add_value_to_diagonal

_supported_modules = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)
_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)

__all__ = [
    'Precondition', 'NaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'DiagNaturalGradient'
]


class Precondition:
    def __init__(self):
        pass

    def update_curvature(self, inputs=None, targets=None, data_loader=None):
        raise NotImplementedError

    def accumulate_curvature(self):
        raise NotImplementedError

    def finalize_accumulation(self):
        raise NotImplementedError

    def reduce_curvature(self):
        raise NotImplementedError

    def update_inv(self, damping=None):
        raise NotImplementedError

    def precondition(self):
        raise NotImplementedError

    def precondition_vector(self, vec):
        raise NotImplementedError


class NaturalGradient(Precondition):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5,
                 ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        self.model = model
        self.modules = [model]
        self.fisher_type = fisher_type
        self.n_mc_samples = n_mc_samples
        self.damping = damping
        super().__init__()
        self.fisher_shape = SHAPE_FULL
        self.fisher_manager = None
        self._pre_inv_postfix = pre_inv_postfix

    def _get_fisher_attr(self, postfix=None):
        if postfix is None:
            return self.fisher_type
        else:
            return f'{self.fisher_type}_{postfix}'

    def _get_fisher(self, module, postfix=None):
        attr = self._get_fisher_attr(postfix)
        fisher = getattr(module, attr, None)
        return fisher

    @property
    def _pre_inv_attr(self):
        return self._get_fisher_attr(self._pre_inv_postfix)

    def _get_pre_inv_fisher(self, module):
        return getattr(module, self._pre_inv_attr, None)

    def _set_fisher(self, module, data, postfix=None):
        attr = self._get_fisher_attr(postfix)
        setattr(module, attr, data)

    def _clear_fisher(self, module, postfix=None):
        attr = self._get_fisher_attr(postfix)
        if hasattr(module, attr):
            delattr(module, attr)

    def update_curvature(self, inputs=None, targets=None, data_loader=None):
        rst = fisher_for_cross_entropy(self.model,
                                       inputs=inputs,
                                       targets=targets,
                                       data_loader=data_loader,
                                       fisher_types=self.fisher_type,
                                       fisher_shapes=self.fisher_shape,
                                       n_mc_samples=self.n_mc_samples)
        self.fisher_manager = rst

    def move_curvature(self, postfix, scale=1., to_pre_inv=False):
        self.accumulate_curvature(postfix, scale, to_pre_inv, replace=True)

    def accumulate_curvature(self, postfix='acc', scale=1., to_pre_inv=False, replace=False):
        if to_pre_inv:
            postfix = self._pre_inv_postfix
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            fisher.scaling(scale)
            fisher_acc = self._get_fisher(module, postfix)
            if fisher_acc is None or replace:
                self._set_fisher(module, fisher, postfix)
            else:
                self._set_fisher(module, fisher_acc + fisher, postfix)
            self._clear_fisher(module)

    def finalize_accumulation(self, postfix='acc'):
        for module in self.modules:
            fisher_acc = self._get_fisher(module, postfix)
            assert fisher_acc is not None
            self._set_fisher(module, fisher_acc)
            self._clear_fisher(module, postfix)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            inv = _cholesky_inv(add_value_to_diagonal(fisher.data, damping))
            setattr(fisher, 'inv', inv)

    def precondition(self):
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        fisher = self._get_pre_inv_fisher(self.model)
        ng = torch.mv(fisher.inv, g)

        pointer = 0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                val = ng[pointer:pointer + numel]
                p.grad.copy_(val.reshape_as(p.grad))
                pointer += numel

        assert pointer == ng.numel()


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_BLOCK_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            g = module.weight.grad.flatten()
            if _bias_requires_grad(module):
                g = torch.cat([g, module.bias.grad.flatten()])
            ng = torch.mv(fisher.inv, g)

            if _bias_requires_grad(module):
                w_numel = module.weight.numel()
                grad_w = ng[:w_numel]
                module.bias.grad.copy_(ng[w_numel:])
            else:
                grad_w = ng
            module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_KRON
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                unit = fisher.unit.data
                f = unit.shape[0]
                dmp = torch.eye(2, device=unit.device, dtype=unit.dtype).repeat(f, 1, 1) * damping
                inv = torch.inverse(fisher.unit.data + dmp)
                setattr(fisher.unit, 'inv', inv)
            else:
                A = fisher.kron.A
                B = fisher.kron.B
                A_eig_mean = A.trace() / A.shape[0]
                B_eig_mean = B.trace() / B.shape[0]
                pi = torch.sqrt(A_eig_mean / B_eig_mean)
                r = damping**0.5

                A_inv = _cholesky_inv(add_value_to_diagonal(A, r * pi))
                B_inv = _cholesky_inv(add_value_to_diagonal(B, r / pi))

                setattr(fisher.kron, 'A_inv', A_inv)
                setattr(fisher.kron, 'B_inv', B_inv)

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                inv = fisher.unit.inv  # (f, 2, 2)
                assert _bias_requires_grad(module)
                grad_w = module.weight.grad  # (f,)
                grad_b = module.bias.grad  # (f,)
                g = torch.stack([grad_w, grad_b], dim=1)  # (f, 2)
                g = g.unsqueeze(2)  # (f, 2, 1)
                ng = torch.matmul(inv, g).squeeze(2)  # (f, 2)
                module.weight.grad.copy_(ng[:, 0])
                module.bias.grad.copy_(ng[:, 1])
            else:
                A_inv = fisher.kron.A_inv
                B_inv = fisher.kron.B_inv
                grad2d = module.weight.grad.view(B_inv.shape[0], -1)
                if _bias_requires_grad(module):
                    grad2d = torch.cat(
                        [grad2d, module.bias.grad.unsqueeze(dim=1)], dim=1)
                ng = B_inv.mm(grad2d).mm(A_inv)
                if _bias_requires_grad(module):
                    grad_w = ng[:, :-1]
                    module.bias.grad.copy_(ng[:, -1])
                else:
                    grad_w = ng
                module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                inv = fisher.unit.inv  # (f, 2, 2)
                assert _bias_requires_grad(module)
                vec_w = vec[idx]  # (f,)
                vec_b = vec[idx + 1]  # (f,)
                v = torch.stack([vec_w, vec_b], dim=1)  # (f, 2)
                v = v.unsqueeze(2)  # (f, 2, 1)
                ng = torch.matmul(inv, v).squeeze(2)  # (f, 2)
                vec[idx].copy_(ng[:, 0])
                vec[idx + 1].copy_(ng[:, 1])
                idx += 2
            else:
                A_inv = fisher.kron.A_inv
                B_inv = fisher.kron.B_inv
                w_idx = idx
                vec2d = vec[w_idx].view(B_inv.shape[0], -1)
                idx += 1
                if _bias_requires_grad(module):
                    vec2d = torch.cat(
                        [vec2d, vec[idx].unsqueeze(dim=1)], dim=1)
                ng = B_inv.mm(vec2d).mm(A_inv)
                if _bias_requires_grad(module):
                    vec_w = ng[:, :-1]
                    vec[idx].copy_(ng[:, -1])
                    idx += 1
                else:
                    vec_w = ng
                vec[w_idx].copy_(vec_w.reshape_as(module.weight.data))

        assert idx == len(vec)


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 pre_inv_postfix=None,
                 n_mc_samples=1,
                 damping=1e-5):
        super().__init__(model, fisher_type, pre_inv_postfix, n_mc_samples, damping)
        self.fisher_shape = SHAPE_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping

        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            elif isinstance(module, _supported_modules):
                diag_w = fisher.diag.weight
                setattr(fisher.diag, 'weight_inv', 1 / (diag_w + damping))
                if _bias_requires_grad(module):
                    diag_b = fisher.diag.bias
                    setattr(fisher.diag, 'bias_inv', 1 / (diag_b + damping))

    def precondition(self):
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            w_inv = fisher.diag.weight_inv
            module.weight.grad.mul_(w_inv)
            if _bias_requires_grad(module):
                b_inv = fisher.diag.bias_inv
                module.bias.grad.mul_(b_inv)

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_pre_inv_fisher(module)
            if fisher is None:
                continue
            assert fisher.diag is not None, module
            vec[idx].mul_(fisher.diag.weight_inv)
            idx += 1
            if _bias_requires_grad(module):
                vec[idx].mul_(fisher.diag.bias_inv)
                idx += 1

        assert idx == len(vec)

    def precondition_vector_module(self, vec, module):
        fisher = self._get_pre_inv_fisher(module)
        assert fisher is not None
        assert fisher.diag is not None, module
        vec[0].mul_(fisher.diag.weight_inv)
        if _bias_requires_grad(module):
            vec[1].mul_(fisher.diag.bias_inv)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad


def _cholesky_inv(X):
    u = torch.cholesky(X)
    return torch.cholesky_inverse(u)

