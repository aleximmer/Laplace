from typing import List
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .precondition import KFAC, DiagNaturalGradient
from .fisher import FISHER_EXACT, FISHER_MC
from .kernel import batch, empirical_implicit_ntk, empirical_class_wise_direct_ntk, get_preconditioned_kernel_fn
from .utils import add_value_to_diagonal, nvtx_range


__all__ = [
    'FROMP',
]


_precond_classes = {'kron': KFAC, 'diag': DiagNaturalGradient}
_fisher_types = {'exact': FISHER_EXACT, 'mc': FISHER_MC}
_kernel_fns = {'implicit': empirical_implicit_ntk, 'class_wise': empirical_class_wise_direct_ntk}


class PastTask:
    def __init__(self, memorable_points, class_ids=None):
        self.memorable_points = memorable_points
        self.kernel_inv = None
        self.mean = None
        self.class_ids = class_ids

    def update_kernel(self, model, kernel_fn, eps=1e-5):
        memorable_points = self.memorable_points
        if isinstance(memorable_points, DataLoader):
            kernel = batch(kernel_fn, model, memorable_points)
        else:
            kernel = kernel_fn(model, memorable_points)
        n, c = kernel.shape[0], kernel.shape[-1]  # (n, n, c, c) or (n, n, c)
        ndim = kernel.ndim
        if ndim == 4:
            kernel = kernel.transpose(1, 2).reshape(n * c, n * c)  # (nc, nc)
        elif ndim == 3:
            kernel = kernel.transpose(0, 2)  # (c, n, n)
        else:
            raise ValueError(f'Invalid kernel ndim: {ndim}. ndim must be 3 or 4.')

        kernel = add_value_to_diagonal(kernel, eps)
        self.kernel_inv = torch.inverse(kernel).detach_()

    @torch.no_grad()
    def update_mean(self, model):
        self.mean = self._evaluate_mean(model)

    def _evaluate_mean(self, model):
        means = []
        memorable_points = self.memorable_points
        if isinstance(memorable_points, DataLoader):
            device = next(model.parameters()).device
            for inputs, _ in self.memorable_points:
                inputs = inputs.to(device)
                means.append(model(inputs))
            return torch.cat(means)  # (n, c)
        else:
            return model(memorable_points)

    def get_penalty(self, model):
        assert self.kernel_inv is not None and self.mean is not None
        kernel_inv = self.kernel_inv  # (nc, nc) or (c, n, n)
        current_mean = self._evaluate_mean(model)  # (n, c)
        b = current_mean - self.mean  # (n, c)
        if kernel_inv.ndim == 2:
            # kernel_inv: (nc, nc)
            b = b.flatten()  # (nc,)
            v = torch.mv(kernel_inv, b)  # (nc,)
        else:
            # kernel_inv: (c, n, n)
            b = b.transpose(0, 1).unsqueeze(2)  # (c, n, 1)
            v = torch.matmul(kernel_inv, b)  # (c, n, 1)
            v = v.transpose(0, 1).flatten()  # (nc,)
            b = b.flatten()  # (nc,)

        return torch.dot(b, v)


class FROMP:
    """
    Implementation of a functional-regularisation method called
    Functional Regularisation of Memorable Past (FROMP):
    Pingbo Pan et al., 2020
    Continual Deep Learning by Functional Regularisation of Memorable Past
    https://arxiv.org/abs/2004.14070

    Example::

        >>> import torch
        >>> from asdfghjkl import FROMP
        >>>
        >>> model = torch.nn.Linear(5, 3)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> fr = FROMP(model, tau=1.)
        >>>
        >>> for data_loader in data_loader_list:
        >>>     for x, y in data_loader:
        >>>         optimizer.zero_grad()
        >>>         loss = loss_fn(model(x), y)
        >>>         if fr.is_ready:
        >>>             loss += fr.get_penalty()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     fr.update_regularization_info(data_loader)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 tau=1.,
                 eps=1e-5,
                 max_tasks_for_penalty=None,
                 n_memorable_points=10,
                 ggn_shape='diag',
                 ggn_type='exact',
                 prior_prec=1e-5,
                 n_mc_samples=1,
                 kernel_type='implicit',
                 ):
        assert ggn_type in _fisher_types, f'ggn_type: {ggn_type} is not supported.' \
                                          f' choices: {list(_fisher_types.keys())}'
        assert ggn_shape in _precond_classes, f'ggn_shape: {ggn_shape} is not supported.' \
                                              f' choices: {list(_precond_classes.keys())}'
        assert kernel_type in _kernel_fns, f'kernel_type: {kernel_type} is not supported.' \
                                           f' choices: {list(_kernel_fns.keys())}'

        self.model = model
        self.tau = tau
        self.eps = eps
        self.max_tasks_for_penalty = max_tasks_for_penalty
        self.n_memorable_points = n_memorable_points

        if isinstance(model, DDP):
            # As DDP disables hook functions required for Fisher calculation,
            # the underlying module will be used instead.
            model_precond = model.module
        else:
            model_precond = model
        self.precond = _precond_classes[ggn_shape](model_precond,
                                                   fisher_type=_fisher_types[ggn_type],
                                                   pre_inv_postfix='all_tasks_ggn',
                                                   n_mc_samples=n_mc_samples,
                                                   damping=prior_prec)
        self.kernel_fn = get_preconditioned_kernel_fn(_kernel_fns[kernel_type], self.precond)
        self.observed_tasks: List[PastTask] = []

    @property
    def is_ready(self):
        return len(self.observed_tasks) > 0

    def update_regularization_info(self,
                                   data_loader: DataLoader,
                                   class_ids: List[int] = None,
                                   memorable_points_as_tensor=True,
                                   is_distributed=False):
        model = self.model
        if isinstance(model, DDP):
            # As DDP disables hook functions required for Kernel calculation,
            # the underlying module will be used instead.
            model = model.module
        model.eval()

        # update GGN and inverse for the current task
        with customize_head(model, class_ids):
            self.precond.update_curvature(data_loader=data_loader)
        if is_distributed:
            self.precond.reduce_curvature()
        self.precond.accumulate_curvature(to_pre_inv=True)
        self.precond.update_inv()

        # register the current task with the memorable points
        with customize_head(model, class_ids):
            memorable_points = collect_memorable_points(model,
                                                        data_loader,
                                                        self.n_memorable_points,
                                                        memorable_points_as_tensor,
                                                        is_distributed)
        self.observed_tasks.append(PastTask(memorable_points, class_ids))

        # update information (kernel & mean) for each observed task
        for task in self.observed_tasks:
            with customize_head(model, task.class_ids, softmax=True):
                task.update_kernel(model, self.kernel_fn)
                task.update_mean(model)

    def get_penalty(self, tau=None, eps=None, max_tasks=None, cholesky=False):
        assert self.is_ready, 'Functional regularization is not ready yet, ' \
                              'call FROMP.update_regularization_info(data_loader).'
        if tau is None:
            tau = self.tau
        if eps is None:
            eps = self.eps
        if max_tasks is None:
            max_tasks = self.max_tasks_for_penalty
        model = self.model
        model.eval()
        observed_tasks = self.observed_tasks

        # collect indices of tasks to calculate regularization penalty
        n_observed_tasks = len(observed_tasks)
        indices = list(range(n_observed_tasks))
        if max_tasks and max_tasks < n_observed_tasks:
            import random
            indices = random.sample(indices, max_tasks)

        # get regularization penalty on all the selected tasks
        with disable_broadcast_buffers(model):
            total_penalty = 0
            for idx in indices:
                task = observed_tasks[idx]
                with customize_head(model, task.class_ids, softmax=True):
                    total_penalty += task.get_penalty(model, eps=eps, cholesky=cholesky)

        return 0.5 * tau * total_penalty


@torch.no_grad()
def collect_memorable_points(model,
                             data_loader: DataLoader,
                             n_memorable_points,
                             as_tensor=True,
                             is_distributed=False):
    device = next(model.parameters()).device
    dataset = data_loader.dataset

    # create a data loader w/o shuffling so that indices in the dataset are stored
    assert data_loader.batch_size is not None, 'DataLoader w/o batch_size is not supported.'
    if is_distributed:
        indices = range(dist.get_rank(), len(dataset), dist.get_world_size())
        dataset = Subset(dataset, indices)
    no_shuffle_loader = DataLoader(dataset,
                                   batch_size=data_loader.batch_size,
                                   num_workers=data_loader.num_workers,
                                   pin_memory=True,
                                   drop_last=False,
                                   shuffle=False)
    # collect Hessian trace
    hessian_traces = []
    for inputs, _ in no_shuffle_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)  # (n, c)
        diag_hessian = probs - probs * probs  # (n, c)
        hessian_traces.append(diag_hessian.sum(dim=1))  # [(n,)]
    hessian_traces = torch.cat(hessian_traces)

    # sort indices by Hessian trace
    indices = torch.argsort(hessian_traces, descending=True).cpu()
    top_indices = indices[:n_memorable_points]

    if as_tensor:
        # crate a Tensor for memorable points on model's device
        memorable_points = [dataset[idx][0] for idx in top_indices]
        return torch.stack(memorable_points).to(device)
    else:
        # create a DataLoader for memorable points
        memorable_points = Subset(dataset, top_indices)
        batch_size = min(n_memorable_points, data_loader.batch_size)
        return DataLoader(memorable_points,
                          batch_size=batch_size,
                          pin_memory=True,
                          drop_last=False,
                          shuffle=False)


@contextmanager
def customize_head(module: torch.nn.Module, class_ids: List[int] = None, softmax=False):

    def forward_hook(module, input, output):
        if class_ids is not None:
            output = output[:, class_ids]
        if softmax:
            return F.softmax(output, dim=1)
        else:
            return output

    handle = module.register_forward_hook(forward_hook)
    yield
    handle.remove()
    del forward_hook


@contextmanager
def disable_broadcast_buffers(module):
    tmp = False
    if isinstance(module, DDP):
        tmp = module.broadcast_buffers
        module.broadcast_buffers = False
    yield
    if isinstance(module, DDP):
        module.broadcast_buffers = tmp
