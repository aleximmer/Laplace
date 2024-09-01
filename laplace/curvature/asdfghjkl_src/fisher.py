from contextlib import contextmanager
import numpy as np

import torch
import torch.nn.functional as F
from .core import extend
from .utils import disable_param_grad
from .gradient import data_loader_gradient
from .operations import *
from .symmatrix import SymMatrix, Kron, Diag, UnitWise
from .matrices import *
from .mvp import power_method, conjugate_gradient_method

_SHAPE_TO_OP = {
    SHAPE_FULL: OP_BATCH_GRADS,  # full
    SHAPE_BLOCK_DIAG: OP_BATCH_GRADS,  # block-diagonal
    SHAPE_KRON: OP_COV_KRON,  # Kronecker-factored
    SHAPE_DIAG: OP_COV_DIAG,  # diagonal
}

_COV_FULL = 'cov_full'
_CVP_FULL = 'cvp_full'
_COV_BLOCK_DIAG = 'cov_block_diag'
_CVP_BLOCK_DIAG = 'cvp_block_diag'

__all__ = [
    'fisher_for_cross_entropy',
    'fvp_for_cross_entropy',
    'zero_fisher',
    'zero_fvp',
    'fisher_for_cross_entropy_eigenvalues',
    'fisher_free_for_cross_entropy',
    'woodbury_ifvp'
]

_supported_types = [FISHER_EXACT, FISHER_MC, COV]
_supported_types_for_eig = _supported_types
_supported_shapes = [SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG]
_supported_shapes_for_eig = [SHAPE_FULL, SHAPE_BLOCK_DIAG]


def fisher_for_cross_entropy(
    model,
    fisher_types,
    fisher_shapes,
    inputs=None,
    targets=None,
    data_loader=None,
    stats_name=None,
    compute_param_grad=False,
    n_mc_samples=1,
    is_distributed=False,
    all_reduce=False,
    is_master=True,
    matrix_manager=None,
):
    if isinstance(fisher_types, str):
        fisher_types = [fisher_types]
    if isinstance(fisher_shapes, str):
        fisher_shapes = [fisher_shapes]
    # remove duplicates
    fisher_types = set(fisher_types)
    fisher_shapes = set(fisher_shapes)
    for ftype in fisher_types:
        assert ftype in _supported_types, \
            f'Invalid fisher_type: {ftype}. ' \
            f'fisher_type must be in {_supported_types}.'
    for fshape in fisher_shapes:
        assert fshape in _supported_shapes, \
            f'Invalid fisher_shape: {fshape}. ' \
            f'fisher_shape must be in {_supported_shapes}.'

    zero_fisher(model, fisher_types)

    # setup operations for mammoth_utils.autograd.extend
    op_names = [_SHAPE_TO_OP[shape] for shape in fisher_shapes]
    if compute_param_grad:
        assert COV in fisher_types, \
            f'"{COV}" must be in fisher_types when compute_param_grad is True.'
        if data_loader is not None:
            op_names.append(OP_ACCUMULATE_GRADS)  # accumulate gradient

    # setup matrix manager as needed
    if matrix_manager is None:
        matrix_manager = MatrixManager(model, fisher_types)

    kwargs = dict(
        compute_full_fisher=SHAPE_FULL in fisher_shapes,
        compute_block_diag_fisher=SHAPE_BLOCK_DIAG in fisher_shapes,
        compute_param_grad=compute_param_grad,
        n_mc_samples=n_mc_samples
    )

    if data_loader is not None:
        # accumulate fisher for an epoch
        device = next(model.parameters()).device
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with extend(model, op_names):
                _fisher_for_cross_entropy(
                    model, fisher_types, inputs, targets, **kwargs
                )
            if stats_name is not None:
                matrix_manager.accumulate_matrices(stats_name)
        if compute_param_grad:
            data_loader_gradient(
                model,
                data_loader,
                has_accumulated=True,
                is_distributed=is_distributed,
                all_reduce=all_reduce,
                is_master=is_master
            )
    else:
        # compute fisher for a single batch
        assert inputs is not None
        with extend(model, op_names):
            _fisher_for_cross_entropy(
                model, fisher_types, inputs, targets, **kwargs
            )

    # reduce matrices
    if is_distributed:
        matrix_manager.reduce_matrices(stats_name, is_master, all_reduce)

    return matrix_manager


def zero_fisher(module, fisher_types):
    for child in module.children():
        zero_fisher(child, fisher_types)
    for ftype in fisher_types:
        if hasattr(module, ftype):
            delattr(module, ftype)


def zero_fvp(module, fisher_types):
    for child in module.children():
        zero_fvp(child, fisher_types)
    for ftype in fisher_types:
        attr = _get_fvp_attr(ftype)
        if hasattr(module, attr):
            delattr(module, attr)


def _check_fisher_type_shape(fisher_type, fisher_shape):
    assert fisher_type in _supported_types_for_eig, \
        f'Invalid fisher_type: {fisher_type}. ' \
        f'fisher_type must be in {_supported_types_for_eig}.'
    assert fisher_shape in _supported_shapes_for_eig, \
        f'Invalid fisher_shape: {fisher_shape}. ' \
        f'fisher_shape must be in {_supported_shapes_for_eig}.'


def fisher_for_cross_entropy_eigenvalues(
        model,
        fisher_type,
        fisher_shape,
        data_loader=None,
        inputs=None,
        targets=None,
        n_mc_samples=1,
        top_n=1,
        max_iters=100,
        tol=1e-3,
        is_distributed=False,
        print_progress=False
):
    _check_fisher_type_shape(fisher_type, fisher_shape)

    def fvp_fn(vec, x, y):
        return fvp_for_cross_entropy(vec,
                                     model,
                                     fisher_type,
                                     fisher_shape,
                                     inputs=x,
                                     targets=y,
                                     n_mc_samples=n_mc_samples)

    # for making MC samplings at each iteration deterministic
    random_seed = torch.rand(1) * 100 if fisher_type == FISHER_MC else None

    eigvals, eigvecs = power_method(fvp_fn,
                                    model,
                                    data_loader=data_loader,
                                    inputs=inputs,
                                    targets=targets,
                                    top_n=top_n,
                                    max_iters=max_iters,
                                    tol=tol,
                                    is_distributed=is_distributed,
                                    print_progress=print_progress,
                                    random_seed=random_seed
                                    )

    return eigvals, eigvecs


def fisher_free_for_cross_entropy(
        model,
        b,
        fisher_type,
        fisher_shape,
        data_loader=None,
        inputs=None,
        targets=None,
        init_x=None,
        damping=1e-3,
        n_mc_samples=1,
        max_iters=None,
        tol=1e-8,
        preconditioner=None,
        is_distributed=False,
        print_progress=False,
        random_seed=None,
        save_log=False
):
    _check_fisher_type_shape(fisher_type, fisher_shape)

    def fvp_fn(vec, x, y):
        return fvp_for_cross_entropy(vec,
                                     model,
                                     fisher_type,
                                     fisher_shape,
                                     inputs=x,
                                     targets=y,
                                     n_mc_samples=n_mc_samples)

    # for making MC samplings at each iteration deterministic
    if fisher_type == FISHER_MC and random_seed is None:
        random_seed = int(torch.rand(1) * 100)

    return conjugate_gradient_method(fvp_fn,
                                     b,
                                     data_loader=data_loader,
                                     inputs=inputs,
                                     targets=targets,
                                     init_x=init_x,
                                     damping=damping,
                                     max_iters=max_iters,
                                     tol=tol,
                                     preconditioner=preconditioner,
                                     is_distributed=is_distributed,
                                     print_progress=print_progress,
                                     random_seed=random_seed,
                                     save_log=save_log)


def fvp_for_cross_entropy(
    vec,
    model,
    fisher_type,
    fisher_shape,
    inputs,
    targets=None,
    n_mc_samples=1
):
    compute_full_fvp = compute_block_diag_fvp = False
    if fisher_shape == SHAPE_FULL:
        compute_full_fvp = True
    elif fisher_shape == SHAPE_BLOCK_DIAG:
        compute_block_diag_fvp = True
    else:
        raise ValueError(f'Invalid fisher_shape: {fisher_shape}.')

    zero_fvp(model, [fisher_type])

    with extend(model, OP_BATCH_GRADS):
        _fisher_for_cross_entropy(
            model, [fisher_type],
            inputs,
            targets,
            compute_full_fvp=compute_full_fvp,
            compute_block_diag_fvp=compute_block_diag_fvp,
            vec=vec,
            n_mc_samples=n_mc_samples
        )

    if fisher_shape == SHAPE_FULL:
        return getattr(model, _get_fvp_attr(fisher_type))
    else:
        rst = []
        for module in model.modules():
            fvp = getattr(module, _get_fvp_attr(fisher_type), None)
            if fvp is not None:
                rst.extend(fvp)
        return rst


def _fisher_for_cross_entropy(
    model,
    fisher_types,
    inputs,
    targets=None,
    compute_param_grad=False,
    compute_full_fisher=False,
    compute_full_fvp=False,
    compute_block_diag_fisher=False,
    compute_block_diag_fvp=False,
    vec=None,
    n_mc_samples=1
):
    logits = model(inputs)
    log_probs = F.log_softmax(logits, dim=1)
    probs = None

    def loss_and_backward(target):
        model.zero_grad(set_to_none=True)
        loss = F.nll_loss(log_probs, target, reduction='sum')
        loss.backward(retain_graph=True)
        if compute_full_fisher:
            _full_covariance(model)
        if compute_full_fvp:
            _full_cvp(model, vec)
        if compute_block_diag_fisher:
            _block_diag_covariance(model)
        if compute_block_diag_fvp:
            _block_diag_cvp(model, vec)

    if FISHER_MC in fisher_types:
        probs = F.softmax(logits, dim=1)
        _fisher_mc(loss_and_backward, model, probs, n_mc_samples)

    if FISHER_EXACT in fisher_types:
        if probs is None:
            probs = F.softmax(logits, dim=1)
        _fisher_exact(loss_and_backward, model, probs)

    if COV in fisher_types:
        assert targets is not None, 'targets must be specified for computing covariance.'
        _covariance(loss_and_backward, model, targets, compute_param_grad)


def _module_batch_grads(model):
    rst = []
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        batch_grads = operation.get_op_results()[OP_BATCH_GRADS]
        rst.append((module, batch_grads))
    return rst


def _module_batch_flatten_grads(model):
    rst = []
    for module, batch_grads in _module_batch_grads(model):
        batch_flatten_grads = torch.cat(
            [g.flatten(start_dim=1) for g in batch_grads.values()],
            dim=1
        )
        rst.append((module, batch_flatten_grads))
    return rst


def _module_batch_gvp(model, vec):
    rst = []
    pointer = 0
    for module, batch_grads in _module_batch_grads(model):
        batch_gvp = None
        for b_g in batch_grads.values():
            v = vec[pointer]
            b_gvp = b_g.mul(v.unsqueeze(0)).flatten(start_dim=1).sum(1)  # n
            if batch_gvp is None:
                batch_gvp = b_gvp
            else:
                batch_gvp += b_gvp
            pointer += 1
        rst.append((module, batch_gvp))
    assert pointer == len(vec)
    return rst


def _full_covariance(model):
    batch_all_g = []
    for _, batch_g in _module_batch_flatten_grads(model):
        batch_all_g.append(batch_g)
    batch_all_g = torch.cat(batch_all_g, dim=1)  # n x p_all
    cov_full = torch.matmul(batch_all_g.T, batch_all_g)  # p_all x p_all
    setattr(model, _COV_FULL, cov_full)


def _block_diag_covariance(model):
    for module, batch_g in _module_batch_flatten_grads(model):
        cov_block = torch.matmul(batch_g.T, batch_g)  # p_all x p_all
        setattr(module, _COV_BLOCK_DIAG, cov_block)


def _full_cvp(model, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    # compute batched (g^t)v
    batch_all_gvp = None
    for module, batch_gvp in _module_batch_gvp(model, vec):
        if batch_all_gvp is None:
            batch_all_gvp = batch_gvp
        else:
            batch_all_gvp += batch_gvp

    # compute cvp = sum[g(g^t)v]
    cvp = []
    for module, batch_grads in _module_batch_grads(model):
        for b_g in batch_grads.values():
            cvp.append(torch.einsum('n...,n->...', b_g, batch_all_gvp))

    setattr(model, _CVP_FULL, cvp)


def _block_diag_cvp(model, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    batch_gvp_dict = {k: v for k, v in _module_batch_gvp(model, vec)}
    for module, batch_grads in _module_batch_grads(model):
        cvp = []
        # compute cvp = sum[g(g^t)v]
        batch_gvp = batch_gvp_dict[module]
        for b_g in batch_grads.values():
            cvp.append(torch.einsum('n...,n->...', b_g, batch_gvp))

        setattr(module, _CVP_BLOCK_DIAG, cvp)


def _fisher_mc(loss_and_backward, model, probs, n_mc_samples=1):
    dist = torch.distributions.Categorical(probs)
    _targets = dist.sample((n_mc_samples, ))
    for i in range(n_mc_samples):
        loss_and_backward(_targets[i])
        _register_fisher(
            model,
            FISHER_MC,
            scale=1 / n_mc_samples,
            accumulate=True
        )


def _fisher_exact(loss_and_backward, model, probs):
    _, n_classes = probs.shape
    probs, _targets = torch.sort(probs, dim=1, descending=True)
    sqrt_probs = torch.sqrt(probs)
    for i in range(n_classes):
        with _grads_scale(model, sqrt_probs[:, i]):
            loss_and_backward(_targets[:, i])
        _register_fisher(
            model, FISHER_EXACT, accumulate=True
        )


def _covariance(loss_and_backward, model, targets, compute_param_grad=False):
    if compute_param_grad:
        loss_and_backward(targets)
    else:
        with disable_param_grad(model):
            loss_and_backward(targets)
    _register_fisher(model, COV)


@contextmanager
def _grads_scale(model, scale):
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        operation.grads_scale = scale

    yield

    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        operation.grads_scale = None


def _register_fisher(model, fisher_type, scale=1., accumulate=False):
    """
    module.fisher_{fisher_type} = op_results
    op_results = {
        'diag': {'weight': torch.Tensor, 'bias': torch.Tensor},
        'kron': {'A': torch.Tensor, 'B': torch.Tensor},
        'block_diag': torch.Tensor,
        'unit_wise': torch.Tensor,
    }
    """
    device = next(model.parameters()).device
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        op_results = operation.get_op_results()
        kron = diag = unit = None
        if OP_COV_KRON in op_results:
            rst = op_results[OP_COV_KRON]
            kron = Kron(rst['A'], rst['B'], device=device)
        if OP_COV_DIAG in op_results:
            rst = op_results[OP_COV_DIAG]
            diag = Diag(
                rst.get('weight', None), rst.get('bias', None), device=device
            )
        if OP_COV_UNIT_WISE in op_results:
            rst = op_results[OP_COV_UNIT_WISE]
            unit = UnitWise(rst, device=device)
        operation.clear_op_results()
        # move block_diag/kron/diag fisher
        _accumulate_fisher(
            module,
            _COV_BLOCK_DIAG,
            fisher_type,
            kron=kron,
            diag=diag,
            unit=unit,
            scale=scale,
            accumulate=accumulate
        )
        # move block_diag fvp
        _accumulate_fvp(
            module, _CVP_BLOCK_DIAG, fisher_type, scale, accumulate
        )

    # move full fisher
    _accumulate_fisher(
        model, _COV_FULL, fisher_type, scale=scale, accumulate=accumulate
    )
    # move full fvp
    _accumulate_fvp(model, _CVP_FULL, fisher_type, scale, accumulate)


def _accumulate_fisher(
    module,
    data_src_attr,
    dst_attr,
    kron=None,
    diag=None,
    unit=None,
    scale=1.,
    accumulate=False
):
    data = getattr(module, data_src_attr, None)
    if all(v is None for v in [data, kron, diag, unit]):
        return
    device = next(module.parameters()).device
    fisher = SymMatrix(data, kron, diag, unit, device=device)
    fisher.scaling(scale)
    dst_fisher = getattr(module, dst_attr, None)
    if (dst_fisher is None) or (not accumulate):
        setattr(module, dst_attr, fisher)
    else:
        # accumulate fisher
        dst_fisher += fisher
        if dst_fisher.has_kron:
            # not accumulate kron.A
            dst_fisher.kron.A = fisher.kron.A
        setattr(module, dst_attr, dst_fisher)

    if data is not None:
        delattr(module, data_src_attr)


def _accumulate_fvp(module, src_attr, fisher_type, scale=1., accumulate=False):
    dst_attr = _get_fvp_attr(fisher_type)
    cvp = getattr(module, src_attr, None)
    if cvp is None:
        return
    cvp = [v * scale for v in cvp]
    dst_fvp = getattr(module, dst_attr, None)
    if (dst_fvp is None) or (not accumulate):
        setattr(module, dst_attr, cvp)
    else:
        dst_fvp = [u.add(v) for u, v in zip(dst_fvp, cvp)]
        setattr(module, dst_attr, dst_fvp)

    delattr(module, src_attr)


def _get_fvp_attr(fisher_type):
    return f'{fisher_type}_vp'


def woodbury_ifvp(
        vec,
        model,
        inputs,
        targets,
        loss_fn,
        damping=1e-5,
        data_average=True,
        fisher_shape=None,
):
    """
    Calculate inverse-empirical Fisher vector product by using the Woodbury matrix identity
    """
    assert damping > 0, 'Damping value has to be positive.'

    with extend(model, OP_BATCH_GRADS):
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets, reduction='sum')
        loss.backward()

        batch_g_all = []
        for _, batch_g in _module_batch_flatten_grads(model):
            batch_g_all.append(batch_g)
        grads = torch.cat(batch_g_all, dim=1).T  # (p, n)

    p, n = grads.shape
    if data_average:
        grads /= np.sqrt(n)
    assert vec.shape == (p,)
    gram = torch.matmul(grads.T, grads)  # (n, n)
    inv = torch.inverse(gram + torch.eye(n) * damping)  # (n, n)
    b = torch.matmul(inv, torch.matmul(grads.T, vec))

    return (vec - torch.matmul(grads, b)) / damping  # (p,)

