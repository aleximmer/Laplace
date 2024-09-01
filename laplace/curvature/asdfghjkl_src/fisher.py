from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .core import extend
from .matrices import (
    COV,
    FISHER_EXACT,
    FISHER_MC,
    SHAPE_BLOCK_DIAG,
    SHAPE_DIAG,
    SHAPE_FULL,
    SHAPE_KRON,
    MatrixManager,
)
from .operations import (
    OP_BATCH_GRADS,
    OP_COV_DIAG,
    OP_COV_KRON,
    OP_COV_UNIT_WISE,
)
from .symmatrix import Diag, Kron, SymMatrix, UnitWise
from .utils import disable_param_grad

_SHAPE_TO_OP = {
    SHAPE_FULL: OP_BATCH_GRADS,  # full
    SHAPE_BLOCK_DIAG: OP_BATCH_GRADS,  # block-diagonal
    SHAPE_KRON: OP_COV_KRON,  # Kronecker-factored
    SHAPE_DIAG: OP_COV_DIAG,  # diagonal
}

_COV_FULL = "cov_full"
_COV_BLOCK_DIAG = "cov_block_diag"

__all__ = [
    "fisher_for_cross_entropy",
]

_supported_types = [FISHER_EXACT, FISHER_MC, COV]
_supported_shapes = [SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG]


def fisher_for_cross_entropy(
    model,
    fisher_types,
    fisher_shapes,
    inputs=None,
    targets=None,
    data_loader=None,
    stats_name=None,
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
        assert ftype in _supported_types, (
            f"Invalid fisher_type: {ftype}. "
            f"fisher_type must be in {_supported_types}."
        )
    for fshape in fisher_shapes:
        assert fshape in _supported_shapes, (
            f"Invalid fisher_shape: {fshape}. "
            f"fisher_shape must be in {_supported_shapes}."
        )

    zero_fisher(model, fisher_types)

    # setup operations for mammoth_utils.autograd.extend
    op_names = [_SHAPE_TO_OP[shape] for shape in fisher_shapes]

    # setup matrix manager as needed
    if matrix_manager is None:
        matrix_manager = MatrixManager(model, fisher_types)

    kwargs = dict(
        compute_full_fisher=SHAPE_FULL in fisher_shapes,
        compute_block_diag_fisher=SHAPE_BLOCK_DIAG in fisher_shapes,
        n_mc_samples=n_mc_samples,
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
    else:
        # compute fisher for a single batch
        assert inputs is not None
        with extend(model, op_names):
            _fisher_for_cross_entropy(model, fisher_types, inputs, targets, **kwargs)

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


def _fisher_for_cross_entropy(
    model,
    fisher_types,
    inputs,
    targets=None,
    compute_full_fisher=False,
    compute_block_diag_fisher=False,
    n_mc_samples=1,
):
    logits = model(inputs)
    log_probs = F.log_softmax(logits, dim=1)
    probs = None

    def loss_and_backward(target):
        model.zero_grad(set_to_none=True)
        loss = F.nll_loss(log_probs, target, reduction="sum")
        loss.backward(retain_graph=True)
        if compute_full_fisher:
            _full_covariance(model)
        if compute_block_diag_fisher:
            _block_diag_covariance(model)

    if FISHER_MC in fisher_types:
        probs = F.softmax(logits, dim=1)
        _fisher_mc(loss_and_backward, model, probs, n_mc_samples)

    if FISHER_EXACT in fisher_types:
        if probs is None:
            probs = F.softmax(logits, dim=1)
        _fisher_exact(loss_and_backward, model, probs)

    if COV in fisher_types:
        assert (
            targets is not None
        ), "targets must be specified for computing covariance."
        _covariance(loss_and_backward, model, targets)


def _module_batch_grads(model):
    rst = []
    for module in model.modules():
        operation = getattr(module, "operation", None)
        if operation is None:
            continue
        batch_grads = operation.get_op_results()[OP_BATCH_GRADS]
        rst.append((module, batch_grads))
    return rst


def _module_batch_flatten_grads(model):
    rst = []
    for module, batch_grads in _module_batch_grads(model):
        batch_flatten_grads = torch.cat(
            [g.flatten(start_dim=1) for g in batch_grads.values()], dim=1
        )
        rst.append((module, batch_flatten_grads))
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


def _fisher_mc(loss_and_backward, model, probs, n_mc_samples=1):
    dist = torch.distributions.Categorical(probs)
    _targets = dist.sample(torch.Size((n_mc_samples,)))
    for i in range(n_mc_samples):
        loss_and_backward(_targets[i])
        _register_fisher(model, FISHER_MC, scale=1 / n_mc_samples, accumulate=True)


def _fisher_exact(loss_and_backward, model, probs):
    _, n_classes = probs.shape
    probs, _targets = torch.sort(probs, dim=1, descending=True)
    sqrt_probs = torch.sqrt(probs)
    for i in range(n_classes):
        with _grads_scale(model, sqrt_probs[:, i]):
            loss_and_backward(_targets[:, i])
        _register_fisher(model, FISHER_EXACT, accumulate=True)


def _covariance(loss_and_backward, model, targets):
    with disable_param_grad(model):
        loss_and_backward(targets)
    _register_fisher(model, COV)


@contextmanager
def _grads_scale(model, scale):
    for module in model.modules():
        operation = getattr(module, "operation", None)
        if operation is None:
            continue
        operation.grads_scale = scale

    yield

    for module in model.modules():
        operation = getattr(module, "operation", None)
        if operation is None:
            continue
        operation.grads_scale = None


def _register_fisher(model, fisher_type, scale=1.0, accumulate=False):
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
        operation = getattr(module, "operation", None)
        if operation is None:
            continue
        op_results = operation.get_op_results()
        kron = diag = unit = None
        if OP_COV_KRON in op_results:
            rst = op_results[OP_COV_KRON]
            kron = Kron(rst["A"], rst["B"], device=device)
        if OP_COV_DIAG in op_results:
            rst = op_results[OP_COV_DIAG]
            diag = Diag(rst.get("weight", None), rst.get("bias", None), device=device)
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
            accumulate=accumulate,
        )

    # move full fisher
    _accumulate_fisher(
        model, _COV_FULL, fisher_type, scale=scale, accumulate=accumulate
    )


def _accumulate_fisher(
    module,
    data_src_attr,
    dst_attr,
    kron=None,
    diag=None,
    unit=None,
    scale=1.0,
    accumulate=False,
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
            # do not accumulate kron.A
            dst_fisher.kron.A = fisher.kron.A
        setattr(module, dst_attr, dst_fisher)

    if data is not None:
        delattr(module, data_src_attr)
