import torch
from .symmatrix import SymMatrix, Diag
from .matrices import SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_DIAG, HESSIAN, MatrixManager
from .mvp import power_method, conjugate_gradient_method

__all__ = [
    'hessian_eigenvalues',
    'hessian',
    'hessian_for_loss',
    'hessian_free'
]
_supported_shapes = [SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_DIAG]


def hessian_eigenvalues(
    model,
    loss_fn,
    data_loader=None,
    inputs=None,
    targets=None,
    top_n=1,
    max_iters=100,
    tol=1e-3,
    is_distributed=False,
    print_progress=False
):
    def hvp_fn(vec, x, y):
        model.zero_grad()
        loss = loss_fn(model(x), y)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
        return hvp(vec, grads, params)

    eigvals, eigvecs = power_method(hvp_fn,
                                    model,
                                    data_loader=data_loader,
                                    inputs=inputs,
                                    targets=targets,
                                    top_n=top_n,
                                    max_iters=max_iters,
                                    tol=tol,
                                    is_distributed=is_distributed,
                                    print_progress=print_progress)

    return eigvals, eigvecs


def hessian_free(
        model,
        loss_fn,
        b,
        data_loader=None,
        inputs=None,
        targets=None,
        init_x=None,
        damping=1e-3,
        max_iters=None,
        tol=1e-8,
        is_distributed=False,
        print_progress=False,
):
    def hvp_fn(vec, x, y):
        model.zero_grad()
        loss = loss_fn(model(x), y)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
        return hvp(vec, grads, params)

    return conjugate_gradient_method(hvp_fn,
                                     b,
                                     data_loader=data_loader,
                                     inputs=inputs,
                                     targets=targets,
                                     init_x=init_x,
                                     damping=damping,
                                     max_iters=max_iters,
                                     tol=tol,
                                     is_distributed=is_distributed,
                                     print_progress=print_progress)


def hvp(vec, grads, params):
    Hv = torch.autograd.grad(grads, inputs=params, grad_outputs=vec)
    return Hv


def hessian_for_loss(
    model,
    loss_fn,
    hessian_shapes,
    inputs=None,
    targets=None,
    data_loader=None,
    stats_name=None,
    is_distributed=False,
    all_reduce=False,
    is_master=True,
    matrix_manager=None,
):
    if isinstance(hessian_shapes, str):
        hessian_shapes = [hessian_shapes]
    # remove duplicates
    hessian_shapes = set(hessian_shapes)
    for hshape in hessian_shapes:
        assert hshape in _supported_shapes, f'Invalid hessian_shape: {hshape}. hessian_shape must be in {_supported_shapes}.'

    # setup matrix manager as needed
    if matrix_manager is None:
        matrix_manager = MatrixManager(model, HESSIAN)

    if data_loader is not None:
        device = next(model.parameters()).device
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets)
        matrix_manager.accumulate_matrices(
            stats_name, scale=1 / len(data_loader)
        )
    else:
        assert inputs is not None and targets is not None
        _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets)

    # reduce matrices
    if is_distributed:
        matrix_manager.reduce_matrices(stats_name, is_master, all_reduce)

    return matrix_manager


def _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets):
    model.zero_grad()
    loss = loss_fn(model(inputs), targets)
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]

    # full
    if SHAPE_FULL in hessian_shapes:
        full_hess = hessian(loss, params)
        setattr(model, 'hessian', SymMatrix(data=full_hess, device=device))
    else:
        full_hess = None

    if SHAPE_BLOCK_DIAG not in hessian_shapes \
            and SHAPE_DIAG not in hessian_shapes:
        return

    idx = 0
    for module in model.modules():
        w = getattr(module, 'weight', None)
        b = getattr(module, 'bias', None)
        params = [p for p in [w, b] if p is not None and p.requires_grad]
        if len(params) == 0:
            continue

        # module hessian
        if full_hess is None:
            m_hess = hessian(loss, params)
        else:
            m_numel = sum([p.numel() for p in params])
            m_hess = full_hess[idx:idx + m_numel, idx:idx + m_numel]
            idx += m_numel

        # block-diagonal
        if SHAPE_BLOCK_DIAG in hessian_shapes:
            setattr(module, 'hessian', SymMatrix(data=m_hess, device=device))

        # diagonal
        if SHAPE_DIAG in hessian_shapes:
            m_hess = torch.diag(m_hess)
            _idx = 0
            w_hess = b_hess = None
            if w is not None and w.requires_grad:
                w_numel = w.numel()
                w_hess = m_hess[_idx:_idx + w_numel].view_as(w)
                _idx += w_numel
            if b is not None and b.requires_grad:
                b_numel = b.numel()
                b_hess = m_hess[_idx:_idx + b_numel].view_as(b)
                _idx += b_numel
            diag = Diag(weight=w_hess, bias=b_hess, device=device)
            if hasattr(module, 'hessian'):
                module.hessian.diag = diag
            else:
                setattr(module, 'hessian', SymMatrix(diag=diag, device=device))


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/hessian.py
def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(
            output, inp, create_graph=True, allow_unused=allow_unused
        )
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = _gradient(
                    grad[j],
                    inputs[i:],
                    retain_graph=True,
                    create_graph=create_graph
                )[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/gradient.py
def _gradient(
    outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False
):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph
    )
    grads = [
        x if x is not None else torch.zeros_like(y) for x,
        y in zip(grads, inputs)
    ]
    return torch.cat([x.contiguous().view(-1) for x in grads])
