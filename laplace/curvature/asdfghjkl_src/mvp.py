import torch
import torch.distributed as dist

__all__ = [
    'power_method',
]


def power_method(
    mvp_fn,
    model,
    data_loader=None,
    inputs=None,
    targets=None,
    top_n=1,
    max_iters=100,
    tol=1e-3,
    is_distributed=False,
    print_progress=False,
    random_seed=None
):
    # adopted from https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
    assert top_n >= 1
    assert max_iters >= 1

    params = [p for p in model.parameters() if p.requires_grad]

    def _report(message):
        if print_progress:
            print(message)

    def _call_mvp(v):
        return mvp(
            mvp_fn,
            v,
            data_loader=data_loader,
            inputs=inputs,
            targets=targets,
            random_seed=random_seed,
            is_distributed=is_distributed
        )

    eigvals = []
    eigvecs = []
    for i in range(top_n):
        _report(f'start power iteration for lambda({i+1}).')
        vec = [torch.randn_like(p) for p in params]
        if is_distributed:
            vec = _flatten_parameters(vec)
            dist.broadcast(vec, src=0)
            vec = _unflatten_like_parameters(vec, params)

        eigval = 0
        last_eigval = 0
        # power iteration
        for j in range(max_iters):
            vec = _orthnormal(vec, eigvecs)
            Mv = _call_mvp(vec)
            eigval = _group_product(Mv, vec)
            if isinstance(eigval, torch.Tensor):
                eigval = eigval.item()
            if j > 0:
                diff = abs(eigval - last_eigval) / (abs(last_eigval) + 1e-6)
                _report(f'{j}/{max_iters} diff={diff}')
                if diff < tol:
                    break
            last_eigval = eigval
            vec = Mv
        eigvals.append(eigval)
        eigvecs.append(vec)

    # sort both in descending order
    eigvals, eigvecs = (list(t) for t in zip(*sorted(zip(eigvals, eigvecs))[::-1]))

    return eigvals, eigvecs


def mvp(
    mvp_fn,
    vec,
    data_loader=None,
    inputs=None,
    targets=None,
    random_seed=None,
    damping=None,
    is_distributed=False
):
    if random_seed:
        # for matrices that are not deterministic (e.g., fisher_mc)
        torch.manual_seed(random_seed)

    if data_loader is not None:
        Mv = _data_loader_mvp(mvp_fn, vec, data_loader)
    else:
        assert inputs is not None
        Mv = mvp_fn(vec, inputs, targets)

    if damping:
        Mv = _group_add(Mv, vec, damping)

    if is_distributed:
        Mv = _all_reduce_params(Mv)

    return Mv


def _data_loader_mvp(mvp_fn, vec, data_loader):
    device = vec[0].device
    Mv = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _Mv = mvp_fn(vec, inputs, targets)
        if not Mv:
            Mv = _Mv
        else:
            Mv = [mv.add(_mv) for mv, _mv in zip(Mv, _Mv)]

    Mv = [mv.div(len(data_loader)) for mv in Mv]

    return Mv


def _all_reduce_params(params):
    world_size = dist.get_world_size()
    # pack
    packed_tensor = _flatten_parameters(params)
    # all-reduce
    dist.all_reduce(packed_tensor)
    # unpack
    rst = _unflatten_like_parameters(packed_tensor.div(world_size), params)

    dist.barrier()

    return rst


def _flatten_parameters(params):
    vec = []
    for param in params:
        vec.append(param.flatten())
    return torch.cat(vec)


def _unflatten_like_parameters(vec, params):
    pointer = 0
    rst = []
    for param in params:
        numel = param.numel()
        rst.append(vec[pointer:pointer + numel].view_as(param))
        pointer += numel
    return rst


def _group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def _group_add(xs, ys, alpha=1.):
    return [x.add(y.mul(alpha)) for x, y in zip(xs, ys)]


def _normalization(v):
    s = _group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def _orthnormal(w, v_list):
    for v in v_list:
        w = _group_add(w, v, alpha=-_group_product(w, v))
    return _normalization(w)
