import torch
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX


def cleanup(module):
    for child in module.children():
        cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)


def Jacobians(model, data):
    # Jacobians are batch x output x params
    model = extend(model)
    to_stack = []
    for i in range(model.output_size):
        model.zero_grad()
        out = model(data)
        with backpack(BatchGrad()):
            if model.output_size > 1:
                out[:, i].sum().backward()
            else:
                out.sum().backward()
            to_cat = []
            for param in model.parameters():
                to_cat.append(param.grad_batch.detach().reshape(data.shape[0], -1))
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)
        if i == 0:
            f = out.detach()

    # cleanup
    model.zero_grad()
    CTX.remove_hooks()
    cleanup(model)
    if model.output_size > 1:
        return torch.stack(to_stack, dim=2).transpose(1, 2), f
    else:
        return Jk.unsqueeze(-1).transpose(1, 2), f


def LLJacobians(model, data):
    f, phi = model.forward_with_features(data)
    bsize = len(data)
    output_size = f.shape[-1]

    # calculate Jacobians using the feature vector 'phi'
    identity = torch.eye(output_size, device=data.device).unsqueeze(0).tile(bsize, 1, 1)
    # Jacobians are batch x output x params
    Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
    if model.last_layer.bias is not None:
        Js = torch.cat([Js, identity], dim=2)

    return Js, f
