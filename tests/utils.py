import torch
import numpy as np


def get_psd_matrix(dim):
    X = torch.randn(dim, dim*3)
    return X @ X.T / (dim * 3)

def grad(model):
    return torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()

def jacobians_naive(model, data):
    model.zero_grad()
    f = model(data)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = (i != (f.shape[0] - 1) or j != (f.shape[1] - 1))
                f[i, j].backward(retain_graph=rg)
                Jij = grad(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = (i != (f.shape[0] - 1))
            f[i].backward(retain_graph=rg)
            jacs = grad(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs).transpose(1, 2)
    return Jacs.detach(), f.detach()


def jacobians_naive_aug(model, data):
    model.zero_grad()
    f = model(data).mean(dim=1)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = (i != (f.shape[0] - 1) or j != (f.shape[1] - 1))
                f[i, j].backward(retain_graph=rg)
                Jij = grad(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = (i != (f.shape[0] - 1))
            f[i].backward(retain_graph=rg)
            jacs = grad(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs).transpose(1, 2)
    return Jacs.detach(), f.detach()

    
def gradients_naive_aug(model, data, targets, likelihood):
    if likelihood == 'classification':
        lossf = torch.nn.CrossEntropyLoss(reduction='sum')
    elif likelihood == 'regression':
        lossf = torch.nn.MSELoss(reduction='sum')
    output = model(data).mean(dim=1)
    grads = list()
    loss = torch.zeros(0)
    for i in range(len(targets)):
        model.zero_grad()
        lossi = lossf(output[i].unsqueeze(0), targets[i].unsqueeze(0)) 
        lossi.backward(retain_graph=True)
        loss += lossi.detach()
        grads.append(grad(model))
    return torch.stack(grads), loss
 