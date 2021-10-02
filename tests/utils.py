import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def toy_regression_dataset(sigma, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    # create simple sinusoid data set
    X_train = (torch.rand(n_train) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.linspace(-5, 13, n_test).unsqueeze(-1)  # +-5 on top of the training X-range

    return X_train, y_train, train_loader, X_test


def toy_model(train_loader: DataLoader, n_epochs=100, fit=True):
    model = torch.nn.Sequential(torch.nn.Linear(1, 50),
                                torch.nn.Tanh(),
                                torch.nn.Linear(50, 1))
    if fit:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
        for i in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    return model


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
